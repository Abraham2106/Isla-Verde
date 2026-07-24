#!/usr/bin/env python3
"""ISLA VERDE v3.0 — Pipeline clásico: red ICE 230kV → QUBO/Ising.

Construye el grafo de transmisión 230 kV del ICE desde CSV oficiales,
extrae instancias NISQ (MVP-8/STD-12/LARGE-16), formula Max-Cut como
QUBO e Ising, calcula líneas base clásicas y exporta JSON por instancia.

Uso: python modelador_red.py --data-dir ./data --out-dir ./scratch
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cvxpy as cp
import h3
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger("isla_verde.modelador_red")


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DEFAULT_INSTANCES: Mapping[str, tuple[str, ...]] = {
    "mvp8": (
        "Arenal", "Cañas", "Garabito", "Barranca",
        "La Garita", "La Caja", "Lindora", "Belen",
    ),
    "std12": (
        "Arenal", "Cañas", "Garabito", "Barranca",
        "La Garita", "La Caja", "Lindora", "Belen",
        "Coyol", "San Miguel", "El Este", "Tejar",
    ),
    "large16": (
        "Arenal", "Cañas", "Garabito", "Barranca",
        "La Garita", "La Caja", "Lindora", "Belen",
        "Coyol", "San Miguel", "El Este", "Tejar",
        "Tarbaca", "Higuito", "Coronado", "Ribera",
    ),
}

# Corredor de respaldo con longitudes reales medidas.
# Lindora-La Caja: dos circuitos paralelos con pesos sumados.
FALLBACK_EDGES_M: tuple[tuple[str, str, float], ...] = (
    ("Arenal", "Garabito", 58158.6),
    ("Arenal", "Lindora", 122582.0),
    ("Garabito", "Cañas", 61285.8),
    ("Barranca", "Garabito", 7949.7),
    ("Barranca", "La Garita", 41536.5),
    ("La Garita", "Lindora", 20828.0),
    ("Lindora", "La Caja", 5916.9),
    ("La Caja", "Belen", 3733.1),
)
FALLBACK_PARALLEL_LENGTHS_M: tuple[float, float] = (5860.3, 5973.5)


@dataclass(frozen=True)
class Config:
    """Configuración inmutable del pipeline."""
    data_dir: Path = Path("/workspace/knowledge/")
    out_dir: Path = Path("/workspace/scratch/")
    seed: int = 42
    voltage: int = 230
    weight_numerator: float = 1.0e5
    instances: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_INSTANCES)
    )
    critical_node: str = "La Caja"
    generator_anchor: str = "Arenal"
    generator_anchors: tuple[str, ...] = ("Arenal", "Garabito")
    generator_shares: tuple[float, ...] = (0.6, 0.4)
    h3_resolution: int = 5
    alpha: float | None = None
    beta: float | None = None
    subs_filename: str = "Subestaciones.csv"
    lines_filename: str = "LineasDeTransmision.csv"


# ---------------------------------------------------------------------------
# Normalización de nombres y parseo de circuitos
# ---------------------------------------------------------------------------

_PAREN_SUFFIX_RE = re.compile(r"\s*\(.*\)\s*$")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_diacritics(s: str) -> str:
    """Quita diacríticos via NFD + eliminación de combining marks."""
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize_name(s: str) -> str:
    """Normalización canónica para vincular subestaciones entre CSVs."""
    s = _strip_diacritics(str(s).strip().lower())
    s = _PAREN_SUFFIX_RE.sub("", s)
    s = _TRAILING_DIGITS_RE.sub("", s).strip()
    return _WHITESPACE_RE.sub(" ", s)


def normalize_province(s: str) -> str:
    """Normaliza nombre de provincia."""
    return _WHITESPACE_RE.sub(
        " ", _strip_diacritics(str(s).strip().lower())
    ).title()


def build_name_map(official_names: list[str]) -> dict[str, str]:
    """Mapeo nombre_normalizado → nombre_oficial con alias 'la X' → 'X'."""
    name_map: dict[str, str] = {}
    for official in official_names:
        key = normalize_name(official)
        if key in name_map and name_map[key] != official:
            logger.warning(
                "Colisión de normalización: %r y %r → %r",
                name_map[key], official, key,
            )
        name_map[key] = official

    for key in sorted(name_map):
        if key.startswith("la "):
            alias = key[3:]
            if alias and alias not in name_map:
                name_map[alias] = name_map[key]
    return name_map


@dataclass(frozen=True)
class CircuitResolution:
    """Resultado de parsear un nombre de circuito."""
    raw: str
    status: str  # "ok", "no-hyphen", "unknown-endpoint"
    endpoints: tuple[str, str] | None = None
    unknown_endpoints: tuple[str, ...] = ()


def parse_circuit(raw: str, name_map: Mapping[str, str]) -> CircuitResolution:
    """Parsea 'Sub1-Sub2' resolviendo ambos extremos contra name_map."""
    raw = str(raw).strip()
    if "-" not in raw:
        return CircuitResolution(raw=raw, status="no-hyphen")

    best_unknown: tuple[str, ...] | None = None
    for pos, ch in enumerate(raw):
        if ch != "-":
            continue
        left = normalize_name(raw[:pos])
        right = normalize_name(raw[pos + 1:])
        if not left or not right:
            continue
        unknown = tuple(k for k in (left, right) if k not in name_map)
        if not unknown:
            return CircuitResolution(
                raw=raw, status="ok",
                endpoints=(name_map[left], name_map[right]),
            )
        if best_unknown is None or len(unknown) < len(best_unknown):
            best_unknown = unknown

    return CircuitResolution(
        raw=raw, status="unknown-endpoint",
        unknown_endpoints=best_unknown or (),
    )


def mercator_to_latlon(x: float, y: float) -> tuple[float, float]:
    """Convierte coordenadas Web Mercator a lat/lon."""
    R = 6_378_137.0
    lon = math.degrees(x / R)
    lat = math.degrees(math.atan(math.sinh(y / R)))
    return lat, lon


# ---------------------------------------------------------------------------
# Álgebra QUBO/Ising
# ---------------------------------------------------------------------------


def build_qubo(
    edges: Sequence[tuple[int, int, float]],
    balances: np.ndarray,
    alpha: float,
    beta: float,
    critical_idx: int,
    anchor_idx: int,
) -> tuple[np.ndarray, float]:
    """Construye matriz Q del QUBO: H_cut + α·H_balance + β·H_critical."""
    n = balances.shape[0]
    Q = np.zeros((n, n), dtype=np.float64)

    # H_cut = -Σ W(x_i + x_j - 2·x_i·x_j): minimizar maximiza el corte
    for i, j, w in edges:
        Q[i, i] -= w
        Q[j, j] -= w
        Q[i, j] += 2.0 * w

    # H_balance = α·(Σ B_i·x_i)²
    for i in range(n):
        Q[i, i] += alpha * balances[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * alpha * balances[i] * balances[j]

    # H_critical = β·(x_c - x_g)²
    c, g = critical_idx, anchor_idx
    lo, hi = (c, g) if c < g else (g, c)
    Q[c, c] += beta
    Q[g, g] += beta
    Q[lo, hi] -= 2.0 * beta

    return Q, 0.0


def qubo_to_ising(
    Q: np.ndarray, qubo_offset: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convierte QUBO a Ising via x = (1-s)/2."""
    diag = np.diag(Q).copy()
    upper = np.triu(Q, k=1)
    h_vec = -diag / 2.0 - (upper.sum(axis=1) + upper.sum(axis=0)) / 4.0
    J_upper = upper / 4.0
    offset = qubo_offset + diag.sum() / 2.0 + upper.sum() / 4.0
    return h_vec, J_upper, offset


def qubo_energies(
    Q: np.ndarray, offset: float, x_batch: np.ndarray,
) -> np.ndarray:
    """E(x) = x^T·Q·x + offset para lote de filas binarias (m, n)."""
    x = x_batch.astype(np.float64)
    return ((x @ Q) * x).sum(axis=1) + offset


def ising_energies(
    h: np.ndarray, J: np.ndarray, offset: float, s_batch: np.ndarray,
) -> np.ndarray:
    """E(s) = h·s + s^T·J·s + offset para lote de espines (m, n)."""
    s = s_batch.astype(np.float64)
    return ((s @ J) * s).sum(axis=1) + s @ h + offset


def enumerate_bitstrings(n: int, fix_first_zero: bool = False) -> np.ndarray:
    """Genera todas las 2^n (o 2^(n-1) si fix_first_zero) asignaciones."""
    free = n - 1 if fix_first_zero else n
    ints = np.arange(1 << free, dtype=np.uint64)
    bits = (
        (ints[:, None] >> np.arange(free, dtype=np.uint64)[None, :]) & 1
    ).astype(np.float64)
    if fix_first_zero:
        bits = np.hstack([np.zeros((bits.shape[0], 1)), bits])
    return bits


def cut_values(
    edges: Sequence[tuple[int, int, float]], x_batch: np.ndarray,
) -> np.ndarray:
    """Corte ponderado Σ W(x_i + x_j - 2·x_i·x_j) por lote."""
    x = x_batch.astype(np.float64)
    total = np.zeros(x.shape[0], dtype=np.float64)
    for i, j, w in edges:
        total += w * (x[:, i] + x[:, j] - 2.0 * x[:, i] * x[:, j])
    return total


def bitstring(x: np.ndarray) -> str:
    """Vector binario como cadena '01...'."""
    return "".join(str(int(round(b))) for b in x)


def build_maxcut_ising(
    edges: Sequence[tuple[int, int, float]], n: int,
) -> tuple[np.ndarray, float]:
    """Ising de Max-Cut puro (h=0, J=w/2) para benchmark QAOA."""
    J = np.zeros((n, n))
    for i, j, w in edges:
        J[i, j] += w / 2.0
    offset = -float(sum(w for _, _, w in edges)) / 2.0
    return J, offset


class CalibrationError(RuntimeError):
    """La restricción crítica no se pudo imponer duplicando β."""


# ---------------------------------------------------------------------------
# Modelo de instancia
# ---------------------------------------------------------------------------


@dataclass
class InstanceModel:
    """Modelo completo de una instancia NISQ."""
    tier: str
    variable_order: list[str]
    graph: nx.Graph
    edges_idx: list[tuple[int, int, float]]
    balances: np.ndarray
    alpha: float
    beta: float
    q_matrix: np.ndarray
    qubo_offset: float
    h_vec: np.ndarray
    j_upper: np.ndarray
    ising_offset: float
    maxcut_j: np.ndarray
    maxcut_offset: float
    h_total_energies: np.ndarray
    h_total_optimum: float
    h_total_argmin: np.ndarray
    critical_satisfied: bool
    baselines: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------


class ICEPowerGridModeler:
    """Pipeline: CSV → grafo → QUBO/Ising → baselines → export."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.source: str = "real"
        self.graph: nx.Graph = nx.Graph()
        self.excluded: list[CircuitResolution] = []
        self.instances: dict[str, InstanceModel] = {}
        self.instance_errors: dict[str, str] = {}
        self.checks: list[tuple[int, str, str, bool | None, str]] = []

    # -- Ingesta y grafo ------------------------------------------------------

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Lee los CSV oficiales del ICE. Retorna None si no se encuentran."""
        subs_path = self.cfg.data_dir / self.cfg.subs_filename
        lines_path = self.cfg.data_dir / self.cfg.lines_filename
        try:
            subs = pd.read_csv(subs_path, encoding="utf-8-sig")
            lines = pd.read_csv(lines_path, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            logger.warning("Fallo al leer CSV (%s: %s)", type(exc).__name__, exc)
            return None

        required_subs = {"X", "Y", "Subestacio", "Provincia"}
        required_lines = {"Voltaje", "Circuito", "Shape__Length"}
        missing_s = required_subs - set(subs.columns)
        missing_l = required_lines - set(lines.columns)
        if missing_s or missing_l:
            raise ValueError(
                f"Columnas requeridas ausentes. "
                f"{self.cfg.subs_filename}: esperado {sorted(required_subs)}, "
                f"encontrado {sorted(subs.columns)}; "
                f"{self.cfg.lines_filename}: esperado {sorted(required_lines)}, "
                f"encontrado {sorted(lines.columns)}"
            )
        return subs, lines

    def build_graph(self, subs: pd.DataFrame, lines: pd.DataFrame) -> None:
        """Construye grafo 230kV: nodos=subestaciones, aristas=líneas."""
        cfg = self.cfg
        name_map = build_name_map(subs["Subestacio"].tolist())
        graph = nx.Graph()

        for _, row in subs.iterrows():
            lat, lon = mercator_to_latlon(float(row["X"]), float(row["Y"]))
            graph.add_node(
                str(row["Subestacio"]),
                province=normalize_province(row["Provincia"]),
                x_merc=float(row["X"]), y_merc=float(row["Y"]),
                lat=lat, lon=lon,
            )

        filtered = lines[lines["Voltaje"] == cfg.voltage]
        logger.info(
            "Filtradas %d/%d filas a %d kV",
            len(filtered), len(lines), cfg.voltage,
        )

        for _, row in filtered.iterrows():
            res = parse_circuit(row["Circuito"], name_map)
            if res.status != "ok":
                self.excluded.append(res)
                continue
            u, v = res.endpoints  # type: ignore[misc]
            length_m = float(row["Shape__Length"])
            weight = cfg.weight_numerator / length_m

            if graph.has_edge(u, v):
                # Circuitos paralelos: capacidades se suman
                data = graph[u][v]
                data["weight"] += weight
                data["circuits"].append(res.raw)
                data["lengths_m"].append(length_m)
                data["parallel"] = True
            else:
                graph.add_edge(
                    u, v, weight=weight, circuits=[res.raw],
                    lengths_m=[length_m], parallel=False,
                )

        # Quitar subestaciones sin circuito a este nivel de tensión
        isolated = [n for n in graph.nodes if graph.degree(n) == 0]
        graph.remove_nodes_from(isolated)

        # Longitud equivalente: numerador / peso sumado
        for _, _, data in graph.edges(data=True):
            data["length_m"] = cfg.weight_numerator / data["weight"]

        self.graph = graph
        self._check_graph_invariants()

    def _check_graph_invariants(self) -> None:
        """Verifica invariantes esperadas del grafo completo."""
        g = self.graph
        actual = (
            g.number_of_nodes(), g.number_of_edges(),
            nx.is_connected(g) if g.number_of_nodes() else False,
            sum(1 for _, _, d in g.edges(data=True) if d["parallel"]),
            len(self.excluded),
        )
        expected = (46, 58, True, 3, 4)
        ok = actual == expected
        if not ok:
            logger.warning(
                "Deriva de invariantes: esperado %s, obtenido %s",
                expected, actual,
            )
        self.checks.append((
            8, "Invariantes del grafo (46/58/conexo/3/4)",
            "soft", ok, f"esperado {expected}, obtenido {actual}",
        ))

    def build_fallback_graph(self) -> None:
        """Grafo sintético de respaldo basado en corredor ICE 230kV."""
        logger.warning("[FALLBACK] Usando proxy sintético del corredor ICE 230kV")
        self.source = "fallback"
        cfg = self.cfg
        graph = nx.Graph()

        for u, v, length_m in FALLBACK_EDGES_M:
            if (u, v) == ("Lindora", "La Caja"):
                l1, l2 = FALLBACK_PARALLEL_LENGTHS_M
                weight = cfg.weight_numerator / l1 + cfg.weight_numerator / l2
                graph.add_edge(
                    u, v, weight=weight,
                    circuits=["Lindora-La Caja", "Lindora-La Caja2"],
                    lengths_m=[l1, l2], parallel=True,
                    length_m=cfg.weight_numerator / weight,
                )
            else:
                graph.add_edge(
                    u, v, weight=cfg.weight_numerator / length_m,
                    circuits=[f"{u}-{v}"], lengths_m=[length_m],
                    parallel=False, length_m=length_m,
                )

        pos = nx.spring_layout(graph, seed=cfg.seed)
        for node in graph.nodes:
            graph.nodes[node].update(
                province="Sintética", lat=0.0, lon=0.0,
                x_merc=float(pos[node][0]), y_merc=float(pos[node][1]),
            )
        self.graph = graph
        self.checks.append((
            8, "Invariantes del grafo (46/58/conexo/3/4)",
            "soft", None, "omitido: proxy de respaldo activo",
        ))

    # -- Extracción de instancias ---------------------------------------------

    def extract_instances(self) -> dict[str, nx.Graph]:
        """Extrae subgrafos inducidos para cada tier de instancia."""
        subgraphs: dict[str, nx.Graph] = {}
        for tier, node_set in self.cfg.instances.items():
            missing = [n for n in node_set if n not in self.graph]
            if missing:
                suggestions = {
                    n: difflib.get_close_matches(n, list(self.graph.nodes), n=3)
                    for n in missing
                }
                msg = f"nodos no resueltos {missing}; cercanos: {suggestions}"
                self.instance_errors[tier] = msg
                logger.error("Instancia %s omitida: %s", tier, msg)
                continue
            sub = nx.Graph(self.graph.subgraph(node_set))
            if not nx.is_connected(sub):
                components = [sorted(c) for c in nx.connected_components(sub)]
                raise AssertionError(
                    f"Instancia {tier} desconectada: {components}"
                )
            subgraphs[tier] = sub
            logger.info(
                "Instancia %s: %d nodos, %d aristas, conexa",
                tier, sub.number_of_nodes(), sub.number_of_edges(),
            )
        return subgraphs

    # -- Balance de potencia sintético ----------------------------------------

    def assign_balances(self, nodes: Sequence[str]) -> dict[str, float]:
        """Asigna balance generación-demanda sintético con suma cero exacta."""
        anchors = [a for a in self.cfg.generator_anchors if a in nodes]
        if not anchors:
            raise ValueError(
                f"Ningún ancla generadora de {self.cfg.generator_anchors} "
                f"presente en {sorted(nodes)}"
            )
        shares = np.array(
            [s for a, s in zip(self.cfg.generator_anchors,
                               self.cfg.generator_shares) if a in nodes],
            dtype=np.float64,
        )
        shares /= shares.sum()

        load_nodes = sorted(n for n in nodes if n not in anchors)
        draws = self.rng.uniform(50.0, 150.0, size=len(load_nodes))
        total_load = float(draws.sum())

        balances = {n: -float(d) for n, d in zip(load_nodes, draws)}
        for anchor, share in zip(anchors, shares):
            balances[anchor] = float(share) * total_load
        balances[load_nodes[-1]] -= math.fsum(balances.values())
        assert abs(math.fsum(balances.values())) < 1e-9
        return balances

    # -- Formulación QUBO/Ising -----------------------------------------------

    def formulate_instance(self, tier: str, subgraph: nx.Graph) -> InstanceModel:
        """Formula QUBO/Ising con auto-calibración de β."""
        cfg = self.cfg
        variable_order = sorted(subgraph.nodes)
        index = {name: i for i, name in enumerate(variable_order)}

        edges_idx = sorted(
            (min(index[u], index[v]), max(index[u], index[v]),
             float(d["weight"]))
            for u, v, d in subgraph.edges(data=True)
        )
        balance_map = self.assign_balances(variable_order)
        balances = np.array([balance_map[n] for n in variable_order])

        total_weight = float(sum(w for _, _, w in edges_idx))
        sum_b2 = float((balances ** 2).sum())
        alpha = (cfg.alpha if cfg.alpha is not None
                 else total_weight / (sum_b2 / 4.0))
        beta = cfg.beta if cfg.beta is not None else 2.0 * total_weight

        critical_idx = index[cfg.critical_node]
        anchor_idx = index[cfg.generator_anchor]
        n = len(variable_order)
        all_states = enumerate_bitstrings(n)

        # Auto-calibración: duplicar β hasta satisfacer restricción crítica
        MAX_DOUBLINGS = 3
        satisfied = False
        Q = np.zeros((0, 0))
        q_offset = 0.0
        energies = np.zeros(0)
        argmin_x = np.zeros(0)

        for attempt in range(1 + MAX_DOUBLINGS):
            Q, q_offset = build_qubo(
                edges_idx, balances, alpha, beta, critical_idx, anchor_idx,
            )
            energies = qubo_energies(Q, q_offset, all_states)
            argmin_x = all_states[int(np.argmin(energies))]
            satisfied = bool(argmin_x[critical_idx] == argmin_x[anchor_idx])
            if satisfied:
                break
            if attempt < MAX_DOUBLINGS:
                logger.warning(
                    "[%s] restriccion critica violada con beta=%.6g, "
                    "duplicando (intento %d/%d)",
                    tier, beta, attempt + 1, MAX_DOUBLINGS,
                )
                beta *= 2.0

        if not satisfied:
            raise CalibrationError(
                f"[{tier}] x_c == x_g insatisfecha tras {MAX_DOUBLINGS} "
                f"duplicaciones de beta (beta={beta:.6g}, alpha={alpha:.6g})"
            )

        h_vec, j_upper, ising_offset = qubo_to_ising(Q, q_offset)
        mc_j, mc_offset = build_maxcut_ising(edges_idx, n)

        return InstanceModel(
            tier=tier, variable_order=variable_order, graph=subgraph,
            edges_idx=edges_idx, balances=balances, alpha=alpha, beta=beta,
            q_matrix=Q, qubo_offset=q_offset,
            h_vec=h_vec, j_upper=j_upper, ising_offset=ising_offset,
            maxcut_j=mc_j, maxcut_offset=mc_offset,
            h_total_energies=energies,
            h_total_optimum=float(energies.min()),
            h_total_argmin=argmin_x,
            critical_satisfied=satisfied,
        )

    # -- Líneas base clásicas -------------------------------------------------

    def compute_baselines(self, model: InstanceModel) -> None:
        """Calcula fuerza bruta, greedy y Goemans-Williamson."""
        n = len(model.variable_order)
        edges = model.edges_idx

        # Fuerza bruta sobre 2^(n-1), x_0 fijo en 0
        half = enumerate_bitstrings(n, fix_first_zero=True)
        cuts = cut_values(edges, half)
        best_idx = int(np.argmax(cuts))
        brute_cut, brute_x = float(cuts[best_idx]), half[best_idx]

        greedy_cut, greedy_x = self._greedy_maxcut(edges, n)
        gw_result = self._goemans_williamson(edges, n, roundings=50)

        model.baselines = {
            "maxcut": {
                "brute_force": {
                    "cut": brute_cut, "bitstring": bitstring(brute_x),
                },
                "greedy": {
                    "cut": greedy_cut, "bitstring": bitstring(greedy_x),
                },
                "goemans_williamson": gw_result,
            },
            "h_total": {
                "optimum_energy": model.h_total_optimum,
                "bitstring": bitstring(model.h_total_argmin),
                "critical_constraint_satisfied": model.critical_satisfied,
            },
        }
        logger.info(
            "[%s] baselines: brute=%.4f greedy=%.4f GW=%.4f | H_total=%.4f",
            model.tier, brute_cut, greedy_cut, gw_result["cut"],
            model.h_total_optimum,
        )

    def _greedy_maxcut(
        self, edges: Sequence[tuple[int, int, float]], n: int,
    ) -> tuple[float, np.ndarray]:
        """Greedy: inicio aleatorio + mejor flip individual."""
        x = self.rng.integers(0, 2, size=n).astype(np.float64)
        incident: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i, j, w in edges:
            incident[i].append((j, w))
            incident[j].append((i, w))

        for _ in range(10 * (1 << n)):
            gains = np.array([
                sum(w if x[i] == x[j] else -w for j, w in incident[i])
                for i in range(n)
            ])
            best = int(np.argmax(gains))
            if gains[best] <= 1e-12:
                break
            x[best] = 1.0 - x[best]

        if x[0] == 1.0:
            x = 1.0 - x
        return float(cut_values(edges, x[None, :])[0]), x

    def _goemans_williamson(
        self, edges: Sequence[tuple[int, int, float]], n: int,
        roundings: int,
    ) -> dict[str, Any]:
        """Goemans-Williamson via relajación SDP + redondeo aleatorio."""
        gram = cp.Variable((n, n), symmetric=True)
        constraints = [gram >> 0, cp.diag(gram) == 1]
        objective = cp.Maximize(
            sum(w * (1 - gram[i, j]) for i, j, w in edges) / 2
        )
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        if gram.value is None:
            raise RuntimeError(
                f"SDP sin solución (status: {problem.status})"
            )

        G = np.asarray((gram.value + gram.value.T) / 2.0)
        eigvals, eigvecs = np.linalg.eigh(G)
        vectors = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))

        best_cut, best_x = -np.inf, np.zeros(n)
        for _ in range(roundings):
            hyperplane = self.rng.standard_normal(n)
            signs = np.sign(vectors @ hyperplane)
            signs[signs == 0] = 1.0
            x = (1.0 - signs) / 2.0
            if x[0] == 1.0:
                x = 1.0 - x
            cut = float(cut_values(edges, x[None, :])[0])
            if cut > best_cut:
                best_cut, best_x = cut, x

        return {
            "cut": best_cut, "bitstring": bitstring(best_x),
            "roundings": roundings,
        }

    # -- Capa geoespacial H3 --------------------------------------------------

    def build_h3_layer(self) -> dict[str, Any] | None:
        """Agrega subestaciones por celda hexagonal H3."""
        if self.source == "fallback":
            logger.warning("Fallback sin coordenadas reales: H3 omitido")
            return None

        balances = self.assign_balances(sorted(self.graph.nodes))

        cell_members: dict[str, list[str]] = {}
        for node in sorted(self.graph.nodes):
            data = self.graph.nodes[node]
            cell = str(h3.latlng_to_cell(
                data["lat"], data["lon"], self.cfg.h3_resolution,
            ))
            cell_members.setdefault(cell, []).append(node)

        node_cell = {n: c for c, ms in cell_members.items() for n in ms}
        super_edges: dict[tuple[str, str], dict[str, Any]] = {}
        for u, v, data in self.graph.edges(data=True):
            cu, cv = node_cell[u], node_cell[v]
            if cu == cv:
                continue
            key = (min(cu, cv), max(cu, cv))
            entry = super_edges.setdefault(
                key, {"weight": 0.0, "circuits": []},
            )
            entry["weight"] += float(data["weight"])
            entry["circuits"].extend(data["circuits"])

        return {
            "h3_resolution": self.cfg.h3_resolution,
            "note": (
                "Propuesta de escalado. Superaristas provienen exclusivamente "
                "de líneas 230 kV cruzando celdas; adyacencia H3 NO es "
                "conexión eléctrica. Valores B son sintéticos."
            ),
            "supernodes": [
                {
                    "cell": cell, "members": members,
                    "B": math.fsum(balances[m] for m in members),
                    "synthetic": True,
                }
                for cell, members in sorted(cell_members.items())
            ],
            "superedges": [
                {"u": u, "v": v,
                 "weight": e["weight"], "circuits": e["circuits"]}
                for (u, v), e in sorted(super_edges.items())
            ],
        }

    # -- Exportación JSON -----------------------------------------------------

    def export(self, h3_layer: dict[str, Any] | None) -> list[Path]:
        """Exporta JSON por instancia, grafo completo e índice."""
        out = self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": self.source,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.cfg.seed,
            "weight_rule": "inverse_length_proxy",
            "excluded_circuits": [r.raw for r in self.excluded],
            "library_versions": {
                "pandas": pd.__version__,
                "networkx": nx.__version__,
                "numpy": np.__version__,
            },
        }
        written: list[Path] = []

        for tier, model in self.instances.items():
            index = {name: i for i, name in enumerate(model.variable_order)}
            n = len(model.variable_order)

            payload = {
                "metadata": metadata,
                "variable_order": model.variable_order,
                "nodes": [
                    {
                        "name": name,
                        "province": self.graph.nodes[name]["province"],
                        "lat": self.graph.nodes[name]["lat"],
                        "lon": self.graph.nodes[name]["lon"],
                        "B": float(model.balances[index[name]]),
                        "synthetic": True,
                        "is_critical": name == self.cfg.critical_node,
                        "is_generator_anchor": (
                            name in self.cfg.generator_anchors
                        ),
                    }
                    for name in model.variable_order
                ],
                "edges": [
                    {
                        "u": u, "v": v,
                        "weight": float(d["weight"]),
                        "length_m": float(d["length_m"]),
                        "circuits": list(d["circuits"]),
                        "parallel": bool(d["parallel"]),
                    }
                    for u, v, d in sorted(
                        model.graph.edges(data=True),
                        key=lambda e: (e[0], e[1]),
                    )
                ],
                "qubo": {
                    "Q_upper": model.q_matrix.tolist(),
                    "offset": model.qubo_offset,
                    "alpha": model.alpha,
                    "beta": model.beta,
                },
                "ising": {
                    "h": model.h_vec.tolist(),
                    "J_upper": model.j_upper.tolist(),
                    "offset": model.ising_offset,
                    "problema": (
                        "H_total (corte + α·balance² + β·crítico); "
                        "comparar contra baselines.h_total, NO contra "
                        "baselines.maxcut"
                    ),
                },
                "ising_maxcut": {
                    "h": [0.0] * n,
                    "J_upper": model.maxcut_j.tolist(),
                    "offset": model.maxcut_offset,
                    "problema": (
                        "Max-Cut puro (min H == -corte máximo); "
                        "benchmark contra baselines.maxcut.brute_force"
                    ),
                },
                "baselines": model.baselines,
            }
            path = out / f"isla_verde_{tier}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(path)
            logger.info("Exportado %s", path)

        # Grafo completo
        export_graph = nx.Graph()
        for node, data in self.graph.nodes(data=True):
            export_graph.add_node(node, **{
                "province": data["province"],
                "x_merc": data["x_merc"], "y_merc": data["y_merc"],
                "lat": data["lat"], "lon": data["lon"],
            })
        for u, v, data in self.graph.edges(data=True):
            export_graph.add_edge(u, v, **{
                "weight": float(data["weight"]),
                "length_m": float(data["length_m"]),
                "circuits": list(data["circuits"]),
                "parallel": bool(data["parallel"]),
            })

        node_link = nx.node_link_data(export_graph, edges="links")
        graph_path = out / "isla_verde_full_graph.json"
        graph_path.write_text(
            json.dumps({"metadata": metadata, "graph": node_link},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(graph_path)

        if h3_layer is not None:
            h3_path = out / "isla_verde_h3_layer.json"
            h3_path.write_text(
                json.dumps({"metadata": metadata, **h3_layer},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(h3_path)

        index_path = out / "isla_verde_index.json"
        index_path.write_text(
            json.dumps({
                "metadata": metadata,
                "instances": {
                    t: f"isla_verde_{t}.json" for t in self.instances
                },
                "instance_errors": self.instance_errors,
                "full_graph": "isla_verde_full_graph.json",
                "h3_layer": ("isla_verde_h3_layer.json"
                             if h3_layer is not None else None),
                "figure": "isla_verde_red_230kv.png",
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(index_path)
        return written

    # -- Visualización --------------------------------------------------------

    def visualize(self) -> Path:
        """Genera mapa estático de la red 230kV coloreada por provincia."""
        graph = self.graph
        pos = {
            n: (d["x_merc"], d["y_merc"])
            for n, d in graph.nodes(data=True)
        }
        provinces = sorted({d["province"] for _, d in graph.nodes(data=True)})
        cmap = plt.get_cmap("tab10")
        pcolor = {p: cmap(i % 10) for i, p in enumerate(provinces)}

        members = (
            set().union(*(set(m.variable_order)
                          for m in self.instances.values()))
            if self.instances else set()
        )
        max_w = max(d["weight"] for _, _, d in graph.edges(data=True))
        widths = [
            0.4 + 4.6 * min(d["weight"] / max_w, 1.0)
            for _, _, d in graph.edges(data=True)
        ]

        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_edges(
            graph, pos, ax=ax, width=widths, edge_color="0.55",
        )
        non_members = [n for n in graph.nodes if n not in members]
        member_nodes = [n for n in graph.nodes if n in members]

        nx.draw_networkx_nodes(
            graph, pos, nodelist=non_members, ax=ax, node_size=55,
            node_color=[pcolor[graph.nodes[n]["province"]]
                        for n in non_members],
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=member_nodes, ax=ax, node_size=110,
            node_color=[pcolor[graph.nodes[n]["province"]]
                        for n in member_nodes],
            edgecolors="black", linewidths=1.6,
        )
        for name in member_nodes:
            ax.annotate(
                name, pos[name], textcoords="offset points",
                xytext=(4, 4), fontsize=8,
            )

        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=c, label=p)
            for p, c in pcolor.items()
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  title="Provincia")
        ax.set_title(
            f"ICE {self.cfg.voltage} kV "
            f"({graph.number_of_nodes()} nodos, "
            f"{graph.number_of_edges()} aristas); "
            f"borde negro = instancias NISQ"
        )
        ax.set_xlabel("Web Mercator X (m)")
        ax.set_ylabel("Web Mercator Y (m)")
        ax.set_aspect("equal")
        fig.tight_layout()

        path = self.cfg.out_dir / "isla_verde_red_230kv.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Figura guardada en %s", path)
        return path

    # -- Auto-verificación ----------------------------------------------------

    def run_verification(self) -> bool:
        """Suite de verificación: consistencia QUBO/Ising y baselines."""
        checks = self.checks
        for tier, model in self.instances.items():
            n = len(model.variable_order)
            index = {name: i for i, name in enumerate(model.variable_order)}
            c_idx = index[self.cfg.critical_node]
            g_idx = index[self.cfg.generator_anchor]

            # Conectividad
            checks.append((
                1, f"[{tier}] instancia conexa", "hard",
                nx.is_connected(model.graph), f"{n} nodos",
            ))

            # QUBO == fórmula directa
            samples = self.rng.integers(0, 2, size=(25, n)).astype(float)
            e_qubo = qubo_energies(model.q_matrix, model.qubo_offset, samples)
            cut = cut_values(model.edges_idx, samples)
            balance = (samples @ model.balances) ** 2
            critical = (samples[:, c_idx] - samples[:, g_idx]) ** 2
            e_direct = -cut + model.alpha * balance + model.beta * critical
            err_q = float(np.abs(e_direct - e_qubo).max())
            checks.append((
                2, f"[{tier}] QUBO == formula directa", "hard",
                err_q < 1e-6, f"max |dE| = {err_q:.3e}",
            ))

            # Ising == QUBO en espines mapeados
            spins = 1.0 - 2.0 * samples
            e_ising = ising_energies(
                model.h_vec, model.j_upper, model.ising_offset, spins,
            )
            err_i = float(np.abs(e_ising - e_qubo).max())
            checks.append((
                3, f"[{tier}] Ising == QUBO mapeado", "hard",
                err_i < 1e-6, f"max |dE| = {err_i:.3e}",
            ))

            # Ising Max-Cut puro == -corte
            e_mc = ising_energies(
                np.zeros(n), model.maxcut_j, model.maxcut_offset, spins,
            )
            err_mc = float(np.abs(e_mc + cut).max())
            checks.append((
                3, f"[{tier}] Ising Max-Cut == -corte", "hard",
                err_mc < 1e-6, f"max |dE| = {err_mc:.3e}",
            ))

            # Fuerza bruta >= greedy
            brute = model.baselines["maxcut"]["brute_force"]["cut"]
            greedy = model.baselines["maxcut"]["greedy"]["cut"]
            checks.append((
                4, f"[{tier}] brute >= greedy", "hard",
                brute >= greedy - 1e-9,
                f"brute {brute:.4f} vs greedy {greedy:.4f}",
            ))

            # GW >= 0.878 × óptimo
            gw = model.baselines["maxcut"]["goemans_williamson"]
            ratio = gw["cut"] / brute if brute > 0 else 1.0
            checks.append((
                5, f"[{tier}] GW >= 0.878 x optimo", "soft",
                ratio >= 0.878, f"ratio = {ratio:.4f}",
            ))

            # Suma B == 0
            b_sum = abs(math.fsum(model.balances.tolist()))
            checks.append((
                6, f"[{tier}] sum(B) == 0", "hard",
                b_sum < 1e-9, f"|sum B| = {b_sum:.3e}",
            ))

            # Restricción crítica satisfecha
            checks.append((
                7, f"[{tier}] x_c == x_g en óptimo", "hard",
                model.critical_satisfied, f"beta = {model.beta:.6g}",
            ))

        checks.sort(key=lambda c: c[0])
        width = max(len(c[1]) for c in checks)
        print()
        print(f"{'#':>2}  {'check':<{width}}  {'tipo':<4}  "
              f"{'estado':<6}  detalle")
        print("-" * (width + 40))
        hard_ok = True
        for num, desc, level, passed, detail in checks:
            status = ("SKIP" if passed is None
                      else "PASS" if passed
                      else "WARN" if level == "soft" else "FAIL")
            if status == "FAIL":
                hard_ok = False
            print(f"{num:>2}  {desc:<{width}}  {level:<4}  "
                  f"{status:<6}  {detail}")
        print("-" * (width + 40))
        print(f"resultado: {'PASS' if hard_ok else 'FAIL'}")
        return hard_ok

    # -- Orquestación ---------------------------------------------------------

    def run(self) -> int:
        """Ejecuta el pipeline completo."""
        logger.info(
            "ISLA VERDE Fase 1 | seed=%d | data=%s | out=%s",
            self.cfg.seed, self.cfg.data_dir, self.cfg.out_dir,
        )
        loaded = self.load_data()
        if loaded is None:
            self.build_fallback_graph()
        else:
            self.build_graph(*loaded)

        subgraphs = self.extract_instances()
        for tier, subgraph in subgraphs.items():
            model = self.formulate_instance(tier, subgraph)
            self.compute_baselines(model)
            self.instances[tier] = model

        h3_layer = self.build_h3_layer()
        self.export(h3_layer)
        self.visualize()
        return 0 if self.run_verification() else 1


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> Config:
    """Construye Config desde argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="ISLA VERDE Fase 1: red ICE 230kV → QUBO/Ising",
    )
    d = Config()
    parser.add_argument(
        "--data-dir", type=Path, default=d.data_dir,
        help="directorio con los CSV del ICE",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=d.out_dir,
        help="directorio de salida",
    )
    parser.add_argument(
        "--seed", type=int, default=d.seed,
        help="semilla del RNG (default: 42)",
    )
    args = parser.parse_args(argv)
    return Config(data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args(argv)
    try:
        return ICEPowerGridModeler(cfg).run()
    except (CalibrationError, AssertionError, ValueError) as exc:
        logger.error("Pipeline abortado: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
