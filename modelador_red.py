#!/usr/bin/env python3
"""ISLA VERDE v3.0 - Fase 1 / Phase 1: Capa clasica de datos y grafo.

ES: Construye el grafo de transmision 230 kV del ICE desde los CSV oficiales,
    extrae tres instancias NISQ verificadas (MVP-8 / STD-12 / LARGE-16),
    formula el Max-Cut restringido como QUBO e Ising, calcula lineas base
    clasicas y exporta un JSON por instancia para el equipo cuantico
    (
EN: Builds the ICE 230 kV transmission graph from the official CSVs,
    extracts three verified NISQ instances (MVP-8 / STD-12 / LARGE-16),
    formulates the constrained Max-Cut as QUBO and Ising, computes classical
    baselines, and exports one JSON bundle per instance for the quantum team
 

Modulos / Modules (M1 -> M11 desde main() / from main()):
    M1  Configuracion / Config (dataclass congelada, argparse)
    M2  Normalizacion de nombres y parseo de circuitos / Name normalization
    M3  Construccion del grafo / Graph construction
    M4  Extraccion de instancias / Instance extraction
    M5  Balance de potencia sintetico / Synthetic power balance
    M6  Constructor QUBO/Ising con autocalibracion / QUBO/Ising builder
    M7  Lineas base clasicas / Classical baselines
    M8  Capa geoespacial H3 (opcional) / H3 layer (optional)
    M9  Exportacion JSON / JSON export
    M10 Visualizacion matplotlib / Static visualization
    M11 Suite de autoverificacion / Self-verification suite

Uso / Usage:
    python3 modelador_red.py --data-dir /ruta/csvs --out-dir /ruta/salida


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
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


try:  # Goemans-Williamson SDP (M7 Track A iii)
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError: 
    cp = None  # type: ignore[assignment]
    CVXPY_AVAILABLE = False

try:  # Capa geoespacial H3 / H3 geospatial layer (M8)
    import h3

    H3_AVAILABLE = True
except ImportError:  
    h3 = None  # type: ignore[assignment]
    H3_AVAILABLE = False

logger = logging.getLogger("isla_verde.modelador_red")


MERCATOR_R = 6378137.0

# ---------------------------------------------------------------------------
# M1 - Configuracion / Configuration
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

# ES: Corredor de respaldo con longitudes reales medidas. Lindora-La Caja se
#     implementa como DOS circuitos paralelos con pesos sumados
#     (100000/5860.3 + 100000/5973.5); 5916.9 m es solo la aproximacion
#     armonica comentada y NO se usa para el peso.
# EN: Fallback corridor with real measured lengths. Lindora-La Caja is
#     implemented as TWO parallel circuits with summed weights
#     (100000/5860.3 + 100000/5973.5); 5916.9 m is only the commented
#     harmonic approximation and is NOT used for the weight.
FALLBACK_EDGES_M: tuple[tuple[str, str, float], ...] = (
    ("Arenal", "Garabito", 58158.6),
    ("Arenal", "Lindora", 122582.0),
    ("Garabito", "Cañas", 61285.8),
    ("Barranca", "Garabito", 7949.7),
    ("Barranca", "La Garita", 41536.5),
    ("La Garita", "Lindora", 20828.0),
    ("Lindora", "La Caja", 5916.9),  # par paralelo, ver nota / parallel pair
    ("La Caja", "Belen", 3733.1),
)
FALLBACK_PARALLEL_LENGTHS_M: tuple[float, float] = (5860.3, 5973.5)


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("/workspace/knowledge/")
    out_dir: Path = Path("/workspace/scratch/")
    seed: int = 42
    voltage: int = 230
    weight_numerator: float = 1.0e5
    instances: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_INSTANCES)
    )
    critical_node: str = "La Caja"
    generator_anchor: str = "Arenal"  # pareja de la restriccion critica / (g)
    generator_anchors: tuple[str, ...] = ("Arenal", "Garabito")
    generator_shares: tuple[float, ...] = (0.6, 0.4)  # hidro/termico estilizado
    h3_resolution: int = 5
    alpha: float | None = None
    beta: float | None = None
    subs_filename: str = "Subestaciones.csv"
    lines_filename: str = "LineasDeTransmision.csv"


# ---------------------------------------------------------------------------
# M2 - Normalizacion de nombres y parseo / Name normalization and parsing
# ---------------------------------------------------------------------------

_PAREN_SUFFIX_RE = re.compile(r"\s*\(.*\)\s*$")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(s: str) -> str:
    """ES: Normalizacion canonica, identica en ambos CSV. Orden exacto: strip
    -> minusculas -> NFD -> quitar diacriticos -> quitar sufijo entre
    parentesis -> quitar digitos finales -> colapsar espacios.
    EN: Canonical normalization, identical on both CSV sides. Exact order:
    strip -> lowercase -> NFD -> strip combining marks -> strip parenthetical
    suffix -> strip trailing digits -> collapse whitespace."""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _PAREN_SUFFIX_RE.sub("", s)
    s = _TRAILING_DIGITS_RE.sub("", s).strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def build_name_map(official_names: Iterable[str]) -> dict[str, str]:
  
    name_map: dict[str, str] = {}
    for official in official_names:
        key = normalize_name(official)
        if key in name_map and name_map[key] != official:
            logger.warning(
                "Colision de normalizacion / normalization collision: "
                "%r y/and %r -> %r", name_map[key], official, key,
            )
        name_map[key] = official
    for key in sorted(name_map):  # orden estable / stable iteration order
        if key.startswith("la "):
            alias = key[3:]
            if alias and alias not in name_map:
                name_map[alias] = name_map[key]
            elif alias in name_map and name_map[alias] != name_map[key]:
                logger.warning(
                    "Alias %r de/for %r colisiona con/collides with %r; "
                    "se mantiene el existente / keeping existing",
                    alias, name_map[key], name_map[alias],
                )
    return name_map


class SkipReason(str, Enum):
    

    NO_HYPHEN = "no-hyphen"
    UNKNOWN_ENDPOINT = "unknown-endpoint"


@dataclass(frozen=True)
class CircuitResolution:


    raw: str
    status: str  # "ok" o/or valor de SkipReason / SkipReason value
    endpoints: tuple[str, str] | None = None  # nombres oficiales / official
    unknown_endpoints: tuple[str, ...] = ()


def parse_circuit(raw: str, name_map: Mapping[str, str]) -> CircuitResolution:
    
    raw = str(raw).strip()
    if "-" not in raw:
        return CircuitResolution(raw=raw, status=SkipReason.NO_HYPHEN.value)

    best_unknown: tuple[str, ...] | None = None
    for pos, ch in enumerate(raw):
        if ch != "-":
            continue
        left, right = normalize_name(raw[:pos]), normalize_name(raw[pos + 1:])
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
        raw=raw,
        status=SkipReason.UNKNOWN_ENDPOINT.value,
        unknown_endpoints=best_unknown or (),
    )


def normalize_province(s: str) -> str:
  
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _WHITESPACE_RE.sub(" ", s)
    return s.title()


def mercator_to_latlon(x: float, y: float) -> tuple[float, float]:
    
    lon = math.degrees(x / MERCATOR_R)
    lat = math.degrees(math.atan(math.sinh(y / MERCATOR_R)))
    return lat, lon


# ---------------------------------------------------------------------------
# M6 - Algebra QUBO/Ising 
# M6 - QUBO/Ising algebra 
# ---------------------------------------------------------------------------


def build_qubo(
    edges: Sequence[tuple[int, int, float]],
    balances: np.ndarray,
    alpha: float,
    beta: float,
    critical_idx: int,
    anchor_idx: int,
) -> tuple[np.ndarray, float]:

    n = balances.shape[0]
    q_matrix = np.zeros((n, n), dtype=np.float64)

    # ES: H_cut = -sum W(x_i + x_j - 2 x_i x_j) = -(corte ponderado):
    #     minimizar MAXIMIZA el peso total cortado (Max-Cut, el benchmark que
    #     exige el reto). Nota fisica honesta: con W = 1/longitud como proxy
    #     de acoplamiento, Max-Cut tiende a cortar las lineas MAS acopladas,
    #     no las debiles; "preservar acoples fuertes y romper solo los
    #     debiles" seria un Min-Cut restringido, que es un problema distinto.
    # EN: H_cut = -sum W(x_i + x_j - 2 x_i x_j) = -(weighted cut):
    #     minimizing MAXIMIZES total cut weight (Max-Cut, the challenge
    #     benchmark). Honest physics note: with W = 1/length as a coupling
    #     proxy, Max-Cut tends to sever the MOST strongly coupled lines, not
    #     the weak ones; "preserve strong couplings, break only weak ones"
    #     would be a constrained Min-Cut, which is a different problem.
    for i, j, w in edges:
        q_matrix[i, i] -= w          # parte lineal / linear part -W(x_i+x_j)
        q_matrix[j, j] -= w
        q_matrix[i, j] += 2.0 * w    # parte cuadratica / quadratic +2W x_i x_j

    # ES: H_balance = alpha*(sum B_i x_i)^2. Con sum B_i = 0 por construccion,
    #     penalizar la isla A cubre ambas islas. Expansion con x_i^2 = x_i:
    #     diagonal B_i^2 + cruzados 2 B_i B_j.
    # EN: H_balance = alpha*(sum B_i x_i)^2. With sum B_i = 0 by construction,
    #     penalizing island A covers both islands. Expansion with x_i^2 = x_i:
    #     diagonal B_i^2 + cross terms 2 B_i B_j.
    for i in range(n):
        q_matrix[i, i] += alpha * balances[i] ** 2   # autodesbalance / self
    for i in range(n):
        for j in range(i + 1, n):
            q_matrix[i, j] += 2.0 * alpha * balances[i] * balances[j]  # cruz

    # ES: H_critical = beta*(x_c - x_g)^2 = beta*(x_c + x_g - 2 x_c x_g):
    #     cero si carga critica y ancla generadora comparten isla; +beta si no.
    #     beta se calibra para que violar nunca convenga.
    # EN: H_critical = beta*(x_c - x_g)^2: zero when critical load and its
    #     generator anchor share an island, +beta otherwise. beta is
    #     calibrated so violation can never pay off.
    c, g = critical_idx, anchor_idx
    lo, hi = (c, g) if c < g else (g, c)
    q_matrix[c, c] += beta
    q_matrix[g, g] += beta
    q_matrix[lo, hi] -= 2.0 * beta

  
    offset = 0.0
    return q_matrix, offset


def qubo_to_ising(
    q_matrix: np.ndarray, qubo_offset: float
) -> tuple[np.ndarray, np.ndarray, float]:
    
    diag = np.diag(q_matrix).copy()
    upper = np.triu(q_matrix, k=1)
    # x_i = (1 - s_i)/2 ; x_i x_j = (1 - s_i - s_j + s_i s_j)/4
    h_vec = -diag / 2.0 - (upper.sum(axis=1) + upper.sum(axis=0)) / 4.0
    j_upper = upper / 4.0
    offset = qubo_offset + diag.sum() / 2.0 + upper.sum() / 4.0
    return h_vec, j_upper, offset


def qubo_energies(
    q_matrix: np.ndarray, offset: float, x_batch: np.ndarray
) -> np.ndarray:
    """ES: E(x) = x^T Q x + offset para un lote de filas binarias (m, n).
    EN: E(x) = x^T Q x + offset for a batch of binary rows (m, n)."""
    x = x_batch.astype(np.float64)
    return ((x @ q_matrix) * x).sum(axis=1) + offset


def ising_energies(
    h_vec: np.ndarray, j_upper: np.ndarray, offset: float, s_batch: np.ndarray
) -> np.ndarray:
    """ES: E(s) = h.s + s^T J s + offset para un lote de espines (m, n).
    EN: E(s) = h.s + s^T J s + offset for a batch of spin rows (m, n)."""
    s = s_batch.astype(np.float64)
    return ((s @ j_upper) * s).sum(axis=1) + s @ h_vec + offset


def enumerate_bitstrings(n_vars: int, fix_first_zero: bool = False) -> np.ndarray:
 
    free = n_vars - 1 if fix_first_zero else n_vars
    ints = np.arange(1 << free, dtype=np.uint64)
    bits = ((ints[:, None] >> np.arange(free, dtype=np.uint64)[None, :]) & 1)
    bits = bits.astype(np.float64)
    if fix_first_zero:
        bits = np.hstack([np.zeros((bits.shape[0], 1)), bits])
    return bits


def cut_values(
    edges: Sequence[tuple[int, int, float]], x_batch: np.ndarray
) -> np.ndarray:
    """ES: Corte ponderado sum_e W_e (x_i + x_j - 2 x_i x_j) por lote.
    EN: Weighted cut sum_e W_e (x_i + x_j - 2 x_i x_j) for a batch of rows."""
    x = x_batch.astype(np.float64)
    total = np.zeros(x.shape[0], dtype=np.float64)
    for i, j, w in edges:
        xi, xj = x[:, i], x[:, j]
        total += w * (xi + xj - 2.0 * xi * xj)
    return total


def bitstring(x_vec: np.ndarray) -> str:
    """ES: Vector binario como cadena '01...' en el orden de variables.
    EN: Binary vector as a '01...' string in variable order."""
    return "".join(str(int(round(b))) for b in x_vec)


class CalibrationError(RuntimeError):
    """ES: La restriccion critica no se pudo imponer duplicando beta.
    EN: Critical constraint could not be enforced by doubling beta."""


# ---------------------------------------------------------------------------
# Orquestador / Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class InstanceModel:
   

    tier: str
    variable_order: list[str]
    graph: nx.Graph
    edges_idx: list[tuple[int, int, float]]  # (i, j, W_ij), i < j
    balances: np.ndarray                     # B_i en orden de variables, suma 0
    alpha: float
    beta: float
    q_matrix: np.ndarray
    qubo_offset: float
    h_vec: np.ndarray
    j_upper: np.ndarray
    ising_offset: float
    h_total_energies: np.ndarray             # tabla 2^n completa / full table
    h_total_optimum: float
    h_total_argmin: np.ndarray
    critical_satisfied: bool
    baselines: dict[str, Any] = field(default_factory=dict)


class ICEPowerGridModeler:


    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # ES: Unico RNG del programa. / EN: Single program-wide RNG.
        self.rng = np.random.default_rng(cfg.seed)
        self.source: str = "real"
        self.graph: nx.Graph = nx.Graph()
        self.excluded: list[CircuitResolution] = []
        self.instances: dict[str, InstanceModel] = {}
        self.instance_errors: dict[str, str] = {}
        self.checks: list[tuple[int, str, str, bool | None, str]] = []
        # (numero, descripcion, "hard"/"soft", paso(None=omitido), detalle)
        # (number, description, "hard"/"soft", passed(None=skipped), detail)

    # -- M2/M3 ----------------------------------------------------------------

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        
        subs_path = self.cfg.data_dir / self.cfg.subs_filename
        lines_path = self.cfg.data_dir / self.cfg.lines_filename
        try:
            # ES: utf-8-sig: ambos archivos traen BOM. / EN: both carry a BOM.
            subs = pd.read_csv(subs_path, encoding="utf-8-sig")
            lines = pd.read_csv(lines_path, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            logger.warning("Fallo al leer CSV / CSV load failed (%s: %s)",
                           type(exc).__name__, exc)
            return None

        required_subs = {"X", "Y", "Subestacio", "Provincia"}
        required_lines = {"Voltaje", "Circuito", "Shape__Length"}
        missing_subs = required_subs - set(subs.columns)
        missing_lines = required_lines - set(lines.columns)
        if missing_subs or missing_lines:
            # ES: Nombrar esperado vs encontrado; jamas adivinar una columna.
            # EN: Name expected vs found; never guess a substitute column.
            raise ValueError(
                "Columnas requeridas ausentes / required columns missing. "
                f"{self.cfg.subs_filename}: expected {sorted(required_subs)}, "
                f"found {sorted(subs.columns)}; "
                f"{self.cfg.lines_filename}: expected {sorted(required_lines)}, "
                f"found {sorted(lines.columns)}"
            )
        # ES: SHAPE_STLe es inconfiable (razon 0.005-4.9 vs Shape__Length) y
        #     deliberadamente nunca se lee.
        # EN: SHAPE_STLe is unreliable (0.005-4.9 ratio vs Shape__Length) and
        #     is deliberately never read.
        return subs, lines

    def build_graph(self, subs: pd.DataFrame, lines: pd.DataFrame) -> None:
        """ES: M3: filtro 230 kV, resolucion de extremos, pesos, fusion de
        circuitos paralelos.
        EN: M3: 230 kV filter, endpoint resolution, weights, parallel merge."""
        cfg = self.cfg
        name_map = build_name_map(subs["Subestacio"].tolist())

        graph = nx.Graph()
        for _, row in subs.iterrows():
            lat, lon = mercator_to_latlon(float(row["X"]), float(row["Y"]))
            graph.add_node(
                str(row["Subestacio"]),
                province=normalize_province(row["Provincia"]),
                x_merc=float(row["X"]),
                y_merc=float(row["Y"]),
                lat=lat,
                lon=lon,
            )

        filtered = lines[lines["Voltaje"] == cfg.voltage]
        logger.info(
            "Filtradas / filtered %d/%d filas de circuito a %d kV",
            len(filtered), len(lines), cfg.voltage,
        )
        for _, row in filtered.iterrows():
            resolution = parse_circuit(row["Circuito"], name_map)
            if resolution.status != "ok":
                self.excluded.append(resolution)
                logger.warning(
                    "Circuito excluido / excluded circuit %r (%s%s)",
                    resolution.raw, resolution.status,
                    f": {', '.join(resolution.unknown_endpoints)}"
                    if resolution.unknown_endpoints else "",
                )
                continue
            u, v = resolution.endpoints  # type: ignore[misc]
            length_m = float(row["Shape__Length"])  # longitud oficial / authoritative
            weight = cfg.weight_numerator / length_m
            if graph.has_edge(u, v):
                # ES: Circuitos paralelos: las capacidades de transferencia se
                #     suman, asi que los pesos se SUMAN en una sola arista.
                # EN: Parallel circuits: transfer capacities add, so weights
                #     are SUMMED into one edge.
                data = graph[u][v]
                data["weight"] += weight
                data["circuits"].append(resolution.raw)
                data["lengths_m"].append(length_m)
                data["parallel"] = True
            else:
                graph.add_edge(
                    u, v,
                    weight=weight,
                    circuits=[resolution.raw],
                    lengths_m=[length_m],
                    parallel=False,
                )

        # ES: Quitar subestaciones sin circuito 230 kV (solo 138 kV o aisladas
        #     a este nivel de tension; irrelevantes para el modelo).
        # EN: Drop substations with no 230 kV circuit (138 kV-only or isolated
        #     at this voltage level; irrelevant to the model).
        isolated = [n for n in graph.nodes if graph.degree(n) == 0]
        graph.remove_nodes_from(isolated)
        logger.info("Eliminadas / dropped %d subestaciones sin circuito %d kV",
                    len(isolated), cfg.voltage)

        # ES: Longitud equivalente por arista: numerador / peso sumado. Para
        #     una linea es la longitud medida; para paralelas es la combinacion
        #     armonica, consistente con impedancias en paralelo.
        # EN: Equivalent length per edge: numerator / summed weight. Single
        #     circuit -> measured length; parallels -> harmonic combination,
        #     consistent with impedances in parallel.
        for _, _, data in graph.edges(data=True):
            data["length_m"] = cfg.weight_numerator / data["weight"]

        self.graph = graph
        self._check_graph_invariants()

    def _check_graph_invariants(self) -> None:
        graph = self.graph
        n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
        connected = nx.is_connected(graph) if n_nodes else False
        n_parallel = sum(
            1 for _, _, d in graph.edges(data=True) if d["parallel"]
        )
        n_excluded = len(self.excluded)
        expected = (46, 58, True, 3, 4)
        actual = (n_nodes, n_edges, connected, n_parallel, n_excluded)
        if actual != expected:
            logger.warning(
                "Deriva de invariantes / invariant drift: esperado/expected "
                "(nodes, edges, connected, parallel, exclusions)=%s, "
                "obtenido/got %s", expected, actual,
            )
        else:
            logger.info(
                "Invariantes del grafo OK / graph invariants hold: 46 nodos, "
                "58 aristas, conexo, 3 pares paralelos, 4 circuitos excluidos"
            )
        ok = actual == expected
        detail = f"expected {expected}, got {actual}"
        self.checks.append((8, "Full-graph invariants (46/58/connected/3/4)",
                            "soft", ok, detail))

    def build_fallback_graph(self) -> None:
      
        logger.warning("[FALLBACK] Using synthetic ICE 230 kV corridor proxy")
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
            graph.nodes[node]["province"] = "Sintetica"
            graph.nodes[node]["x_merc"] = float(pos[node][0])
            graph.nodes[node]["y_merc"] = float(pos[node][1])
            graph.nodes[node]["lat"] = 0.0
            graph.nodes[node]["lon"] = 0.0
        self.graph = graph
        self.checks.append((8, "Full-graph invariants (46/58/connected/3/4)",
                            "soft", None, "skipped: fallback proxy in use"))

    # -- M4 -------------------------------------------------------------------

    def extract_instances(self) -> dict[str, nx.Graph]:
        
        subgraphs: dict[str, nx.Graph] = {}
        for tier, node_set in self.cfg.instances.items():
            missing = [n for n in node_set if n not in self.graph]
            if missing:
                suggestions = {
                    n: difflib.get_close_matches(n, list(self.graph.nodes), n=3)
                    for n in missing
                }
                message = f"unresolved nodes {missing}; closest: {suggestions}"
                self.instance_errors[tier] = message
                logger.error("Instancia %s omitida / skipped: %s", tier, message)
                continue
            subgraph = nx.Graph(self.graph.subgraph(node_set))
            if not nx.is_connected(subgraph):
                components = [sorted(c) for c in nx.connected_components(subgraph)]
                raise AssertionError(
                    f"Instancia/instance {tier} desconectada/disconnected "
                    f"(componentes/components: {components}); los conjuntos "
                    "estan verificados: el procesamiento previo se rompio / "
                    "node sets are spec-verified, upstream processing broke"
                )
            subgraphs[tier] = subgraph
            logger.info(
                "Instancia / instance %s: %d nodos, %d aristas inducidas, conexa",
                tier, subgraph.number_of_nodes(), subgraph.number_of_edges(),
            )
        return subgraphs

    # -- M5 -------------------------------------------------------------------

    def assign_balances(self, nodes: Sequence[str]) -> dict[str, float]:
      
        anchors = [a for a in self.cfg.generator_anchors if a in nodes]
        if not anchors:
            raise ValueError(
                f"Ningun ancla de generacion / no generator anchor of "
                f"{self.cfg.generator_anchors} present in {sorted(nodes)}"
            )
        shares = np.array(
            [s for a, s in zip(self.cfg.generator_anchors,
                               self.cfg.generator_shares) if a in nodes],
            dtype=np.float64,
        )
        shares = shares / shares.sum()

        load_nodes = sorted(n for n in nodes if n not in anchors)
        draws = self.rng.uniform(50.0, 150.0, size=len(load_nodes))
        total_load = float(draws.sum())

        balances = {n: -float(d) for n, d in zip(load_nodes, draws)}
        for anchor, share in zip(anchors, shares):
            balances[anchor] = float(share) * total_load
        residual = math.fsum(balances.values())
        balances[load_nodes[-1]] -= residual  # suma cero exacta / exact zero
        assert abs(math.fsum(balances.values())) < 1e-9
        return balances

    # -- M6 -------------------------------------------------------------------

    def formulate_instance(self, tier: str, subgraph: nx.Graph) -> InstanceModel:
       
        cfg = self.cfg
        # ES: Orden fijo = nombres ordenados, registrado en el export.
        # EN: Fixed order = sorted node names, recorded in the export.
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
        sum_b_squared = float((balances ** 2).sum())
        alpha = cfg.alpha if cfg.alpha is not None else (
            total_weight / (sum_b_squared / 4.0)  # Var analitica / analytic
        )
        beta = cfg.beta if cfg.beta is not None else 2.0 * total_weight
        if beta < 2.0 * total_weight:
            logger.warning(
                "[%s] beta configurado %.6g bajo el piso de seguridad / "
                "configured beta below safety floor 2*sumW=%.6g",
                tier, beta, 2.0 * total_weight,
            )

        critical_idx = index[cfg.critical_node]
        anchor_idx = index[cfg.generator_anchor]
        n_vars = len(variable_order)
        all_states = enumerate_bitstrings(n_vars)

        q_matrix = np.zeros((0, 0))
        qubo_offset = 0.0
        energies = np.zeros(0)
        argmin_x = np.zeros(0)
        satisfied = False
        for attempt in range(4):  # solve inicial + max 3 duplicaciones / doublings
            q_matrix, qubo_offset = build_qubo(
                edges_idx, balances, alpha, beta, critical_idx, anchor_idx
            )
            energies = qubo_energies(q_matrix, qubo_offset, all_states)
            argmin_x = all_states[int(np.argmin(energies))]
            satisfied = bool(argmin_x[critical_idx] == argmin_x[anchor_idx])
            if satisfied:
                break
            logger.warning(
                "[%s] restriccion critica violada en el optimo con beta=%.6g; "
                "duplicando / critical constraint violated, doubling "
                "(intento/attempt %d/3)", tier, beta, attempt + 1,
            )
            beta *= 2.0
        if not satisfied:
            raise CalibrationError(
                f"[{tier}] x_c == x_g insatisfecha tras 3 duplicaciones de "
                f"beta / unsatisfied after 3 beta doublings (beta={beta:.6g}, "
                f"alpha={alpha:.6g}, sum W={total_weight:.6g}, "
                f"optimum={float(energies.min()):.6g}, "
                f"bitstring={bitstring(argmin_x)}, "
                f"c={cfg.critical_node}, g={cfg.generator_anchor})"
            )
        logger.info(
            "[%s] calibrado / calibrated alpha=%.6g beta=%.6g "
            "(sum W=%.6g, sum B^2=%.6g)",
            tier, alpha, beta, total_weight, sum_b_squared,
        )

        h_vec, j_upper, ising_offset = qubo_to_ising(q_matrix, qubo_offset)
        return InstanceModel(
            tier=tier,
            variable_order=variable_order,
            graph=subgraph,
            edges_idx=edges_idx,
            balances=balances,
            alpha=alpha,
            beta=beta,
            q_matrix=q_matrix,
            qubo_offset=qubo_offset,
            h_vec=h_vec,
            j_upper=j_upper,
            ising_offset=ising_offset,
            h_total_energies=energies,
            h_total_optimum=float(energies.min()),
            h_total_argmin=argmin_x,
            critical_satisfied=satisfied,
        )

    # -- M7 -------------------------------------------------------------------

    def compute_baselines(self, model: InstanceModel) -> None:
     
        n_vars = len(model.variable_order)
        edges = model.edges_idx

        # (i) ES: fuerza bruta exacta sobre 2^(n-1), x_0 fijo en 0.
        #     EN: exact brute force over 2^(n-1) states, x_0 fixed to 0.
        half_states = enumerate_bitstrings(n_vars, fix_first_zero=True)
        cuts = cut_values(edges, half_states)
        best_idx = int(np.argmax(cuts))
        brute_cut = float(cuts[best_idx])
        brute_x = half_states[best_idx]

        # (ii) ES: greedy: inicio aleatorio sembrado + mejor flip individual.
        #      EN: greedy: seeded random start + best-improvement single flips.
        greedy_cut, greedy_x = self._greedy_maxcut(edges, n_vars)

        # (iii) ES: GW via relajacion SDP, mejor de 50 redondeos.
        #       EN: GW via SDP relaxation, best of 50 roundings.
        gw_result = self._goemans_williamson(edges, n_vars, roundings=50)

        model.baselines = {
            "maxcut": {
                "brute_force": {"cut": brute_cut, "bitstring": bitstring(brute_x)},
                "greedy": {"cut": greedy_cut, "bitstring": bitstring(greedy_x)},
                "goemans_williamson": gw_result,
            },
            "h_total": {
                "optimum_energy": model.h_total_optimum,
                "bitstring": bitstring(model.h_total_argmin),
                "critical_constraint_satisfied": model.critical_satisfied,
            },
        }
        gw_text = ("unavailable" if gw_result is None
                   else f"{gw_result['cut']:.4f}")
        logger.info(
            "[%s] baselines: brute cut=%.4f greedy=%.4f GW=%s | "
            "H_total optimum=%.4f",
            model.tier, brute_cut, greedy_cut, gw_text, model.h_total_optimum,
        )

    def _greedy_maxcut(
        self, edges: Sequence[tuple[int, int, float]], n_vars: int
    ) -> tuple[float, np.ndarray]:
        
        x = self.rng.integers(0, 2, size=n_vars).astype(np.float64)
        incident: list[list[tuple[int, float]]] = [[] for _ in range(n_vars)]
        for i, j, w in edges:
            incident[i].append((j, w))
            incident[j].append((i, w))
        # ES: Cota generosa; el lazo sale antes. / EN: generous bound; exits early.
        for _ in range(10 * (1 << n_vars)):
            gains = np.array([
                sum(w if x[i] == x[j] else -w for j, w in incident[i])
                for i in range(n_vars)
            ])
            best = int(np.argmax(gains))
            if gains[best] <= 1e-12:
                break
            x[best] = 1.0 - x[best]
        if x[0] == 1.0:
            x = 1.0 - x
        cut = float(cut_values(edges, x[None, :])[0])
        return cut, x

    def _goemans_williamson(
        self, edges: Sequence[tuple[int, int, float]], n_vars: int,
        roundings: int,
    ) -> dict[str, Any] | None:
       
        if not CVXPY_AVAILABLE:
            logger.warning(
                "cvxpy no disponible / unavailable: GW baseline = null"
            )
            return None
        try:
            gram = cp.Variable((n_vars, n_vars), symmetric=True)
            constraints = [gram >> 0, cp.diag(gram) == 1]
            objective = cp.Maximize(
                sum(w * (1 - gram[i, j]) for i, j, w in edges) / 2
            )
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)
            if gram.value is None:
                raise RuntimeError(f"SDP sin solucion / solver returned no "
                                   f"value (status: {problem.status})")
        except Exception as exc:  # disponibilidad de solver varia / env varies
            logger.warning(
                "GW SDP fallo / failed (%s: %s); resultado null",
                type(exc).__name__, exc,
            )
            return None

        gram_value = np.asarray((gram.value + gram.value.T) / 2.0)
        eigvals, eigvecs = np.linalg.eigh(gram_value)
        vectors = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))

        best_cut = -np.inf
        best_x = np.zeros(n_vars)
        for _ in range(roundings):
            hyperplane = self.rng.standard_normal(n_vars)
            signs = np.sign(vectors @ hyperplane)
            signs[signs == 0] = 1.0
            x = (1.0 - signs) / 2.0
            # ES: Canonizar: el corte es invariante al complemento.
            # EN: Canonicalize: cut invariant under complement.
            if x[0] == 1.0:
                x = 1.0 - x
            cut = float(cut_values(edges, x[None, :])[0])
            if cut > best_cut:
                best_cut, best_x = cut, x
        return {
            "cut": best_cut,
            "bitstring": bitstring(best_x),
            "roundings": roundings,
        }

    # -- M8 -------------------------------------------------------------------

    def build_h3_layer(self) -> dict[str, Any] | None:
       
        if not H3_AVAILABLE:
            logger.warning("h3 no disponible / unavailable: capa omitida / skipped")
            return None
        if self.source == "fallback":
            logger.warning("Fallback sin coordenadas reales / no real "
                           "coordinates: H3 omitido / skipped")
            return None
        try:
            # ES: Nombres de API h3 v4 / v3. / EN: h3 v4 / v3 API names.
            latlng_to_cell = getattr(h3, "latlng_to_cell", None) or getattr(
                h3, "geo_to_h3"
            )
            # ES: Balances sinteticos del grafo completo, mismo RNG unico
            #     (determinista: corre despues de todas las instancias).
            # EN: Full-graph synthetic balances, same single RNG
            #     (deterministic: runs after all instances are formulated).
            balances = self.assign_balances(sorted(self.graph.nodes))

            cell_members: dict[str, list[str]] = {}
            for node in sorted(self.graph.nodes):
                data = self.graph.nodes[node]
                cell = str(latlng_to_cell(
                    data["lat"], data["lon"], self.cfg.h3_resolution
                ))
                cell_members.setdefault(cell, []).append(node)

            node_cell = {n: c for c, ms in cell_members.items() for n in ms}
            super_edges: dict[tuple[str, str], dict[str, Any]] = {}
            for u, v, data in self.graph.edges(data=True):
                cu, cv = node_cell[u], node_cell[v]
                if cu == cv:
                    # ES: Circuito interno al supernodo. / EN: co-cell circuit.
                    continue
                key = (min(cu, cv), max(cu, cv))
                entry = super_edges.setdefault(
                    key, {"weight": 0.0, "circuits": []}
                )
                entry["weight"] += float(data["weight"])
                entry["circuits"].extend(data["circuits"])

            layer = {
                "h3_resolution": self.cfg.h3_resolution,
                "note": (
                    "Scaling proposal only. Supernode edges derive exclusively "
                    "from physical 230 kV lines crossing cell boundaries; H3 "
                    "adjacency is NOT electrical connection. B values are "
                    "synthetic."
                ),
                "supernodes": [
                    {
                        "cell": cell,
                        "members": members,
                        "B": math.fsum(balances[m] for m in members),
                        "synthetic": True,
                    }
                    for cell, members in sorted(cell_members.items())
                ],
                "superedges": [
                    {"u": u, "v": v,
                     "weight": entry["weight"], "circuits": entry["circuits"]}
                    for (u, v), entry in sorted(super_edges.items())
                ],
            }
            multi = [c for c, ms in cell_members.items() if len(ms) > 1]
            logger.info(
                "Capa H3 / H3 layer (res %d): %d supernodos (%d multimiembro), "
                "%d superaristas de %d circuitos fisicos",
                self.cfg.h3_resolution, len(cell_members), len(multi),
                len(super_edges), self.graph.number_of_edges(),
            )
            return layer
        except Exception as exc:  # capa opcional: degradar / degrade, never crash
            logger.warning(
                "Capa H3 fallo / H3 layer failed (%s: %s); omitida / skipped",
                type(exc).__name__, exc,
            )
            return None

    # -- M9 -------------------------------------------------------------------

    def export(self, h3_layer: dict[str, Any] | None) -> list[Path]:
       
        out_dir = self.cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_utc = datetime.now(timezone.utc).isoformat()
        metadata = {
            "source": self.source,
            "generated_utc": generated_utc,
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
            # ES: Ising de Max-Cut PURO (sin alpha/beta): C(x) = sum w(1-z_i z_j)/2,
            #     asi que max C == min [sum (w/2) z_i z_j - sum w/2]. Es el
            #     Hamiltoniano que debe usar QAOA para el benchmark contra
            #     brute_force.cut; el "ising" completo (con penalizaciones)
            #     es un problema distinto y se compara contra h_total.
            # EN: PURE Max-Cut Ising (no alpha/beta): the Hamiltonian QAOA must
            #     use for the benchmark against brute_force.cut; the full
            #     "ising" (with penalties) is a different problem, compared
            #     against h_total instead.
            n_vars_export = len(model.variable_order)
            maxcut_j_upper = np.zeros((n_vars_export, n_vars_export))
            for i, j, w in model.edges_idx:
                maxcut_j_upper[i, j] += w / 2.0
            maxcut_offset = -float(sum(w for _, _, w in model.edges_idx)) / 2.0
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
                        "is_generator_anchor": name in self.cfg.generator_anchors,
                    }
                    for name in model.variable_order
                ],
                "edges": [
                    {
                        "u": u,
                        "v": v,
                        "weight": float(d["weight"]),
                        "length_m": float(d["length_m"]),
                        "circuits": list(d["circuits"]),
                        "parallel": bool(d["parallel"]),
                    }
                    for u, v, d in sorted(
                        model.graph.edges(data=True), key=lambda e: (e[0], e[1])
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
                    "problema": "H_total (corte + alpha*balance^2 + beta*critico); "
                                "comparar contra baselines.h_total, NO contra "
                                "baselines.maxcut",
                },
                "ising_maxcut": {
                    "h": [0.0] * n_vars_export,
                    "J_upper": maxcut_j_upper.tolist(),
                    "offset": maxcut_offset,
                    "problema": "Max-Cut puro (min H == -corte maximo); "
                                "benchmark contra baselines.maxcut.brute_force",
                },
                "baselines": model.baselines,
            }
            path = out_dir / f"isla_verde_{tier}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(path)
            logger.info("Exportado / exported %s", path)

      
        export_graph = nx.Graph()
        for node, data in self.graph.nodes(data=True):
            export_graph.add_node(node, **{
                "province": data["province"], "x_merc": data["x_merc"],
                "y_merc": data["y_merc"], "lat": data["lat"], "lon": data["lon"],
            })
        for u, v, data in self.graph.edges(data=True):
            export_graph.add_edge(u, v, **{
                "weight": float(data["weight"]),
                "length_m": float(data["length_m"]),
                "circuits": list(data["circuits"]),
                "parallel": bool(data["parallel"]),
            })
        try:
            node_link = nx.node_link_data(export_graph, edges="links")
        except TypeError:  # networkx < 3.2 no acepta el kwarg / lacks the kwarg
            node_link = nx.node_link_data(export_graph)
        graph_path = out_dir / "isla_verde_full_graph.json"
        graph_path.write_text(
            json.dumps({"metadata": metadata, "graph": node_link},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(graph_path)

        if h3_layer is not None:
            h3_path = out_dir / "isla_verde_h3_layer.json"
            h3_path.write_text(
                json.dumps({"metadata": metadata, **h3_layer},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(h3_path)

        index_payload = {
            "metadata": metadata,
            "instances": {
                tier: f"isla_verde_{tier}.json" for tier in self.instances
            },
            "instance_errors": self.instance_errors,
            "full_graph": "isla_verde_full_graph.json",
            "h3_layer": ("isla_verde_h3_layer.json"
                         if h3_layer is not None else None),
            "figure": "isla_verde_red_230kv.png",
        }
        index_path = out_dir / "isla_verde_index.json"
        index_path.write_text(
            json.dumps(index_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(index_path)
        return written

    # -- M10 ------------------------------------------------------------------

    def visualize(self) -> Path:
        
        graph = self.graph
        positions = {
            n: (d["x_merc"], d["y_merc"]) for n, d in graph.nodes(data=True)
        }
        provinces = sorted({d["province"] for _, d in graph.nodes(data=True)})
        colormap = plt.get_cmap("tab10")
        province_color = {
            p: colormap(i % 10) for i, p in enumerate(provinces)
        }
        members = set().union(
            *(set(m.variable_order) for m in self.instances.values())
        ) if self.instances else set()

        max_weight = max(d["weight"] for _, _, d in graph.edges(data=True))
        widths = [
            0.4 + 4.6 * min(d["weight"] / max_weight, 1.0)  # con tope / capped
            for _, _, d in graph.edges(data=True)
        ]

        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_edges(
            graph, positions, ax=ax, width=widths, edge_color="0.55"
        )
        non_members = [n for n in graph.nodes if n not in members]
        member_nodes = [n for n in graph.nodes if n in members]
        nx.draw_networkx_nodes(
            graph, positions, nodelist=non_members, ax=ax, node_size=55,
            node_color=[province_color[graph.nodes[n]["province"]]
                        for n in non_members],
        )
        nx.draw_networkx_nodes(
            graph, positions, nodelist=member_nodes, ax=ax, node_size=110,
            node_color=[province_color[graph.nodes[n]["province"]]
                        for n in member_nodes],
            edgecolors="black", linewidths=1.6,
        )
        # ES: Solo los miembros de instancias llevan etiqueta.
        # EN: Only instance members are labeled.
        for name in member_nodes:
            x, y = positions[name]
            ax.annotate(
                name, (x, y), textcoords="offset points", xytext=(4, 4),
                fontsize=8,
            )
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=color,
                       label=province)
            for province, color in province_color.items()
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  title="Provincia")
        ax.set_title(
            f"ICE {self.cfg.voltage} kV transmission graph "
            f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} "
            f"edges); outlined = NISQ instance members"
        )
        ax.set_xlabel("Web Mercator X (m)")
        ax.set_ylabel("Web Mercator Y (m)")
        ax.set_aspect("equal")
        fig.tight_layout()
        path = self.cfg.out_dir / "isla_verde_red_230kv.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Figura guardada / figure saved to %s", path)
        return path

    # -- M11 ------------------------------------------------------------------

    def run_verification(self) -> bool:
      
        checks = self.checks

        for tier, model in self.instances.items():
            n_vars = len(model.variable_order)
            index = {n: i for i, n in enumerate(model.variable_order)}
            c_idx = index[self.cfg.critical_node]
            g_idx = index[self.cfg.generator_anchor]

            # 1. ES: Conectividad de instancia, duro. / EN: connectivity, hard.
            connected = nx.is_connected(model.graph)
            checks.append((1, f"[{tier}] instance connected", "hard",
                           connected, f"{n_vars} nodes"))

            # 2. ES: Consistencia QUBO vs formula directa, duro.
            #    EN: QUBO energy consistency vs direct formula, hard.
            samples = self.rng.integers(0, 2, size=(25, n_vars)).astype(float)
            e_qubo = qubo_energies(model.q_matrix, model.qubo_offset, samples)
            cut = cut_values(model.edges_idx, samples)
            balance = (samples @ model.balances) ** 2
            critical = (samples[:, c_idx] - samples[:, g_idx]) ** 2
            e_direct = -cut + model.alpha * balance + model.beta * critical
            max_err_q = float(np.abs(e_direct - e_qubo).max())
            checks.append((2, f"[{tier}] QUBO == direct formula (25 samples)",
                           "hard", max_err_q < 1e-6,
                           f"max |dE| = {max_err_q:.3e}"))

            # 3. ES: Ising igual a QUBO en espines mapeados, duro.
            #    EN: Ising energy equals QUBO at mapped spins, hard.
            spins = 1.0 - 2.0 * samples
            e_ising = ising_energies(
                model.h_vec, model.j_upper, model.ising_offset, spins
            )
            max_err_i = float(np.abs(e_ising - e_qubo).max())
            checks.append((3, f"[{tier}] Ising == QUBO at mapped spins",
                           "hard", max_err_i < 1e-6,
                           f"max |dE| = {max_err_i:.3e}"))

            # 3b. ES: Ising Max-Cut puro (h=0, J=w/2) == -corte, duro: el
            #     Hamiltoniano del benchmark QAOA debe reproducir el corte.
            #     EN: pure Max-Cut Ising (h=0, J=w/2) == -cut, hard: the QAOA
            #     benchmark Hamiltonian must reproduce the cut exactly.
            maxcut_j = np.zeros((n_vars, n_vars))
            for i_e, j_e, w_e in model.edges_idx:
                maxcut_j[i_e, j_e] += w_e / 2.0
            maxcut_off = -float(sum(w for _, _, w in model.edges_idx)) / 2.0
            e_maxcut = ising_energies(
                np.zeros(n_vars), maxcut_j, maxcut_off, spins
            )
            max_err_mc = float(np.abs(e_maxcut + cut).max())
            checks.append((3, f"[{tier}] Max-Cut Ising == -cut (25 samples)",
                           "hard", max_err_mc < 1e-6,
                           f"max |dE| = {max_err_mc:.3e}"))

            # 4. ES: Fuerza bruta >= greedy, duro. / EN: brute >= greedy, hard.
            brute = model.baselines["maxcut"]["brute_force"]["cut"]
            greedy = model.baselines["maxcut"]["greedy"]["cut"]
            checks.append((4, f"[{tier}] brute-force cut >= greedy cut",
                           "hard", brute >= greedy - 1e-9,
                           f"brute {brute:.4f} vs greedy {greedy:.4f}"))

            # 5. ES: GW >= 0.878 x optimo, suave (garantia en esperanza).
            #    EN: GW >= 0.878 x optimum, soft (guarantee in expectation).
            gw = model.baselines["maxcut"]["goemans_williamson"]
            if gw is None:
                checks.append((5, f"[{tier}] GW >= 0.878 x optimum", "soft",
                               None, "skipped: cvxpy/solver unavailable"))
            else:
                ratio = gw["cut"] / brute if brute > 0 else 1.0
                checks.append((5, f"[{tier}] GW >= 0.878 x optimum", "soft",
                               ratio >= 0.878, f"ratio = {ratio:.4f}"))

            # 6. ES: Suma de B cero por instancia, duro.
            #    EN: sum B == 0 per instance, hard.
            b_sum = abs(math.fsum(model.balances.tolist()))
            checks.append((6, f"[{tier}] sum(B) == 0 (tol 1e-9)", "hard",
                           b_sum < 1e-9, f"|sum B| = {b_sum:.3e}"))

            # 7. ES: Restriccion critica en el optimo de H_total, duro.
            #    EN: critical constraint at the H_total optimum, hard.
            checks.append((7, f"[{tier}] x_c == x_g at H_total optimum",
                           "hard", model.critical_satisfied,
                           f"beta = {model.beta:.6g}"))

        checks.sort(key=lambda c: c[0])
        width = max(len(c[1]) for c in checks)
        print()
        print(f"{'#':>2}  {'check':<{width}}  {'type':<4}  {'status':<6}  detail")
        print("-" * (width + 40))
        hard_ok = True
        for num, description, level, passed, detail in checks:
            status = ("SKIP" if passed is None
                      else "PASS" if passed
                      else "WARN" if level == "soft" else "FAIL")
            if status == "FAIL":
                hard_ok = False
            print(f"{num:>2}  {description:<{width}}  {level:<4}  "
                  f"{status:<6}  {detail}")
        print("-" * (width + 40))
        print(f"overall: {'PASS' if hard_ok else 'FAIL'}")
        return hard_ok

    # -- Orquestacion / Orchestration -----------------------------------------

    def run(self) -> int:
        """ES: Ejecuta M1 -> M11; retorna el codigo de salida del proceso.
        EN: Executes M1 -> M11; returns the process exit code."""
        logger.info(
            "ISLA VERDE Fase/Phase 1 | seed=%d | data_dir=%s | out_dir=%s",
            self.cfg.seed, self.cfg.data_dir, self.cfg.out_dir,
        )
        loaded = self.load_data()
        if loaded is None:
            self.build_fallback_graph()
        else:
            self.build_graph(*loaded)

        subgraphs = self.extract_instances()
        # ES: Orden fijo del dict: mvp8, std12, large16 (RNG determinista).
        # EN: Fixed dict order: mvp8, std12, large16 (deterministic RNG use).
        for tier, subgraph in subgraphs.items():
            model = self.formulate_instance(tier, subgraph)
            self.compute_baselines(model)
            self.instances[tier] = model

        h3_layer = self.build_h3_layer()
        self.export(h3_layer)
        self.visualize()

        passed = self.run_verification()
        return 0 if passed else 1


# ---------------------------------------------------------------------------
# Punto de entrada / Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> Config:
    """ES: Construye la Config con overrides de CLI (data_dir, out_dir, seed).
    EN: Builds the run Config from CLI overrides (data_dir, out_dir, seed)."""
    parser = argparse.ArgumentParser(
        description="ISLA VERDE Fase 1 / Phase 1: red ICE 230 kV -> QUBO/Ising"
    )
    defaults = Config()
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir,
                        help="directorio con los dos CSV del ICE / CSV dir")
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir,
                        help="directorio de salida / output dir")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="semilla del RNG unico / single RNG seed (42)")
    args = parser.parse_args(argv)
    return Config(data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed)


def main(argv: Sequence[str] | None = None) -> int:
    """ES: Punto de entrada CLI; retorna el codigo de salida.
    EN: CLI entry point; returns the process exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = parse_args(argv)
    try:
        return ICEPowerGridModeler(cfg).run()
    except (CalibrationError, AssertionError, ValueError) as exc:
        logger.error("Pipeline abortado / aborted: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
