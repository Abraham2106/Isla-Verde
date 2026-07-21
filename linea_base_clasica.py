#!/usr/bin/env python3
"""
Linea base clasica v3 - Proyecto Isla Verde (alineada a la rubrica oficial)
=============================================================================

Rubrica oficial del reto (textual):
    "La linea base es el algoritmo de redondeo SDP de Goemans-Williamson
    (GW) (razon de aproximacion mayor o igual a 0.878). Tambien debe
    reportarse una linea base voraz (greedy) (razon aprox. 0.5).
    Referencia: Goemans & Williamson (1995), JACM 42(6)."

Este modulo entrega EXACTAMENTE eso, sin agregados fuera de rubrica:

    1. Goemans-Williamson -> relajacion SDP + redondeo aleatorio,
                              garantia teorica r >= 0.878
    2. Greedy              -> busqueda local voraz, referencia r ~ 0.5

Autoria e independencia de la instancia:
    Este script NO reconstruye el grafo desde los CSV (evita divergencia
    con el grafo oficial). En su lugar, lee los JSON que ya exporta
    'modelador_red.py' -- las mismas instancias verificadas mvp8/std12/
    large16 que usara el equipo de QAOA -- y usa su 'brute_force.cut' ya
    calculado como referencia del optimo, para que la razon de
    aproximacion sea comparable 1:1 con lo que reporte QAOA.

    La implementacion de GW y Greedy en si es propia (no importa
    modelador_red.py), pero el Greedy aqui es mas fuerte que el de
    modelador_red.py: aquel corre un unico inicio aleatorio + flips;
    este corre multiples reinicios aleatorios + flips y se queda con el
    mejor, lo que reduce la varianza y sube el piso del ~0.5 esperado.

Uso:
    python3 linea_base_clasica_v3.py --scratch-dir ./scratch \
        --tiers mvp8 std12 large16

Dependencias obligatorias: numpy, networkx
Dependencias opcionales:   cvxpy (Goemans-Williamson)

Referencia:
    Goemans, M. X., & Williamson, D. P. (1995). Improved approximation
    algorithms for maximum cut and satisfiability problems using
    semidefinite programming. Journal of the ACM, 42(6), 1115-1145.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger("isla_verde.linea_base_clasica_v3")

# ES: Dependencia opcional: degrada con warning, nunca detiene el
#     pipeline (mismo patron que modelador_red.py).
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depende del entorno
    cp = None
    CVXPY_AVAILABLE = False

GW_THEORETICAL_GUARANTEE = 0.878   # Goemans & Williamson, 1995, JACM 42(6)
GREEDY_EXPECTED_RATIO = 0.5        # cota clasica de la heuristica voraz
DEFAULT_TIERS: tuple[str, ...] = ("mvp8", "std12", "large16")


# ---------------------------------------------------------------------------
# 1. Configuracion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    scratch_dir: Path = Path("./scratch")
    tiers: tuple[str, ...] = DEFAULT_TIERS
    n_redondeos_gw: int = 200
    n_corridas_gw: int = 10
    n_reinicios_greedy: int = 30
    seed: int = 42


# ---------------------------------------------------------------------------
# 2. Carga de la instancia oficial (sin reconstruir el grafo por separado)
# ---------------------------------------------------------------------------
def cargar_instancia(scratch_dir: Path, tier: str) -> tuple[nx.Graph, float] | None:
    """
    Lee 'isla_verde_{tier}.json' (exportado por modelador_red.py) y arma
    el grafo Max-Cut EXACTAMENTE con esos nodos y pesos -- la misma
    instancia que usara QAOA -- en vez de re-parsear los CSV con logica
    propia (evita que dos implementaciones de normalizacion de nombres
    diverjan en un caso limite). Devuelve tambien el optimo exacto
    (baselines.maxcut.brute_force.cut) ya calculado en ese JSON, para
    usarlo como referencia de la razon de aproximacion.

    Devuelve None, con warning, si el archivo no existe o le faltan los
    campos esperados: nunca detiene la comparacion de los demas tiers.
    """
    path = scratch_dir / f"isla_verde_{tier}.json"
    if not path.exists():
        logger.warning(
            "[%s] no se encontro %s. Corre primero: "
            "python3 modelador_red.py --out-dir %s", tier, path, scratch_dir,
        )
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        G = nx.Graph()
        for nodo in payload["variable_order"]:
            G.add_node(nodo)
        for arista in payload["edges"]:
            G.add_edge(arista["u"], arista["v"], weight=float(arista["weight"]))
        optimo = float(payload["baselines"]["maxcut"]["brute_force"]["cut"])
    except KeyError as exc:
        logger.warning("[%s] %s no tiene el campo esperado %s", tier, path, exc)
        return None

    return G, optimo


# ---------------------------------------------------------------------------
# 3. Greedy voraz (mas fuerte que el de modelador_red.py: multiples reinicios)
# ---------------------------------------------------------------------------
def greedy_maxcut(G: nx.Graph, n_reinicios: int, seed: int) -> tuple[float, dict[str, int]]:
    """
    Busqueda local voraz: parte de una asignacion aleatoria y voltea
    nodos mientras eso mejore el corte, hasta un optimo local (mismo
    principio que modelador_red.py). La diferencia es que ESTA version
    corre 'n_reinicios' arranques aleatorios independientes y se queda
    con el mejor de todos -- modelador_red.py corre un unico arranque.
    Mas reinicios reducen la varianza y suben el piso de la razon de
    aproximacion por encima del ~0.5 esperado en el peor caso.
    """
    rng = np.random.default_rng(seed)
    nodos = list(G.nodes())
    mejor_corte_global = 0.0
    mejor_asignacion_global: dict[str, int] = {}

    for _ in range(n_reinicios):
        asignacion = {nodo: int(rng.integers(0, 2)) for nodo in nodos}

        mejorando = True
        while mejorando:
            mejorando = False
            for nodo in nodos:
                ganancia = sum(
                    G[nodo][vecino]["weight"] * (1 if asignacion[vecino] == asignacion[nodo] else -1)
                    for vecino in G.neighbors(nodo)
                )
                if ganancia > 1e-9:
                    asignacion[nodo] = 1 - asignacion[nodo]
                    mejorando = True

        corte = sum(
            data["weight"]
            for u, v, data in G.edges(data=True)
            if asignacion[u] != asignacion[v]
        )
        if corte > mejor_corte_global:
            mejor_corte_global = corte
            mejor_asignacion_global = asignacion.copy()

    return mejor_corte_global, mejor_asignacion_global


# ---------------------------------------------------------------------------
# 4. Goemans-Williamson (Goemans & Williamson, 1995, JACM 42(6))
# ---------------------------------------------------------------------------
def goemans_williamson_maxcut(G: nx.Graph, n_redondeos: int, seed: int) -> dict[str, Any] | None:
    """
    Relajacion semidefinida (SDP) de Max-Cut + redondeo por hiperplano
    aleatorio (Goemans & Williamson, 1995): cada nodo se representa como
    un vector unitario v_i que maximiza sum_ij w_ij (1 - v_i.v_j)/2; se
    prueban 'n_redondeos' hiperplanos aleatorios y se conserva el mejor
    corte. Garantia teorica en esperanza: E[corte] >= 0.878 * OPT.

    Devuelve None (nunca lanza excepcion) si cvxpy no esta disponible o
    el solver falla -- mismo patron de manejo de errores que
    modelador_red.py usa para sus dependencias opcionales.
    """
    if not CVXPY_AVAILABLE:
        logger.warning("cvxpy no disponible: Goemans-Williamson baseline = null")
        return None

    nodos = list(G.nodes())
    n = len(nodos)
    idx = {nodo: i for i, nodo in enumerate(nodos)}

    try:
        W = np.zeros((n, n))
        for u, v, data in G.edges(data=True):
            W[idx[u], idx[v]] = data["weight"]
            W[idx[v], idx[u]] = data["weight"]

        Y = cp.Variable((n, n), PSD=True)
        objetivo = cp.Maximize(0.25 * cp.sum(cp.multiply(W, 1 - Y)))
        restricciones = [cp.diag(Y) == 1]
        problema = cp.Problem(objetivo, restricciones)
        problema.solve(solver=cp.SCS)

        if Y.value is None:
            raise RuntimeError(f"SDP sin solucion (status: {problema.status})")

        Y_val = (Y.value + Y.value.T) / 2
        eigvals, eigvecs = np.linalg.eigh(Y_val)
        eigvals = np.clip(eigvals, 0, None)
        V = eigvecs @ np.diag(np.sqrt(eigvals))
    except Exception as exc:  # disponibilidad de solver varia por entorno
        logger.warning(
            "Goemans-Williamson SDP fallo (%s: %s); resultado null",
            type(exc).__name__, exc,
        )
        return None

    rng = np.random.default_rng(seed)
    mejor_corte = 0.0
    mejor_asignacion: dict[str, int] = {}
    for _ in range(n_redondeos):
        r = rng.normal(size=n)
        r /= np.linalg.norm(r)
        signos = np.sign(V @ r)
        signos[signos == 0] = 1

        asignacion = {nodo: int(signos[idx[nodo]] > 0) for nodo in nodos}
        corte = sum(
            data["weight"]
            for u, v, data in G.edges(data=True)
            if asignacion[u] != asignacion[v]
        )
        if corte > mejor_corte:
            mejor_corte = corte
            mejor_asignacion = asignacion.copy()

    return {"cut": mejor_corte, "assignment": mejor_asignacion, "sdp_bound": problema.value}


def gw_con_estadisticas(G: nx.Graph, n_corridas: int, n_redondeos: int, seed: int) -> dict[str, float] | None:
    """Corre GW varias veces (semillas distintas) y reporta media, desv.
    estandar y mejor corte -- necesario porque el redondeo es aleatorio.
    Devuelve None si GW no esta disponible."""
    cortes = []
    for i in range(n_corridas):
        resultado = goemans_williamson_maxcut(G, n_redondeos=n_redondeos, seed=seed + i)
        if resultado is None:
            return None
        cortes.append(resultado["cut"])
    cortes = np.array(cortes)
    return {"mean": float(cortes.mean()), "std": float(cortes.std()), "best": float(cortes.max())}


# ---------------------------------------------------------------------------
# 5. Reporte
# ---------------------------------------------------------------------------
def imprimir_tabla(tier: str, optimo: float, greedy_cut: float,
                    gw_stats: dict[str, float] | None) -> None:
    print(f"\n=== Instancia: {tier} ===")
    print(f"{'Metodo':<26}{'Corte':>12}{'Razon r':>12}{'Garantia teorica':>18}")
    print("-" * 68)
    print(f"{'Optimo (referencia)':<26}{optimo:>12.4f}{'1.0000':>12}{'':>18}")

    r_greedy = greedy_cut / optimo if optimo > 0 else float("nan")
    print(f"{'Greedy':<26}{greedy_cut:>12.4f}{r_greedy:>12.4f}{f'~{GREEDY_EXPECTED_RATIO}':>18}")

    if gw_stats is None:
        print(f"{'Goemans-Williamson':<26}{'N/D':>12}{'N/D':>12}{'':>18}")
    else:
        r_gw = gw_stats["best"] / optimo if optimo > 0 else float("nan")
        garantia_texto = f">= {GW_THEORETICAL_GUARANTEE}"
        print(f"{'Goemans-Williamson':<26}{gw_stats['best']:>12.4f}{r_gw:>12.4f}{garantia_texto:>18}")
        std_texto = f"+/- {gw_stats['std']:.4f}"
        print(f"{'  (10 corridas: media/std)':<26}{gw_stats['mean']:>12.4f}{std_texto:>12}")


# ---------------------------------------------------------------------------
# 6. Orquestacion
# ---------------------------------------------------------------------------
def run(cfg: Config) -> int:
    logger.info(
        "ISLA VERDE | linea base clasica v3 (rubrica oficial: GW + Greedy) | "
        "scratch_dir=%s | tiers=%s", cfg.scratch_dir, cfg.tiers,
    )

    alguna_instancia = False
    for tier in cfg.tiers:
        cargado = cargar_instancia(cfg.scratch_dir, tier)
        if cargado is None:
            continue
        G, optimo = cargado
        alguna_instancia = True

        greedy_cut, _ = greedy_maxcut(G, n_reinicios=cfg.n_reinicios_greedy, seed=cfg.seed)
        gw_stats = gw_con_estadisticas(
            G, n_corridas=cfg.n_corridas_gw, n_redondeos=cfg.n_redondeos_gw, seed=cfg.seed,
        )
        imprimir_tabla(tier, optimo, greedy_cut, gw_stats)

    if not alguna_instancia:
        logger.error(
            "Ninguna instancia disponible. Corre primero: "
            "python3 modelador_red.py --out-dir %s", cfg.scratch_dir,
        )
        return 1

    print(f"\nReferencia: Goemans, M. X., & Williamson, D. P. (1995). "
          f"Improved approximation algorithms for maximum cut and "
          f"satisfiability problems using semidefinite programming. "
          f"Journal of the ACM, 42(6), 1115-1145.")
    return 0


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Linea base clasica v3 - GW + Greedy (rubrica oficial)"
    )
    defaults = Config()
    parser.add_argument("--scratch-dir", type=Path, default=defaults.scratch_dir)
    parser.add_argument("--tiers", nargs="+", default=list(defaults.tiers))
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    return Config(scratch_dir=args.scratch_dir, tiers=tuple(args.tiers), seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    import sys
    sys.exit(main())
