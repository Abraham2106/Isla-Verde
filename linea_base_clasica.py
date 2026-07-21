
#!/usr/bin/env python3
"""
Linea base clasica v3 - Proyecto Isla Verde (alineada a la rubrica oficial)
=============================================================================

Rubrica oficial del reto (textual):
    "La linea base es el algoritmo de redondeo SDP de Goemans-Williamson
    (GW) (razon de aproximacion mayor o igual a 0.878). Tambien debe
    reportarse una linea base voraz (greedy) (razon aprox. 0.5).
    Referencia: Goemans & Williamson (1995), JACM 42(6)."

Este modulo entrega EXACTAMENTE eso (GW + Greedy, nada fuera de rubrica),
con cinco mejoras de calidad sobre la version base:

    1. Brecha contra la cota SDP: que tan cerca esta el mejor redondeo del
       limite teorico superior de la relajacion (sdp_bound).
    2. Verificacion empirica de la garantia teorica de GW (r_medio >= 0.878)
       reportada como PASS/WARN, mismo estilo que modelador_red.py.
    3. Distribucion de resultados en la instancia std12: Greedy se corre
       muchas veces con semillas distintas y se grafica su distribucion de
       cortes contra la de los redondeos de GW, para ilustrar por que GW
       es mas confiable (menor varianza).
    4. Estabilidad de la particion entre redondeos de GW: mide si distintos
       redondeos convergen a la MISMA particion (alta estabilidad) o a
       varias particiones optimas distintas (baja estabilidad) -- una
       metrica sobre la particion, no solo sobre el valor del corte.
    5. Pulido local post-GW: la particion que entrega el mejor redondeo de
       GW se pasa por la misma busqueda local de flips que usa Greedy.
       Nunca empeora el resultado (solo mejora o queda igual) y reutiliza
       el mismo motor de Greedy sobre el resultado de GW.

Autoria e independencia de la instancia:
    No se reconstruye el grafo desde los CSV. Se leen los JSON que ya
    exporta 'modelador_red.py' (instancias verificadas mvp8/std12/large16,
    las mismas que usara QAOA) y se usa su 'brute_force.cut' ya calculado
    como referencia del optimo.

Eficiencia: el SDP de GW se resuelve UNA sola vez por instancia (es la
parte cara); todos los redondeos reutilizan esa misma solucion, en vez de
resolver el SDP una vez por corrida como en versiones anteriores.

Uso:
    python3 linea_base_clasica_v3.py --scratch-dir ./scratch \
        --tiers mvp8 std12 large16 --distribucion-tier std12

Dependencias obligatorias: numpy, networkx
Dependencias opcionales:   cvxpy (Goemans-Williamson), matplotlib (grafica)

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

# ES: Dependencias opcionales: degradan con warning, nunca detienen el
#     pipeline (mismo patron que modelador_red.py).
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depende del entorno
    cp = None
    CVXPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - depende del entorno
    plt = None
    MATPLOTLIB_AVAILABLE = False

GW_THEORETICAL_GUARANTEE = 0.878   # Goemans & Williamson, 1995, JACM 42(6)
GREEDY_EXPECTED_RATIO = 0.5        # cota clasica (informativa) de la heuristica voraz
DEFAULT_TIERS: tuple[str, ...] = ("mvp8", "std12", "large16")


# ---------------------------------------------------------------------------
# 1. Configuracion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    scratch_dir: Path = Path("./scratch")
    tiers: tuple[str, ...] = DEFAULT_TIERS
    n_redondeos_gw: int = 300
    n_reinicios_greedy: int = 30
    n_muestras_estabilidad: int = 50
    distribucion_tier: str | None = "std12"
    n_repeticiones_distribucion: int = 30
    seed: int = 42


# ---------------------------------------------------------------------------
# 2. Carga de la instancia oficial (sin reconstruir el grafo por separado)
# ---------------------------------------------------------------------------
def cargar_instancia(scratch_dir: Path, tier: str) -> tuple[nx.Graph, float] | None:
    """Lee 'isla_verde_{tier}.json' (exportado por modelador_red.py) y
    arma el grafo EXACTAMENTE con esos nodos y pesos -- la misma
    instancia que usara QAOA. Devuelve tambien el optimo exacto ya
    calculado en ese JSON. Devuelve None, con warning, si falta el
    archivo o algun campo esperado."""
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


def valor_corte(G: nx.Graph, asignacion: dict[str, int]) -> float:
    """Corte ponderado de una particion dada."""
    return sum(
        data["weight"]
        for u, v, data in G.edges(data=True)
        if asignacion[u] != asignacion[v]
    )


# ---------------------------------------------------------------------------
# 3. Busqueda local de flips (motor compartido: nucleo de Greedy Y pulido post-GW)
# ---------------------------------------------------------------------------
def busqueda_local_flips(G: nx.Graph, asignacion_inicial: dict[str, int]) -> tuple[float, dict[str, int]]:
    """Voltea nodos mientras eso mejore el corte, hasta un optimo local.
    Se usa como nucleo de Greedy (arrancando de una asignacion aleatoria)
    Y como pulido final sobre el resultado de Goemans-Williamson
    (arrancando de la particion que entrego el mejor redondeo): el
    resultado nunca empeora respecto al punto de partida, solo mejora o
    queda igual."""
    asignacion = dict(asignacion_inicial)
    nodos = list(G.nodes())

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

    return valor_corte(G, asignacion), asignacion


# ---------------------------------------------------------------------------
# 4. Greedy voraz (multiples reinicios + busqueda local)
# ---------------------------------------------------------------------------
def greedy_maxcut(G: nx.Graph, n_reinicios: int, seed: int) -> dict[str, Any]:
    """Corre 'n_reinicios' arranques aleatorios independientes, cada uno
    llevado a un optimo local via busqueda_local_flips, y se queda con
    el mejor. Mas fuerte que un unico arranque (menor varianza, mejor
    piso de razon de aproximacion). Devuelve tambien la lista completa
    de cortes de cada reinicio, para poder graficar su distribucion."""
    rng = np.random.default_rng(seed)
    nodos = list(G.nodes())

    mejor_corte = 0.0
    mejor_asignacion: dict[str, int] = {}
    todos_los_cortes = []

    for _ in range(n_reinicios):
        inicio = {nodo: int(rng.integers(0, 2)) for nodo in nodos}
        corte, asignacion = busqueda_local_flips(G, inicio)
        todos_los_cortes.append(corte)
        if corte > mejor_corte:
            mejor_corte = corte
            mejor_asignacion = asignacion

    return {"cut": mejor_corte, "assignment": mejor_asignacion, "all_cuts": todos_los_cortes}


# ---------------------------------------------------------------------------
# 5. Goemans-Williamson: SDP resuelto UNA vez, reutilizado en muchos redondeos
# ---------------------------------------------------------------------------
def resolver_relajacion_sdp(G: nx.Graph) -> dict[str, Any] | None:
    """Resuelve la relajacion SDP de Max-Cut (Goemans & Williamson, 1995)
    UNA sola vez: cada nodo -> vector unitario v_i que maximiza
    sum_ij w_ij (1 - v_i.v_j)/2. Devuelve los vectores reconstruidos y la
    cota SDP (limite superior del optimo real). Devuelve None (nunca
    lanza excepcion) si cvxpy no esta disponible o el solver falla --
    mismo patron de manejo de errores que modelador_red.py."""
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

    return {"V": V, "idx": idx, "nodos": nodos, "sdp_bound": problema.value}


def redondeo_unico(G: nx.Graph, sdp: dict[str, Any], seed: int) -> tuple[float, dict[str, int]]:
    """Un unico redondeo por hiperplano aleatorio sobre la solucion SDP
    ya resuelta: elige un vector aleatorio r y separa los nodos segun el
    signo de v_i . r."""
    rng = np.random.default_rng(seed)
    V, idx, nodos = sdp["V"], sdp["idx"], sdp["nodos"]
    n = len(nodos)

    r = rng.normal(size=n)
    r /= np.linalg.norm(r)
    signos = np.sign(V @ r)
    signos[signos == 0] = 1

    asignacion = {nodo: int(signos[idx[nodo]] > 0) for nodo in nodos}
    return valor_corte(G, asignacion), asignacion


def canonizar_particion(asignacion: dict[str, int], referencia: dict[str, int]) -> dict[str, int]:
    """El corte es invariante ante complementar TODA la particion (cambiar
    todos los 0 por 1 y viceversa da el mismo corte). Para comparar dos
    particiones de forma justa, se complementa 'asignacion' si eso la
    acerca mas a 'referencia' antes de medir similitud."""
    nodos = list(referencia.keys())
    coincide = sum(1 for n in nodos if asignacion[n] == referencia[n])
    if coincide < len(nodos) / 2:
        return {n: 1 - v for n, v in asignacion.items()}
    return asignacion


def evaluar_goemans_williamson(G: nx.Graph, n_redondeos: int, seed: int,
                                n_muestras_estabilidad: int) -> dict[str, Any] | None:
    """Resuelve el SDP una vez y genera 'n_redondeos' redondeos
    independientes reutilizandolo (barato, ya que la parte cara -- el
    SDP -- solo se resuelve una vez). Devuelve:
        - mejor/peor corte encontrado (y sus particiones)
        - media y desviacion estandar de TODOS los redondeos
        - brecha contra la cota SDP: (sdp_bound - mejor_corte) / sdp_bound
        - estabilidad de la particion: que tan seguido distintos
          redondeos convergen a la MISMA particion (canonizada), medida
          sobre una muestra de 'n_muestras_estabilidad' redondeos
    """
    sdp = resolver_relajacion_sdp(G)
    if sdp is None:
        return None

    cortes = []
    asignaciones = []
    for i in range(n_redondeos):
        corte, asignacion = redondeo_unico(G, sdp, seed=seed + i)
        cortes.append(corte)
        asignaciones.append(asignacion)

    cortes_arr = np.array(cortes)
    idx_mejor = int(np.argmax(cortes_arr))
    idx_peor = int(np.argmin(cortes_arr))
    mejor_corte, mejor_asignacion = cortes[idx_mejor], asignaciones[idx_mejor]
    peor_corte = cortes[idx_peor]

    # --- estabilidad de la particion: acuerdo promedio con la de mejor corte ---
    muestra = asignaciones[:n_muestras_estabilidad]
    acuerdos = []
    for asignacion in muestra:
        canonizada = canonizar_particion(asignacion, mejor_asignacion)
        n_nodos = len(mejor_asignacion)
        coincidencias = sum(
            1 for nodo in mejor_asignacion if canonizada[nodo] == mejor_asignacion[nodo]
        )
        acuerdos.append(coincidencias / n_nodos)
    estabilidad_promedio = float(np.mean(acuerdos))

    # --- pulido local post-GW: aplica busqueda de flips sobre el mejor redondeo ---
    corte_pulido, asignacion_pulida = busqueda_local_flips(G, mejor_asignacion)

    return {
        "cut": mejor_corte,
        "worst_cut": peor_corte,
        "assignment": mejor_asignacion,
        "all_cuts": cortes,
        "mean": float(cortes_arr.mean()),
        "std": float(cortes_arr.std()),
        "sdp_bound": sdp["sdp_bound"],
        "gap_vs_sdp_bound": (sdp["sdp_bound"] - mejor_corte) / sdp["sdp_bound"] if sdp["sdp_bound"] > 0 else float("nan"),
        "estabilidad_particion": estabilidad_promedio,
        "polished_cut": corte_pulido,
        "polished_assignment": asignacion_pulida,
    }


# ---------------------------------------------------------------------------
# 6. Reporte por instancia
# ---------------------------------------------------------------------------
def imprimir_tabla(tier: str, optimo: float, greedy: dict[str, Any],
                    gw: dict[str, Any] | None) -> None:
    print(f"\n=== Instancia: {tier} ===")
    print(f"{'Metodo':<28}{'Corte':>12}{'Razon r':>12}{'Chequeo':>14}")
    print("-" * 66)
    print(f"{'Optimo (referencia)':<28}{optimo:>12.4f}{'1.0000':>12}{'':>14}")

    r_greedy = greedy["cut"] / optimo if optimo > 0 else float("nan")
    print(f"{'Greedy (mejor de N)':<28}{greedy['cut']:>12.4f}{r_greedy:>12.4f}"
          f"{f'~{GREEDY_EXPECTED_RATIO} (ref.)':>14}")

    if gw is None:
        print(f"{'Goemans-Williamson':<28}{'N/D':>12}{'N/D':>12}{'N/D':>14}")
        return

    r_mejor = gw["cut"] / optimo if optimo > 0 else float("nan")
    r_peor = gw["worst_cut"] / optimo if optimo > 0 else float("nan")
    r_medio = gw["mean"] / optimo if optimo > 0 else float("nan")

    # --- 2. verificacion empirica de la garantia teorica (sobre la media, que
    #        es donde aplica la garantia E[corte] >= 0.878 * OPT) ---
    cumple_garantia = r_medio >= GW_THEORETICAL_GUARANTEE
    estado = "PASS" if cumple_garantia else "WARN"

    print(f"{'Goemans-Williamson (mejor)':<28}{gw['cut']:>12.4f}{r_mejor:>12.4f}{'':>14}")
    print(f"{'  peor redondeo':<28}{gw['worst_cut']:>12.4f}{r_peor:>12.4f}{'':>14}")
    print(f"{'  media (redondeos)':<28}{gw['mean']:>12.4f}{r_medio:>12.4f} "
          f"{f'{estado} (>=0.878)':>14}")
    print(f"{'  desviacion estandar':<28}{gw['std']:>12.4f}")

    # --- 1. brecha contra la cota SDP ---
    print(f"\n  Cota SDP (limite teorico superior): {gw['sdp_bound']:.4f}")
    print(f"  Brecha del mejor redondeo vs cota SDP: {gw['gap_vs_sdp_bound'] * 100:.2f}%")

    # --- 4. estabilidad de la particion ---
    print(f"  Estabilidad de la particion (acuerdo promedio entre redondeos): "
          f"{gw['estabilidad_particion'] * 100:.1f}%")

    # --- 5. pulido local post-GW ---
    mejora_pulido = gw["polished_cut"] - gw["cut"]
    r_pulido = gw["polished_cut"] / optimo if optimo > 0 else float("nan")
    print(f"  Pulido local post-GW: {gw['polished_cut']:.4f}  (razon r = {r_pulido:.4f}, "
          f"mejora de {mejora_pulido:.4f} sobre el redondeo crudo)")


# ---------------------------------------------------------------------------
# 7. Grafica de distribucion (Greedy vs GW) para una instancia especifica
# ---------------------------------------------------------------------------
def graficar_distribucion(tier: str, cortes_greedy: list[float], cortes_gw: list[float],
                           optimo: float, out_path: Path) -> Path | None:
    """Histograma comparando la distribucion de cortes de Greedy (muchos
    reinicios) contra la de los redondeos de GW, para una instancia
    especifica (por defecto std12). Ilustra visualmente por que GW es
    mas confiable: su distribucion deberia estar mas concentrada y mas
    cerca del optimo que la de Greedy."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib no disponible: se omite la grafica de distribucion")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cortes_greedy, bins=15, alpha=0.6, label="Greedy (reinicios)", color="#888888")
    ax.hist(cortes_gw, bins=15, alpha=0.6, label="Goemans-Williamson (redondeos)", color="#1f77b4")
    ax.axvline(optimo, color="black", linestyle="--", linewidth=1.5, label="Optimo exacto")
    ax.set_xlabel("Corte")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribucion de resultados — instancia {tier}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Grafica de distribucion guardada en %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 8. Orquestacion
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

        greedy = greedy_maxcut(G, n_reinicios=cfg.n_reinicios_greedy, seed=cfg.seed)
        gw = evaluar_goemans_williamson(
            G, n_redondeos=cfg.n_redondeos_gw, seed=cfg.seed,
            n_muestras_estabilidad=cfg.n_muestras_estabilidad,
        )
        imprimir_tabla(tier, optimo, greedy, gw)

        # --- 3. distribucion (por defecto, solo para la instancia configurada) ---
        if tier == cfg.distribucion_tier and gw is not None:
            greedy_extra = greedy_maxcut(
                G, n_reinicios=cfg.n_repeticiones_distribucion, seed=cfg.seed + 1000,
            )
            out_path = cfg.scratch_dir / f"distribucion_{tier}.png"
            graficar_distribucion(
                tier, greedy_extra["all_cuts"], gw["all_cuts"], optimo, out_path,
            )

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
    parser.add_argument("--distribucion-tier", type=str, default=defaults.distribucion_tier,
                        help="tier sobre el que graficar la distribucion Greedy vs GW")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    return Config(
        scratch_dir=args.scratch_dir,
        tiers=tuple(args.tiers),
        distribucion_tier=args.distribucion_tier,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    import sys
    sys.exit(main())
