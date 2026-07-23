  #!/usr/bin/env python3
"""ISLA VERDE v3.0 - Fase 2 / Phase 2: Comparacion QAOA vs lineas base clasicas.

ES: Lee los JSON por instancia exportados por 'modelador_red.py' (que ya
    incluyen fuerza bruta, greedy y Goemans-Williamson bajo
    baselines.maxcut), incorpora los resultados de QAOA entregados por el
    equipo cuantico, calcula el approximation ratio de cada metodo contra el
    optimo exacto, e imprime una tabla y una figura comparativa por
    instancia (mvp8 / std12 / large16).
EN: Reads the per-instance JSON files exported by 'modelador_red.py' (which
    already include brute force, greedy, and Goemans-Williamson under
    baselines.maxcut), ingests the QAOA results supplied by the quantum
    team, computes each method's approximation ratio against the exact
    optimum, and prints a table plus a comparison figure per instance
    (mvp8 / std12 / large16).

Uso / Usage:
    python3 comparar_qaoa.py --scratch-dir /ruta/scratch \
        --qaoa-results /ruta/qaoa_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger("isla_verde.comparar_qaoa")

DEFAULT_TIERS: tuple[str, ...] = ("mvp8", "std12", "large16")
# ES: Garantia INFERIOR en esperanza de GW (piso del peor caso, NO un techo:
#     superarla en instancias faciles es legitimo). / EN: GW lower bound in
# expectation (worst-case floor, NOT a ceiling).
GW_THEORETICAL_GUARANTEE = 0.878  # Goemans & Williamson, 1995
# ES: Ningun corte puede superar el optimo exacto de fuerza bruta; un ratio
#     mayor que 1 + tolerancia solo puede ser un bug (instancias distintas,
#     denominador equivocado o cherry-picking) y se rechaza.
# EN: No cut can exceed the exact brute-force optimum; ratio > 1 + tolerance
#     can only be a bug (mismatched instances, wrong denominator, or
#     cherry-picking) and is rejected.
RATIO_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Config:
    scratch_dir: Path = Path("./scratch")
    qaoa_results_path: Path | None = None
    tiers: tuple[str, ...] = DEFAULT_TIERS
    out_filename: str = "comparacion_qaoa_vs_clasico.png"


def load_qaoa_results(path: Path | None, tiers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    empty = {tier: {"cut": None, "p": None} for tier in tiers}
    if path is None:
        logger.warning(
            "Sin --qaoa-results / no --qaoa-results given: todos los tiers "
            "quedan pendientes / all tiers marked pending"
        )
        return empty
    if not path.exists():
        logger.warning(
            "No se encontro %s / file not found: todos los tiers pendientes "
            "/ all tiers pending", path,
        )
        return empty

    # utf-8-sig: tolera el BOM que agregan editores/PowerShell en Windows.
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    results = dict(empty)
    for tier in tiers:
        if tier in raw and raw[tier].get("cut") is not None:
            results[tier] = {
                "cut": float(raw[tier]["cut"]),
                "p": raw[tier].get("p"),
                "objetivo": raw[tier].get("objetivo"),
                "instance_sha256": raw[tier].get("instance_sha256"),
            }
            if results[tier]["objetivo"] not in (None, "valor_esperado"):
                logger.warning(
                    "[%s] el JSON de QAOA declara objetivo=%r; la cifra "
                    "valida para comparar es el VALOR ESPERADO del corte, "
                    "no el mejor shot / expected cut required, not best "
                    "sample", tier, results[tier]["objetivo"],
                )
        else:
            logger.warning("[%s] sin resultado QAOA en %s / no QAOA result "
                           "in %s: pendiente / pending", tier, path, path)
    return results


def load_baselines(
    scratch_dir: Path, tier: str
) -> tuple[dict[str, Any], str] | None:
    """Devuelve (baselines.maxcut, sha256 del archivo de instancia) para
    poder validar que QAOA corrio sobre EXACTAMENTE esta instancia (los
    resultados de Baseline/ sobre el grafo completo de 46 nodos NO son
    comparables con un tier de 8-16)."""
    path = scratch_dir / f"isla_verde_{tier}.json"
    if not path.exists():
        logger.warning(
            "[%s] no se encontro %s / not found. Corre primero / run first: "
            "python3 modelador_red.py --out-dir %s", tier, path, scratch_dir,
        )
        return None

    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))

    baselines = payload.get("baselines", {}).get("maxcut")
    if baselines is None:
        logger.warning("[%s] %s no contiene baselines.maxcut / missing "
                       "baselines.maxcut", tier, path)
        return None
    return baselines, hashlib.sha256(raw).hexdigest()


@dataclass
class TierComparison:
    tier: str
    optimum: float
    greedy_cut: float
    greedy_r: float
    gw_cut: float | None
    gw_r: float | None
    qaoa_cut: float | None
    qaoa_p: int | None
    qaoa_r: float | None


def compute_ratios(tier: str, baselines: dict[str, Any], qaoa: dict[str, Any]) -> TierComparison:
    optimum = float(baselines["brute_force"]["cut"])
    greedy_cut = float(baselines["greedy"]["cut"])
    gw = baselines["goemans_williamson"]
    gw_cut = float(gw["cut"]) if gw is not None else None
    qaoa_cut = qaoa["cut"]
    qaoa_p = qaoa.get("p")

    return TierComparison(
        tier=tier,
        optimum=optimum,
        greedy_cut=greedy_cut,
        greedy_r=greedy_cut / optimum if optimum > 0 else float("nan"),
        gw_cut=gw_cut,
        gw_r=(gw_cut / optimum) if (gw_cut is not None and optimum > 0) else None,
        qaoa_cut=qaoa_cut,
        qaoa_p=qaoa_p,
        qaoa_r=(qaoa_cut / optimum) if (qaoa_cut is not None and optimum > 0) else None,
    )


def format_ratio(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def print_table(comparison: TierComparison) -> None:
    width = 24
    print(f"\n=== Instancia / instance: {comparison.tier} ===")
    print(f"{'Metodo / method':<{width}}{'Corte / cut':>14}{'Razon r':>12}")
    print("-" * (width + 26))
    print(f"{'Optimo (fuerza bruta)':<{width}}{comparison.optimum:>14.4f}{'1.0000':>12}")
    print(f"{'Greedy':<{width}}{comparison.greedy_cut:>14.4f}{comparison.greedy_r:>12.4f}")

    gw_cut_text = f"{comparison.gw_cut:.4f}" if comparison.gw_cut is not None else "N/D"
    print(f"{'Goemans-Williamson':<{width}}{gw_cut_text:>14}{format_ratio(comparison.gw_r):>12}")

    if comparison.qaoa_cut is not None:
        etiqueta = (f"QAOA E[corte] (p={comparison.qaoa_p})"
                    if comparison.qaoa_p else "QAOA E[corte]")
        print(f"{etiqueta:<{width}}{comparison.qaoa_cut:>14.4f}{comparison.qaoa_r:>12.4f}")
    else:
        print(f"{'QAOA':<{width}}{'(pendiente)':>14}{'(pendiente)':>12}")


def plot_comparison(comparisons: list[TierComparison], out_path: Path) -> Path:
    tiers = [c.tier for c in comparisons]
    metodos = [
        ("greedy_r", "Greedy", "#888888"),
        ("gw_r", "Goemans-Williamson", "#1f77b4"),
        ("qaoa_r", "QAOA (valor esperado)", "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    n_metodos = len(metodos)
    ancho = 0.8 / n_metodos
    posiciones = range(len(tiers))

    for offset, (campo, etiqueta, color) in enumerate(metodos):
        valores = [getattr(c, campo) or 0.0 for c in comparisons]
        pos_offset = [p + offset * ancho for p in posiciones]
        ax.bar(pos_offset, valores, width=ancho, label=etiqueta, color=color)

    ax.axhline(GW_THEORETICAL_GUARANTEE, color="black", linestyle="--",
               linewidth=1,
               label=f"Cota GW {GW_THEORETICAL_GUARANTEE} "
                     "(piso en esperanza, no techo)")
    ax.set_xticks([p + ancho * (n_metodos - 1) / 2 for p in posiciones])
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Razon de aproximacion / approximation ratio (r = corte/optimo)")
    ax.set_title("QAOA vs lineas base clasicas — Proyecto Isla Verde")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Figura guardada / figure saved to %s", out_path)
    return out_path


def run(cfg: Config) -> int:
    logger.info(
        "ISLA VERDE Fase/Phase 2 | scratch_dir=%s | qaoa_results=%s",
        cfg.scratch_dir, cfg.qaoa_results_path,
    )
    qaoa_results = load_qaoa_results(cfg.qaoa_results_path, cfg.tiers)

    comparisons: list[TierComparison] = []
    violaciones: list[str] = []
    for tier in cfg.tiers:
        cargado = load_baselines(cfg.scratch_dir, tier)
        if cargado is None:
            continue
        baselines, instancia_sha = cargado

        # ES: QAOA debe haber corrido sobre EXACTAMENTE esta instancia; los
        #     resultados del grafo completo (Baseline/) no son comparables.
        # EN: QAOA must have run on EXACTLY this instance; full-graph results
        #     (Baseline/) are not comparable with a tier.
        sha_qaoa = qaoa_results[tier].get("instance_sha256")
        if sha_qaoa is not None and sha_qaoa != instancia_sha:
            logger.error(
                "[%s] el resultado QAOA proviene de OTRA instancia (sha256 "
                "%s... != %s...): no comparable / QAOA result computed on a "
                "DIFFERENT instance, not comparable",
                tier, sha_qaoa[:12], instancia_sha[:12],
            )
            violaciones.append(f"{tier}: instancia distinta (sha256)")

        comparison = compute_ratios(tier, baselines, qaoa_results[tier])
        comparisons.append(comparison)
        print_table(comparison)

        # ES: Guardrail: contra el optimo exacto, r > 1 es imposible.
        # EN: Guardrail: against the exact optimum, r > 1 is impossible.
        for metodo, ratio in (("greedy", comparison.greedy_r),
                              ("goemans_williamson", comparison.gw_r),
                              ("qaoa", comparison.qaoa_r)):
            if ratio is not None and ratio > 1.0 + RATIO_TOLERANCE:
                logger.error(
                    "[%s] ratio %s = %.6f > 1: imposible contra el optimo "
                    "exacto; revisa instancia, denominador o cherry-picking "
                    "/ impossible vs exact optimum; check instance, "
                    "denominator, or cherry-picking", tier, metodo, ratio,
                )
                violaciones.append(f"{tier}: {metodo} r={ratio:.6f} > 1")

    if not comparisons:
        logger.error(
            "Ninguna instancia disponible / no instances available. Corre "
            "primero / run first: python3 modelador_red.py --out-dir %s",
            cfg.scratch_dir,
        )
        return 1

    out_path = cfg.scratch_dir / cfg.out_filename
    plot_comparison(comparisons, out_path)

    pendientes = [c.tier for c in comparisons if c.qaoa_cut is None]
    if pendientes:
        logger.warning(
            "[PENDIENTE / PENDING] falta resultado QAOA para / missing QAOA "
            "result for: %s. Pasa --qaoa-results con el JSON del equipo "
            "cuantico / pass --qaoa-results with the quantum team's JSON.",
            ", ".join(pendientes),
        )

    if violaciones:
        logger.error(
            "COMPARACION INVALIDA / INVALID COMPARISON: %s",
            "; ".join(violaciones),
        )
        return 2
    return 0


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="ISLA VERDE Fase 2 / Phase 2: QAOA vs lineas base clasicas"
    )
    defaults = Config()
    parser.add_argument("--scratch-dir", type=Path, default=defaults.scratch_dir,
                        help="directorio con los isla_verde_{tier}.json")
    parser.add_argument("--qaoa-results", type=Path, default=None,
                        help="JSON con los resultados de QAOA por tier")
    parser.add_argument("--tiers", nargs="+", default=list(defaults.tiers),
                        help=f"tiers a comparar (default: {list(defaults.tiers)})")
    args = parser.parse_args(argv)
    return Config(
        scratch_dir=args.scratch_dir,
        qaoa_results_path=args.qaoa_results,
        tiers=tuple(args.tiers),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
