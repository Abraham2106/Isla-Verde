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
GW_THEORETICAL_GUARANTEE = 0.878  # Goemans & Williamson, 1995


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

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results = dict(empty)
    for tier in tiers:
        if tier in raw and raw[tier].get("cut") is not None:
            results[tier] = {"cut": float(raw[tier]["cut"]), "p": raw[tier].get("p")}
        else:
            logger.warning("[%s] sin resultado QAOA en %s / no QAOA result "
                           "in %s: pendiente / pending", tier, path, path)
    return results


def load_baselines(scratch_dir: Path, tier: str) -> dict[str, Any] | None:
    path = scratch_dir / f"isla_verde_{tier}.json"
    if not path.exists():
        logger.warning(
            "[%s] no se encontro %s / not found. Corre primero / run first: "
            "python3 modelador_red.py --out-dir %s", tier, path, scratch_dir,
        )
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    baselines = payload.get("baselines", {}).get("maxcut")
    if baselines is None:
        logger.warning("[%s] %s no contiene baselines.maxcut / missing "
                       "baselines.maxcut", tier, path)
        return None
    return baselines


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
        etiqueta = f"QAOA (p={comparison.qaoa_p})" if comparison.qaoa_p else "QAOA"
        print(f"{etiqueta:<{width}}{comparison.qaoa_cut:>14.4f}{comparison.qaoa_r:>12.4f}")
    else:
        print(f"{'QAOA':<{width}}{'(pendiente)':>14}{'(pendiente)':>12}")


def plot_comparison(comparisons: list[TierComparison], out_path: Path) -> Path:
    tiers = [c.tier for c in comparisons]
    metodos = [
        ("greedy_r", "Greedy", "#888888"),
        ("gw_r", "Goemans-Williamson", "#1f77b4"),
        ("qaoa_r", "QAOA", "#d62728"),
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
               linewidth=1, label=f"Garantia teorica GW ({GW_THEORETICAL_GUARANTEE})")
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
    for tier in cfg.tiers:
        baselines = load_baselines(cfg.scratch_dir, tier)
        if baselines is None:
            continue
        comparison = compute_ratios(tier, baselines, qaoa_results[tier])
        comparisons.append(comparison)
        print_table(comparison)

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
