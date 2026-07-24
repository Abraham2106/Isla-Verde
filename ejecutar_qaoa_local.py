#!/usr/bin/env python3
"""Barrido QAOA reproducible en Qulacs; nunca envía trabajos a H2."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / "scratch" / ".config"))

import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qaoa import (
    cargar_instancia,
    construir_circuito_qaoa,
    evaluar_angulos_local,
    obtener_backend_local,
)

logger = logging.getLogger("isla_verde.qaoa_local")


def _corrida(
    instancia: dict[str, Any],
    backend: Any,
    *,
    p: int,
    shots: int,
    maxiter: int,
    seed_init: int,
    seed_sampling: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed_init)
    x0 = rng.uniform(0.0, np.pi, size=2 * p)
    best_cut = -np.inf
    best_params = x0.copy()

    def objective(params: np.ndarray) -> float:
        nonlocal best_cut, best_params
        cut, _, _ = evaluar_angulos_local(
            params, instancia, backend, shots, p, seed=seed_sampling
        )
        if cut > best_cut:
            best_cut = float(cut)
            best_params = np.asarray(params, dtype=float).copy()
        return -float(cut)

    result = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    cut, best_sample, bits = evaluar_angulos_local(
        best_params, instancia, backend, shots, p, seed=seed_sampling
    )
    ratio = float(cut / instancia["optimum"])
    if not 0.0 <= ratio <= 1.0 + 1e-9:
        raise ValueError(f"{instancia['tier']} p={p}: razón inválida {ratio}")

    circuit = construir_circuito_qaoa(
        instancia["n"],
        instancia["h"],
        instancia["J_upper"],
        best_params[:p],
        best_params[p:],
    )
    if len(best_params) != 2 * p:
        raise AssertionError("QAOA debe tener exactamente 2p parámetros")

    return {
        "seed_initialization": seed_init,
        "seed_sampling": seed_sampling,
        "params": best_params.tolist(),
        "cut_expected": float(cut),
        "ratio": ratio,
        "best_sample": {"cut": float(best_sample), "bits": bits},
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
        },
        "circuit": {
            "qubits": int(circuit.n_qubits),
            "layers_p": p,
            "parameter_count": len(best_params),
            "gates": len(circuit.get_commands()),
            "depth": int(circuit.depth()),
        },
    }


def ejecutar(
    scratch_dir: Path,
    tiers: Sequence[str],
    p_values: Sequence[int],
    *,
    shots: int,
    n_runs: int,
    maxiter: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if shots < 1 or n_runs < 1 or maxiter < 1:
        raise ValueError("shots, n_runs y maxiter deben ser positivos")
    if any(p < 1 for p in p_values):
        raise ValueError("p debe ser >= 1")
    if n_runs < 5:
        logger.warning("n_runs=%d: diagnóstico smoke, no cifra de informe", n_runs)

    backend = obtener_backend_local()
    rng = np.random.default_rng(seed)
    sweep: dict[str, list[dict[str, Any]]] = {}
    for tier in tiers:
        instance = cargar_instancia(scratch_dir, tier)
        if instance is None:
            raise FileNotFoundError(f"Falta isla_verde_{tier}.json")
        series = []
        for p in p_values:
            runs = []
            logger.info("[%s] p=%d, corridas=%d", tier, p, n_runs)
            for run_number in range(1, n_runs + 1):
                run = _corrida(
                    instance,
                    backend,
                    p=p,
                    shots=shots,
                    maxiter=maxiter,
                    seed_init=int(rng.integers(0, 2**31 - 1)),
                    seed_sampling=int(rng.integers(0, 2**31 - 1)),
                )
                run["run"] = run_number
                runs.append(run)
            cuts = np.asarray([run["cut_expected"] for run in runs])
            ratios = np.asarray([run["ratio"] for run in runs])
            item = {
                "tier": tier,
                "qubits": int(instance["n"]),
                "p": p,
                "shots": shots,
                "n_runs": n_runs,
                "maxiter": maxiter,
                "backend": "local-qulacs",
                "objetivo": "valor_esperado",
                "instance_sha256": instance["instance_sha256"],
                "optimum": float(instance["optimum"]),
                "cut_mean": float(cuts.mean()),
                "cut_std": float(cuts.std()),
                "ratio_mean": float(ratios.mean()),
                "ratio_std": float(ratios.std()),
                "runs": runs,
            }
            series.append(item)
            logger.info(
                "[%s] p=%d: r=%.4f ± %.4f",
                tier,
                p,
                item["ratio_mean"],
                item["ratio_std"],
            )
        sweep[tier] = series
    return sweep


def guardar_figura(sweep: dict[str, list[dict[str, Any]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier, series in sweep.items():
        ax.errorbar(
            [item["p"] for item in series],
            [item["ratio_mean"] for item in series],
            yerr=[item["ratio_std"] for item in series],
            marker="o",
            capsize=4,
            label=tier,
        )
    ax.axhline(0.878, color="black", linestyle="--", label="cota GW 0.878")
    ax.set(xlabel="Capas p", ylabel="r = E[corte] / óptimo", ylim=(0, 1.05))
    ax.set_title("QAOA local (Qulacs): razón de aproximación vs p")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / "scratch")
    parser.add_argument("--out", type=Path, default=ROOT / "scratch/qaoa_barrido_p.json")
    parser.add_argument(
        "--comparison-out", type=Path, default=ROOT / "scratch/qaoa_results.json"
    )
    parser.add_argument("--tiers", nargs="+", default=["mvp8"])
    parser.add_argument("--p-values", nargs="+", type=int, default=[1])
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    try:
        sweep = ejecutar(
            args.scratch_dir,
            args.tiers,
            args.p_values,
            shots=args.shots,
            n_runs=args.n_runs,
            maxiter=args.maxiter,
            seed=args.seed,
        )
        payload = {
            "schema_version": 1,
            "artifact_type": "isla_verde.qaoa_local_sweep",
            "external_jobs_submitted": False,
            "config": vars(args) | {"scratch_dir": str(args.scratch_dir), "out": str(args.out), "comparison_out": str(args.comparison_out)},
            "results": sweep,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        flat = {}
        for tier, series in sweep.items():
            selected = max(series, key=lambda item: item["p"])
            flat[tier] = {
                key: selected[key]
                for key in (
                    "cut_mean",
                    "p",
                    "objetivo",
                    "instance_sha256",
                    "n_runs",
                    "shots",
                    "backend",
                    "ratio_mean",
                    "ratio_std",
                )
            }
            flat[tier]["cut"] = flat[tier].pop("cut_mean")
        args.comparison_out.write_text(
            json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        guardar_figura(sweep, args.scratch_dir / "qaoa_ratio_vs_p.png")
    except (AssertionError, FileNotFoundError, ImportError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
