#!/usr/bin/env python3
"""Punto de entrada único del pipeline reproducible de Isla Verde."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("isla_verde.main")

PROFILES = {
    "smoke": {
        "tiers": ("mvp8",),
        "p_values": (1,),
        "shots": 128,
        "n_runs": 1,
        "maxiter": 6,
    },
    "report": {
        "tiers": ("mvp8", "std12", "large16"),
        "p_values": (1, 2, 3),
        "shots": 1000,
        "n_runs": 5,
        "maxiter": 40,
    },
}


def _run(name: str, script: str, arguments: Sequence[str], out_dir: Path) -> None:
    command = (
        [sys.executable, "-m", *arguments]
        if script == "-m"
        else [sys.executable, str(ROOT / script), *arguments]
    )
    env = os.environ.copy()
    env.setdefault("XDG_CONFIG_HOME", str(out_dir / ".config"))
    logger.info("[%s] %s", name, " ".join(command))
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} terminó con código {result.returncode}")


def _validate(out_dir: Path, tiers: Sequence[str], profile: str) -> dict:
    sweep_path = out_dir / "qaoa_barrido_p.json"
    comparison_path = out_dir / "comparacion_qaoa_vs_clasico.png"
    required = [out_dir / "isla_verde_index.json", sweep_path, comparison_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Faltan artefactos: {missing}")

    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    checks = []
    for tier in tiers:
        instance_path = out_dir / f"isla_verde_{tier}.json"
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
        optimum = float(instance["baselines"]["maxcut"]["brute_force"]["cut"])
        greedy = float(instance["baselines"]["maxcut"]["greedy"]["cut"])
        gw = instance["baselines"]["maxcut"]["goemans_williamson"]
        if optimum <= 0 or greedy > optimum + 1e-6:
            raise RuntimeError(f"{tier}: baseline inválida")
        if gw and float(gw["cut"]) > optimum + 1e-6:
            raise RuntimeError(f"{tier}: GW supera el óptimo exacto")

        expected_sha = hashlib.sha256(instance_path.read_bytes()).hexdigest()
        for item in sweep["results"][tier]:
            if item["instance_sha256"] != expected_sha:
                raise RuntimeError(f"{tier}: QAOA usó otra instancia")
            if not 0 <= item["ratio_mean"] <= 1 + 1e-9:
                raise RuntimeError(f"{tier}: razón QAOA fuera de [0,1]")
            if profile == "report" and item["n_runs"] < 5:
                raise RuntimeError(f"{tier}: report requiere >=5 corridas")
            for run in item["runs"]:
                if run["circuit"]["parameter_count"] != 2 * item["p"]:
                    raise RuntimeError(f"{tier}: parámetros QAOA != 2p")
        checks.append({"tier": tier, "optimum": optimum, "status": "PASS"})

    summary = {
        "status": "PASS",
        "profile": profile,
        "tiers": list(tiers),
        "external_jobs_submitted": False,
        "checks": checks,
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    profile = PROFILES[args.profile]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not args.skip_install:
            _run(
                "dependencias",
                "-m",
                ["pip", "install", "-r", str(ROOT / "requirements.txt")],
                out_dir,
            )
        _run(
            "modelado",
            "modelador_red.py",
            ["--data-dir", str(args.data_dir.resolve()), "--out-dir", str(out_dir), "--seed", str(args.seed)],
            out_dir,
        )
        _run(
            "baselines",
            "linea_base_clasica.py",
            ["--scratch-dir", str(out_dir), "--tiers", *profile["tiers"], "--seed", str(args.seed)],
            out_dir,
        )
        _run(
            "qaoa-local",
            "ejecutar_qaoa_local.py",
            [
                "--scratch-dir", str(out_dir),
                "--out", str(out_dir / "qaoa_barrido_p.json"),
                "--comparison-out", str(out_dir / "qaoa_results.json"),
                "--tiers", *profile["tiers"],
                "--p-values", *(str(p) for p in profile["p_values"]),
                "--shots", str(profile["shots"]),
                "--n-runs", str(profile["n_runs"]),
                "--maxiter", str(profile["maxiter"]),
                "--seed", str(args.seed),
            ],
            out_dir,
        )
        _run(
            "comparación",
            "comparar_qaoa.py",
            [
                "--scratch-dir", str(out_dir),
                "--qaoa-results", str(out_dir / "qaoa_results.json"),
                "--tiers", *profile["tiers"],
            ],
            out_dir,
        )
        summary = _validate(out_dir, profile["tiers"], args.profile)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Pipeline abortado: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
