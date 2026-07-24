#!/usr/bin/env python3
"""ISLA VERDE - Construccion del circuito QAOA.

Contenido:
    1. Carga de instancias desde los JSON de 'modelador_red.py'.
    2. Corte ponderado de un bitstring (misma definicion que la Fase 1).
    3. Circuito QAOA de p capas construido con pytket desde el Ising (h, J).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pytket import Circuit
from pytket.backends.backendresult import BackendResult

import qnexus as qnx

logger = logging.getLogger("isla_verde.cuantico")

# Fase 3: BackendConfig objetivo. H2-Emulator esta alojado en Nexus (gratis,
# facturado en segundos). H2-1E/H2-2E son los emuladores en la nube de
# Quantinuum (facturados en HQC) y requieren aprovisionamiento aparte.
NEXUS_PROJECT = "isla-verde"
EMULATOR_NAME = "H2-Emulator"


# ===========================================================================
# 1. Carga de la instancia (JSON producido por modelador_red.py)
# ===========================================================================
def cargar_instancia(scratch_dir: Path, tier: str) -> dict[str, Any] | None:
    """Lee isla_verde_{tier}.json y construye el Ising de Max-Cut PURO
    (h=0, J_ij=w_ij/2) desde las aristas: es el mismo problema cuyo optimo
    exacto calcula la fuerza bruta, asi numerador y denominador de r son
    comparables. El "ising" completo del JSON (con alpha/beta) codifica el
    Hamiltoniano restringido, que es OTRO problema, y no se usa aqui.
    None con warning si falta el archivo."""
    path = scratch_dir / f"isla_verde_{tier}.json"
    if not path.exists():
        logger.warning("[%s] no se encontro %s. Corre primero la etapa "
                       "clasica (modelador_red.py --out-dir %s)",
                       tier, path, scratch_dir)
        return None

    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))

    try:
        variable_order = payload["variable_order"]
        edges = payload["edges"]
        optimum = float(payload["baselines"]["maxcut"]["brute_force"]["cut"])
    except KeyError as exc:
        logger.warning("[%s] %s no tiene el campo esperado %s", tier, path, exc)
        return None

    n = len(variable_order)
    indice = {name: q for q, name in enumerate(variable_order)}
    j_upper = np.zeros((n, n), dtype=np.float64)
    for arista in edges:
        i, j = sorted((indice[arista["u"]], indice[arista["v"]]))
        j_upper[i, j] += float(arista["weight"]) / 2.0
    h = np.zeros(n, dtype=np.float64)

    # Contraste contra el ising_maxcut exportado por modelador_red.py.
    exportado = payload.get("ising_maxcut")
    if exportado is not None and not np.allclose(
            j_upper, np.asarray(exportado["J_upper"], dtype=np.float64),
            atol=1e-9):
        logger.error("[%s] ising_maxcut del JSON no coincide con las "
                     "aristas; se usa el derivado de las aristas", tier)

    # ES: Normalizacion del Hamiltoniano del circuito: |J| alcanza ~10^2 y
    #     COBYLA veria un paisaje ultra-oscilatorio en gamma. Escalar H_C por
    #     una constante solo reparametriza gamma (misma familia de estados);
    #     el corte reportado se evalua siempre desde las aristas sin escalar.
    escala = float(np.abs(j_upper).max())
    if escala <= 0.0:
        escala = 1.0

    return {"tier": tier, "variable_order": variable_order,
            "h": h / escala, "J_upper": j_upper / escala,
            "escala_hamiltoniano": escala, "edges": edges,
            "optimum": optimum, "n": n,
            "instance_sha256": hashlib.sha256(raw).hexdigest()}


# ===========================================================================
# 2. Corte ponderado de un bitstring (misma definicion que la Fase 1)
# ===========================================================================
def valor_corte(edges: list[dict], bits: dict[str, int]) -> float:
    """Suma de pesos de las aristas cuyos extremos caen en islas distintas.
    Identica a la usada por greedy y GW, para que los numeros sean
    directamente comparables. NO es la energia del Ising."""
    total = 0.0
    for arista in edges:
        u, v, w = arista["u"], arista["v"], float(arista["weight"])
        if bits[u] != bits[v]:
            total += w
    return total


# ===========================================================================
# 3. Circuito QAOA de p capas desde el Hamiltoniano Ising
# ===========================================================================
def construir_circuito_qaoa(n: int, h: np.ndarray, j_upper: np.ndarray,
                             gammas: np.ndarray, betas: np.ndarray) -> Circuit:
    """QAOA de p capas. Por cada capa k:
        Hamiltoniano de costo (gamma_k):
            campo local h_i -> Rz(2*gamma_k*h_i)
            acople J_ij     -> CX(i,j) . Rz(2*gamma_k*J_ij) . CX(i,j)
        Hamiltoniano de mezcla (beta_k): Rx(2*beta_k) en cada qubit.
    Antes: Hadamard en todos (superposicion uniforme). Al final: medicion.
    Los angulos de pytket van en unidades de PI, de ahi la division."""
    p = len(gammas)
    circ = Circuit(n, n)

    for q in range(n):
        circ.H(q)

    for k in range(p):
        gamma, beta = float(gammas[k]), float(betas[k])
        for i in range(n):
            if abs(h[i]) > 1e-12:
                circ.Rz(2.0 * gamma * h[i] / np.pi, i)
        for i in range(n):
            for j in range(i + 1, n):
                jij = j_upper[i, j]
                if abs(jij) > 1e-12:
                    circ.CX(i, j)
                    circ.Rz(2.0 * gamma * jij / np.pi, j)
                    circ.CX(i, j)
        for q in range(n):
            circ.Rx(2.0 * beta / np.pi, q)

    for q in range(n):
        circ.Measure(q, q)

    return circ


# ===========================================================================
# 4. Ejecucion del circuito QAOA en Quantinuum Nexus (H2)
# ===========================================================================
# Fases 1-2: autenticacion y proyecto activo. qnx.login() es un no-op
# silencioso dentro de un notebook de Nexus Jupyterhub; en un entorno
# externo abre el flujo de login por navegador (o usar
# qnx.login_with_credentials()).
def obtener_proyecto_nexus(nombre: str = NEXUS_PROJECT):
    """Fase 1 (auth) + Fase 2 (proyecto activo)."""
    qnx.login()
    proyecto = qnx.projects.get_or_create(name=nombre)
    qnx.context.set_active_project(proyecto)
    return proyecto


def ejecutar_circuito_h2(circ: Circuit, n_shots: int, proyecto,
                          device_name: str = EMULATOR_NAME,
                          optimisation_level: int = 2,
                          timeout: int = 600) -> BackendResult:
    """Fases 3-5: BackendConfig, upload, compilacion y ejecucion en Nexus.

    Usa las convenience functions qnx.compile()/qnx.execute() (bloqueantes,
    con timeout) en vez de manejar a mano start_compile_job/
    start_execute_job + jobs.wait_for + jobs.results + download_result: son
    el mismo flujo pero en una sola llamada cada una, tal como lo documenta
    qnexus para jobs que se espera terminen rapido."""
    sufijo = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    config = qnx.QuantinuumConfig(device_name=device_name)

    ref_circ = qnx.circuits.upload(
        circuit=circ, name=f"qaoa-{sufijo}", project=proyecto)

    compilados = qnx.compile(
        programs=[ref_circ],
        name=f"qaoa-compile-{sufijo}",
        optimisation_level=optimisation_level,
        backend_config=config,
        project=proyecto,
        timeout=timeout,
    )

    resultados = qnx.execute(
        programs=compilados,
        name=f"qaoa-execute-{sufijo}",
        n_shots=[n_shots],
        backend_config=config,
        project=proyecto,
        timeout=timeout,
    )
    return resultados[0]


def evaluar_angulos_h2(params: np.ndarray, instancia: dict, proyecto,
                       shots: int, p: int,
                       device_name: str = EMULATOR_NAME
                       ) -> tuple[float, float, dict[str, int]]:
    """Fase 6: parseo de resultados. Corre el circuito QAOA con los angulos
    dados en H2 y devuelve (corte_promedio, mejor_corte, mejor_bitstring),
    reutilizando valor_corte() para que la cifra sea comparable con las
    lineas base clasicas."""
    n = instancia["n"]
    variable_order = instancia["variable_order"]
    edges = instancia["edges"]
    gammas, betas = params[:p], params[p:]

    circ = construir_circuito_qaoa(n, instancia["h"], instancia["J_upper"],
                                    gammas, betas)
    resultado = ejecutar_circuito_h2(circ, shots, proyecto,
                                     device_name=device_name)
    counts = resultado.get_counts()

    suma, total = 0.0, 0
    mejor_corte, mejor_bits = -np.inf, {}
    for lectura, veces in counts.items():
        bits = {variable_order[q]: int(lectura[q]) for q in range(n)}
        corte = valor_corte(edges, bits)
        suma += corte * veces
        total += veces
        if corte > mejor_corte:
            mejor_corte, mejor_bits = corte, bits

    return (suma / total if total else 0.0), mejor_corte, mejor_bits
