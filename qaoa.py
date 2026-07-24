#!/usr/bin/env python3
"""ISLA VERDE - Circuito QAOA + ejecucion local y en Quantinuum Nexus (H2).

Contenido:
    1. Carga de instancias desde los JSON de 'modelador_red.py'.
    2. Corte ponderado de un bitstring (misma definicion que la Fase 1).
    3. Circuito QAOA de p capas construido con pytket desde el Ising (h, J).
    4. Backend local (Qulacs) para el bucle de optimizacion de angulos.
    5. Ejecucion del circuito QAOA en Quantinuum Nexus (H2), via qnexus.
    6. Una corrida hibrida: optimizacion local + validacion final en H2.
    7. Estadistica sobre n_runs corridas (media +/- desviacion estandar).
    8. Barrido de p, figura "razon de aproximacion vs p" y export a JSON.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from pytket import Circuit
from pytket.backends.backendresult import BackendResult

# qnexus solo hace falta para el camino H2 (seccion 5 en adelante). Se importa
# de forma perezosa para que la construccion del circuito, el backend local y
# la formulacion QUBO/Ising se puedan importar y usar sin qnexus instalado
# (util para notebooks academicos o entornos sin acceso a Nexus).
try:
    import qnexus as qnx
except ImportError:  # pragma: no cover
    qnx = None


def _requiere_qnexus():
    """Falla con un mensaje claro si se usa una funcion H2 sin qnexus."""
    if qnx is None:
        raise ImportError(
            "Esta funcion necesita qnexus (camino H2), que no esta instalado. "
            "Instalar con 'pip install qnexus' o correr dentro del entorno de "
            "Quantinuum Nexus. Las funciones locales (circuito, Qulacs, QUBO) "
            "no requieren qnexus.")
    return qnx


logger = logging.getLogger("isla_verde.cuantico")

# Fase 3: BackendConfig objetivo. H2-Emulator esta alojado en Nexus (gratis,
# facturado en segundos). H2-1E/H2-2E son los emuladores en la nube de
# Quantinuum (facturados en HQC) y requieren aprovisionamiento aparte.
NEXUS_PROJECT = "isla-verde"
EMULATOR_NAME = "H2-Emulator"

# Cota de Goemans-Williamson (1995): garantia INFERIOR en esperanza del
# peor caso, NO un techo. Superarla en instancias faciles es normal y no
# demuestra ventaja cuantica. Se dibuja como referencia en la figura r vs p.
GW_GUARANTEE = 0.878

# Referencia al ULTIMO execute job enviado a H2. Si una espera truena por
# TimeoutError dentro de una funcion (y el ref local se pierde junto con la
# excepcion), el job sigue vivo en Nexus y puede recuperarse sin reenviar:
#     import qaoa
#     resultado = qaoa.esperar_resultado_h2_paciente(qaoa.ULTIMO_REF_JOB)
ULTIMO_REF_JOB = None


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
# 4. Backend local para el bucle de optimizacion (Paso 7.1)
# ===========================================================================
def obtener_backend_local():
    """QulacsBackend: simulador local, sin red, sin costo. Este es el
    backend que se usa DENTRO del bucle de COBYLA/BFGS (~40 evaluaciones
    por corrida): mandar cada una de esas evaluaciones a H2 por HTTPS
    seria demasiado lento (ver evaluar_angulos_h2 mas abajo, que si va a
    H2, pero solo se llama UNA VEZ por corrida, al final)."""
    from pytket.extensions.qulacs import QulacsBackend
    return QulacsBackend()


def evaluar_angulos_local(params: np.ndarray, instancia: dict, backend,
                          shots: int, p: int, seed: int | None = None
                          ) -> tuple[float, float, dict[str, int]]:
    """Corre el circuito QAOA con los angulos dados en el backend local y
    devuelve (corte_promedio, mejor_corte, mejor_bitstring). Misma logica
    de parseo que evaluar_angulos_h2, pero sin pasar por Nexus.

    `seed` fija el muestreo de shots de Qulacs: sin el, dos corridas con la
    misma semilla de angulos dan cifras distintas (el muestreo es aleatorio),
    lo que rompe la reproducibilidad que exige la rubrica. Con un seed fijo,
    cada circuito produce siempre los mismos counts; distintos angulos siguen
    dando counts distintos, pero de forma determinista y reproducible."""
    n = instancia["n"]
    variable_order = instancia["variable_order"]
    edges = instancia["edges"]
    gammas, betas = params[:p], params[p:]

    circ = construir_circuito_qaoa(n, instancia["h"], instancia["J_upper"],
                                    gammas, betas)
    circ = backend.get_compiled_circuit(circ)
    if seed is None:
        counts = backend.run_circuit(circ, n_shots=shots).get_counts()
    else:
        counts = backend.run_circuit(circ, n_shots=shots,
                                     seed=seed).get_counts()

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


# ===========================================================================
# 5. Ejecucion del circuito QAOA en Quantinuum Nexus (H2)
# ===========================================================================
# Fases 1-2: autenticacion y proyecto activo. qnx.login() es un no-op
# silencioso dentro de un notebook de Nexus Jupyterhub; en un entorno
# externo abre el flujo de login por navegador (o usar
# qnx.login_with_credentials()).
def obtener_proyecto_nexus(nombre: str = NEXUS_PROJECT):
    """Fase 1 (auth) + Fase 2 (proyecto activo)."""
    qnx = _requiere_qnexus()
    qnx.login()
    proyecto = qnx.projects.get_or_create(name=nombre)
    qnx.context.set_active_project(proyecto)
    return proyecto


def enviar_circuito_h2(circ: Circuit, n_shots: int, proyecto,
                        device_name: str = EMULATOR_NAME,
                        optimisation_level: int = 2,
                        timeout_compile: int = 300):
    """Fases 3-4: BackendConfig, upload y compilacion (bloqueante, via la
    convenience qnx.compile) + envio del execute job SIN esperarlo.

    A diferencia de la convenience qnx.execute(), start_execute_job()
    devuelve la referencia al job de inmediato: si el emulador esta en cola
    (comun cuando hay varios equipos usando H2-Emulator a la vez en el
    hackathon) y qnx.execute() truena por timeout, se pierde esa referencia
    y solo queda volver a enviar el job (empeorando la cola). Con
    start_execute_job se conserva el ref y se puede consultar/esperar de
    nuevo sobre EL MISMO job con esperar_resultado_h2()."""
    qnx = _requiere_qnexus()
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
        timeout=timeout_compile,
    )

    global ULTIMO_REF_JOB
    ref_execute_job = qnx.start_execute_job(
        programs=compilados,
        name=f"qaoa-execute-{sufijo}",
        n_shots=[n_shots],
        backend_config=config,
        project=proyecto,
    )
    ULTIMO_REF_JOB = ref_execute_job
    return ref_execute_job


def esperar_resultado_h2(ref_execute_job, timeout: int | None = None
                          ) -> BackendResult:
    """Fase 5: espera (o vuelve a esperar) el mismo execute job y descarga
    el resultado. Si truena por timeout, ref_execute_job sigue siendo
    valido: se puede volver a llamar esta funcion sin reenviar el job."""
    qnx = _requiere_qnexus()
    qnx.jobs.wait_for(ref_execute_job, timeout=timeout)
    return qnx.jobs.results(ref_execute_job)[0].download_result()


def esperar_resultado_h2_paciente(ref_execute_job, max_espera: int = 1800,
                                   intervalo: int = 30) -> BackendResult:
    """Variante de esperar_resultado_h2 para colas largas del emulador
    compartido: polling suave (un GET de status cada `intervalo` segundos)
    hasta `max_espera` segundos en total, sin websockets. Si se agota el
    tiempo lanza TimeoutError, pero el ref sigue valido: se puede volver a
    llamar esta funcion (o usar qaoa.ULTIMO_REF_JOB) sin reenviar el job."""
    qnx = _requiere_qnexus()
    limite = time.monotonic() + max_espera
    while True:
        estado = qnx.jobs.status(ref_execute_job).status
        nombre = getattr(estado, "value", str(estado))
        if nombre == "COMPLETED":
            break
        # Estados terminales de fallo documentados por Nexus: ademas de
        # ERROR/CANCELLED, TERMINATED (el job se detuvo) y DEPLETED (se agoto
        # la cuota/creditos). En cualquiera de ellos el job NO va a completar,
        # asi que se corta de inmediato en vez de seguir esperando en vano.
        if nombre in ("ERROR", "CANCELLED", "TERMINATED", "DEPLETED"):
            raise RuntimeError(f"El job termino en estado {nombre}")
        if time.monotonic() >= limite:
            raise TimeoutError(
                f"El job sigue en {nombre} tras {max_espera}s; el ref sigue "
                "valido: vuelve a llamar esperar_resultado_h2_paciente() "
                "para seguir esperando sin reenviar el job")
        time.sleep(intervalo)
    return qnx.jobs.results(ref_execute_job)[0].download_result()


def ejecutar_circuito_h2(circ: Circuit, n_shots: int, proyecto,
                          device_name: str = EMULATOR_NAME,
                          optimisation_level: int = 2,
                          timeout: int = 600) -> BackendResult:
    """Atajo: enviar_circuito_h2() + esperar_resultado_h2() en una sola
    llamada. Si esperas que el emulador este congestionado (cola larga),
    usa las dos funciones por separado para no perder el ref del job."""
    ref_execute_job = enviar_circuito_h2(
        circ, n_shots, proyecto, device_name=device_name,
        optimisation_level=optimisation_level)
    return esperar_resultado_h2(ref_execute_job, timeout=timeout)


def evaluar_angulos_h2(params: np.ndarray, instancia: dict, proyecto,
                       shots: int, p: int,
                       device_name: str = EMULATOR_NAME,
                       timeout_h2: int = 1800
                       ) -> tuple[float, float, dict[str, int]]:
    """Fase 6: parseo de resultados. Corre el circuito QAOA con los angulos
    dados en H2 y devuelve (corte_promedio, mejor_corte, mejor_bitstring),
    reutilizando valor_corte() para que la cifra sea comparable con las
    lineas base clasicas. Usa la espera paciente (polling suave) porque la
    cola del emulador compartido puede tardar varios minutos por job."""
    n = instancia["n"]
    variable_order = instancia["variable_order"]
    edges = instancia["edges"]
    gammas, betas = params[:p], params[p:]

    circ = construir_circuito_qaoa(n, instancia["h"], instancia["J_upper"],
                                    gammas, betas)
    ref_job = enviar_circuito_h2(circ, shots, proyecto,
                                 device_name=device_name)
    logger.info("Job H2 enviado: %s (si la espera truena, recuperar con "
                "qaoa.ULTIMO_REF_JOB)", ref_job.annotations.name)
    resultado = esperar_resultado_h2_paciente(ref_job, max_espera=timeout_h2)
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


# ===========================================================================
# 6. Una corrida hibrida: optimizacion local + validacion final en H2
# ===========================================================================
def una_ejecucion_hibrida(instancia: dict, backend_local, proyecto,
                          shots: int, p: int, x0: np.ndarray,
                          optimizer: str = "COBYLA",
                          device_name: str = EMULATOR_NAME,
                          timeout_h2: int = 1800,
                          seed_local: int | None = None) -> dict[str, Any]:
    """Paso 7.2. El objetivo es el VALOR ESPERADO del corte (estandar QAOA);
    como scipy minimiza, se minimiza su negativo. TODAS las evaluaciones
    dentro del bucle de COBYLA/BFGS van a backend_local (Qulacs, rapido, sin
    red). Al terminar, los mejores angulos encontrados se REEVALUAN UNA SOLA
    VEZ en H2 (evaluar_angulos_h2) y esa es la cifra que se reporta.

    `seed_local` fija el muestreo de Qulacs durante la optimizacion para que
    la corrida sea reproducible (mismos angulos -> misma trayectoria)."""
    mejor = {"esperado": -np.inf, "params": x0.copy()}

    def objetivo(params: np.ndarray) -> float:
        esperado, _, _ = evaluar_angulos_local(
            params, instancia, backend_local, shots, p, seed=seed_local)
        if esperado > mejor["esperado"]:
            mejor.update(esperado=esperado, params=params.copy())
        return -esperado

    if optimizer.upper() == "BFGS":
        minimize(objetivo, x0, method="BFGS", options={"maxiter": 40})
    else:
        minimize(objetivo, x0, method="COBYLA",
                 options={"maxiter": 40, "rhobeg": 0.5})

    esperado_h2, mejor_muestra_h2, bits_h2 = evaluar_angulos_h2(
        mejor["params"], instancia, proyecto, shots, p,
        device_name=device_name, timeout_h2=timeout_h2)

    optimo = instancia["optimum"]
    return {
        "cut_local": float(mejor["esperado"]),  # mejor E[corte] visto en local
        "cut_h2": float(esperado_h2),           # cifra principal reportada
        "ratio_h2": float(esperado_h2 / optimo) if optimo > 0 else float("nan"),
        "params": [float(v) for v in mejor["params"]],
        "mejor_muestra_h2": {"cut": float(mejor_muestra_h2), "bits": bits_h2},
    }


# ===========================================================================
# 7. n_runs corridas hibridas con arranques distintos -> estadistica
# ===========================================================================
def resolver_instancia_hibrida(instancia: dict, backend_local, proyecto,
                               shots: int, p: int, n_runs: int,
                               optimizer: str = "COBYLA", seed: int = 42,
                               device_name: str = EMULATOR_NAME,
                               timeout_h2: int = 1800) -> dict[str, Any]:
    """Paso 7.4a. Corre n_runs corridas hibridas independientes (arranques
    x0 distintos, derivados de la semilla) y agrega media y desviacion
    estandar sobre las n_runs cifras de H2. La rubrica exige n_runs >= 5;
    cada corrida hace UNA llamada a H2 (contar el tiempo de cola).

    La cifra por corrida es el VALOR ESPERADO del corte reevaluado en H2
    con los mejores angulos (nunca el mejor shot, que es solo diagnostico
    y esta sesgado alto por construccion)."""
    tier = instancia["tier"]
    if n_runs < 5:
        logger.warning("[%s] n_runs=%d < 5: la rubrica exige media +/- "
                       "desviacion estandar sobre >= 5 ejecuciones. Usar "
                       "n_runs >= 5 para las cifras del informe.",
                       tier, n_runs)
    rng = np.random.default_rng(seed)

    corridas = []
    for run in range(n_runs):
        x0 = rng.uniform(0.0, np.pi, size=2 * p)
        # Semilla local por corrida, derivada de la maestra (reproducible):
        # fija el muestreo de Qulacs durante la optimizacion de esta corrida.
        seed_local = int(rng.integers(0, 2**31 - 1))
        res = una_ejecucion_hibrida(instancia, backend_local, proyecto,
                                    shots, p, x0, optimizer=optimizer,
                                    device_name=device_name,
                                    timeout_h2=timeout_h2,
                                    seed_local=seed_local)
        res["run"] = run + 1
        res["seed_local"] = seed_local
        corridas.append(res)
        logger.info("[%s] p=%d run %d/%d: E[cut] H2=%.4f  r=%.4f  "
                    "(local=%.4f)", tier, p, run + 1, n_runs,
                    res["cut_h2"], res["ratio_h2"], res["cut_local"])

    cortes = np.array([c["cut_h2"] for c in corridas])
    ratios = np.array([c["ratio_h2"] for c in corridas])
    return {
        "tier": tier, "p": p, "n_runs": n_runs, "optimizer": optimizer,
        "shots": shots, "seed": seed,
        "objetivo": "valor_esperado",
        "backend": f"h2-{device_name}",
        "instance_sha256": instancia["instance_sha256"],
        "optimum": instancia["optimum"],
        # cifra principal: E[corte] en H2 por corrida, media/std entre corridas
        "cut_mean": float(cortes.mean()), "cut_std": float(cortes.std()),
        "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std()),
        # datos secundarios, solo para transparencia
        "ratio_best": float(ratios.max()), "ratio_worst": float(ratios.min()),
        # procedencia por corrida: angulos finales y cifras individuales
        "runs": corridas,
    }


# ===========================================================================
# 8. Barrido de p, figura "razon de aproximacion vs p" y export a JSON
# ===========================================================================
def barrer_p_hibrido(scratch_dir: Path, tiers: list[str],
                     p_values: list[int], backend_local, proyecto,
                     shots: int, n_runs: int, optimizer: str = "COBYLA",
                     seed: int = 42, device_name: str = EMULATOR_NAME,
                     timeout_h2: int = 1800
                     ) -> dict[str, list[dict[str, Any]]]:
    """Paso 7.4b. Para cada instancia y cada p, corre el bloque completo de
    n_runs corridas hibridas. Total de jobs H2 = len(tiers disponibles) *
    len(p_values) * n_runs; con la cola compartida, estimar varios minutos
    por job. Devuelve {tier: [res_p1, res_p2, ...]}."""
    resultados: dict[str, list[dict[str, Any]]] = {}
    for tier in tiers:
        instancia = cargar_instancia(scratch_dir, tier)
        if instancia is None:
            continue
        serie = []
        for p in p_values:
            logger.info("[%s] p=%d (%d qubits, %d corridas)...",
                        tier, p, instancia["n"], n_runs)
            res = resolver_instancia_hibrida(
                instancia, backend_local, proyecto, shots, p, n_runs,
                optimizer=optimizer, seed=seed, device_name=device_name,
                timeout_h2=timeout_h2)
            serie.append(res)
            print(f"  [{tier}] p={p}: r = {res['ratio_mean']:.4f} "
                  f"+/- {res['ratio_std']:.4f}   "
                  f"(mejor {res['ratio_best']:.4f}, "
                  f"peor {res['ratio_worst']:.4f})")
        resultados[tier] = serie
    return resultados


def _baselines_ratios(scratch_dir: Path, tier: str
                      ) -> tuple[float | None, float | None]:
    """Lee del JSON de la instancia los ratios MEDIDOS de las lineas base
    clasicas: (greedy_r, gw_r) = (corte / optimo). None si falta el dato."""
    path = scratch_dir / f"isla_verde_{tier}.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_bytes().decode("utf-8"))
    maxcut = payload.get("baselines", {}).get("maxcut")
    if not maxcut:
        return None, None
    optimo = float(maxcut["brute_force"]["cut"])
    if optimo <= 0:
        return None, None
    greedy_r = float(maxcut["greedy"]["cut"]) / optimo
    gw = maxcut.get("goemans_williamson")
    gw_r = (float(gw["cut"]) / optimo) if gw else None
    return greedy_r, gw_r


def graficar_r_vs_p(resultados: dict[str, list[dict[str, Any]]],
                    out_path: Path, scratch_dir: Path | None = None
                    ) -> Path | None:
    """Figura de la rubrica: razon de aproximacion vs p con barras de error
    (desviacion estandar sobre las n_runs corridas).

    Si se pasa scratch_dir, ademas de la cota TEORICA 0.878 se dibujan los
    ratios MEDIDOS de greedy y GW por instancia (lineas horizontales en el
    color de cada tier). Esto evita que se lea el 0.878 como techo: en estas
    instancias faciles greedy y GW ya alcanzan ~1.0, y mostrarlo es la
    comparacion honesta que pide la rubrica."""
    if not resultados:
        logger.warning("Sin resultados: no se genera figura")
        return None

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 5))
    handles_tier = []
    for tier, serie in resultados.items():
        ps = [r["p"] for r in serie]
        medias = [r["ratio_mean"] for r in serie]
        stds = [r["ratio_std"] for r in serie]
        linea = ax.errorbar(ps, medias, yerr=stds, marker="o", capsize=4,
                            label=tier)
        color = linea[0].get_color()
        handles_tier.append(Line2D([0], [0], color=color, marker="o",
                                   label=tier))
        # Ratios medidos de las lineas base clasicas para este tier.
        if scratch_dir is not None:
            greedy_r, gw_r = _baselines_ratios(scratch_dir, tier)
            if greedy_r is not None:
                ax.axhline(greedy_r, color=color, linestyle=":", linewidth=1.2)
            if gw_r is not None:
                ax.axhline(gw_r, color=color, linestyle="-.", linewidth=1.2)

    ax.axhline(GW_GUARANTEE, color="black", linestyle="--", linewidth=1)

    # Leyenda en dos bloques: colores = instancia; estilos = metodo.
    handles_estilo = [
        Line2D([0], [0], color="gray", marker="o", linestyle="-",
               label="QAOA (media +/- desv. est.)"),
        Line2D([0], [0], color="gray", linestyle=":",
               label="Greedy (medido)"),
        Line2D([0], [0], color="gray", linestyle="-.",
               label="Goemans-Williamson (medido)"),
        Line2D([0], [0], color="black", linestyle="--",
               label=f"Cota GW teorica {GW_GUARANTEE} (piso, no techo)"),
    ]

    todos_p = sorted({r["p"] for serie in resultados.values() for r in serie})
    ax.set_xticks(todos_p)
    ax.set_xlabel("Numero de capas p")
    ax.set_ylabel("Razon de aproximacion r = corte / optimo")
    ax.set_title("QAOA en H2 via qnexus: razon de aproximacion vs p")
    ax.set_ylim(0, 1.05)
    leyenda_tier = ax.legend(handles=handles_tier, loc="lower left",
                             fontsize=8, title="Instancia")
    ax.add_artist(leyenda_tier)
    ax.legend(handles=handles_estilo, loc="lower right", fontsize=8,
              title="Metodo")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Figura guardada en %s", out_path)
    return out_path


def guardar_resultados(resultados: dict[str, list[dict[str, Any]]],
                       out_path: Path) -> Path:
    """Exporta el barrido completo (con procedencia por corrida) a JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    logger.info("Resultados guardados en %s", out_path)
    return out_path


def resultados_para_comparador(resultados: dict[str, list[dict[str, Any]]],
                               p_reporte: int | None = None
                               ) -> dict[str, dict[str, Any]]:
    """Adaptador de formato. barrer_p_hibrido() / guardar_resultados()
    producen el barrido completo {tier: [res_p1, res_p2, ...]}, pero
    comparar_qaoa.py espera el formato PLANO {tier: {"cut", "p", ...}} con
    una sola cifra por instancia. Esta funcion selecciona, por cada tier, el
    resultado del p a reportar (por defecto el p mas grande del barrido) y
    devuelve el diccionario plano.

    La cifra es cut_mean (la MEDIA del valor esperado del corte en H2 sobre
    las n_runs corridas), nunca el mejor shot: reportar el maximo seria
    cherry-picking, marcado como red flag por la rubrica."""
    plano: dict[str, dict[str, Any]] = {}
    for tier, serie in resultados.items():
        if not serie:
            continue
        if p_reporte is None:
            res = max(serie, key=lambda r: r["p"])
        else:
            res = next((r for r in serie if r["p"] == p_reporte), None)
            if res is None:
                logger.warning("[%s] no hay resultado para p=%d; se omite del "
                               "JSON del comparador", tier, p_reporte)
                continue
        plano[tier] = {
            "cut": res["cut_mean"],            # media de E[corte] en H2
            "p": res["p"],
            "objetivo": res["objetivo"],       # "valor_esperado"
            "instance_sha256": res["instance_sha256"],
            "n_runs": res["n_runs"],
            "shots": res["shots"],
            "backend": res.get("backend"),
            "ratio_mean": res["ratio_mean"],
            "ratio_std": res["ratio_std"],
        }
    return plano


def guardar_para_comparador(resultados: dict[str, list[dict[str, Any]]],
                            out_path: Path,
                            p_reporte: int | None = None) -> Path:
    """Escribe el JSON PLANO que consume comparar_qaoa.py (--qaoa-results).
    Correr el comparador con:
        python comparar_qaoa.py --scratch-dir scratch --tiers mvp8 std12 \\
            --qaoa-results scratch/qaoa_results.json"""
    plano = resultados_para_comparador(resultados, p_reporte=p_reporte)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plano, f, indent=2, ensure_ascii=False)
    logger.info("JSON para el comparador guardado en %s", out_path)
    return out_path
