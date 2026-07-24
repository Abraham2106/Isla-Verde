#!/usr/bin/env python3
"""ISLA VERDE - Codigo de deteccion de errores "Iceberg" [[k+2,k,2]] sobre QAOA.

Capa OPCIONAL de proteccion de circuitos. NO modifica qaoa.py: importa de el
la carga de instancia, el valor de corte y la cota GW, y ofrece una version
CODIFICADA del circuito QAOA de Max-Cut con post-seleccion por sindrome.

Referencias:
    Self, Benedetti, Amaro, "Protecting Expressive Circuits with a Quantum
        Error Detection Code" (arXiv:2211.06703). Define el codigo Iceberg.
    He, Amaro, Shaydulin, Pistoia, "Performance of QAOA with Quantum Error
        Detection" (arXiv:2409.12104). Aplica Iceberg a QAOA en H2 y observa
        mejora hasta 20 qubits logicos frente al circuito sin codificar.

Idea del codigo (n = k+2 fisicos: k datos 0..k-1, top t=k, bottom b=k+1):
    Estabilizadores  SX = X^{(x)n},  SZ = Z^{(x)n}  (espacio codigo = +1 de ambos).
    Operadores logicos:  X_i = X_i X_t,   Z_i = Z_i Z_b.
    Acople logico:       Z_i Z_j = Z_i Z_j  (SIN sobrecosto: 1 puerta ZZ nativa).
    Mezcla logica:       exp(-i b X_i) = exp(-i b X_i X_t)  (1 puerta XX de 2 qubits).
    Estado inicial QAOA |+>^k logico = H^{(x)n} GHZ_Z  (GHZ + Hadamard en todos).

Post-seleccion (QED): si en alguna medida de sindrome SX o SZ se lee -1, el
disparo se DESCARTA. No hay decodificacion: solo se conservan los disparos
limpios (a costa de una tasa de descarte que crece con qubits/profundidad).

Presupuesto de qubits: el circuito usa n+2 = k+4 qubits fisicos (2 ancillas de
sindrome reutilizadas). Emulador H2 de Nexus: tope de 20 qubits, asi que
mvp8 -> 12, std12 -> 16, large16 -> 20 (justo en el limite; instancias
mayores no caben codificadas en el emulador alojado).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from pytket import Circuit, OpType

# Reutilizamos el nucleo sin tocarlo: misma instancia, mismo valor de corte,
# misma cota GW. Asi el corte codificado es directamente comparable con QAOA
# sin codificar y con las lineas base clasicas.
from qaoa import cargar_instancia, valor_corte, GW_GUARANTEE  # noqa: F401

logger = logging.getLogger("isla_verde.iceberg")

_TOL = 1e-12


# ===========================================================================
# 1. Construccion del circuito QAOA codificado con el codigo Iceberg
# ===========================================================================
def construir_circuito_qaoa_iceberg(instancia: dict, gammas: np.ndarray,
                                    betas: np.ndarray,
                                    rondas_sindrome: int = 0
                                    ) -> tuple[Circuit, dict[str, Any]]:
    """QAOA de p=len(gammas) capas codificado en el Iceberg [[k+2,k,2]].

    Layout fisico:  datos 0..k-1,  top t=k,  bottom b=k+1,
                    ancillas de sindrome a1=k+2 (SZ), a2=k+3 (SX).

    `rondas_sindrome` = numero de rondas de sindrome MID-circuito repartidas
    entre las capas QAOA (requieren backend con medida intermedia + reset, p.
    ej. el emulador H2; Qulacs local no las soporta -> usar 0). Ademas de esas,
    SIEMPRE se hace una ronda terminal de SX y la medida destructiva final en Z
    (de la que se deriva SZ y los bits logicos), como en la Fig. 1(e) del paper.

    Devuelve (circuito, meta) donde meta describe la posicion de cada bit
    clasico para que decodificar_disparo() sepa leer el sindrome y los datos.
    """
    k = instancia["n"]              # numero de qubits logicos = nodos del grafo
    if k % 2 != 0:
        raise ValueError(f"El codigo Iceberg exige k par; instancia k={k}. "
                         "Anade un nodo aislado (peso 0) para completar a par.")
    n = k + 2                       # qubits fisicos de datos+top+bottom
    t, b = k, k + 1                 # top y bottom
    a1, a2 = n, n + 1               # ancillas de sindrome (SZ, SX)
    h = np.asarray(instancia["h"], dtype=float)
    J = np.asarray(instancia["J_upper"], dtype=float)
    p = len(gammas)

    # Bits clasicos: 2 por ronda mid (SX,SZ) + 1 SX terminal + n medida final.
    n_cbits = 2 * rondas_sindrome + 1 + n
    circ = Circuit(n + 2, n_cbits)  # n datos + 2 ancillas de sindrome
    cbit = 0                        # contador de bits clasicos
    sindrome_bits: list[int] = []   # bits que deben leer 0 (SX/SZ mid-circuito)

    def zz(alpha: float, i: int, j: int) -> None:
        # exp(-i alpha/2 * Z_i Z_j) nativo en iones atrapados (Molmer-Sorensen).
        circ.add_gate(OpType.ZZPhase, alpha / np.pi, [i, j])

    def xx(alpha: float, i: int, j: int) -> None:
        circ.add_gate(OpType.XXPhase, alpha / np.pi, [i, j])

    # --- Inicializacion logica |+>^k = H^{(x)n} GHZ_Z ----------------------
    # GHZ_Z: t en |+>, CX(t -> resto) => (|0..0> + |1..1>)/raiz2 sobre los n.
    circ.H(t)
    circ.CX(t, b)
    for i in range(k):
        circ.CX(t, i)
    for q in range(n):             # H en todos: GHZ_Z -> |+>^k logico
        circ.H(q)

    # --- Capas QAOA sobre operadores logicos -------------------------------
    def ronda_sindrome_mid() -> None:
        # Sindrome intermedio con ancillas reutilizadas (necesita reset).
        nonlocal cbit
        # SZ = paridad Z de los n fisicos -> ancilla a1 en |0>, CX(q -> a1).
        circ.add_gate(OpType.Reset, [a1])
        for q in range(n):
            circ.CX(q, a1)
        circ.Measure(a1, cbit); sindrome_bits.append(cbit); cbit += 1
        # SX = X^{(x)n} -> ancilla a2 en |+>, CX(a2 -> q), medir a2 en X.
        circ.add_gate(OpType.Reset, [a2])
        circ.H(a2)
        for q in range(n):
            circ.CX(a2, q)
        circ.H(a2)
        circ.Measure(a2, cbit); sindrome_bits.append(cbit); cbit += 1

    cortes_por_ronda = _repartir(p, rondas_sindrome)
    capa = 0
    for bloque in cortes_por_ronda:
        for _ in range(bloque):
            gamma, beta = float(gammas[capa]), float(betas[capa])
            # Costo: campo local h_i -> Z_i Z_b ; acople J_ij -> Z_i Z_j.
            for i in range(k):
                if abs(h[i]) > _TOL:
                    zz(2.0 * gamma * h[i], i, b)
            for i in range(k):
                for j in range(i + 1, k):
                    if abs(J[i, j]) > _TOL:
                        zz(2.0 * gamma * J[i, j], i, j)
            # Mezcla: exp(-i beta X_i) = exp(-i beta X_i X_t) por qubit logico.
            for i in range(k):
                xx(2.0 * beta, i, t)
            capa += 1
        if capa < p:               # sindrome intermedio entre bloques de capas
            ronda_sindrome_mid()

    # --- Ronda terminal de SX + medida destructiva (deriva SZ y bits) ------
    if rondas_sindrome > 0:        # a2 solo necesita reset si ya se uso antes
        circ.add_gate(OpType.Reset, [a2])
    circ.H(a2)
    for q in range(n):
        circ.CX(a2, q)
    circ.H(a2)
    circ.Measure(a2, cbit)
    sx_terminal = cbit; cbit += 1

    datos_bits: dict[int, int] = {}
    for q in range(n):             # medida destructiva de los n fisicos en Z
        circ.Measure(q, cbit)
        datos_bits[q] = cbit; cbit += 1

    meta = {"k": k, "n": n, "t": t, "b": b,
            "variable_order": instancia["variable_order"],
            "sindrome_bits": sindrome_bits, "sx_terminal": sx_terminal,
            "datos_bits": datos_bits, "n_cbits": cbit}
    return circ, meta


def _repartir(p: int, rondas: int) -> list[int]:
    """Divide p capas en (rondas+1) bloques casi iguales; los sindromes
    intermedios van entre bloques. rondas=0 -> un solo bloque de p capas."""
    bloques = rondas + 1
    base, resto = divmod(p, bloques)
    return [base + (1 if i < resto else 0) for i in range(bloques)]


# ===========================================================================
# 2. Decodificacion y post-seleccion de los disparos
# ===========================================================================
def decodificar_counts_iceberg(counts: dict[tuple[int, ...], int], meta: dict
                               ) -> tuple[dict[str, dict[str, int]], int, int]:
    """Filtra los disparos que pasan la deteccion de errores y los decodifica
    a bits logicos. Un disparo se DESCARTA si algun bit de sindrome (SX/SZ
    intermedio o SX terminal) es 1, o si la paridad SZ de la medida final es
    impar. Para los que pasan, el bit logico i = z_i XOR z_b.

    Devuelve (counts_logicos, n_conservados, n_descartados) donde
    counts_logicos mapea la clave de bitstring logico a {bits, veces}."""
    k, b = meta["k"], meta["b"]
    sindrome_bits = meta["sindrome_bits"]
    sx_terminal = meta["sx_terminal"]
    datos_bits = meta["datos_bits"]
    variable_order = meta["variable_order"]

    logicos: dict[str, dict[str, int]] = {}
    conservados = descartados = 0
    for lectura, veces in counts.items():
        # 1) sindromes intermedios y SX terminal deben ser 0.
        if any(lectura[c] for c in sindrome_bits) or lectura[sx_terminal]:
            descartados += veces
            continue
        # 2) paridad SZ de la medida destructiva (todos los n fisicos) par.
        z = [lectura[datos_bits[q]] for q in range(meta["n"])]
        if sum(z) % 2 != 0:
            descartados += veces
            continue
        # 3) decodificar: bit logico i = z_i XOR z_b.
        bits = {variable_order[i]: z[i] ^ z[b] for i in range(k)}
        clave = "".join(str(bits[v]) for v in variable_order)
        entrada = logicos.setdefault(clave, {"bits": bits, "veces": 0})
        entrada["veces"] += veces
        conservados += veces

    return logicos, conservados, descartados


# ===========================================================================
# 3. Evaluacion de angulos (local, con post-seleccion) - espejo de qaoa.py
# ===========================================================================
def evaluar_angulos_local_iceberg(params: np.ndarray, instancia: dict, backend,
                                  shots: int, p: int, seed: int | None = None,
                                  rondas_sindrome: int = 0
                                  ) -> tuple[float, float, dict[str, int], float]:
    """Analogo codificado de qaoa.evaluar_angulos_local. Corre el circuito
    Iceberg en el backend local, post-selecciona y devuelve
    (corte_promedio, mejor_corte, mejor_bitstring, tasa_descarte).

    El corte se evalua SIEMPRE con qaoa.valor_corte sobre las aristas, para que
    la cifra sea identica en definicion a QAOA sin codificar, greedy y GW."""
    edges = instancia["edges"]
    gammas, betas = params[:p], params[p:]

    circ, meta = construir_circuito_qaoa_iceberg(
        instancia, gammas, betas, rondas_sindrome=rondas_sindrome)
    circ = backend.get_compiled_circuit(circ)
    if seed is None:
        counts = backend.run_circuit(circ, n_shots=shots).get_counts()
    else:
        counts = backend.run_circuit(circ, n_shots=shots, seed=seed).get_counts()

    logicos, conservados, descartados = decodificar_counts_iceberg(counts, meta)
    total = conservados + descartados
    tasa_descarte = (descartados / total) if total else 1.0

    if conservados == 0:           # todos los disparos descartados
        return 0.0, 0.0, {}, tasa_descarte

    suma = 0.0
    mejor_corte, mejor_bits = -np.inf, {}
    for entrada in logicos.values():
        corte = valor_corte(edges, entrada["bits"])
        suma += corte * entrada["veces"]
        if corte > mejor_corte:
            mejor_corte, mejor_bits = corte, entrada["bits"]

    return suma / conservados, mejor_corte, mejor_bits, tasa_descarte


# ===========================================================================
# 4. Una corrida local codificada: optimizacion de angulos + post-seleccion
# ===========================================================================
def una_corrida_local_iceberg(instancia: dict, backend, shots: int, p: int,
                              x0: np.ndarray, seed_local: int | None = None,
                              maxiter: int = 30, rondas_sindrome: int = 0
                              ) -> dict[str, Any]:
    """Optimiza los angulos QAOA maximizando el corte esperado POST-SELECCIONADO
    (COBYLA). Espejo local de qaoa.una_ejecucion_hibrida, sin H2. Devuelve la
    cifra principal (corte esperado), la razon frente al optimo y la tasa de
    descarte con los angulos optimos."""
    mejor = {"esperado": -np.inf, "params": x0.copy()}

    def objetivo(params: np.ndarray) -> float:
        esperado, _, _, _ = evaluar_angulos_local_iceberg(
            params, instancia, backend, shots, p, seed=seed_local,
            rondas_sindrome=rondas_sindrome)
        if esperado > mejor["esperado"]:
            mejor.update(esperado=esperado, params=params.copy())
        return -esperado

    minimize(objetivo, x0, method="COBYLA",
             options={"maxiter": maxiter, "rhobeg": 0.5})

    esperado, mejor_corte, bits, tasa = evaluar_angulos_local_iceberg(
        mejor["params"], instancia, backend, shots, p, seed=seed_local,
        rondas_sindrome=rondas_sindrome)
    optimo = instancia["optimum"]
    return {
        "cut": float(esperado),
        "ratio": float(esperado / optimo) if optimo > 0 else float("nan"),
        "best_cut": float(mejor_corte),
        "discard_rate": float(tasa),
        "params": [float(v) for v in mejor["params"]],
        "mejor_bits": bits,
    }


# ===========================================================================
# 5. Demo: comparacion local codificado (Iceberg) vs sin codificar
# ===========================================================================
def demo(scratch_dir: Path, tier: str = "mvp8", p: int = 1, shots: int = 2000,
         seed: int = 42, maxiter: int = 25) -> dict[str, Any] | None:
    """Comparacion rapida en el simulador local (sin ruido, sin Nexus):
    QAOA codificado con Iceberg vs QAOA sin codificar, misma instancia y
    mismos angulos iniciales. Sin ruido las razones deben ser parecidas; la
    utilidad del Iceberg aparece con ruido (emulador H2 / hardware). Aqui se
    valida que el pipeline codificado corre y post-selecciona correctamente."""
    from qaoa import evaluar_angulos_local, obtener_backend_local

    instancia = cargar_instancia(scratch_dir, tier)
    if instancia is None:
        return None
    backend = obtener_backend_local()
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, np.pi, size=2 * p)
    seed_local = int(rng.integers(0, 2**31 - 1))

    # Codificado (Iceberg, con post-seleccion).
    enc = una_corrida_local_iceberg(instancia, backend, shots, p, x0,
                                    seed_local=seed_local, maxiter=maxiter)

    # Sin codificar (nucleo qaoa.py), mismos angulos iniciales.
    mejor = {"esperado": -np.inf, "params": x0.copy()}

    def objetivo(params):
        esp, _, _ = evaluar_angulos_local(params, instancia, backend, shots, p,
                                          seed=seed_local)
        if esp > mejor["esperado"]:
            mejor.update(esperado=esp, params=params.copy())
        return -esp

    minimize(objetivo, x0, method="COBYLA",
             options={"maxiter": maxiter, "rhobeg": 0.5})
    esp_plain, _, _ = evaluar_angulos_local(
        mejor["params"], instancia, backend, shots, p, seed=seed_local)
    optimo = instancia["optimum"]

    resumen = {
        "tier": tier, "p": p, "shots": shots, "optimum": optimo,
        "qubits_logicos": instancia["n"],
        "qubits_fisicos_iceberg": instancia["n"] + 4,   # n+2 + 2 ancillas
        "instance_sha256": instancia["instance_sha256"],
        "encoded": {"cut": enc["cut"], "ratio": enc["ratio"],
                    "discard_rate": enc["discard_rate"]},
        "unencoded": {"cut": float(esp_plain),
                      "ratio": float(esp_plain / optimo) if optimo > 0
                      else float("nan")},
        "gw_guarantee": GW_GUARANTEE,
        "nota": ("Simulador local SIN ruido: paridad esperada. La ventaja del "
                 "Iceberg (razon mayor a igual profundidad) se mide en el "
                 "emulador H2 con ruido; ver arXiv:2409.12104."),
    }
    return resumen


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Demo local del codigo Iceberg [[k+2,k,2]] sobre QAOA "
                    "Max-Cut de Isla Verde (codificado vs sin codificar).")
    parser.add_argument("--scratch-dir", default="scratch", type=Path)
    parser.add_argument("--tier", default="mvp8")
    parser.add_argument("-p", type=int, default=1)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON de salida (por defecto "
                             "<scratch>/iceberg_<tier>.json)")
    args = parser.parse_args()

    resumen = demo(args.scratch_dir, tier=args.tier, p=args.p,
                   shots=args.shots, maxiter=args.maxiter)
    if resumen is None:
        raise SystemExit(f"No se encontro la instancia '{args.tier}'. Corre "
                         "primero modelador_red.py para generar los JSON.")

    out = args.out or (args.scratch_dir / f"iceberg_{args.tier}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    e, u = resumen["encoded"], resumen["unencoded"]
    print(f"\n[{resumen['tier']}] p={resumen['p']}  "
          f"({resumen['qubits_logicos']} qubits logicos -> "
          f"{resumen['qubits_fisicos_iceberg']} fisicos con Iceberg)")
    print(f"  Iceberg     : r = {e['ratio']:.4f}   "
          f"(descarte {e['discard_rate']*100:.1f}%)")
    print(f"  Sin codificar: r = {u['ratio']:.4f}")
    print(f"  Resultado guardado en {out}")
