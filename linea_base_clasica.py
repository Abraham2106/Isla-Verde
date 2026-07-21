import numpy as np
import networkx as nx
import cvxpy as cp


# ---------------------------------------------------------------------------
# 1. Carga del grafo (conectar aqui con modelador_red.py)
# ---------------------------------------------------------------------------
def cargar_grafo():
    G = nx.Graph()
    edges = [
        ("SUB_A", "SUB_B", 4),
        ("SUB_A", "SUB_C", 2),
        ("SUB_B", "SUB_C", 5),
        ("SUB_B", "SUB_D", 3),
        ("SUB_C", "SUB_D", 1),
        ("SUB_D", "SUB_E", 6),
    ]
    G.add_weighted_edges_from(edges)
    return G


# ---------------------------------------------------------------------------
# 2. Greedy
# ---------------------------------------------------------------------------
def greedy_maxcut(G, n_restarts=50, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    mejor_corte = 0
    mejor_particion = None

    for _ in range(n_restarts):
        orden = rng.permutation(nodes)
        asignacion = {}
        for nodo in orden:
            ganancia_0 = sum(
                G[nodo][vecino]["weight"]
                for vecino in G.neighbors(nodo)
                if vecino in asignacion and asignacion[vecino] != 0
            )
            ganancia_1 = sum(
                G[nodo][vecino]["weight"]
                for vecino in G.neighbors(nodo)
                if vecino in asignacion and asignacion[vecino] != 1
            )
            asignacion[nodo] = 0 if ganancia_0 >= ganancia_1 else 1

        corte = sum(
            data["weight"]
            for u, v, data in G.edges(data=True)
            if asignacion[u] != asignacion[v]
        )
        if corte > mejor_corte:
            mejor_corte = corte
            mejor_particion = asignacion.copy()

    return mejor_corte, mejor_particion


# ---------------------------------------------------------------------------
# 3. Goemans-Williamson
# ---------------------------------------------------------------------------
def goemans_williamson_maxcut(G, n_redondeos=500, seed=0):
    nodos = list(G.nodes())
    n = len(nodos)
    idx = {nodo: i for i, nodo in enumerate(nodos)}

    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        W[idx[u], idx[v]] = data["weight"]
        W[idx[v], idx[u]] = data["weight"]

    Y = cp.Variable((n, n), PSD=True)
    objetivo = cp.Maximize(0.25 * cp.sum(cp.multiply(W, 1 - Y)))
    restricciones = [cp.diag(Y) == 1]
    problema = cp.Problem(objetivo, restricciones)
    problema.solve(solver=cp.SCS)

    Y_val = Y.value
    Y_val = (Y_val + Y_val.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Y_val)
    eigvals = np.clip(eigvals, 0, None)
    V = eigvecs @ np.diag(np.sqrt(eigvals))

    rng = np.random.default_rng(seed)
    mejor_corte = 0
    mejor_particion = None
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
            mejor_particion = asignacion.copy()

    cota_sdp = problema.value
    return mejor_corte, mejor_particion, cota_sdp


# ---------------------------------------------------------------------------
# 4. Ejecucion multiple de GW para reportar media +/- desviacion estandar
# ---------------------------------------------------------------------------
def gw_con_estadisticas(G, n_corridas=10, n_redondeos=500):
    cortes = []
    for semilla in range(n_corridas):
        corte, _, _ = goemans_williamson_maxcut(G, n_redondeos=n_redondeos, seed=semilla)
        cortes.append(corte)

    cortes = np.array(cortes)
    return cortes.mean(), cortes.std(), cortes.max()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    G = cargar_grafo()
    print(f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas\n")

    corte_greedy, particion_greedy = greedy_maxcut(G)
    print(f"[Greedy]  Corte = {corte_greedy}")
    print(f"  Particion: {particion_greedy}\n")

    corte_gw, particion_gw, cota_sdp = goemans_williamson_maxcut(G)
    print(f"[Goemans-Williamson] Corte = {corte_gw}")
    print(f"  Cota SDP (limite superior del optimo real): {cota_sdp:.4f}")
    print(f"  Particion: {particion_gw}\n")

    media, std, maximo = gw_con_estadisticas(G, n_corridas=10)
    print("[Goemans-Williamson] Estadisticas sobre 10 corridas:")
    print(f"  Media = {media:.2f}  |  Desv. estandar = {std:.2f}  |  Mejor = {maximo:.2f}\n")

    print("=== Resumen para el equipo de QAOA ===")
    print(f"Cota superior (SDP): {cota_sdp:.4f}")
    print(f"Greedy:              corte = {corte_greedy}")
    print(f"GW (mejor de 10):    corte = {maximo:.0f}")
    print("Usar la cota SDP como referencia de 'optimo' si el grafo es")
    print("demasiado grande para fuerza bruta exacta.")


main()
