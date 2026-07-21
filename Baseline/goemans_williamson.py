"""
Implementación completa de Goemans-Williamson SDP
para Max-Cut.

Características:

- NetworkX para grafos
- CVXPY para SDP
- NumPy para álgebra lineal
- Matplotlib para visualización
- CLI mediante argparse

Autor:
    Implementación académica

"""

"""
python goemans_williamson.py --graph ..\scratch\isla_verde_full_graph.json --iterations 500 --runs 30 --output results_GW
"""

import json
import csv
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path


import numpy as np
import networkx as nx
import cvxpy as cp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt




# ============================================================
# VISUALIZACION
# ============================================================


def draw_graph(G, filename):
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=600)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------


def draw_cut(G, result, filename):
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)
    group0 = []
    group1 = []
    partition = partition_dict(result)
    for node, value in partition.items():
        if value == 1:
            group1.append(node)
        else:
            group0.append(node)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(
        G, pos, nodelist=group0, node_color="red", node_size=700, label="Grupo -1"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=group1, node_color="blue", node_size=700, label="Grupo +1"
    )
    nx.draw_networkx_labels(G, pos)
    plt.legend()
    plt.title(f"Max-Cut = {result.best_cut}")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------


def draw_histogram(result, filename):
    plt.figure(figsize=(15, 10))
    plt.hist(result.cuts, bins=20)
    plt.xlabel("Valor del corte")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de cortes")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# ============================================================
# RESULTADOS
# ============================================================


@dataclass
class GWResult:
    best_cut: float
    mean_cut: float
    std_cut: float
    sdp_value: float
    runtime: float
    nodes: list
    partition: list
    cuts: list


# ============================================================
# CLASE PRINCIPAL
# ============================================================


class GoemansWilliamson:
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    # --------------------------------------------------------
    # CARGAR GRAFO
    # --------------------------------------------------------

    def load_graph(self, filename):
        """
        Carga un grafo desde JSON personalizado.
        Formato esperado:
        {
            metadata:{},
            graph:{
                nodes:[
                    {
                        id:"Nodo",
                        ...
                    }
                ],
                links:[
                    {
                        source:"NodoA",
                        target:"NodoB",
                        weight:valor
                    }
                ]
            }
        }

        """
        with open(filename, "r", encoding="utf8") as f:
            data = json.load(f)
        graph_data = data["graph"]
        G = nx.Graph()
        # -----------------------------
        # cargar nodos
        # -----------------------------

        for node in graph_data["nodes"]:
            node_id = node["id"]
            attributes = node.copy()
            # quitamos id porque NetworkX
            # ya lo usa como clave
            attributes.pop("id", None)
            G.add_node(node_id, **attributes)
        # -----------------------------
        # cargar enlaces
        # -----------------------------
        for edge in graph_data["links"]:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("weight", 1),
                length_m=edge.get("length_m"),
                circuits=edge.get("circuits", []),
                parallel=edge.get("parallel", False),
            )
        return G

    # --------------------------------------------------------
    # MATRIZ DE PESOS
    # --------------------------------------------------------
    def adjacency_matrix(self, G):
        nodes = list(G.nodes())
        W = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
        return W, nodes

    # --------------------------------------------------------
    # SDP
    # --------------------------------------------------------
    def solve_sdp(self, W):
        n = W.shape[0]
        X = cp.Variable((n, n), PSD=True)
        constraints = [cp.diag(X) == 1]
        objective = cp.Maximize(0.25 * cp.sum(cp.multiply(W, (1 - X))))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return X.value, problem.value

    # --------------------------------------------------------
    # FACTORIZACION
    # --------------------------------------------------------
    def extract_vectors(self, X):
        eigenvalues, eigenvectors = np.linalg.eigh(X)
        eigenvalues[eigenvalues < 0] = 0
        V = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        return V

    # --------------------------------------------------------
    # REDONDEO
    # --------------------------------------------------------
    def random_rounding(self, V):
        r = np.random.randn(V.shape[1])
        r /= np.linalg.norm(r)
        result = np.sign(V @ r)
        result[result == 0] = 1
        return result.astype(int)

    # --------------------------------------------------------
    # VALOR DEL CORTE
    # --------------------------------------------------------
    def cut_value(self, G, partition, nodes):
        index = {node: i for i, node in enumerate(nodes)}
        total = 0
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1)
            if partition[index[u]] != partition[index[v]]:
                total += weight
        return total

    # --------------------------------------------------------
    # SOLVER PRINCIPAL
    # --------------------------------------------------------
    def solve(self, G, iterations=100):
        W, nodes = self.adjacency_matrix(G)
        X, sdp_value = self.solve_sdp(W)
        V = self.extract_vectors(X)
        best_partition = None
        best_cut = -1
        cuts = []
        for _ in range(iterations):
            partition = self.random_rounding(V)
            value = self.cut_value(G, partition, nodes)
            cuts.append(value)
            if value > best_cut:
                best_cut = value
                best_partition = partition
        return (best_partition, best_cut, sdp_value, cuts, nodes)


# ============================================================
# BENCHMARK MULTIPLE
# ============================================================


def benchmark(gw, G, runs=20, iterations=200):
    """
    Ejecuta múltiples veces Goemans-Williamson
    para analizar estabilidad.
    """
    start = time.time()
    cuts = []
    best_cut = -1
    best_partition = None
    best_nodes = None
    sdp_value = None
    for i in range(runs):
        partition, cut, sdp, current_cuts, nodes = gw.solve(G, iterations)
        cuts.extend(current_cuts)
        if cut > best_cut:
            best_cut = cut
            best_partition = partition
            best_nodes = nodes
            sdp_value = sdp
    runtime = time.time() - start
    return GWResult(
        best_cut=best_cut,
        mean_cut=float(np.mean(cuts)),
        std_cut=float(np.std(cuts)),
        sdp_value=float(sdp_value),
        runtime=runtime,
        nodes=best_nodes,
        partition=best_partition.tolist(),
        cuts=cuts,
    )


# ============================================================
# CONVERTIR PARTICION
# ============================================================


def partition_dict(result):
    """
    Convierte:
    nodes=[A,B,C]
    partition=[1,-1,1]

    en:
    {
       A:1,
       B:-1,
       C:1
    }
    """
    return {node: group for node, group in zip(result.nodes, result.partition)}


# ============================================================
# EXPORTAR RESULTADOS
# ============================================================


def save_results(result, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    # -------------------------
    # JSON
    # -------------------------
    json_file = folder / "experiment.json"
    with open(json_file, "w", encoding="utf8") as f:
        json.dump(asdict(result), f, indent=4)
    # -------------------------
    # CSV
    # -------------------------
    csv_file = folder / "experiment.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "cut"])
        for i, value in enumerate(result.cuts):
            writer.writerow([i, value])


# ============================================================
# EJECUCION DESDE CONSOLA
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Goemans-Williamson Max-Cut SDP")
    parser.add_argument("--graph", required=True, help="Archivo JSON del grafo")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    gw = GoemansWilliamson(seed=args.seed)
    print("Cargando grafo...")
    G = gw.load_graph(args.graph)
    print("Ejecutando SDP...")
    result = benchmark(gw, G, runs=args.runs, iterations=args.iterations)
    print("\n======================")
    print("RESULTADOS")
    print("======================")

    print("\n" + "=" * 50)
    print(" RESULTADOS GOEMANS-WILLIAMSON SDP")
    print("=" * 50)

    print("\n--- Información del grafo ---")
    print(f"Nodos: {len(G.nodes)}")
    print(f"Aristas: {len(G.edges)}")

    print("\n--- Optimización SDP ---")
    print(f"Valor SDP: {result.sdp_value:.3f}")
    print("Cota GW (87.8%): " f"{0.878 * result.sdp_value:.3f}")

    print("\n--- Resultados del corte ---")
    print(f"Mejor corte: {result.best_cut:.3f}")
    print(f"Promedio cortes: {result.mean_cut:.3f}")
    print(f"Desv. estándar: {result.std_cut:.3f}")
    ratio = result.best_cut / result.sdp_value
    print(f"Ratio SDP: {ratio:.3%}")

    print("\n--- Estabilidad ---")
    print(f"Cantidad muestras: {len(result.cuts)}")
    print(f"Peor corte: {min(result.cuts):.3f}")
    print(f"Mejor corte: {max(result.cuts):.3f}")

    print("\n--- Tiempo ejecución ---")
    print(f"Tiempo total: {result.runtime:.3f} segundos")

    print("\n--- Partición encontrada ---")
    grupo_a = [
        node for node, value in zip(result.nodes, result.partition) if value == 1
    ]
    grupo_b = [
        node for node, value in zip(result.nodes, result.partition) if value == -1
    ]
    print(f"\nGrupo A ({len(grupo_a)} nodos):")
    print(grupo_a)
    print(f"\nGrupo B ({len(grupo_b)} nodos):")
    print(grupo_b)
    print("\n" + "=" * 50)

    gap = result.sdp_value - result.best_cut
    print(f"Gap SDP: {gap:.4f}")

    output = Path(args.output)
    save_results(result, output)
    draw_graph(G, output / "original_graph.png")
    draw_cut(G, result, output / "best_cut.png")
    draw_histogram(result, output / "histogram.png")
    print("\nArchivos guardados en:", output)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
