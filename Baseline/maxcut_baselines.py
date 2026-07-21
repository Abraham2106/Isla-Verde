"""
maxcut_baselines.py
Baselines clásicos para Max-Cut:
1. Greedy Max-Cut
2. Fuerza Bruta Max-Cut
Compatible con Goemans-Williamson SDP.
"""

"""
python maxcut_baselines.py --graph ..\scratch\isla_verde_full_graph.json --algorithm greedy --output results_greedy
python maxcut_baselines.py --graph ..\scratch\isla_verde_full_graph.json --algorithm bruteforce --output results_bruteforce
"""

import json
import csv
import time
import argparse
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
import networkx as nx
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from goemans_williamson import draw_graph, draw_cut, draw_histogram

# ==========================================================
# RESULTADO COMUN
# ==========================================================


@dataclass
class MaxCutResult:
    algorithm: str
    best_cut: float
    mean_cut: float
    std_cut: float
    runtime: float
    nodes: list
    partition: list
    cuts: list
    sdp_value: float = None


# ==========================================================
# CARGAR GRAFO
# ==========================================================


def load_graph(filename):
    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)
    graph = data["graph"]
    G = nx.Graph()
    # nodos
    for node in graph["nodes"]:
        attrs = node.copy()
        node_id = attrs.pop("id")
        G.add_node(node_id, **attrs)
    # enlaces
    for edge in graph["links"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge.get("weight", 1),
            length_m=edge.get("length_m", None),
        )
    return G


# ==========================================================
# VALOR DEL CORTE
# ==========================================================


def cut_value(G, partition):
    total = 0
    for u, v, data in G.edges(data=True):
        if partition[u] != partition[v]:
            total += data.get("weight", 1)
    return total


# ==========================================================
# GREEDY MAX CUT
# ==========================================================


class GreedyMaxCut:
    def solve(self, G):
        start = time.time()
        nodes = list(G.nodes())
        partition = {}
        for node in nodes:
            gain_plus = 0
            gain_minus = 0
            for neighbor in G.neighbors(node):
                if neighbor in partition:
                    weight = G[node][neighbor].get("weight", 1)
                    if partition[neighbor] == 1:
                        gain_minus += weight
                    else:
                        gain_plus += weight
            if gain_plus > gain_minus:
                partition[node] = 1
            else:
                partition[node] = -1
        cut = cut_value(G, partition)
        return MaxCutResult(
            algorithm="Greedy Max-Cut",
            best_cut=cut,
            mean_cut=cut,
            std_cut=0,
            runtime=time.time() - start,
            nodes=nodes,
            partition=[partition[n] for n in nodes],
            cuts=[cut],
        )


# ==========================================================
# FUERZA BRUTA
# ==========================================================


class BruteForceMaxCut:
    def solve(self, G):
        start = time.time()
        nodes = list(G.nodes())
        n = len(nodes)
        if n > 25:
            raise ValueError("Fuerza bruta limitada a n<=25")
        best_cut = -1
        best_partition = None
        cuts = []
        # eliminamos simetría
        # primer nodo fijo
        for bits in itertools.product([-1, 1], repeat=n - 1):
            partition = [1] + list(bits)
            assignment = {nodes[i]: partition[i] for i in range(n)}
            value = cut_value(G, assignment)
            cuts.append(value)
            if value > best_cut:
                best_cut = value
                best_partition = partition
        return MaxCutResult(
            algorithm="Brute Force Max-Cut",
            best_cut=best_cut,
            mean_cut=sum(cuts) / len(cuts),
            std_cut=0,
            runtime=time.time() - start,
            nodes=nodes,
            partition=best_partition,
            cuts=cuts,
        )


# ==========================================================
# INFORME CONSOLA
# ==========================================================


def print_results(result, G):
    print("\n" + "=" * 55)
    print(result.algorithm)
    print("=" * 55)
    print("\n--- Información del grafo ---")
    print(f"Nodos:              {len(G.nodes)}")
    print(f"Aristas:            {len(G.edges)}")
    print("\n--- Max Cut ---")
    print(f"Mejor corte:        {result.best_cut:.4f}")
    print(f"Promedio:           {result.mean_cut:.4f}")
    print(f"Desv estándar:      {result.std_cut:.4f}")
    print("\n--- Estadísticas ---")
    print(f"Muestras:           {len(result.cuts)}")
    print(f"Peor corte:         {min(result.cuts):.4f}")
    print(f"Mejor corte:        {max(result.cuts):.4f}")
    print("\n--- Tiempo ---")
    print(f"Tiempo:             {result.runtime:.4f}s")
    print("=" * 55)


# ==========================================================
# GUARDAR RESULTADOS
# ==========================================================


def save_results(result, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    with open(folder / "experiment.json", "w", encoding="utf8") as f:
        json.dump(asdict(result), f, indent=4)
    with open(folder / "cuts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "cut"])
        for i, c in enumerate(result.cuts):
            writer.writerow([i, c])


# ==========================================================
# CLI
# ==========================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--algorithm", choices=["greedy", "bruteforce"], required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    G = load_graph(args.graph)
    if args.algorithm == "greedy":
        solver = GreedyMaxCut()
    else:
        solver = BruteForceMaxCut()
    result = solver.solve(G)
    print_results(result, G)
    output = Path(args.output)
    save_results(result, output)
    draw_graph(G, output / "original_graph.png")
    draw_cut(G, result, output / "best_cut.png")
    draw_histogram(result, output / "histogram.png")
    print("\nArchivos guardados en:", output)


if __name__ == "__main__":
    main()
