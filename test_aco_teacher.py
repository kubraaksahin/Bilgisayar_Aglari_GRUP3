import pandas as pd
import networkx as nx

from aco import run_aco
from Metric_Evaluation import evaluate_path
from config import (
    W_DELAY,
    W_RELIABILITY,
    W_RESOURCE,
    ACO_N_ANTS,
    ACO_N_ITERATIONS,
    TEACHER_NODE_FILE,
    TEACHER_EDGE_FILE,
    TEACHER_DEMAND_FILE,
)


def build_graph_from_teacher(node_csv: str, edge_csv: str) -> nx.Graph:
    """
    Hocanın NodeData ve EdgeData dosyalarından grafı oluşturur.
    NodeData kolonları: node_id; s_ms; r_node
    EdgeData kolonları: src; dst; capacity_mbps; delay_ms; r_link
    (Eğer sizde isimler farklıysa burayı ona göre düzeltin.)
    """
    df_nodes = pd.read_csv(node_csv, sep=";", decimal=",")
    df_edges = pd.read_csv(edge_csv, sep=";", decimal=",")

    G = nx.Graph()

    # Düğümler
    for _, row in df_nodes.iterrows():
        nid = int(row["node_id"])
        G.add_node(
            nid,
            s_ms=float(row["s_ms"]),
            r_node=float(row["r_node"]),
        )

    # Kenarlar
    for _, row in df_edges.iterrows():
        u = int(row["src"])
        v = int(row["dst"])
        G.add_edge(
            u,
            v,
            capacity_mbps=float(row["capacity_mbps"]),
            delay_ms=float(row["delay_ms"]),
            r_link=float(row["r_link"]),
        )

    return G


def main():
    node_csv = TEACHER_NODE_FILE
    edge_csv = TEACHER_EDGE_FILE
    demand_csv = TEACHER_DEMAND_FILE

    # Talepleri oku
    df_demands = pd.read_csv(demand_csv, sep=";", decimal=",")
    G = build_graph_from_teacher(node_csv, edge_csv)

    print(f"Teacher graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    print(f"Number of demands: {len(df_demands)}")

    results = []

    # İstersen test için kısıtlayabilirsin:
    # max_demands = min(10, len(df_demands))
    max_demands = len(df_demands)

    for demand_idx in range(max_demands):
        row = df_demands.iloc[demand_idx]

        src = int(row["src"])
        dst = int(row["dst"])

        # Hocanın DemandData dosyasındaki bant genişliği kolonunun adı:
        # genelde "demand_mbps" oluyor; farklıysa burayı uyarlarsın.
        if "demand_mbps" in df_demands.columns:
            B = float(row["demand_mbps"])
        elif "demand" in df_demands.columns:
            B = float(row["demand"])
        else:
            raise KeyError("Demand kolon adı 'demand_mbps' veya 'demand' değil, kontrol et.")

        print(f"\nDemand #{demand_idx}: {src} -> {dst} (B={B} Mbps)")

        best_path, best_cost = run_aco(
            G,
            src,
            dst,
            B,
            n_ants=ACO_N_ANTS,
            n_iterations=ACO_N_ITERATIONS,
            seed=2025 + demand_idx,
        )

        if best_path is None:
            print("  No feasible path found.")
            results.append({
                "demand_index": demand_idx,
                "src": src,
                "dst": dst,
                "demand_mbps": B,
                "best_path": None,
                "TotalDelay": None,
                "TotalReliability": None,
                "ReliabilityCost": None,
                "ResourceCost": None,
                "TotalCost": None,
                "Feasible": False,
            })
            continue

        metrics = evaluate_path(
            G,
            best_path,
            demand_B=B,
            w_delay=W_DELAY,
            w_rel=W_RELIABILITY,
            w_res=W_RESOURCE,
        )

        print(
            f"  best_cost={metrics['TotalCost']:.4f}, "
            f"delay={metrics['TotalDelay']:.2f} ms, "
            f"R={metrics['TotalReliability']:.4f}"
        )

        results.append({
            "demand_index": demand_idx,
            "src": src,
            "dst": dst,
            "demand_mbps": B,
            "best_path": "->".join(map(str, best_path)),
            "TotalDelay": metrics["TotalDelay"],
            "TotalReliability": metrics["TotalReliability"],
            "ReliabilityCost": metrics["ReliabilityCost"],
            "ResourceCost": metrics["ResourceCost"],
            "TotalCost": metrics["TotalCost"],
            "Feasible": metrics["Feasible"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("aco_results_teacher.csv", index=False)
    print("\nACO teacher results saved to aco_results_teacher.csv")


if __name__ == "__main__":
    main()
