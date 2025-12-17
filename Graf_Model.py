import argparse
import os
import io

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ---------- IO ----------

def read_node_file(path: str, n: int) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=';', decimal=',', dtype={'node_id': int})
    expected = pd.DataFrame({'node_id': list(range(n))})
    return expected.merge(df, on='node_id', how='left')


def to_semicolon_decimal_comma(df: pd.DataFrame, path: str):
    buf = io.StringIO()
    df.to_csv(buf, sep=';', index=False, float_format='%.6f')
    text = buf.getvalue().replace('.', ',')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


# ---------- GRAPH ----------

def ensure_connected_by_adding_edges(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G

    for i in range(len(comps) - 1):
        a = int(rng.choice(list(comps[i])))
        b = int(rng.choice(list(comps[i + 1])))
        if a != b:
            G.add_edge(a, b)

    if not nx.is_connected(G):
        return ensure_connected_by_adding_edges(G, rng)

    return G


# ---------- ATTRIBUTES ----------

def assign_node_attrs(df_nodes: pd.DataFrame, n: int, rng: np.random.Generator, force_random=False) -> pd.DataFrame:
    s_list, r_list = [], []

    has_s = 's_ms' in df_nodes.columns
    has_r = 'r_node' in df_nodes.columns

    for i in range(n):
        s = df_nodes.at[i, 's_ms'] if has_s else np.nan
        r = df_nodes.at[i, 'r_node'] if has_r else np.nan

        if pd.isna(s) or force_random:
            s = round(rng.uniform(0.5, 2.0), 2)
        if pd.isna(r) or force_random:
            r = round(rng.uniform(0.95, 0.999), 3)

        s_list.append(float(s))
        r_list.append(float(r))

    return pd.DataFrame({
        'node_id': range(n),
        's_ms': s_list,
        'r_node': r_list
    })


def assign_edge_attrs(G: nx.Graph, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for u, v in G.edges():
        rows.append({
            'src': int(u),
            'dst': int(v),
            'capacity_mbps': int(rng.integers(100, 1001)),
            'delay_ms': round(float(rng.uniform(3.0, 15.0)), 1),
            'r_link': round(float(rng.uniform(0.95, 0.999)), 3)
        })
    return pd.DataFrame(rows)


# ---------- PLOT ----------

def plot_network(G: nx.Graph, node_df: pd.DataFrame, out_path: str, seed=None):
    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, seed=seed)

    reliabilities = (
        node_df.set_index('node_id')
        .reindex(sorted(G.nodes()))['r_node']
        .fillna(0.96)
        .values
    )

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=40,
        node_color=reliabilities,
        cmap=plt.cm.viridis,
        vmin=0.95, vmax=0.999
    )

    nx.draw_networkx_edges(G, pos, alpha=0.15)
    plt.colorbar(nodes, label='node reliability')
    plt.title(f'Generated ER Network (N={G.number_of_nodes()}, E={G.number_of_edges()})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser(description='Generate connected ER network with node/link attributes.')
    parser.add_argument('--nodes', type=str, default=None, help='Optional input node CSV')
    parser.add_argument('--out-nodes', type=str, default='generated_nodes.csv')
    parser.add_argument('--edges', type=str, default='generated_edges.csv')
    parser.add_argument('--out', type=str, default='network_plot.png')
    parser.add_argument('--n', type=int, default=250)
    parser.add_argument('--p', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--force-random-nodes', action='store_true')

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # --- Nodes input ---
    if args.nodes:
        df_nodes_in = read_node_file(args.nodes, args.n)
    else:
        df_nodes_in = pd.DataFrame({'node_id': range(args.n)})

    # --- Graph ---
    print(f"Generating Erdős–Rényi G({args.n}, {args.p})")
    G = nx.gnp_random_graph(args.n, args.p, seed=args.seed)

    if G.number_of_edges() == 0:
        for _ in range(max(1, args.n // 2)):
            u, v = rng.integers(0, args.n, size=2)
            if u != v:
                G.add_edge(int(u), int(v))

    G = ensure_connected_by_adding_edges(G, rng)
    print(f"Final graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # --- Attributes ---
    df_nodes_out = assign_node_attrs(df_nodes_in, args.n, rng, args.force_random_nodes)
    df_edges_out = assign_edge_attrs(G, rng)

    # --- Save ---
    to_semicolon_decimal_comma(df_nodes_out, args.out_nodes)
    to_semicolon_decimal_comma(df_edges_out, args.edges)

    print(f"Nodes saved to {args.out_nodes}")
    print(f"Edges saved to {args.edges}")

    try:
        plot_network(G, df_nodes_out, args.out, seed=args.seed)
        print(f"Plot saved to {args.out}")
    except Exception as e:
        print("Plotting failed:", e)


if __name__ == '__main__':
    main()
