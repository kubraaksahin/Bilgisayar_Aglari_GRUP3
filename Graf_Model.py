#!/usr/bin/env python3
"""
generate_network.py

Kullanım örneği:
python generate_network.py \
  --nodes BSM307_317_Guz2025_TermProject_NodeData.csv \
  --demands BSM307_317_Guz2025_TermProject_DemandData.csv \
  --out network_plot.png \
  --n 250 --p 0.4 --seed 42

Açıklama:
- Eğer --nodes dosyası verilirse, mevcut node özellikleri okunur; eksik veya yoksa rastgele atama yapılır.
- Kenarlar (edges) Erdős–Rényi G(n,p) ile üretilir (N ve P parametreleriyle).
- Oluşturulan grafiğin bağlı (connected) olması sağlanır. Ayrıca verilen (varsa) demand (S-D) çiftleri arasında yol yoksa, bileşenler arası rastgele kenarlar eklenerek yollar garanti edilir.
- Her kenara rastgele özellikler atanır: capacity [100,1000] Mbps, delay [3,15] ms, reliability [0.95,0.999].
- Her düğüme processing delay [0.5,2.0] ms ve node reliability [0.95,0.999] atanır (varsa input ile override edilir).
- Çıktılar: node CSV (semicolon, decimal comma), edge CSV (aynı format) ve ağ görsellemesi (PNG).
"""

import argparse
import os
import io

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def read_node_file(path: str, n: int) -> pd.DataFrame:
    # CSV: node_id;s_ms;r_node with semicolon and comma decimals
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=';', decimal=',', dtype={'node_id': int})
    # ensure all node ids 0..n-1 exist; if missing, create rows with NaN for attributes
    expected = pd.DataFrame({'node_id': list(range(n))})
    df = expected.merge(df, on='node_id', how='left')
    return df


def read_demands(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=';', decimal=',', dtype={'src': int, 'dst': int})
    return df


def ensure_connected_by_adding_edges(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    # Eğer bağlı değilse, bileşenleri birbirine rastgele bağlayarak connected yap
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    # bağlamak için her ardışık bileşenden birer node al ve aralarına kenar ekle
    for i in range(len(comps) - 1):
        comp_a = list(comps[i])
        comp_b = list(comps[i + 1])
        a = int(rng.choice(comp_a))
        b = int(rng.choice(comp_b))
        if a != b:
            G.add_edge(a, b)
    # tekrar kontrol; hala parçalıysa döngü (nadiren)
    if not nx.is_connected(G):
        return ensure_connected_by_adding_edges(G, rng)
    return G


def ensure_sd_paths(G: nx.Graph, demands: pd.DataFrame, rng: np.random.Generator) -> nx.Graph:
    # Her talep (src,dst) için bir yol varsa devam et; yoksa bileşenleri bağlayan ek kenarlar ekle
    if demands is None or demands.empty:
        return G
    for _, row in demands.iterrows():
        s = int(row['src'])
        d = int(row['dst'])
        if s not in G.nodes() or d not in G.nodes():
            continue
        if not nx.has_path(G, s, d):
            # s'nin bileşeninden bir node ile d'nin bileşeninden bir node seçip bağla
            comp_s = next(c for c in nx.connected_components(G) if s in c)
            comp_d = next(c for c in nx.connected_components(G) if d in c)
            a = int(rng.choice(list(comp_s)))
            b = int(rng.choice(list(comp_d)))
            if a != b:
                G.add_edge(a, b)
    # garanti için bağlantı kontrolü
    if not nx.is_connected(G):
        G = ensure_connected_by_adding_edges(G, rng)
    return G


def assign_node_attrs(df_nodes: pd.DataFrame, n: int, rng: np.random.Generator, force_random: bool = False) -> pd.DataFrame:
    # df_nodes has columns node_id, s_ms, r_node, may contain NaN
    s_list = []
    r_list = []
    # normalize column names if present
    has_s = 's_ms' in df_nodes.columns
    has_r = 'r_node' in df_nodes.columns
    for i in range(n):
        val_s = df_nodes.at[i, 's_ms'] if has_s else None
        val_r = df_nodes.at[i, 'r_node'] if has_r else None
        if pd.isna(val_s) or force_random:
            s = round(rng.uniform(0.5, 2.0), 2)
        else:
            s = float(val_s)
        if pd.isna(val_r) or force_random:
            r = round(rng.uniform(0.95, 0.999), 3)
        else:
            r = float(val_r)
        s_list.append(s)
        r_list.append(r)
    df_nodes_out = pd.DataFrame({
        'node_id': list(range(n)),
        's_ms': s_list,
        'r_node': r_list
    })
    return df_nodes_out


def assign_edge_attrs(G: nx.Graph, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for u, v in G.edges():
        # keep ordering src<=dst for consistency (optional)
        rows.append({
            'src': int(u),
            'dst': int(v),
            'capacity_mbps': int(rng.integers(100, 1001)),  # [100,1000]
            'delay_ms': round(float(rng.uniform(3.0, 15.0)), 1),
            'r_link': round(float(rng.uniform(0.95, 0.999)), 3)
        })
    df_edges = pd.DataFrame(rows, columns=['src', 'dst', 'capacity_mbps', 'delay_ms', 'r_link'])
    return df_edges


def to_semicolon_decimal_comma(df: pd.DataFrame, path: str):
    # pandas to_csv with sep=';' then replace '.' -> ',' for decimals
    buf = io.StringIO()
    # keep integer columns as-is; format floats with dot then replace to comma
    df.to_csv(buf, sep=';', index=False, float_format='%.6f')
    text = buf.getvalue()
    # replace decimal point with comma (this is simple but OK for our numeric-only CSV)
    text = text.replace('.', ',')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def plot_network(G: nx.Graph, node_df: pd.DataFrame, out_path: str, figsize=(16, 12), seed=None):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=seed)
    # node color by reliability
    reliabilities = node_df.set_index('node_id')['r_node'].reindex(sorted(G.nodes())).fillna(0.96).values
    nodes = nx.draw_networkx_nodes(G, pos,
                                   nodelist=sorted(G.nodes()),
                                   node_size=40,
                                   node_color=reliabilities,
                                   cmap=plt.cm.viridis,
                                   vmin=0.95, vmax=0.999)
    nx.draw_networkx_edges(G, pos, alpha=0.15)
    plt.colorbar(nodes, label='node reliability')
    plt.axis('off')
    plt.title('Generated ER network (N={} , E={})'.format(G.number_of_nodes(), G.number_of_edges()))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate random ER network with node/link attributes, ensure connectivity/S-D paths.')
    parser.add_argument('--nodes', type=str, default=None, help='Input node CSV (semicolon, decimal comma). Optional.')
    parser.add_argument('--edges', type=str, default='generated_edges.csv', help='Output edge CSV path.')
    parser.add_argument('--demands', type=str, default=None, help='Demand CSV (semicolon, decimal comma). Optional.')
    parser.add_argument('--out', type=str, default='network_plot.png', help='Output network plot PNG.')
    parser.add_argument('--n', type=int, default=250, help='Node count (N).')
    parser.add_argument('--p', type=float, default=0.4, help='Edge probability (P) for Erdős-Rényi.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (int).')
    parser.add_argument('--force-random-nodes', action='store_true', help='Ignore input node attributes and force random assignment.')
    parser.add_argument('--out-nodes', type=str, default='generated_nodes.csv', help='Output node CSV path.')
    parser.add_argument('--demand-count', type=int, default=30,help='Number of random demands to generate if --demands is not provided.')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) read nodes if provided
    if args.nodes:
        try:
            df_nodes_in = read_node_file(args.nodes, args.n)
        except FileNotFoundError:
            print("Warning: nodes file not found:", args.nodes)
            df_nodes_in = pd.DataFrame({'node_id': list(range(args.n))})
    else:
        df_nodes_in = pd.DataFrame({'node_id': list(range(args.n))})

    # 2) read demands if provided
    demands = None
    if args.demands:
        try:
             demands = read_demands(args.demands)
             print(f"Demands loaded from file ({len(demands)} rows).")
        except FileNotFoundError:
             print("Warning: demands file not found:", args.demands)
             demands = None


    # 3) generate ER graph
    print(f"Generating Erdős–Rényi G({args.n},{args.p}) ...")
    G = nx.gnp_random_graph(args.n, args.p, seed=args.seed)

    # If graph has no edges (extremely unlikely for p=0.4,n=250) ensure some edges
    if G.number_of_edges() == 0:
        print("Graph had no edges, adding random edges.")
        for _ in range(max(1, args.n // 2)):
            u = int(rng.integers(0, args.n))
            v = int(rng.integers(0, args.n))
            if u != v:
                G.add_edge(u, v)

    # 4) ensure connectivity (or at least S-D paths)
    G = ensure_connected_by_adding_edges(G, rng)
    if demands is not None:
        G = ensure_sd_paths(G, demands, rng)

    print(f"Final graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # 4.5) Generate random demands if not provided
    if demands is None:
         print(f"Generating {args.demand_count} random demands...")
         demand_set = set()
         max_tries = args.demand_count * 10
         tries = 0

    while len(demand_set) < args.demand_count and tries < max_tries:
        tries += 1
        s = int(rng.integers(0, args.n))
        d = int(rng.integers(0, args.n))
        if s == d:
            continue
        if not nx.has_path(G, s, d):
            continue
        demand_val = int(rng.integers(10, 201))
        demand_set.add((s, d, demand_val))

    if len(demand_set) < args.demand_count:
        raise RuntimeError("Could not generate enough valid demands.")

    demands = pd.DataFrame(
        list(demand_set),
        columns=['src', 'dst', 'demand_mbps']
    )

    to_semicolon_decimal_comma(demands, 'generated_demands.csv')
    print(f"Random demands saved to generated_demands.csv ({len(demands)} demands).")


    # 5) assign node attributes (use input if exists)
    df_nodes_out = assign_node_attrs(df_nodes_in, args.n, rng, force_random=args.force_random_nodes)

    # 6) assign edge attributes
    df_edges_out = assign_edge_attrs(G, rng)

    # 7) Validate demands: keep only demands with existing nodes; warn if no path (should be none)
    if demands is not None:
        valid_demands = []
        for _, r in demands.iterrows():
            s = int(r['src']); d = int(r['dst'])
            if s not in G.nodes() or d not in G.nodes():
                print(f"Demand ({s}->{d}) has node outside graph, skipping.")
                continue
            if not nx.has_path(G, s, d):
                print(f"Warning: no path for demand ({s}->{d}) — this should not happen after enforcement.")
            valid_demands.append({'src': s, 'dst': d, 'demand_mbps': int(r['demand_mbps'])})
        df_demands_out = pd.DataFrame(valid_demands)
        if df_demands_out.empty:
            print("No valid demands after filtering.")
        else:
            to_semicolon_decimal_comma(df_demands_out, 'generated_demands.csv')
            print("generated demands saved to generated_demands.csv")
    else:
        df_demands_out = None

    # 8) save nodes and edges with semicolon separator and comma decimals
    to_semicolon_decimal_comma(df_nodes_out, args.out_nodes)
    to_semicolon_decimal_comma(df_edges_out, args.edges)

    print(f"Nodes saved to {args.out_nodes}")
    print(f"Edges saved to {args.edges}")

    # 9) plot and save network visualization
    try:
        plot_network(G, df_nodes_out, args.out, figsize=(18, 14), seed=args.seed)
        print(f"Network plot saved to {args.out}")
    except Exception as e:
        print("Plotting failed:", e)


if __name__ == '__main__':
    main()