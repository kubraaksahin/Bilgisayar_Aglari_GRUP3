import argparse
import json
import csv
import os
from statistics import mean
import networkx as nx
import numpy as np

# Görselleştirme için gerekli kütüphaneler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False

def generate_connected_erdos_renyi(n, p, rng, max_attempts=50):
    """
    Erdos-Renyi G(n, p) grafiğini üretir ve bağlı bir grafik elde etmeye çalışır.
    Eğer max_attempts kadar bağlı graf elde edilemezse, son üretilen grafın
    bileşenlerini birbirine rastgele kenarlar ekleyerek bağlar.
    """
    attempt = 0
    last_G = None
    seed_int = None
    while attempt < max_attempts:
        seed_int = int(rng.integers(0, 2**31 - 1))
        G = nx.erdos_renyi_graph(n, p, seed=seed_int)
        last_G = G
        if nx.is_connected(G):
            return G, {"method": "regenerated_connected", "attempts": attempt + 1, "seed": seed_int}
        attempt += 1

    # Fallback: bileşenleri birbirine bağla
    comps = list(nx.connected_components(last_G))
    if len(comps) > 1:
        comp_list = [list(c) for c in comps]
        for i in range(len(comp_list) - 1):
            u = int(rng.choice(comp_list[i]))
            v = int(rng.choice(comp_list[i + 1]))
            last_G.add_edge(u, v)
    return last_G, {"method": "connected_by_linking_components", "attempts": attempt, "seed": seed_int}

def annotate_graph(G, rng):
    """
    Düğüm ve kenar özniteliklerini rastgele atar:
    - node: processing_delay_ms (0.5 - 2.0 ms), node_reliability (0.95 - 0.999)
    - edge: bandwidth_mbps (100 - 1000), link_delay_ms (3 - 15), link_reliability (0.95 - 0.999)
    """
    for node in G.nodes():
        processing_delay_ms = float(rng.uniform(0.5, 2.0))
        node_reliability = float(rng.uniform(0.95, 0.999))
        G.nodes[node]["processing_delay_ms"] = round(processing_delay_ms, 6)
        G.nodes[node]["node_reliability"] = round(node_reliability, 6)

    for u, v in G.edges():
        bandwidth_mbps = float(rng.uniform(100.0, 1000.0))
        link_delay_ms = float(rng.uniform(3.0, 15.0))
        link_reliability = float(rng.uniform(0.95, 0.999))
        G.edges[u, v]["bandwidth_mbps"] = round(bandwidth_mbps, 6)
        G.edges[u, v]["link_delay_ms"] = round(link_delay_ms, 6)
        G.edges[u, v]["link_reliability"] = round(link_reliability, 6)

    return G

def save_outputs(G, out_prefix):
    """
    Grafiği çeşitli formatlarda kaydeder:
    - {out_prefix}.graphml
    - {out_prefix}_nodes.csv
    - {out_prefix}_edges.csv
    - {out_prefix}_summary.json
    """
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    graphml_path = f"{out_prefix}.graphml"
    nx.write_graphml(G, graphml_path)

    nodes_csv = f"{out_prefix}_nodes.csv"
    with open(nodes_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "processing_delay_ms", "node_reliability", "degree"])
        for node in G.nodes():
            writer.writerow([
                node,
                G.nodes[node].get("processing_delay_ms", ""),
                G.nodes[node].get("node_reliability", ""),
                G.degree[node]
            ])

    edges_csv = f"{out_prefix}_edges.csv"
    with open(edges_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "bandwidth_mbps", "link_delay_ms", "link_reliability"])
        for u, v in G.edges():
            writer.writerow([
                u,
                v,
                G.edges[u, v].get("bandwidth_mbps", ""),
                G.edges[u, v].get("link_delay_ms", ""),
                G.edges[u, v].get("link_reliability", "")
            ])

    # Özet istatistikler
    degrees = [d for _, d in G.degree()]
    node_processing = [G.nodes[n]["processing_delay_ms"] for n in G.nodes()]
    node_rel = [G.nodes[n]["node_reliability"] for n in G.nodes()]
    edge_bw = [G.edges[u, v]["bandwidth_mbps"] for u, v in G.edges()]
    edge_delay = [G.edges[u, v]["link_delay_ms"] for u, v in G.edges()]
    edge_rel = [G.edges[u, v]["link_reliability"] for u, v in G.edges()]

    summary = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": mean(degrees) if degrees else 0,
        "processing_delay_ms": {
            "min": min(node_processing),
            "max": max(node_processing),
            "avg": mean(node_processing)
        },
        "node_reliability": {
            "min": min(node_rel),
            "max": max(node_rel),
            "avg": mean(node_rel)
        },
        "bandwidth_mbps": {
            "min": min(edge_bw) if edge_bw else None,
            "max": max(edge_bw) if edge_bw else None,
            "avg": mean(edge_bw) if edge_bw else None
        },
        "link_delay_ms": {
            "min": min(edge_delay) if edge_delay else None,
            "max": max(edge_delay) if edge_delay else None,
            "avg": mean(edge_delay) if edge_delay else None
        },
        "link_reliability": {
            "min": min(edge_rel) if edge_rel else None,
            "max": max(edge_rel) if edge_rel else None,
            "avg": mean(edge_rel) if edge_rel else None
        }
    }

    summary_path = f"{out_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "graphml": graphml_path,
        "nodes_csv": nodes_csv,
        "edges_csv": edges_csv,
        "summary_json": summary_path
    }

def draw_graph_static(G, out_path, seed=None, dpi=300):
    """
    Matplotlib ile statik PNG üretir. Düğüm boyutları/renkleri ve kenar kalınlıkları
    özniteliklere göre ölçeklenir.
    """
    layout = nx.spring_layout(G, seed=seed)

    node_proc = np.array([G.nodes[n]["processing_delay_ms"] for n in G.nodes()])
    min_proc, max_proc = 0.5, 2.0
    node_sizes = 80 + (node_proc - min_proc) / (max_proc - min_proc + 1e-9) * (800 - 80)

    node_rel = np.array([G.nodes[n]["node_reliability"] for n in G.nodes()])
    cmap_nodes = cm.get_cmap("viridis")
    norm_nodes = colors.Normalize(vmin=0.95, vmax=0.999)
    node_colors = cmap_nodes(norm_nodes(node_rel))

    edge_bw = np.array([G.edges[u, v]["bandwidth_mbps"] for u, v in G.edges()])
    min_bw, max_bw = 100.0, 1000.0
    edge_widths = 0.5 + (edge_bw - min_bw) / (max_bw - min_bw + 1e-9) * (4.0 - 0.5)

    edge_rel = np.array([G.edges[u, v]["link_reliability"] for u, v in G.edges()])
    cmap_edges = cm.get_cmap("Greys")
    norm_edges = colors.Normalize(vmin=0.95, vmax=0.999)
    edge_colors = cmap_edges(norm_edges(edge_rel))

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_edges(G, layout, alpha=0.6, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, layout, node_size=node_sizes, node_color=node_colors, linewidths=0.2, edgecolors='k')
    plt.axis("off")
    plt.title(f"Rastgele ER Grafiği: n={G.number_of_nodes()}, m={G.number_of_edges()}")

    sm = cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.03, pad=0.04)
    cbar.set_label("Düğüm güvenilirliği")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path

def draw_graph_interactive(G, out_path, height="900px", width="100%"):
    """
    pyvis ile etkileşimli HTML (my_topology.html gibi) oluşturur.
    pyvis yüklü değilse fonksiyon hata fırlatır; script içinde try/except ile yakalanır.
    """
    if not _HAS_PYVIS:
        raise RuntimeError("pyvis yüklü değil. Etkileşimli HTML için 'pip install pyvis' kurun.")

    net = Network(height=height, width=width, notebook=False, directed=False)
    net.force_atlas_2based()

    for n in G.nodes():
        pdata = G.nodes[n]
        title = f"node: {n}<br>processing_delay_ms: {pdata.get('processing_delay_ms')}<br>node_reliability: {pdata.get('node_reliability')}"
        size = 10 + (pdata.get("processing_delay_ms", 0.5) - 0.5) / (2.0 - 0.5 + 1e-9) * 40
        color_val = pdata.get("node_reliability", 0.95)
        cmap = cm.get_cmap("viridis")
        norm = colors.Normalize(vmin=0.95, vmax=0.999)
        rgba = cmap(norm(color_val))
        hex_color = colors.to_hex(rgba)
        net.add_node(n, label=str(n), title=title, size=size, color=hex_color)

    for u, v in G.edges():
        ed = G.edges[u, v]
        bw = ed.get("bandwidth_mbps", 100)
        width = 1 + (bw - 100) / (1000 - 100 + 1e-9) * 4
        title = f"bandwidth_mbps: {bw}<br>link_delay_ms: {ed.get('link_delay_ms')}<br>link_reliability: {ed.get('link_reliability')}"
        net.add_edge(u, v, title=title, width=width)

    net.show(out_path)
    return out_path

def check_sd_pairs_connectivity(G, sd_pairs):
    """
    Verilen S-D çiftleri arasında yol olup olmadığını kontrol eder ve sonuçları döner.
    """
    results = []
    for s, d in sd_pairs:
        can = nx.has_path(G, s, d)
        results.append({"s": s, "d": d, "has_path": bool(can)})
    return results

def parse_args():
    """
    Komut satırı argümanlarını ayrıştırır.
    --sample-sd varsayılanı 1 olarak ayarlanmıştır.
    """
    parser = argparse.ArgumentParser(description="Rastgele ER grafiği üretir, öznitelik atar ve isteğe bağlı görselleştirir.")
    parser.add_argument("--n", type=int, default=250, help="Düğüm sayısı (varsayılan 250)")
    parser.add_argument("--p", type=float, default=0.4, help="Kenar oluşturma olasılığı (varsayılan 0.4)")
    parser.add_argument("--seed", type=int, default=None, help="Rastgele seed (int)")
    parser.add_argument("--out", type=str, default="topology", help="Çıktı öneki (varsayılan 'topology')")
    parser.add_argument("--max-attempts", type=int, default=50, help="Bağlı grafik için maksimum deneme sayısı")
    parser.add_argument("--sample-sd", type=int, default=1, help="Rastgele seçilecek S-D çifti sayısı (varsayılan 1)")
    parser.add_argument("--plot", action="store_true", help="Statik PNG görseli kaydet")
    parser.add_argument("--plot-path", type=str, default=None, help="Statik PNG kaydetme yolu (varsayılan: {out}.png)")
    parser.add_argument("--interactive", action="store_true", help="Etkileşimli HTML (pyvis) kaydet")
    parser.add_argument("--interactive-path", type=str, default=None, help="Etkileşimli HTML kaydetme yolu (varsayılan: {out}.html)")
    return parser.parse_args()

def main():
    """
    Programın ana akışı:
    - Argümanları oku
    - RNG oluştur (seed ile tekrar üretilebilirlik)
    - Grafiği üret, öznitelik ata, kaydet, (isteğe bağlı) görselleştir
    - Örnek S-D çiftleri için bağlantı kontrolü yapar
    """
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    
    print(f"Erdos-Renyi G({args.n}, {args.p}) grafiği oluşturuluyor (seed={args.seed}) ...")
    G, gen_info = generate_connected_erdos_renyi(args.n, args.p, rng, max_attempts=args.max_attempts)
    print(f"Graf oluşturma yöntemi: {gen_info}")

    print("Düğümlere ve kenarlara öznitelikler atanıyor...")
    G = annotate_graph(G, rng)

    print("Çıktılar kaydediliyor...")
    outputs = save_outputs(G, args.out)
    print(f"Kaydedilen dosyalar: {outputs}")

    # Statik görsel (matplotlib)
    if args.plot:
        plot_path = args.plot_path or f"{args.out}.png"
        print(f"Statik görsel oluşturuluyor -> {plot_path} ...")
        layout_seed = int(rng.integers(0, 2**31 - 1))
        try:
            p = draw_graph_static(G, plot_path, seed=layout_seed)
            print(f"Statik görsel kaydedildi: {p}")
        except Exception as e:
            print(f"Statik görsel oluşturulamadı: {e}")

    # Etkileşimli görsel (pyvis)
    if args.interactive:
        html_path = args.interactive_path or f"{args.out}.html"
        print(f"Etkileşimli HTML oluşturuluyor -> {html_path} ...")
        try:
            p = draw_graph_interactive(G, html_path)
            print(f"Etkileşimli HTML kaydedildi: {p}")
        except Exception as e:
            print(f"Etkileşimli HTML oluşturulamadı: {e}")
            if not _HAS_PYVIS:
                print("pyvis yüklü değil. Etkileşimli çıktı için 'pip install pyvis' komutunu çalıştırın.")

    # Örnek S-D çiftleri seçimi ve kontrolü
    sd_results = []
    if args.sample_sd > 0:
        all_nodes = list(G.nodes())
        sd_pairs = []
        for _ in range(args.sample_sd):
            s = int(rng.choice(all_nodes))
            d = int(rng.choice(all_nodes))
            while d == s:
                d = int(rng.choice(all_nodes))
            sd_pairs.append((s, d))
        sd_results = check_sd_pairs_connectivity(G, sd_pairs)
        print("Örnek S-D bağlantı sonuçları:")
        for r in sd_results:
            print(r)

    print("İşlem tamamlandı.")

if __name__ == "__main__":
    main()