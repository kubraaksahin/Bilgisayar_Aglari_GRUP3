import time
import math

import networkx as nx
import matplotlib.pyplot as plt

from Graf_Model import ProjeAgi              # graf dosyanın adı buysa böyle kalsın
from Metric_Evaluation import RouteEvaluator # metric dosyanın adı buysa böyle kalsın
from aco_router import AntColonyRouter

from pathlib import Path
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D




def read_int(prompt: str, min_v: int | None = None, max_v: int | None = None) -> int:
    while True:
        s = input(prompt).strip()
        try:
            x = int(s)
        except ValueError:
            print("Lütfen tam sayı girin.")
            continue
        if min_v is not None and x < min_v:
            print(f"En az {min_v} olmalı.")
            continue
        if max_v is not None and x > max_v:
            print(f"En fazla {max_v} olmalı.")
            continue
        return x


def read_float(prompt: str, min_v: float | None = None, max_v: float | None = None) -> float:
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            x = float(s)
        except ValueError:
            print("Lütfen sayı girin. (Örn: 0.33)")
            continue
        if min_v is not None and x < min_v:
            print(f"En az {min_v} olmalı.")
            continue
        if max_v is not None and x > max_v:
            print(f"En fazla {max_v} olmalı.")
            continue
        return x


def path_min_capacity_mbps(G, path) -> float:
    if not path or len(path) < 2:
        return 0.0
    caps = []
    for i in range(len(path) - 1):
        ed = G.get_edge_data(path[i], path[i + 1], default={})
        caps.append(float(ed.get("capacity_mbps", 0.0)))
    return min(caps) if caps else 0.0


def feasible_text(G, path, demand_mbps: float) -> str:
    cap = path_min_capacity_mbps(G, path)
    return "EVET" if cap >= demand_mbps else "HAYIR"


def build_layout_cached(G, seed=42):
    print("🧭 Layout hesaplanıyor (bir kere)...")
    return nx.spring_layout(G, seed=seed)


def save_path_on_full_graph_png(
    G,
    path,
    pos,
    out_file="cozum_rotasi_full_graph.png",
    title="ACO - Bulunan Rota (Full Graph)",
):
    """
    GRAF DEĞİŞTİRMEZ.
    Full graph çok soluk, path kırmızı + beyaz halo.
    Arrow yok.
    """
    if not path or len(path) < 2:
        print("⚠️ Çizim iptal: path boş veya çok kısa.")
        return

    path_edges = list(zip(path[:-1], path[1:]))

    plt.figure(figsize=(18, 12))

    # 1) Arka plan edges (daha da soluk yap)
    bg_edges = nx.draw_networkx_edges(
        G, pos,
        alpha=0.018,
        width=0.25,
        edge_color="gray",
        arrows=False
    )
    if hasattr(bg_edges, "set_zorder"):
        bg_edges.set_zorder(1)

    # 2) Arka plan nodes
    bg_nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=10,
        alpha=0.10,
        node_color="#1f77b4"
    )
    if hasattr(bg_nodes, "set_zorder"):
        bg_nodes.set_zorder(2)

    # 3) Path halo (kalın beyaz)
    halo = nx.draw_networkx_edges(
        G, pos,
        edgelist=path_edges,
        width=10.0,
        alpha=0.95,
        edge_color="white",
        arrows=False
    )
    if hasattr(halo, "set_zorder"):
        halo.set_zorder(9)

    # 4) Path kırmızı (üstte)
    path_lc = nx.draw_networkx_edges(
        G, pos,
        edgelist=path_edges,
        width=5.2,
        alpha=0.98,
        edge_color="red",
        arrows=False
    )
    if hasattr(path_lc, "set_zorder"):
        path_lc.set_zorder(10)
    # Ek bir vurgu: kırmızı çizgiye hafif siyah outline
    try:
        path_lc.set_path_effects([pe.Stroke(linewidth=6.5, foreground="black", alpha=0.25), pe.Normal()])
    except Exception:
        pass

    # 5) Path nodes
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=220, node_color="red", alpha=0.95)

    # 6) Source & Destination belirgin
    src, dst = path[0], path[-1]
    nx.draw_networkx_nodes(G, pos, nodelist=[src], node_size=380, node_color="green", alpha=0.95)
    nx.draw_networkx_nodes(G, pos, nodelist=[dst], node_size=380, node_color="blue", alpha=0.95)

    # 7) S/D label (kutulu)
    labels = {src: f"S:{src}", dst: f"D:{dst}"}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=12,
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.88)
    )

    # 8) Legend (profesyonel dokunuş)
    legend_elems = [
        Line2D([0], [0], color="red", lw=4, label="Path"),
        Line2D([0], [0], marker="o", color="w", label="Source", markerfacecolor="green", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Destination", markerfacecolor="blue", markersize=10),
    ]
    plt.legend(handles=legend_elems, loc="lower right", frameon=True)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=350)
    plt.close()
    print(f"✅ PNG kaydedildi: {out_file}")



def save_path_khop_png(
    G,
    path,
    pos_full,
    k_hops=2,
    out_file="cozum_rotasi_khop.png",
):
    if not path or len(path) < 2:
        print("⚠️ Çizim iptal: path boş veya çok kısa.")
        return

    # k-hop node set
    nodes = set(path)
    frontier = set(path)
    for _ in range(k_hops):
        nxt = set()
        for x in frontier:
            if hasattr(G, "successors"):
                nxt |= set(G.successors(x))
                nxt |= set(G.predecessors(x))
            else:
                nxt |= set(G.neighbors(x))
        nodes |= nxt
        frontier = nxt

    H = G.subgraph(nodes).copy()

    # Aynı global layout üzerinden alt küme pozisyon
    pos = {n: pos_full[n] for n in H.nodes() if n in pos_full}

    path_edges = list(zip(path[:-1], path[1:]))
    src, dst = path[0], path[-1]

    plt.figure(figsize=(16, 11))

    # Arka plan (k-hop) - biraz daha belirgin ama yine soluk
    nx.draw_networkx_edges(H, pos, alpha=0.06, width=0.45, edge_color="gray", arrows=False)
    nx.draw_networkx_nodes(H, pos, node_size=22, alpha=0.18, node_color="#1f77b4")

    # Path halo + red
    nx.draw_networkx_edges(H, pos, edgelist=path_edges, width=10.0, alpha=0.95, edge_color="white", arrows=False)
    lc = nx.draw_networkx_edges(H, pos, edgelist=path_edges, width=5.2, alpha=0.98, edge_color="red", arrows=False)
    try:
        lc.set_path_effects([pe.Stroke(linewidth=6.5, foreground="black", alpha=0.25), pe.Normal()])
    except Exception:
        pass

    nx.draw_networkx_nodes(H, pos, nodelist=path, node_size=240, node_color="red", alpha=0.95)
    nx.draw_networkx_nodes(H, pos, nodelist=[src], node_size=420, node_color="green", alpha=0.95)
    nx.draw_networkx_nodes(H, pos, nodelist=[dst], node_size=420, node_color="blue", alpha=0.95)

    labels = {src: f"S:{src}", dst: f"D:{dst}"}
    nx.draw_networkx_labels(
        H, pos,
        labels=labels,
        font_size=12,
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.88)
    )

    legend_elems = [
        Line2D([0], [0], color="red", lw=4, label="Path"),
        Line2D([0], [0], marker="o", color="w", label="Source", markerfacecolor="green", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Destination", markerfacecolor="blue", markersize=10),
    ]
    plt.legend(handles=legend_elems, loc="lower right", frameon=True)

    plt.title(f"ACO - Bulunan Rota (k-hop={k_hops})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=350)
    plt.close()
    print(f"✅ PNG kaydedildi: {out_file}")





def print_output_schema(src, dst, B, best_path, details, elapsed, aco_params):
    print("\nLütfen 0 ile 249 arasında düğüm numaraları girin.")
    print(f"👉 Başlangıç Düğümü (Kaynak): {src}")
    print(f"👉 Bitiş Düğümü (Hedef): {dst}")
    print(f"📦 Demand (B): {B:.2f} Mbps\n")

    print(f"🧠 Algoritma Çalışıyor... ({aco_params['num_iters']} Nesil/Iterasyon hesaplanacak)")
    print("-" * 35)

    if not best_path or not details or not math.isfinite(details.get("total_cost", float("inf"))):
        print("❌ SONUÇ BULUNAMADI")
        print("-" * 35)
        print(f"⏱️ Hesaplama Süresi: {elapsed:.4f} saniye")
        return

    print("✅ SONUÇ BULUNDU")
    print("-" * 35)
    print(f"⏱️ Hesaplama Süresi: {elapsed:.4f} saniye")
    print(f"🗺️ Rota: {best_path}")
    print(f"💰 Toplam Maliyet Skoru: {details['total_cost']:.4f}")

    bw_ok = feasible_text(G, best_path, B)
    print(f"📶 Bant Genişliği Uygun mu?: {bw_ok}")

    print("\n📊 Metrik Detayları:")
    print(f"  • Toplam Gecikme: {details.get('delay', 0.0):.2f} ms")
    print(f"  • Güvenilirlik Maliyeti: {details.get('reliability_cost', 0.0):.4f}")
    print(f"  • Kaynak Kullanımı: {details.get('resource_cost', 0.0):.4f}")

    print("\n⚙️ Kullanılan ACO Parametreleri:")
    for k in ["num_ants", "num_iters", "alpha", "beta", "rho", "q", "tau0", "max_steps", "elitist", "w_delay", "w_rel", "w_res"]:
        print(f"  • {k}: {aco_params[k]}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    node_csv = BASE_DIR / "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_csv = BASE_DIR / "BSM307_317_Guz2025_TermProject_EdgeData.csv"

    print("--- Veriler Yukleniyor (CSV Uyumlu) ---")
    proje = ProjeAgi(node_csv, edge_csv)
    G = proje.G
    evaluator = RouteEvaluator(G)

    # Layout bir kere (full graph PNG için)
    pos_full = build_layout_cached(G, seed=42)

    # Kullanıcı input
    src = read_int("👉 Başlangıç Düğümü (Kaynak): ", 0, 249)
    dst = read_int("👉 Bitiş Düğümü (Hedef): ", 0, 249)
    B = read_float("📦 Demand (B) Mbps (örn 200): ", 0.0, 10_000.0)

    print("\nAğırlıklar (toplamı 1 önerilir).")
    w_delay = read_float("⚖️ Wdelay (örn 0.33): ", 0.0, 1.0)
    w_rel = read_float("⚖️ Wreliability (örn 0.33): ", 0.0, 1.0)
    w_res = read_float("⚖️ Wresource (örn 0.34): ", 0.0, 1.0)

    w_sum = w_delay + w_rel + w_res
    if abs(w_sum - 1.0) > 1e-6:
        yn = input(f"\n⚠️ Ağırlıklar toplamı {w_sum:.4f}. Normalize edilsin mi? (E/H): ").strip().lower()
        if yn in ("e", "evet", "y", "yes"):
            w_delay /= w_sum
            w_rel /= w_sum
            w_res /= w_sum

    # ACO parametreleri (istersen sabit bırak)
    num_ants = 50
    num_iters = 60
    alpha = 0.8
    beta = 3.0
    rho = 0.1
    q = 0.5
    tau0 = 2.0
    max_steps = 450
    elitist = True

    print(f"\n🧠 Algoritma Çalışıyor... ({num_iters} Nesil/Iterasyon hesaplanacak)")
    t0 = time.time()

    router = AntColonyRouter(
        G, evaluator,
        w_delay=w_delay, w_rel=w_rel, w_res=w_res,
        num_ants=num_ants,
        num_iters=num_iters,
        alpha=alpha,
        beta=beta,
        rho=rho,
        q=q,
        tau0=tau0,
        max_steps=max_steps,
        elitist=elitist,
        seed=42
    )

    best_path, details = router.solve(src, dst, B)
    elapsed = time.time() - t0

    aco_params = {
        "num_ants": num_ants, "num_iters": num_iters,
        "alpha": alpha, "beta": beta, "rho": rho, "q": q, "tau0": tau0,
        "max_steps": max_steps, "elitist": elitist,
        "w_delay": round(w_delay, 4), "w_rel": round(w_rel, 4), "w_res": round(w_res, 4)
    }

    print_output_schema(src, dst, B, best_path, details, elapsed, aco_params)

    # PNG kaydetme (graf dosyasına dokunmadan)
    if best_path and details:
        yn = input("\n🖼️ Rotayı PNG olarak kaydetmek ister misiniz? (E/H): ").strip().lower()
        if yn in ("e", "evet", "y", "yes"):
            mode = input("Mod: 1=FullGraph  2=k-hop (Enter=2): ").strip()
            if mode == "1":
                print("\n🎨 Grafik çiziliyor, lütfen bekleyin...")
                save_path_on_full_graph_png(
                    G, best_path, pos_full,
                    out_file="cozum_rotasi_full_graph.png",
                    title="ACO - Bulunan Rota (Full Graph)"
                )

            else:
                k = read_int("k-hop değeri (öneri 2): ", 1, 6)
                print("\n🎨 Grafik çiziliyor, lütfen bekleyin...")
                save_path_khop_png(G, best_path, pos_full=pos_full, k_hops=k, out_file="cozum_rotasi_khop.png")
