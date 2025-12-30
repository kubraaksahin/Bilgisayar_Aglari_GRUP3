import time
import math
from pathlib import Path

from Graf_Model import ProjeAgi              
from Metric_Evaluation import RouteEvaluator 
from aco_router import AntColonyRouter       

def read_int(prompt: str, min_v: int | None = None, max_v: int | None = None) -> int:
    """Kullanıcıdan güvenli şekilde tam sayı okur (opsiyonel aralık kontrolüyle)."""
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
    """Kullanıcıdan güvenli şekilde float okur (',' yerine '.' destekler)."""
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


# -----------------------------------------------------------------------------
# Bant genişliği uygunluk kontrolü (path feasibility)
# -----------------------------------------------------------------------------


def path_min_capacity_mbps(G, path) -> float:
    """Path üzerindeki minimum kapasiteyi (bottleneck) döndürür."""
    if not path or len(path) < 2:
        return 0.0
    caps = []
    for i in range(len(path) - 1):
        ed = G.get_edge_data(path[i], path[i + 1], default={})
        caps.append(float(ed.get("capacity_mbps", 0.0)))
    return min(caps) if caps else 0.0


def feasible_text(G, path, demand_mbps: float) -> str:
    """Demand’i taşıyabiliyor mu? (min kapasite >= demand) => EVET/HAYIR."""
    cap = path_min_capacity_mbps(G, path)
    return "EVET" if cap >= float(demand_mbps) else "HAYIR"


# -----------------------------------------------------------------------------
# Çıktı formatı (hocanın istediği şemaya yakın, sade)
# -----------------------------------------------------------------------------


def print_output_schema(src, dst, B, best_path, details, elapsed, aco_params, bw_ok_text: str):
    """Sonucu ekrana düzenli bir şema ile basar."""
    print("\nLütfen 0 ile 249 arasında düğüm numaraları girin.")
    print(f"👉 Başlangıç Düğümü (Kaynak): {src}")
    print(f"👉 Bitiş Düğümü (Hedef): {dst}")
    print(f"📦 Demand (B): {B:.2f} Mbps\n")

    print(f"🧠 Algoritma Çalışıyor... ({aco_params['num_iters']} Nesil/Iterasyon hesaplanacak)")
    print("-" * 35)

    # Çözüm yoksa / maliyet geçersizse
    if (not best_path) or (not details) or (not math.isfinite(float(details.get("total_cost", float("inf"))))):
        print("❌ SONUÇ BULUNAMADI")
        print("-" * 35)
        print(f"⏱️ Hesaplama Süresi: {elapsed:.4f} saniye")
        return

    # Çözüm varsa
    print("✅ SONUÇ BULUNDU")
    print("-" * 35)
    print(f"⏱️ Hesaplama Süresi: {elapsed:.4f} saniye")
    print(f"🗺️ Rota: {best_path}")
    print(f"💰 Toplam Maliyet Skoru: {float(details['total_cost']):.4f}")
    print(f"📶 Bant Genişliği Uygun mu?: {bw_ok_text}")

    print("\n📊 Metrik Detayları:")
    print(f"  • Toplam Gecikme: {float(details.get('delay', 0.0)):.2f} ms")
    print(f"  • Güvenilirlik Maliyeti: {float(details.get('reliability_cost', 0.0)):.4f}")
    print(f"  • Kaynak Kullanımı: {float(details.get('resource_cost', 0.0)):.4f}")

    print("\n⚙️ Kullanılan ACO Parametreleri:")
    for k in ["num_ants", "num_iters", "alpha", "beta", "rho", "q", "tau0", "max_steps", "elitist", "w_delay", "w_rel", "w_res"]:
        print(f"  • {k}: {aco_params[k]}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    node_csv = BASE_DIR / "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_csv = BASE_DIR / "BSM307_317_Guz2025_TermProject_EdgeData.csv"

    print("--- Veriler Yukleniyor (CSV Uyumlu) ---")
    proje = ProjeAgi(node_csv, edge_csv)
    G = proje.G
    evaluator = RouteEvaluator(G)

    # Kullanıcıdan temel girişler
    src = read_int("👉 Başlangıç Düğümü (Kaynak): ", 0, 249)
    dst = read_int("👉 Bitiş Düğümü (Hedef): ", 0, 249)
    B = read_float("📦 Demand (B) Mbps (örn 200): ", 0.0, 10_000.0)

    # Çok kriterli maliyet ağırlıkları
    print("\nAğırlıklar (toplamı 1 önerilir).")
    w_delay = read_float("⚖️ Wdelay (örn 0.33): ", 0.0, 1.0)
    w_rel = read_float("⚖️ Wreliability (örn 0.33): ", 0.0, 1.0)
    w_res = read_float("⚖️ Wresource (örn 0.34): ", 0.0, 1.0)

    # Ağırlıklar 1 etmiyorsa kullanıcı isterse normalize edilir
    w_sum = w_delay + w_rel + w_res
    if abs(w_sum - 1.0) > 1e-6 and w_sum > 0:
        yn = input(f"\n⚠️ Ağırlıklar toplamı {w_sum:.4f}. Normalize edilsin mi? (E/H): ").strip().lower()
        if yn in ("e", "evet", "y", "yes"):
            w_delay /= w_sum
            w_rel /= w_sum
            w_res /= w_sum

    # ACO hiperparametreleri (tuning sonrası bulunan en iyi değerlerle güncellenebilir)
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

    # NOT: seed parametresi YOK (aco_router.py içinde SEED=42 sabit)
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
        elitist=elitist
    )

    best_path, details = router.solve(src, dst, B)
    elapsed = time.time() - t0

    aco_params = {
        "num_ants": num_ants,
        "num_iters": num_iters,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "q": q,
        "tau0": tau0,
        "max_steps": max_steps,
        "elitist": elitist,
        "w_delay": round(w_delay, 4),
        "w_rel": round(w_rel, 4),
        "w_res": round(w_res, 4),
    }

    # Bant genişliği uygunluk bilgisi (EVET/HAYIR)
    bw_ok_text = feasible_text(G, best_path, B) if best_path else "HAYIR"

    print_output_schema(src, dst, B, best_path, details, elapsed, aco_params, bw_ok_text)
