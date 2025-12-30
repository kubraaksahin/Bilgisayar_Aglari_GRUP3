import math
import random
import pandas as pd

from Graf_Model import ProjeAgi
from Metric_Evaluation import RouteEvaluator
from aco_router import AntColonyRouter

def load_demands(demand_csv_path: str, limit: int | None = None):
    """DemandData.csv dosyasını okuyup senaryoları döndürür.

    CSV formatı:
      - ayraç: ';'
      - ondalık: ','  (decimal=",")
    """
    df = pd.read_csv(demand_csv_path, sep=";", decimal=",")
    demands = []
    for _, row in df.iterrows():
        demands.append({
            "src": int(row["src"]),
            "dst": int(row["dst"]),
            "demand_mbps": float(row["demand_mbps"])
        })
    return demands[:limit] if limit else demands


def path_min_capacity_mbps(G, path) -> float:
    """Bir path üzerindeki minimum kapasiteyi (bottleneck) döndürür."""
    if not path or len(path) < 2:
        return 0.0
    caps = []
    for i in range(len(path) - 1):
        ed = G.get_edge_data(path[i], path[i + 1], default={})
        caps.append(float(ed.get("capacity_mbps", 0.0)))
    return min(caps) if caps else 0.0


def is_bw_feasible(G, path, demand_mbps: float) -> bool:
    """Bant genişliği uygunluk kontrolü: min(capacity) >= demand."""
    return path_min_capacity_mbps(G, path) >= float(demand_mbps)


def run_single_demand(G, evaluator, demand, aco_params, weights):
    """Tek bir demand senaryosu için ACO'yu çalıştırır.

    - aco_params: ACO hiperparametreleri (num_ants, num_iters, alpha, beta, rho, ...)
    - weights: (w_delay, w_rel, w_res) -> çok kriterli maliyet ağırlıkları

    Başarısız olursa (None, None) döner.
    """
    router = AntColonyRouter(
        G, evaluator,
        w_delay=weights[0], w_rel=weights[1], w_res=weights[2],
        **aco_params
    )

    path, details = router.solve(demand["src"], demand["dst"], demand["demand_mbps"])

    # Yol veya detay yoksa başarısız kabul et
    if not path or not details:
        return None, None

    # Toplam maliyet sayısal değilse başarısız kabul et
    total_cost = float(details.get("total_cost", float("inf")))
    if not math.isfinite(total_cost):
        return None, None

    # Bandwidth uygun değilse (bottleneck < demand) başarısız kabul et
    if not is_bw_feasible(G, path, demand["demand_mbps"]):
        return None, None

    return path, details


def tune_hyperparams_random(
    G,
    evaluator,
    demands,
    weights,
    trials: int = 25,
    tune_demands_count: int = 8,
    repeats: int = 3,
    seed: int = 42,
    fail_penalty: float = 1e6
):
    """Random Search ile ACO hiperparametre optimizasyonu yapar.

    Parametre uzayından rastgele seçim yapılır; her trial için seçilen parametreler
    birkaç demand üzerinde test edilir.

    Skor hesabı:
      - Her demand için (repeats içinde) bulunan en iyi maliyet alınır.
      - Bulunamazsa fail_penalty yazılır.
      - Trial skoru = demand başına en iyi maliyetlerin ortalaması.

    Not:
      - ACO seed'i içeride sabit olduğu için, repeats tekrarları aynı sonuçları
        üretmeye eğilimlidir. Yine de "başarı oranı" gibi raporlamalar için repeats
        yapısı korunmuştur.
    """
    rng = random.Random(seed)

    # Denenecek hiperparametre uzayı
    space = {
        "num_ants": [20, 30, 40, 50, 60, 80],
        "num_iters": [40, 60, 80, 120],
        "alpha": [0.8, 1.0, 1.5, 2.0],
        "beta": [2.0, 3.0, 4.0, 6.0],
        "rho": [0.1, 0.2, 0.3, 0.4],
        "q": [0.5, 1.0, 2.0, 5.0],
        "tau0": [0.5, 1.0, 2.0],
        "max_steps": [300, 450, 600],
        "elitist": [True, False],
    }

    tune_demands = demands[:tune_demands_count] if len(demands) >= tune_demands_count else demands

    best_params = None
    best_score = float("inf")
    best_success_rate = -1.0

    print("\n=== Hiperparametre Optimizasyonu (Random Search) ===")
    print(f"Tuning demands: {len(tune_demands)} | repeats per demand: {repeats} | trials: {trials}")

    for t in range(1, trials + 1):
        # Her trial'da uzaydan rastgele bir kombinasyon seçilir
        params = {k: rng.choice(v) for k, v in space.items()}

        per_demand_best_costs = []
        success_runs = 0
        total_runs = 0

        for d in tune_demands:
            best_cost_for_demand = float("inf")

            # Aynı demand için birkaç tekrar
            for _ in range(repeats):
                total_runs += 1

                path, details = run_single_demand(G, evaluator, d, params, weights)
                if path is None:
                    continue

                success_runs += 1
                cost = float(details["total_cost"])
                if cost < best_cost_for_demand:
                    best_cost_for_demand = cost

            # Demand için en iyi maliyet bulunamadıysa ceza yazar
            if math.isfinite(best_cost_for_demand):
                per_demand_best_costs.append(best_cost_for_demand)
            else:
                per_demand_best_costs.append(fail_penalty)

        score = sum(per_demand_best_costs) / len(per_demand_best_costs)
        success_rate = success_runs / max(1, total_runs)

        print(f"[trial {t:02d}/{trials}] score={score:.4f} success={success_rate:.2%} params={params}")

        # Daha düşük skor daha iyi; eşitse başarı oranı yüksek olan tercih edilir
        if (score < best_score - 1e-9) or (abs(score - best_score) <= 1e-9 and success_rate > best_success_rate):
            best_score = score
            best_success_rate = success_rate
            best_params = params

    print("\n=== EN İYİ PARAMETRELER ===")
    print("best_score:", best_score)
    print("best_success_rate:", f"{best_success_rate:.2%}")
    print("best_params:", best_params)

    return best_params


def evaluate_demands_simplified(
    G,
    evaluator,
    demands,
    aco_params,
    weights,
    repeats: int = 5,
    limit: int = 20
):
    """En iyi aco_params ile DemandData üzerinde sade test çalıştırır."""
    used = demands[:limit]
    print("\n=== DemandData Üzerinde Test (Sade Çıktı) ===")
    print(f"Scenarios: {len(used)} | repeats per scenario: {repeats}")

    for i, d in enumerate(used, start=1):
        best = None
        success = 0

        for _ in range(repeats):
            path, details = run_single_demand(G, evaluator, d, aco_params, weights)
            if path is None:
                continue

            success += 1
            cost = float(details["total_cost"])

            # Bu senaryoda bulunan en iyi (en düşük) maliyetli çözümü saklar
            if best is None or cost < best["cost"]:
                bw_ok = is_bw_feasible(G, path, d["demand_mbps"])
                best = {
                    "path": path,
                    "cost": cost,
                    "delay": float(details.get("delay", 0.0)),
                    "rel_cost": float(details.get("reliability_cost", 0.0)),
                    "res_cost": float(details.get("resource_cost", 0.0)),
                    "bw_ok": bw_ok
                }

        B = float(d["demand_mbps"])

        if best is None:
            print(f"[{i:02d}] S={d['src']} D={d['dst']} B={B} | ❌ BULUNAMADI")
        else:
            bw_text = "EVET" if best["bw_ok"] else "HAYIR"
            print(
                f"[{i:02d}] S={d['src']} D={d['dst']} B={B} | ✅ BULUNDU ({success}/{repeats}) | "
                f"BW_UYGUN?: {bw_text} | Rota: {best['path']} | "
                f"TotalCost={best['cost']:.4f} | Delay={best['delay']:.2f} ms | "
                f"RelCost={best['rel_cost']:.4f} | ResCost={best['res_cost']:.4f}"
            )


if __name__ == "__main__":
    # Dosya isimleri 
    node_csv = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_csv = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    demand_csv = "BSM307_317_Guz2025_TermProject_DemandData.csv"

    print("--- Veriler Yukleniyor (CSV Uyumlu) ---")
    proje = ProjeAgi(node_csv, edge_csv)
    G = proje.G
    evaluator = RouteEvaluator(G)

    # Demand senaryolarını yükle
    demands = load_demands(demand_csv)

    # Çok kriterli maliyet ağırlıkları (Delay / Reliability / Resource)
    weights = (0.33, 0.33, 0.34)

    # 1) Hiperparametre tuning (Random Search)
    best_params = tune_hyperparams_random(
        G, evaluator, demands, weights,
        trials=25,
        tune_demands_count=8,
        repeats=3,
        seed=42,
        fail_penalty=1e6
    )

    # 2) En iyi parametrelerle DemandData test
    evaluate_demands_simplified(
        G, evaluator, demands,
        aco_params=best_params,
        weights=weights,
        repeats=5,
        limit=20
    )
