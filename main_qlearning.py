import os
import sys
import networkx as nx
import math

# Proje modülleri
from Graf_Model import ProjeAgi
from Metric_Evaluation import RouteEvaluator

# Q-Learning ajanını içe aktar
try:
    from q_agent import QLearningAgent
except ImportError:
    from q_agent import QLearningAgent


def main():
    # Girdi olarak kullanılacak CSV dosyaları
    node_file = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_file = "BSM307_317_Guz2025_TermProject_EdgeData.csv"

    # Dosyaların varlığı kontrol ediliyor
    if not os.path.exists(node_file) or not os.path.exists(edge_file):
        print("HATA: CSV dosyaları bulunamadı!")
        return

    print("Grafik yükleniyor...")

    # Grafik yapısı oluşturuluyor
    proje = ProjeAgi(node_file, edge_file)
    G = proje.get_graph()

    # Kullanıcıdan başlangıç ve hedef düğüm bilgileri alınıyor
    try:
        source = int(input("Başlangıç Düğümü ID: "))
        destination = int(input("Hedef Düğüm ID: "))
    except ValueError:
        print("Lütfen sayı girin.")
        return

    # Girilen düğümlerin grafikte olup olmadığı kontrolü
    if source not in G.nodes or destination not in G.nodes:
        print("HATA: Düğüm bulunamadı.")
        return

    print("\n--- METRIK AGIRLIKLARI ---")

    # Çok kriterli değerlendirme için ağırlıklar
    try:
        w_delay = float(input("Delay agirligi (orn 0.3): "))
        w_rel   = float(input("Reliability agirligi (orn 0.5): "))
        w_res   = float(input("Resource agirligi (orn 0.2): "))

        # Ağırlıkların toplamı 1 değilse normalize ediliyor
        total = w_delay + w_rel + w_res
        if abs(total - 1.0) > 1e-6:
            print("Uyari: Agirliklar 1'e esit degil, normalize ediliyor.")
            w_delay /= total
            w_rel   /= total
            w_res   /= total

    except ValueError:
        # Hatalı girişte varsayılan değerler
        print("Hatali giris! Varsayilan agirliklar kullaniliyor.")
        w_delay, w_rel, w_res = 0.25, 0.6, 0.15

    weights = {
        'w_delay': w_delay,
        'w_rel': w_rel,
        'w_res': w_res
    }

    print("Kullanilan agirliklar:", weights)

    # Eğitim adım sayısı
    episodes = 15000
    print(f"\n--- Q-Learning Eğitimi ({episodes} Episode) ---")

    # Q-Learning ajanı oluşturuluyor
    agent = QLearningAgent(
        G,
        source,
        destination,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0
    )

    # Ajan eğitiliyor
    agent.train(episodes, weights)

    # En iyi rota elde ediliyor
    q_path = agent.get_best_path()

    evaluator = RouteEvaluator(G)

    print("\n" + "=" * 60)
    print("           DETAYLI KARŞILAŞTIRMALI SONUÇLAR       ")
    print("=" * 60)

    # Rota ve metrik detaylarını yazdıran yardımcı fonksiyon
    def print_path_details(name, path, results, weights):
        print(f"\n>> {name}")

        if not path:
            print("   Rota bulunamadı!")
            return

        # Rota bilgisi
        print(f"   Rota: {' -> '.join(map(str, path))}")
        print(f"   ------------------------------------------------")

        print("   Kullanilan Agirliklar:")
        print(f"   • Delay:       {weights['w_delay']:.2f}")
        print(f"   • Reliability: {weights['w_rel']:.2f}")
        print(f"   • Resource:    {weights['w_res']:.2f}")

        delay = results['delay']
        rel_cost = results['reliability_cost']
        res_cost = results['resource_cost']
        total_score = results['total_cost']

        # Güvenilirlik oranı log-cost üzerinden hesaplanıyor
        try:
            reliability_rate = math.exp(-rel_cost) * 100
        except OverflowError:
            reliability_rate = 0.0

        print(f"   ------------------------------------------------")
        print(f"   • Toplam Gecikme:      {delay:.4f} ms")
        print(f"   • Güvenilirlik Oranı:  %{reliability_rate:.4f}")
        print(f"   • Kaynak Maliyeti:     {res_cost:.4f}")
        print(f"   ------------------------------------------------")
        print(f"    GENEL SKOR (Cost):   {total_score:.4f}")

    # Q-Learning sonucu geçerli mi kontrol ediliyor
    if q_path and q_path[-1] == destination:
        q_results = evaluator.calculate_total_fitness(q_path, **weights)
        print_path_details("Q-Learning", q_path, q_results, weights)
    else:
        print("\n>> Q-Learning\n   HEDEF ULAŞILAMADI!")
        q_results = {'total_cost': float('inf')}


if __name__ == "__main__":
    main()

