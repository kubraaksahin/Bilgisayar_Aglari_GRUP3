import os
import sys
import networkx as nx
import math
# --- MODÜLLERİ İÇERİ AKTARMA ---
from Graf_Model import ProjeAgi
from Metric_Evaluation import RouteEvaluator
try:
    from q_agent import QLearningAgent
except ImportError:
    from q_agent import QLearningAgent

def main():
    print("=================================================")
    print("   Q-LEARNING ROTALAMA OPTİMİZASYONU      ")
    print("=================================================")

    node_file = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_file = "BSM307_317_Guz2025_TermProject_EdgeData.csv"

    if not os.path.exists(node_file) or not os.path.exists(edge_file):
        print("HATA: CSV dosyaları bulunamadı!")
        return

    print("Grafik yükleniyor...")
    proje = ProjeAgi(node_file, edge_file)
    G = proje.get_graph()
    
    # KULLANICI GİRDİLERİ
    try:
        source = int(input("Başlangıç Düğümü ID: "))
        destination = int(input("Hedef Düğüm ID: "))
    except ValueError:
        print("Lütfen sayı girin.")
        return

    if source not in G.nodes or destination not in G.nodes:
        print("HATA: Düğüm bulunamadı.")
        return

    # AĞIRLIKLAR
        print("\n--- METRIK AGIRLIKLARI ---")
    try:
        w_delay = float(input("Delay agirligi (orn 0.3): "))
        w_rel   = float(input("Reliability agirligi (orn 0.5): "))
        w_res   = float(input("Resource agirligi (orn 0.2): "))

        total = w_delay + w_rel + w_res
        if abs(total - 1.0) > 1e-6:
            print("Uyari: Agirliklar 1'e esit degil, normalize ediliyor.")
            w_delay /= total
            w_rel   /= total
            w_res   /= total

    except ValueError:
        print("Hatali giris! Varsayilan agirliklar kullaniliyor.")
        w_delay, w_rel, w_res = 0.25, 0.6, 0.15

    weights = {
        'w_delay': w_delay,
        'w_rel': w_rel,
        'w_res': w_res
    }

    print("Kullanilan agirliklar:", weights)

    # --- 1. Q-LEARNING KISMI ---
    # Episode sayısını artırdım çünkü 250 düğümlü ağ karmaşık 5000 turda öğrenemedi
    episodes = 15000 
    print(f"\n--- Q-Learning Eğitimi ({episodes} Episode) ---")
    
    # Epsilon decay agent.py içinde değiştirilmeli! (0.999 önerilir)
    agent = QLearningAgent(G, source, destination, alpha=0.1, gamma=0.9, epsilon=1.0)
    agent.train(episodes, weights)
    
    q_path = agent.get_best_path()

    # --- 2. DEĞERLENDİRME VE KIYASLAMA ---
    evaluator = RouteEvaluator(G)
    
    print("\n" + "="*60)
    print("           DETAYLI KARŞILAŞTIRMALI SONUÇLAR       ")
    print("="*60)

    # --- Yardımcı Fonksiyon: Sonuçları Ekrana Bas ---
    def print_path_details(name, path, results, weights):
        print(f"\n>> {name}")
        if not path:
            print("   Rota bulunamadı!")
            return
        
        # 1. Rotayı Yazdır
        print(f"   Rota: {' -> '.join(map(str, path))}")
        print(f"   ------------------------------------------------")

                # Kullanılan Ağırlıkları Yazdır
        print(f"   Kullanilan Agirliklar:")
        print(f"   • Delay:       {weights['w_delay']:.2f}")
        print(f"   • Reliability: {weights['w_rel']:.2f}")
        print(f"   • Resource:    {weights['w_res']:.2f}")

        # 2. Metrikleri Ayrıştır
        delay = results['delay']
        rel_cost = results['reliability_cost']
        res_cost = results['resource_cost']
        total_score = results['total_cost']
        
        # 3. Güvenilirlik Maliyetini Yüzdeye (%) Çevir
        # Matematik: e^(-maliyet) bize olasılığı verir.
        try:
            reliability_rate = math.exp(-rel_cost) * 100
        except OverflowError:
            reliability_rate = 0.0

        # 4. Ekrana Bas
        print(f"   ------------------------------------------------")
        print(f"   • Toplam Gecikme:      {delay:.4f} ms")
        print(f"   • Güvenilirlik Oranı:  %{reliability_rate:.4f}")
        print(f"   • Kaynak Maliyeti:     {res_cost:.4f}")
        print(f"   ------------------------------------------------")
        print(f"   ★ GENEL SKOR (Cost):   {total_score:.4f}")

    # A) Q-Learning Sonucu
    if q_path and q_path[-1] == destination:
        q_results = evaluator.calculate_total_fitness(q_path, **weights)
        print_path_details("Q-Learning", q_path, q_results,weights)
    else:
        print("\n>> Q-Learning\n   HEDEF ULAŞILAMADI!")
        q_results = {'total_cost': float('inf')}

if __name__ == "__main__":

    main()
