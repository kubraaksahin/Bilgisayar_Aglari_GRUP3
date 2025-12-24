import numpy as np
import random
import networkx as nx
from Metric_Evaluation import RouteEvaluator # Senin yazdığın sınıfı çağırıyoruz

class QLearningAgent:
    def __init__(self, graph, source, destination, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.graph = graph
        self.source = source
        self.destination = destination
        
        # Hiperparametreler
        self.alpha = alpha      # Öğrenme hızı
        self.gamma = gamma      # Gelecek odaklılık
        self.epsilon = epsilon  # Keşif oranı (Başlangıçta %100) ajanın acemilik oranı rastgele hareket etme oranı
        self.epsilon_decay = 0.9999 # Her turda azalma hızı
        self.epsilon_min = 0.01    # Minimum keşif oranı
        
        self.q_table = {} # Hafıza Defteri

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, current_node):
        """Epsilon-Greedy ile sonraki düğümü seç"""
        neighbors = list(self.graph.neighbors(current_node))
        if not neighbors: return None

        # 1. Keşif (Exploration): Rastgele git
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(neighbors)
        
        # 2. Sömürü (Exploitation): Bildiğin en iyi yoldan git
        q_values = [self.get_q_value(current_node, n) for n in neighbors]
        max_q = max(q_values)
        
        # En iyi olanların arasından rastgele seç (Eşitlik varsa)
        best_actions = [neighbors[i] for i in range(len(neighbors)) if q_values[i] == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        """Q-Tablosunu güncelle """
        # Gelecekteki en iyi ihtimali bul
        next_neighbors = list(self.graph.neighbors(next_state))
        if next_neighbors:
            max_future_q = max([self.get_q_value(next_state, n) for n in next_neighbors])
        else:
            max_future_q = 0.0

        current_q = self.get_q_value(state, action)
        
        # Formül: Eski Değer + Öğrenme Hızı * (Ödül + Gelecek Tahmini - Eski Değer)
        new_q = current_q + self.alpha * (reward + (self.gamma * max_future_q) - current_q)
        self.q_table[(state, action)] = new_q

    def train(self, episodes, weights):
        print(f"Eğitim başlıyor... Hedef: {episodes} epizot.")
        
        # Alpha'yı sabitleyelim (Yavaş ve emin adımlarla öğrensin)
        self.alpha = 0.1

        for episode in range(episodes):
            current_node = self.source
            path = [current_node]
            step_count = 0
            
            # --- 1. Ajan Rastgele Geziniyor (Veri Topluyor) ---
            while current_node != self.destination and step_count < 100:
                next_node = self.choose_action(current_node)
                
                if next_node is None: break 
                
                # BURADAKİ ANLIK ÖĞRENMEYİ KAPATTIM!
                # Sadece gezip yolu hafızaya atıyor.
                
                current_node = next_node
                path.append(current_node)
                step_count += 1
            
            # --- 2. Hedefe Vardıysa Geriye Dönük Öğreniyor ---
            if current_node == self.destination:
                # Toplam maliyeti hesapla
                # (Evaluator kodunun agent içinde veya dışarıda olmasına göre burayı düzenle)
                evaluator = RouteEvaluator(self.graph) # Eğer sınıf içindeyse self.evaluator
                metrics = evaluator.calculate_total_fitness(path, **weights)
                total_cost = metrics['total_cost']
                
                # --- DÜZELTME BURADA: Üs alma işlemini 2 yaptık ---
                if total_cost > 0:
                    final_reward = (1000.0 / total_cost)
                else:
                    final_reward = 0
                
                # Ödülü sondan başa doğru dağıt (En kararlı yöntem)
                running_reward = final_reward
                
                for i in range(len(path) - 2, -1, -1):
                    u = path[i]
                    v = path[i+1]
                    
                    old_q = self.get_q_value(u, v)
                    # Q-Learning Formülü
                    new_q = old_q + self.alpha * (running_reward - old_q)
                    self.q_table[(u, v)] = new_q
                    
                    # Ödülü sönümle
                    running_reward = running_reward * self.gamma

            # Epsilon Decay (Keşfi azalt)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
     

    def get_best_path(self):
        """
        Bu fonksiyon komşuları ID sırasına dizer, böylece sonuç ASLA değişmez.
        Ayrıca ekrana Q değerlerini basar, böylece ajanın öğrenip öğrenmediğini görürüz.
        """
        path = [self.source]
        current_node = self.source
        
        print(f"\n--- EN İYİ YOL SEÇİLİYOR (Test Aşaması) ---")
        
        while current_node != self.destination:
            # 1. KİLİT NOKTA: Komşuları sırala! (NetworkX rastgele vermesin)
            neighbors = sorted(list(self.graph.neighbors(current_node)))
            
            if not neighbors:
                print("Çıkmaz sokağa girildi.")
                break
            
            # Komşuların Q değerlerini al ve ekrana yaz (Debug)
            q_values = []
            debug_info = []
            
            for n in neighbors:
                val = self.get_q_value(current_node, n)
                q_values.append(val)
                debug_info.append(f"Git->{n}: Puan={val:.4f}")
            
            # Ekrana bas: Ajan şu an ne düşünüyor?
            print(f"Düğüm {current_node} karar anı: {debug_info}")
            
            # Hepsi 0 ise ajan burayı HİÇ öğrenmemiş demektir
            if all(v == 0 for v in q_values):
                print("UYARI: Ajan burası için hiçbir şey öğrenmemiş! Rastgele (ama sıralı) gidiyor.")
            
            # En yüksek puanı bul
            max_q = max(q_values)
            
            # 2. KİLİT NOKTA: Puanları eşit olanlardan her zaman İLKİNİ seç (Sıralı olduğu için hep aynı olur)
            # index() fonksiyonu listedeki 'ilk' en büyüğü bulur.
            best_index = q_values.index(max_q)
            best_node = neighbors[best_index]
            
            # Döngü kontrolü
            if best_node in path:
                print(f"HATA: {best_node} düğümüne geri döndü (Sonsuz Döngü).")
                break
                
            path.append(best_node)
            current_node = best_node
            
            if len(path) > 100:
                print("Rota çok uzadı, kesiliyor.")
                break
                
        return path