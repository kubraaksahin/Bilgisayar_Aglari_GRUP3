#!/usr/bin/env python
# coding: utf-8

# In[8]:


import random
import networkx as nx
import time
import math
from Graf_Model import ProjeAgi
from Metric_Evaluation import RouteEvaluator

class GenetikAlgoritmaYolBulucu:
    def __init__(self, graf, fitness, w_Delay, w_ReliabilityCost, w_Source, 
                 populasyon_boyutu=50, nesil_sayisi=50, mutasyon_orani=0.1, elit_sayisi=2, seed=None):
        self.graf = graf
        self.fitness = fitness
        self.w_Delay = w_Delay
        self.w_ReliabilityCost = w_ReliabilityCost
        self.w_Source = w_Source
        self.populasyon_boyutu = populasyon_boyutu
        self.nesil_sayisi = nesil_sayisi
        self.mutasyon_orani = mutasyon_orani
        self.elit_sayisi = elit_sayisi
        self.rng = random.Random(seed)

    def _rastgele_yol_uret(self, src, dst, talep_mbps):
        for _ in range(100):
            yol = [src]
            ziyaret_edilenler = {src}
            while yol[-1] != dst:
                mevcut = yol[-1]
                
                adaylar = [
                    k for k in self.graf.neighbors(mevcut)
                    if k not in ziyaret_edilenler and 
                    self.graf.get_edge_data(mevcut, k).get("capacity_mbps", 0) >= talep_mbps
                ]
                
                if not adaylar: break
                
                sonraki = self.rng.choice(adaylar)
                yol.append(sonraki)
                ziyaret_edilenler.add(sonraki)
                
                if len(yol) > 250: break 
            
            if yol[-1] == dst: return yol
        return None

    def _fitness_hesapla(self, yol):
        sonuc = self.fitness.calculate_total_fitness(
            yol, self.w_Delay, self.w_ReliabilityCost, self.w_Source
        )
        return sonuc['total_cost']

    def _crossover(self, anne, baba):
        ortak_yollar = list(set(anne) & set(baba))
        if ortak_yollar:
            secim = self.rng.choice(ortak_yollar)
            anne_idx = anne.index(secim)
            baba_idx = baba.index(secim)
            cocuk = anne[:anne_idx + 1] + baba[baba_idx + 1:]
            return cocuk if len(set(cocuk)) == len(cocuk) else anne
        return anne

    def _mutasyon(self, yol, talep_mbps):
        if len(yol) < 3 or self.rng.random() > self.mutasyon_orani:
            return yol
        
        indeks = self.rng.randint(1, len(yol) - 2)
        yeni_kuyruk = self._rastgele_yol_uret(yol[indeks], yol[-1], talep_mbps)
        return yol[:indeks] + yeni_kuyruk if yeni_kuyruk else yol

    def cozum_bul(self, src, dst, talep_mbps):
        populasyon = []
        for _ in range(self.populasyon_boyutu):
            yol = self._rastgele_yol_uret(src, dst, talep_mbps)
            if yol: populasyon.append(yol)
            
        if not populasyon: return None, {}

        for _ in range(self.nesil_sayisi):
           
            puanli_populasyon = sorted([(self._fitness_hesapla(y), y) for y in populasyon], key=lambda x: x[0])
        
            yeni_populasyon = [p[1] for p in puanli_populasyon[:self.elit_sayisi]]
            
            gen_havuzu = [p[1] for p in puanli_populasyon[:len(puanli_populasyon)//2 + 1]] or [p[1] for p in puanli_populasyon]

            while len(yeni_populasyon) < self.populasyon_boyutu:
                anne = self.rng.choice(gen_havuzu)
                baba = self.rng.choice(gen_havuzu)
                cocuk = self._crossover(anne, baba)
                cocuk = self._mutasyon(cocuk, talep_mbps)
                yeni_populasyon.append(cocuk)
            
            populasyon = yeni_populasyon

        en_iyi_yol = min(populasyon, key=self._fitness_hesapla)
        en_iyi_puan = self._fitness_hesapla(en_iyi_yol)
        
        return en_iyi_yol, {'total_cost': en_iyi_puan}

if __name__ == "__main__":
    node_file = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_file = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    
    # AYARLAR
    SRC_NODE = 8        
    DST_NODE = 44       
    TALEP_MBPS = 200.0 
    
    W_DELAY = 0.33
    W_REL   = 0.33
    W_RES   = 0.34

    print("=== 1. AG TOPOLOJISI YUKLENIYOR ===")
    proje = ProjeAgi(node_file, edge_file)
    G = proje.get_graph()
    
    print("\n=== 2. ALGORITMA HAZIRLANIYOR ===")
    route_evaluator = RouteEvaluator(G) 

    ga_solver = GenetikAlgoritmaYolBulucu(
        graf=G,
        fitness=route_evaluator, 
        w_Delay=W_DELAY,           
        w_ReliabilityCost=W_REL,   
        w_Source=W_RES,            
        populasyon_boyutu=100,     
        nesil_sayisi=50,        
        mutasyon_orani=0.10,     
        elit_sayisi=2          
    )

    print(f"\n>> Genetik Algoritma Calisiyor: {SRC_NODE} -> {DST_NODE}")

    if SRC_NODE not in G.nodes or DST_NODE not in G.nodes:
        print("HATA: Dugum bulunamadi!")
    else:
        t1 = time.time()
        en_iyi_yol, _ = ga_solver.cozum_bul(SRC_NODE, DST_NODE, TALEP_MBPS)
        sure = time.time() - t1

        if en_iyi_yol:
            detay = route_evaluator.calculate_total_fitness(en_iyi_yol, W_DELAY, W_REL, W_RES)
           
            try: guvenlik_yuzde = math.exp(-detay['reliability_cost']) * 100
            except: guvenlik_yuzde = 0.0

            print("-" * 40)
            print(f"Toplam Gecikme       : {detay['delay']:.2f} ms")
            print(f"Guvenilirlik         : %{guvenlik_yuzde:.4f}")
            print(f"Kaynak Maliyeti      : {detay['resource_cost']:.4f}")
            print("-" * 40)
            print(f"Fitness (Uygunluk)   : {detay['total_cost']:.4f}")
            print(f"Gecen Sure           : {sure:.4f} saniye")
            print(f"Bulunan Yol          : {' -> '.join(map(str, en_iyi_yol))}")
        else:
            print("\n>>> SONUC: BASARISIZ (Yol bulunamadi)")


# In[ ]:




