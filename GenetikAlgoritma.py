#!/usr/bin/env python
# coding: utf-8

# In[49]:


#!/usr/bin/env python
# coding: utf-8

# In[8]:

import numpy as np
import random
import networkx as nx
import time
import math
from Graf_Model import ProjeAgi
from Metric_Evaluation import RouteEvaluator

seed_degeri = 42

def algoritmayi_baslat():
    random.seed(seed_degeri)
    np.random.seed(seed_degeri)
    print(f"Sistem {seed_degeri} seed'i ile sabitlendi.")
    
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
        for _ in range(100): #100 kere yol bulmaya çalışıyor
            yol = [src] #ilk düğümü direkt yolun içine ekliyor 
            ziyaret_edilenler = {src} # ziyaret edilenler tuple ı içine ekleme yapıyor 
            while yol[-1] != dst: #son düğüme ulaşılıp ulaşılmadığını kontrol ediyor
                mevcut = yol[-1] #son düğümü alıyor
                
                adaylar = [
                    k for k in self.graf.neighbors(mevcut) #komşuları alıyor
                    if k not in ziyaret_edilenler and 
                    self.graf.get_edge_data(mevcut, k).get("capacity_mbps", 0) >= talep_mbps
                ] #ziyaret edilenler listesindeyse ve kapasitesi karşılıyorsa adaylar listesinin içine ekliyor 
                
                if not adaylar: break
                
                sonraki = self.rng.choice(adaylar) #adayların içinden rastgele seçim yapıyor 
                yol.append(sonraki) #sonraki yola geçiyor 
                ziyaret_edilenler.add(sonraki) # ziyaret edilenlere ekliyor 
                
                if len(yol) > 250: break 
            
            if yol[-1] == dst: return yol # son düğüme ulaşınca da buldugu yolu döndürüyor 
        return None

    def _fitness_hesapla(self, yol): # fitness değerini hesaplıyor 
        sonuc = self.fitness.calculate_total_fitness(
            yol, self.w_Delay, self.w_ReliabilityCost, self.w_Source
        )
        return sonuc['total_cost']

    def _crossover(self, anne, baba):
        ortak_yollar = list(set(anne) & set(baba)) #anne ile baba arasından ortak liste alıyor 
        if ortak_yollar: # eger varsa 
            secim = self.rng.choice(ortak_yollar) #ortak yolların içinden rastgele seçim yapıyor 
            anne_idx = anne.index(secim)
            baba_idx = baba.index(secim) #seçilen düğümün indekslerini buluyor 
            cocuk = anne[:anne_idx + 1] + baba[baba_idx + 1:] #annenin o indekse kadar başını babanın da sonunu alıp birleştiriyor 
            return cocuk if len(set(cocuk)) == len(cocuk) else anne #çocukta aynı düğüme 2 defa gidildiyse direkt anneyi döndürüyor
        return anne

    def _mutasyon(self, yol, talep_mbps):
        if len(yol) < 3 or self.rng.random() > self.mutasyon_orani:
            return yol #yol 3 den kısa ise ya da mutasyon oranı rastgele gelen mutasyon oranından kücükse direkt yolu döndürüyor 
        if len(yol) == 3: # yol uzunlugu 3 ise 
            baslangic = yol[0] 
            bitis = yol[-1]
            edge_data = self.graf.get_edge_data(baslangic, bitis)
            if edge_data is not None and edge_data.get("capacity_mbps", 0) >= talep_mbps:
                return [baslangic, bitis]
              # başlangıç bitiş düğümlerini alıp aralarında yol var mı kontrol ediyor eğer yol varsa kapasitesi karşılıyor mu diye kontrol ediyor eğer ikisini de karşılarsa direkt o yolu döndürüyor  
        indeks = self.rng.randint(1, len(yol) - 2) # ilk ve son düğüm haricinde bi düğümden rastgele bi düğüm seçiyo
        yeni_kuyruk = self._rastgele_yol_uret(yol[indeks], yol[-1], talep_mbps) # belirlediğimiz indeksten kesip devamını rastgele şekilde üretiyoruz
        return yol[:indeks] + yeni_kuyruk if yeni_kuyruk else yol 

    def cozum_bul(self, src, dst, talep_mbps):
        populasyon = [] 
        butun_nesiller=[] # boş populasyon ve butun nesilleri tutabilmek için boş liste oluşturuyorum
        for _ in range(self.populasyon_boyutu):
            yol = self._rastgele_yol_uret(src, dst, talep_mbps) # rastgele populasyon boyutu kadar yol üretiyoruz
            if yol: populasyon.append(yol) # eger none donmezse hepsini populaasyona ekliyoruz
                
        if not populasyon: return None, {} 
        for i in range(self.nesil_sayisi): 
           
            puanli_populasyon = sorted([(self._fitness_hesapla(y), y) for y in populasyon], key=lambda x: x[0])
            # populasyonu dönüp her bir yolun fitness degerini buluyoruz hepsini puanlı populasyon listesine ekleyip kücük fitness degerinden büyüğüne sıralama yapıyoruz 
            yeni_populasyon = [p[1] for p in puanli_populasyon[:self.elit_sayisi]] 
            # elit sayısı kadar olan yolları yeni populasyon listesine atıyoruz 
            gen_havuzu = [p[1] for p in puanli_populasyon[:int(len(puanli_populasyon) //2 + 1)]] or [p[1] for p in puanli_populasyon]
            # en iyi genlerin yarısını gen havuzuna atıyoruz 
            while len(yeni_populasyon) < self.populasyon_boyutu: #yeni populasyon istedigimiz poppulasyon sayısına ulaşana kadar dönüyoruz
                anne = self.rng.choice(gen_havuzu) 
                baba = self.rng.choice(gen_havuzu) #anne babaya gen havuzundan rastgele yollar seçiyoruz 
                cocuk = self._crossover(anne, baba) #anne babayı çaprazlıyoruz 
                cocuk = self._mutasyon(cocuk, talep_mbps) # cocuga da mutasyon yapıyopruz
                if cocuk not in yeni_populasyon:
                    yeni_populasyon.append(cocuk) # cocuk yeni populasyonun içinde yoksa ekliyruz
            
            populasyon = yeni_populasyon # populasyona en son buldugumuz populasyonu ekliyoruz 
            en_iyi_yol = min(populasyon, key=self._fitness_hesapla) # hepsinin fitness degerlerine göre sıralama yapıyoruz en iyi yolu buluyoruz  
            butun_nesiller.append(en_iyi_yol) # bütün nesillere en iyi yolu ekliyoruz 
            bu_neslin_skoru = self._fitness_hesapla(en_iyi_yol) # en iyi yolun fitness degerini buluyruz 
            print(f"Nesil {i+1:02d} | En İyi Fitness: {bu_neslin_skoru:.4f} | Yol Uzunluğu: {en_iyi_yol}")
        butun_nesiller_puan_yol=[] # bütün nesilleri tutmalık bi liste oluşturuyruz 
        for eleman in butun_nesiller:
         #   print(eleman)
            yol_listesi = list(eleman) 
            butun_nesiller_puanlari = self._fitness_hesapla(yol_listesi) # bütün nesillerin fitness degerlerini tekrar alıp
            butun_nesiller_puan_yol.append((butun_nesiller_puanlari, yol_listesi)) #bütün nesillerin puan yol şeklinde tutuyoz
        sirali_liste = sorted(butun_nesiller_puan_yol, key=lambda x: x[0]) # sıralayıp en yisini aliyoruz 
        return sirali_liste[0][1], {'total_cost':  sirali_liste[0][0]} # listenin 0.indisinin 1.indisini yani yolunu aliyoruz ve total cost içinde döndürüyoruz  

if __name__ == "__main__":
    node_file = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_file = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    algoritmayi_baslat()
    # AYARLAR
    SRC_NODE = 8 #int(input("Başlangıç düğümü seçiniz:"))        
    DST_NODE = 44 #int(input("Son düğümü seçiniz:"))       
    TALEP_MBPS = 100 #int(input("Talep seçiniz:")) 
    
    W_DELAY = 0.33 #float(input("Delay agirligi "))
    W_REL   = 0.33 #float(input("Reliability agirligi "))
    W_RES   = 0.34 #float(input("Resource agirligi "))

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
        populasyon_boyutu=50,     
        nesil_sayisi=25,        
        mutasyon_orani=0.2,     
        elit_sayisi=1,
        seed = seed_degeri
    
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


# In[ ]:





# In[ ]:




