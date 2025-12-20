import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import time # Süre ölçümü için

class ProjeAgi:
    def __init__(self, node_file, edge_file):
        self.node_file = node_file
        self.edge_file = edge_file
        self.G = nx.DiGraph()
        self.load_data()
        
    def get_graph(self):
        return self.G
    
    def load_data(self):
        print("--- Veriler Yukleniyor (CSV Uyumlu) ---")
        # 1. Dugumleri Yukle
        if os.path.exists(self.node_file):
            df_nodes = pd.read_csv(self.node_file, sep=';', decimal=',')
            for _, row in df_nodes.iterrows():
               
                self.G.add_node(int(row['node_id']), 
                                s_ms=float(row['s_ms']), 
                                r_node=float(row['r_node']))
            print(f"Dugumler tamam: {len(df_nodes)}")
        
        # 2. Kenarlari Yukle
        if os.path.exists(self.edge_file):
            df_edges = pd.read_csv(self.edge_file, sep=';', decimal=',')
            for _, row in df_edges.iterrows():
                self.G.add_edge(int(row['src']), int(row['dst']), 
                                capacity_mbps=float(row['capacity_mbps']), 
                                delay_ms=float(row['delay_ms']), 
                                r_link=float(row['r_link']))
            print(f"Kenarlar tamam: {len(df_edges)}")


    # --- GÖRSELLEŞTİRME VE KAYDETME FONKSİYONU ---
    def save_network_image(self, output_filename="ag_topolojisi_yuksek_cozunurluk.png"):
        """
        Yoğun grafiği hesaplar ve yüksek çözünürlüklü PNG olarak kaydeder.
        Ekrana çizim yapmaz, doğrudan dosyaya yazar.
        """
        start_time = time.time()
        print(f"--- Grafik Hazirlaniyor ({self.G.number_of_edges()} baglanti) ---")
        print("Lutfen bekleyin, bu islem bilgisayar hizina gore 30-90 saniye surebilir...")

        # 1. Tuval boyutunu çok büyük ayarla (Piksellenmeyi önlemek için)
        plt.figure(figsize=(24, 24)) 
        
        # 2. Düzen hesaplama 
        print("Dugum konumlari hesaplaniyor (spring_layout)...")
        try:
            pos = nx.spring_layout(self.G, seed=42, k=0.5, iterations=50, weight=None)
        except Exception as e:
             print(f"Spring layout hatasi, rastgele duzene geciliyor: {e}")
             pos = nx.random_layout(self.G, seed=42)

        print("Cizim elemanlari yerlestiriliyor...")
        
        # 3. Kenarları çiz (Aşırı şeffaf: alpha=0.03 ve ince)
        nx.draw_networkx_edges(self.G, pos, alpha=0.03, edge_color='darkgray', width=0.5, arrows=False)
        
        # 4. Düğümleri çiz (Küçük ve mavi)
        nx.draw_networkx_nodes(self.G, pos, node_size=15, node_color='steelblue', alpha=0.8)
        
        plt.title(f"Proje Ag Topolojisi\nNodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()}", fontsize=16)
        plt.axis('off') # Eksenleri gizle

        # 5. Dosyayı Yüksek Çözünürlükte (DPI=300) Kaydet
        print(f"Dosya diske yaziliyor: {output_filename} ...")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Belleği temizle
        plt.close() 
        
        end_time = time.time()
        print(f"--- ISLEM TAMAMLANDI ---")
        print(f"Gecen sure: {end_time - start_time:.2f} saniye.")
        print(f"Olusturulan dosya: {os.path.abspath(output_filename)}")

# --- MAIN KISMI ---
if __name__ == "__main__":
    # Dosya isimlerinin doğru olduğundan emin ol
    node_csv = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_csv = "BSM307_317_Guz2025_TermProject_EdgeData.csv"

    # 1. Modeli yükle
    proje = ProjeAgi(node_csv, edge_csv)
    
    # 2. GÖRÜNTÜYÜ PNG OLARAK KAYDET

    proje.save_network_image()
