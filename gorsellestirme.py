import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi Oku
df = pd.read_csv("Benchmark_Sonuclari2.csv", sep=";")
sns.set_style("whitegrid")

# GRAFİK 1: Genel Ortalama Süre (Bar Plot)
plt.figure(figsize=(8, 6)) 
sns.barplot(data=df, x='Algorithm', y='Avg_Time',hue='Algorithm',legend=False, palette='viridis')
plt.title("Genel Ortalama Süre (Daha Düşük = Daha İyi)")
plt.ylabel("Saniye")
plt.tight_layout()
plt.savefig("1_Genel_Ortalama_Sure.png")
plt.close()
print("1. Grafik Kaydedildi.")

# GRAFİK 2: Genel Ortalama Maliyet (Bar Plot)
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Algorithm', y='Avg_Cost', hue='Algorithm',legend=False, palette='magma')
plt.title("Genel Ortalama Maliyet (Daha Düşük = Daha İyi)")
plt.ylabel("Maliyet")
plt.tight_layout()
plt.savefig("2_Genel_Ortalama_Maliyet.png")
plt.close()
print("2. Grafik Kaydedildi.")

# GRAFİK 3: Senaryo Bazlı Süre (Line Plot - OK İŞARETLİ)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Scenario_ID', y='Avg_Time', hue='Algorithm', marker='o')
plt.title("Senaryoya Göre Süre Değişimi")
plt.ylabel("Saniye")
plt.xticks(df['Scenario_ID'].unique()) 

plt.annotate('Q-Learning çok hızlı!\n(~0.1 sn)', 
            xy=(5, 0.1),             # Okun ucu
            xytext=(7, 1.5),         # Yazının yeri
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=11,
            color='black')


plt.tight_layout()
plt.savefig("3_Senaryo_Sure_Degisimi.png")
plt.close()
print("3. Grafik Kaydedildi.")

# GRAFİK 4: Senaryo Bazlı Maliyet (Line Plot)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Scenario_ID', y='Avg_Cost', hue='Algorithm', marker='o')
plt.title("Senaryoya Göre Maliyet Değişimi")
plt.ylabel("Maliyet")
plt.xticks(df['Scenario_ID'].unique())

plt.tight_layout()
plt.savefig("4_Senaryo_Maliyet_Degisimi.png")
plt.close()
print("4. Grafik Kaydedildi.")

print("\nTAMAMLANDI! Tüm grafikler ayrı dosyalar olarak oluşturuldu.")