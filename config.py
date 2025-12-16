# config.

# Hocanın dosyaları
TEACHER_NODE_FILE = "BSM307_317_Guz2025_TermProject_NodeData.csv"
TEACHER_EDGE_FILE = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
TEACHER_DEMAND_FILE = "BSM307_317_Guz2025_TermProject_DemandData.csv"


# -----------------------------
# 1) Ortak ağırlık seti (8 tane farklı ağırlık senaryosundan Tüm algoritmalar için seçilen Ağırlıklar)
# -----------------------------
W_DELAY = 0.04
W_RELIABILITY = 0.95
W_RESOURCE = 0.01

# -----------------------------
# 2) ACO Hiperparametre seçimi
# -----------------------------
ACO_N_ANTS = 80       
ACO_N_ITERATIONS = 80
