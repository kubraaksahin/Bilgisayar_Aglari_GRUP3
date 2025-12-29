import sys
import math
import time
import numpy as np
import networkx as nx

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFrame, QGridLayout,
    QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ---- Project Modules (sizde var) ----
from Graf_Model import ProjeAgi
from Metric_Evaluation import RouteEvaluator
from GenetikAlgoritma import GenetikAlgoritmaYolBulucu
from aco_router import AntColonyRouter
from q_agent import QLearningAgent

WIDTH = 1280
HEIGHT = 850

C = {
    'bg_app': "#0B1120", 'bg_panel': "#0f172a", 'bg_card': "#1e293b",
    'bg_lighter': "#334155", 'bg_dark': "#0f172a", 'text_main': "#f8fafc",
    'text_dim': "#94a3b8", 'accent': "#6366f1", 'blue': "#3b82f6", 'green': "#10b981",
    'red': "#ef4444", 'border': "#334155"
}

STYLESHEET = f"""
QMainWindow {{ background-color: {C['bg_app']}; }}
QWidget {{ font-family: 'Segoe UI', sans-serif; color: {C['text_main']}; font-size: 13px; }}
QMessageBox {{ background-color: #1e293b; color: white; }}
QMessageBox QLabel {{ color: #f8fafc; }}
QFrame#Sidebar {{ background-color: {C['bg_panel']}; border-right: 1px solid {C['border']}; }}
QLabel[class="SectionHeader"] {{ color: {C['text_dim']}; font-weight: bold; font-size: 12px; margin-top: 10px; margin-bottom: 5px; }}
QPushButton {{ background-color: {C['bg_app']}; border: 1px solid {C['border']}; color: {C['text_main']}; padding: 8px; border-radius: 4px; font-weight: 500; }}
QPushButton:hover {{ background-color: {C['bg_card']}; border-color: {C['blue']}; }}
QPushButton:pressed {{ background-color: {C['blue']}; color: white; }}
QPushButton#RunBtn {{ background-color: {C['blue']}; border: none; color: white; font-weight: bold; }}
QPushButton#RunBtn:hover {{ background-color: #2563eb; }}
QComboBox {{ background-color: {C['bg_app']}; border: 1px solid {C['border']}; padding: 6px; border-radius: 4px; }}
QComboBox QAbstractItemView {{ background-color: {C['bg_card']}; selection-background-color: {C['blue']}; }}
QSlider::groove:horizontal {{ height: 4px; background: {C['border']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C['blue']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
"""

# -----------------------------
# WORKER THREAD FOR ALGORITHMS
# -----------------------------
class AlgorithmWorker(QThread):
    finished = pyqtSignal(object, object, float)  # path, results, duration
    failed = pyqtSignal(str)

    def __init__(self, algo_type, graph, src, dst, demand, weights):
        super().__init__()
        self.algo_type = algo_type
        self.graph = graph
        self.src = src
        self.dst = dst
        self.demand = demand
        self.weights = weights
        self.evaluator = RouteEvaluator(graph)

    def run(self):
        start_time = time.time()
        path = []
        results = {}

        try:
            if self.algo_type == "Genetik Algoritma":
                solver = GenetikAlgoritmaYolBulucu(
                    self.graph, self.evaluator,
                    w_Delay=self.weights['delay'],
                    w_ReliabilityCost=self.weights['rel'],
                    w_Source=self.weights['res'],
                    populasyon_boyutu=50,
                    nesil_sayisi=30,
                    mutasyon_orani=0.1,
                    elit_sayisi=2,
                    seed=42
                )
                path, _meta = solver.cozum_bul(self.src, self.dst, self.demand)

            elif self.algo_type == "Karınca Kolonisi Optimizasyonu":
                router = AntColonyRouter(
                    self.graph, self.evaluator,
                    w_delay=self.weights['delay'],
                    w_rel=self.weights['rel'],
                    w_res=self.weights['res'],
                    num_ants=30,
                    num_iters=30,
                    alpha=0.8, beta=3.0, rho=0.1, q=0.5, tau0=2.0, max_steps=250, elitist=True
                )
                path, _meta = router.solve(self.src, self.dst, self.demand)

            elif self.algo_type == "Q-Learning":
                agent = QLearningAgent(
                    self.graph, self.src, self.dst,
                    alpha=0.1, gamma=0.9, epsilon=1.0
                )
                w_dict = {'w_delay': self.weights['delay'], 'w_rel': self.weights['rel'], 'w_res': self.weights['res']}
                agent.train(episodes=1000, weights=w_dict)
                path = agent.get_best_path()

            if path and path[-1] == self.dst:
                results = self.evaluator.calculate_total_fitness(
                    path, self.weights['delay'], self.weights['rel'], self.weights['res']
                )
            else:
                path = []

        except Exception as e:
            self.failed.emit(f"Algoritma hatası: {e}")
            return

        duration = time.time() - start_time
        self.finished.emit(path, results, duration)


# -----------------------------
# 3D CANVAS (OpenGL / GPU)
# -----------------------------
class Network3DCanvas(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # GL View
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#050810")
        lay.addWidget(self.view)

        # Optional grid (görsel referans)
        grid = gl.GLGridItem()

        grid.setSize(100, 100)
        grid.setSpacing(10, 10)
        grid.translate(0, 0, -10)
        #self.view.addItem(grid)

        # Data
        self.node_file = "BSM307_317_Guz2025_TermProject_NodeData.csv"
        self.edge_file = "BSM307_317_Guz2025_TermProject_EdgeData.csv"

        try:
            self.proje = ProjeAgi(self.node_file, self.edge_file)
            self.G = self.proje.get_graph()
            print(f"Graph loaded: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        except Exception as e:
            print(f"Error loading graph: {e}")
            self.G = nx.DiGraph()

        self.nodes_list = sorted(list(self.G.nodes()))
        self.node_index = {n: i for i, n in enumerate(self.nodes_list)}

        # Cached layout
        self.pos3d = None

        # GL Items
        self.edges_item = None
        self.nodes_item = None
        self.path_item = None

        # State
        self.current_path = []

        self.build_scene()

    def build_scene(self):
        # Clear previous GL items (keep grid)
        if self.edges_item is not None:
            self.view.removeItem(self.edges_item)
        if self.nodes_item is not None:
            self.view.removeItem(self.nodes_item)
        if self.path_item is not None:
            self.view.removeItem(self.path_item)
            self.path_item = None

        if len(self.G.nodes) == 0:
            return

        # Layout cache (compute once)
        if self.pos3d is None:
            # 2D spring, then lift to 3D with small z jitter
            pos2 = nx.spring_layout(self.G, seed=42, k=0.15, iterations=30)
            xs = np.array([pos2[n][0] for n in self.nodes_list], dtype=float)
            ys = np.array([pos2[n][1] for n in self.nodes_list], dtype=float)

            # Normalize scale
            xs = (xs - xs.mean()) / (xs.std() + 1e-9) * 25
            ys = (ys - ys.mean()) / (ys.std() + 1e-9) * 25
            zs = np.random.RandomState(42).normal(0, 1.5, size=len(self.nodes_list))

            self.pos3d = np.column_stack([xs, ys, zs]).astype(np.float32)

        # Nodes colors (RGBA)
        base = np.array([100/255, 116/255, 139/255, 1.0], dtype=np.float32)
        self.node_colors = np.repeat(base[None, :], len(self.nodes_list), axis=0)

        # Draw edges as ONE OpenGL item (mode='lines' => pairs are independent)
        E = self.G.number_of_edges()
        lines = np.empty((2 * E, 3), dtype=np.float32)

        i = 0
        for u, v in self.G.edges():
            if u in self.node_index and v in self.node_index:
                lines[i] = self.pos3d[self.node_index[u]]
                lines[i + 1] = self.pos3d[self.node_index[v]]
                i += 2

        lines = lines[:i]

        self.edges_item = gl.GLLinePlotItem(
            pos=lines,
            mode='lines',
            width=0.5,
            color=(100/255, 116/255, 139/255, 0.03),
            antialias=False
        )
        self.view.addItem(self.edges_item)

        # Draw nodes (one scatter item)
        self.nodes_item = gl.GLScatterPlotItem(
            pos=self.pos3d,
            size=3,
            color=self.node_colors,
            pxMode=True
        )
        self.view.addItem(self.nodes_item)

        # Camera
        self.view.opts['distance'] = 130
        self.view.opts['fov'] = 60


    def reset_view(self):
        self.current_path = []
        self.build_scene()

    def set_selected(self, src, dst):
        # reset base colors
        base = np.array([100/255, 116/255, 139/255, 1.0], dtype=np.float32)
        self.node_colors[:] = base

        if src in self.node_index:
            self.node_colors[self.node_index[src]] = np.array([16/255, 185/255, 129/255, 1.0], dtype=np.float32)
        if dst in self.node_index:
            self.node_colors[self.node_index[dst]] = np.array([239/255, 68/255, 68/255, 1.0], dtype=np.float32)

        if self.nodes_item is not None:
            self.nodes_item.setData(pos=self.pos3d, color=self.node_colors, size=6)

    def highlight_path(self, path):
        # Remove previous path
        if self.path_item is not None:
            self.view.removeItem(self.path_item)
            self.path_item = None

        self.current_path = path or []

        # Update node colors
        base = np.array([50/255, 65/255, 85/255, 0.6], dtype=np.float32)
        self.node_colors[:] = base

        if not path:
            if self.nodes_item is not None:
                self.nodes_item.setData(pos=self.pos3d, color=self.node_colors, size=3)
            return

        for i, n in enumerate(path):
            if n in self.node_index:
                idx= self.node_index[n]
                if i == 0:
                    self.node_colors[idx] = np.array([16/255, 185/255, 129/255, 1.0], dtype=np.float32)                
                elif i == len(path) - 1: # Hedef
                    self.node_colors[idx] = np.array([239/255, 68/255, 68/255, 1.0], dtype=np.float32)
                else: # Ara Yol
                    self.node_colors[idx] = np.array([59/255, 130/255, 246/255, 1.0], dtype=np.float32)
        
        self.nodes_item.setData(pos=self.pos3d, color=self.node_colors, size=4)
        # Path lines (fast)
        pts = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if u in self.node_index and v in self.node_index:
                pts.append(self.pos3d[self.node_index[u]])
                pts.append(self.pos3d[self.node_index[v]])

        if pts:
            pts = np.array(pts, dtype=np.float32)
            self.path_item = gl.GLLinePlotItem(
                pos=pts,
                mode='lines',
                width=2.5,
                color=(1.0, 0.0, 0.0, 1.0),
                antialias=True
            )
            self.view.addItem(self.path_item)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QoS Simulator - 3D (OpenGL)")
        self.resize(WIDTH, HEIGHT)
        self.setStyleSheet(STYLESHEET)

        cw = QWidget()
        self.setCentralWidget(cw)
        main = QHBoxLayout(cw)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # 3D Canvas
        self.canvas = Network3DCanvas()

        # Sidebar
        sidebar = QFrame(objectName="Sidebar")
        sidebar.setFixedWidth(320)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(20, 25, 20, 25)
        sb.setSpacing(15)

        btn_reset = QPushButton("Yenile / Reset")
        btn_reset.clicked.connect(self.canvas.reset_view)
        sb.addWidget(btn_reset)

        # Selection
        lbl_sel = QLabel("SEÇİM")
        lbl_sel.setProperty("class", "SectionHeader")
        sb.addWidget(lbl_sel)

        self.combos = {}
        nodes_list = self.canvas.nodes_list

        for lbl, key in [("Başlangıç Noktası", "s"), ("Bitiş Noktası", "d")]:
            sb.addWidget(QLabel(lbl))
            cb = QComboBox()
            cb.setEditable(True)
            cb.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
           
            for n in nodes_list:
                cb.addItem(str(n))
            cb.currentTextChanged.connect(self.update_selection_visuals)
            sb.addWidget(cb)
            self.combos[key] = cb

        # Optimization
        lbl_opt = QLabel("OPTİMİZASYON")
        lbl_opt.setProperty("class", "SectionHeader")
        sb.addWidget(lbl_opt)

        sb.addWidget(QLabel("Algoritma"))
        self.cb_algo = QComboBox()
        self.cb_algo.addItems(["Genetik Algoritma", "Karınca Kolonisi Optimizasyonu", "Q-Learning"])
        sb.addWidget(self.cb_algo)

        # Weights
        self.sl_delay, self.sl_rel, self.sl_res = None, None, None
        for txt, attr in [("Gecikme Ağırlığı", "delay"), ("Güvenilirlik Ağırlığı", "rel"), ("Kaynak Ağırlığı", "res")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(txt))
            row.addStretch()
            val = QLabel("0.33", styleSheet=f"color: {C['text_main']}; font-weight: bold;")
            row.addWidget(val)
            sb.addLayout(row)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(33)
            sl.valueChanged.connect(lambda v, l=val: l.setText(f"{v/100:.2f}"))
            sb.addWidget(sl)

            if attr == "delay":
                self.sl_delay = sl
            elif attr == "rel":
                self.sl_rel = sl
            else:
                self.sl_res = sl

        # Run
        self.btn_run = QPushButton("BAŞLAT", objectName="RunBtn")
        self.btn_run.clicked.connect(self.on_run_click)
        sb.addWidget(self.btn_run)
        sb.addStretch()

        main.addWidget(sidebar)

        # Right panel
        right_panel = QFrame()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.addWidget(self.canvas, stretch=1)

        # Metrics panel
        metrics_panel = QFrame()
        metrics_panel.setFixedHeight(140)
        metrics_panel.setStyleSheet(f"background: {C['bg_lighter']}; border-top: 1px solid {C['border']};")
        mp = QGridLayout(metrics_panel)
        mp.setContentsMargins(20, 15, 20, 15)

        self.metric_labels = {}
        self.create_metric_card(mp, "Toplam Gecikme", "---", C['blue'], 0, 0, "delay")
        self.create_metric_card(mp, "Toplam Güvenilirlik", "---", C['green'], 0, 1, "rel")
        self.create_metric_card(mp, "Çalışma Süresi", "---", "#e2e8f0", 0, 2, "time")
        self.create_metric_card(mp, "Kaynak Maliyeti", "---", C['red'], 1, 0, "res")
        self.create_metric_card(mp, "Toplam Maliyet", "---", "#38bdf8", 1, 1, "cost")

        # Path
        frame_path = QFrame()
        frame_path.setStyleSheet(f"background: {C['bg_dark']}; border-radius: 8px; border: 1px solid {C['border']};")
        fpl = QVBoxLayout(frame_path)
        fpl.setContentsMargins(15, 8, 15, 8)
        lbl_pt = QLabel("Yol Özeti")
        lbl_pt.setStyleSheet(f"color: {C['text_dim']}; font-size: 10px; font-weight: bold; border: none;")
        self.lbl_path_val = QLabel("---")
        self.lbl_path_val.setStyleSheet(f"color: {C['text_main']}; font-size: 10px; font-family: 'Consolas'; border: none;")
        fpl.addWidget(lbl_pt)
        fpl.addWidget(self.lbl_path_val)
        mp.addWidget(frame_path, 1, 2)

        rp.addWidget(metrics_panel)
        main.addWidget(right_panel)

    def create_metric_card(self, layout, title, value, color, r, c, key):
        frame = QFrame()
        frame.setStyleSheet(f"background: {C['bg_dark']}; border-radius: 8px; border: 1px solid {C['border']};")
        h = QHBoxLayout(frame)
        v = QVBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {C['text_dim']}; font-size: 10px; font-weight: bold; border: none;")
        val_lbl = QLabel(value)
        val_lbl.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold; border: none;")
        v.addWidget(title_lbl)
        v.addWidget(val_lbl)
        h.addLayout(v)
        layout.addWidget(frame, r, c)
        self.metric_labels[key] = val_lbl

    def update_selection_visuals(self):
        s_text = self.combos['s'].currentText()
        d_text = self.combos['d'].currentText()

        def parse_node(t):
            if not t or t == "--":
                return None
            try:
                return int(t)
            except:
                return None

        s = parse_node(s_text)
        d = parse_node(d_text)
        self.canvas.set_selected(s, d)

    def on_run_click(self):
        try:
            s_text = self.combos['s'].currentText()
            d_text = self.combos['d'].currentText()
            if s_text in ("", "--") or d_text in ("", "--"):
                QMessageBox.warning(self, "Eksik Seçim", "Lütfen Başlangıç ve Bitiş noktalarını seçin.")
                return

            s = int(s_text)
            d = int(d_text)

            if s == d:
                QMessageBox.warning(self, "Geçersiz Seçim", "Başlangıç ve Bitiş noktaları aynı olamaz.")
                return

            algo = self.cb_algo.currentText()

            w_d = self.sl_delay.value() / 100.0
            w_r = self.sl_rel.value() / 100.0
            w_s = self.sl_res.value() / 100.0
            total = w_d + w_r + w_s
            if total <= 0:
                total = 1.0
            weights = {'delay': w_d / total, 'rel': w_r / total, 'res': w_s / total}

            self.btn_run.setEnabled(False)
            self.btn_run.setText("HESAPLANIYOR...")

            self.worker = AlgorithmWorker(algo, self.canvas.G, s, d, 10.0, weights)
            self.worker.finished.connect(self.on_algorithm_finished)
            self.worker.failed.connect(self.on_algorithm_failed)
            self.worker.start()

        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Giriş Hatası: {e}")

    def on_algorithm_failed(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("BAŞLAT")
        QMessageBox.warning(self, "Algoritma Hatası", msg)

    def on_algorithm_finished(self, path, results, duration):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("BAŞLAT")

        # 3D highlight
        self.canvas.highlight_path(path)

        remember_path = path if path else []
        if remember_path:
            path_str = " -> ".join(map(str, remember_path))
            if len(path_str) > 45:
                path_str = path_str[:42] + "..."
            self.lbl_path_val.setText(path_str)
        else:
            self.lbl_path_val.setText("Yol Bulunamadı")

        # Metrics
        self.metric_labels['time'].setText(f"{duration:.3f} s")

        if results and remember_path:
            self.metric_labels['delay'].setText(f"{results['delay']:.2f} ms")
            self.metric_labels['res'].setText(f"{results['resource_cost']:.2f} U")
            self.metric_labels['cost'].setText(f"{results['total_cost']:.4f}")

            try:
                rel_pct = math.exp(-results['reliability_cost']) * 100
            except:
                rel_pct = 0.0
            self.metric_labels['rel'].setText(f"% {rel_pct:.2f}")
        else:
            self.metric_labels['delay'].setText("---")
            self.metric_labels['rel'].setText("---")
            self.metric_labels['res'].setText("---")
            self.metric_labels['cost'].setText("---")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
