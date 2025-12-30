"""
aco_router.py

Bu modül, kapasite kısıtı (capacity >= demand) altında kaynak-hedef arasında
çok kriterli (gecikme + güvenilirlik + kaynak) rota bulmak için
Karınca Kolonisi Optimizasyonu (Ant Colony Optimization - ACO) uygular.

Temel fikir:
  - Karıncalar, feromon (τ) ve sezgisel bilgi (η) ile olasılıksal olarak yol kurar.
  - Her iterasyon sonunda feromonlar buharlaşır (evaporasyon) ve iyi çözümler
    maliyetleriyle ters orantılı feromon bırakır (deposit).
"""

import random
import numpy as np

SEED = 42
_SEED_INITIALIZED = False


def algoritmayi_baslat() -> None:
    """Rastgelelik kaynaklarını sabit SEED ile bir kere sabitle."""
    global _SEED_INITIALIZED
    if _SEED_INITIALIZED:
        return
    random.seed(SEED)
    np.random.seed(SEED)
    _SEED_INITIALIZED = True
    print(f"Sistem {SEED} seed'i ile sabitlendi.")


# Modül import edildiğinde bir kere sabitle.
algoritmayi_baslat()

import math
import networkx as nx


class AntColonyRouter:
    """ACO tabanlı yönlendirme (routing) sınıfı.

    - graph: NetworkX grafı (kenar ve düğüm öznitelikleri CSV'den gelir)
    - evaluator: RouteEvaluator (toplam maliyet ve metrik hesapları)
    - w_delay / w_rel / w_res: çok kriterli hedef fonksiyonu ağırlıkları
    - num_ants / num_iters: bir çözüm aramasında denenen aday sayısı
    """

    def __init__(
        self,
        graph: nx.Graph,
        evaluator,
        w_delay: float = 0.33,
        w_rel: float = 0.33,
        w_res: float = 0.34,
        num_ants: int = 50,
        num_iters: int = 60,
        alpha: float = 0.8,
        beta: float = 3.0,
        rho: float = 0.1,
        q: float = 0.5,
        tau0: float = 2.0,
        max_steps: int = 450,
        elitist: bool = True,
    ):
        
        self.G = graph
        self.evaluator = evaluator

        self.w_delay = float(w_delay)
        self.w_rel = float(w_rel)
        self.w_res = float(w_res)

        self.num_ants = int(num_ants)
        self.num_iters = int(num_iters)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.tau0 = float(tau0)
        self.max_steps = int(max_steps)
        self.elitist = bool(elitist)

        # Seed DIŞARIDAN verilemez; her zaman SEED=42 kullanılır.
        self.rng = random.Random(SEED)

        # Başlangıç feromonu: tüm kenarlarda tau0.
        self.tau = {(u, v): self.tau0 for (u, v) in self.G.edges()}

    def _step_cost(self, u: int, v: int, dst: int) -> float:
        """u -> v adımı için yerel maliyeti hesapla.

        Bu değer, heuristic (η) için kullanılır: η = 1 / step_cost.

        - Delay: kenar gecikmesi + (ara düğüm ise) düğüm işlem gecikmesi
        - Reliability cost: -log(r_link) - log(r_node)
        - Resource cost: 1000 / capacity (kapasite arttıkça maliyet azalır)
        """
        ed = self.G.get_edge_data(u, v, default={})

        delay_ms = float(ed.get("delay_ms", 0.0))
        cap = float(ed.get("capacity_mbps", 1.0))
        r_link = float(ed.get("r_link", 0.99))

        v_data = self.G.nodes[v]
        proc = float(v_data.get("s_ms", 0.0)) if v != dst else 0.0
        r_node = float(v_data.get("r_node", 0.99))

        step_delay = delay_ms + proc

        # Güvenilirlik 0 veya negatifse log tanımsız; 
        if r_link <= 0 or r_node <= 0:
            return float("inf")

        step_rel_cost = (-math.log(r_link)) + (-math.log(r_node))
        step_res_cost = (1000.0 / cap) if cap > 0 else float("inf")

        return (self.w_delay * step_delay) + (self.w_rel * step_rel_cost) + (self.w_res * step_res_cost)

    def _feasible_neighbors(self, u: int, demand_mbps: float, visited: set[int]) -> list[int]:
        """Kapasite kısıtı (capacity >= demand) ve ziyaret seti ile uygun komşuları döndür."""
        if hasattr(self.G, "successors"):
            cand = list(self.G.successors(u))
        else:
            cand = list(self.G.neighbors(u))

        feasible: list[int] = []
        for v in cand:
            if v in visited:
                continue
            ed = self.G.get_edge_data(u, v, default={})
            cap = float(ed.get("capacity_mbps", 0.0))
            if cap >= demand_mbps:
                feasible.append(v)
        return feasible

    def _pick_next(self, u: int, candidates: list[int], dst: int) -> int:
        """ACO geçiş kuralına göre bir sonraki düğümü seç.

        P(u→v) ∝ τ(u,v)^α · η(u,v)^β  (roulette-wheel seçim).
        """
        weights: list[float] = []
        for v in candidates:
            # Feromon etkisi
            tau_uv = (self.tau.get((u, v), self.tau0) ** self.alpha)

            # Sezgisel bilgi (eta): yerel maliyetin tersi
            step_cost = self._step_cost(u, v, dst)
            eta_uv = 1.0 / (1e-9 + step_cost)

            weights.append(tau_uv * (eta_uv ** self.beta))

        s = sum(weights)
        if s <= 0 or math.isinf(s) or math.isnan(s):
            # Ağırlıklar bozuksa (örn inf/NaN), rastgele seçimle devam et.
            return self.rng.choice(candidates)

        # Roulette-wheel: [0, s) aralığında rasgele bir nokta seç
        r = self.rng.random() * s
        acc = 0.0
        for v, w in zip(candidates, weights):
            acc += w
            if acc >= r:
                return v
        return candidates[-1]

    def _construct_path(self, src: int, dst: int, demand_mbps: float) -> list[int] | None:
        """Tek bir karıncanın src→dst arasında yol kurma süreci."""
        if not self.G.has_node(src) or not self.G.has_node(dst):
            return None

        path = [src]
        visited = {src}
        cur = src

        for _ in range(self.max_steps):
            if cur == dst:
                return path

            candidates = self._feasible_neighbors(cur, demand_mbps, visited)
            if not candidates:
                return None

            nxt = self._pick_next(cur, candidates, dst)
            path.append(nxt)
            visited.add(nxt)
            cur = nxt

        return None

    def _evaporate(self) -> None:
        """Feromon buharlaşması: tüm kenarlarda τ ← (1-ρ)·τ."""
        for e in list(self.tau.keys()):
            self.tau[e] *= (1.0 - self.rho)
            if self.tau[e] < 1e-12:
                self.tau[e] = 1e-12  # sayısal stabilite için alt sınır

    def _deposit(self, path: list[int], total_cost: float) -> None:
        """Seçilen path üzerindeki kenarlara feromon ekle: Δτ = q / total_cost."""
        if not path or total_cost <= 0 or math.isinf(total_cost):
            return

        # Maliyet ne kadar düşükse o kadar çok feromon bırakılır
        delta = self.q / total_cost
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            self.tau[e] = self.tau.get(e, self.tau0) + delta

    def solve(self, src: int, dst: int, demand_mbps: float):
        """ACO ana döngüsü: en iyi yolu ve metrik detaylarını döndür."""
        best_path = None
        best_details = None
        best_cost = float("inf")

        for _it in range(self.num_iters):
            ant_solutions: list[tuple[list[int], float]] = []

            # Aynı iterasyonda num_ants karınca, mevcut feromon matrisiyle yol dener.
            for _a in range(self.num_ants):
                path = self._construct_path(src, dst, demand_mbps)
                if not path:
                    continue

                details = self.evaluator.calculate_total_fitness(
                    path,
                    w_delay=self.w_delay,
                    w_rel=self.w_rel,
                    w_res=self.w_res
                )
                cost = float(details.get("total_cost", float("inf")))

                ant_solutions.append((path, cost))

                # Global en iyi çözümü tut
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    best_details = details

            # 1) Evaporasyon (tüm kenarlarda)
            self._evaporate()

            # Hiç çözüm bulunamadıysa bu iterasyonda deposit yok
            if not ant_solutions:
                continue

            # 2) Deposit: elitist ise sadece iterasyonun en iyisi, değilse hepsi
            if self.elitist:
                path, cost = min(ant_solutions, key=lambda x: x[1])
                self._deposit(path, cost)
            else:
                for path, cost in ant_solutions:
                    self._deposit(path, cost)

        return best_path, best_details

