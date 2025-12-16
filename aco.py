# aco.py

import math
import random
from typing import List, Optional, Tuple

import networkx as nx

from Metric_Evaluation import evaluate_path
from config import ACO_N_ANTS, ACO_N_ITERATIONS


class AntColonyRouting:
    def __init__(
        self,
        G: nx.Graph,
        source: int,
        destination: int,
        demand_B: float,
        n_ants: int = ACO_N_ANTS,
        n_iterations: int = ACO_N_ITERATIONS,
        alpha: float = 1.0,          # feromonun önemi
        beta: float = 2.0,           # sezgisel bilginin önemi
        rho: float = 0.5,            # buharlaşma katsayısı
        q: float = 1.0,              # feromon güncelleme katsayısı
        max_steps_factor: float = 3, # max adım = factor * |V|
        seed: Optional[int] = None,
    ) -> None:
        self.G = G
        self.source = int(source)
        self.destination = int(destination)
        self.demand_B = float(demand_B)

        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.max_steps = int(max_steps_factor * max(1, self.G.number_of_nodes()))

        self.rng = random.Random(seed)

        # Feromon tablosu (her kenar için)
        self.pheromone = {}
        self._initialize_pheromones()

        # En iyi global çözüm
        self.best_path: Optional[List[int]] = None
        self.best_cost: float = float("inf")

    # -------------------------------------------------------------------------
    # Pheromone & heuristic
    # -------------------------------------------------------------------------

    def _edge_key(self, u: int, v: int) -> Tuple[int, int]:
        """Undirected graph için canonical edge key."""
        return (u, v) if u <= v else (v, u)

    def _initialize_pheromones(self) -> None:
        """Tüm kenarlara başlangıç feromon değeri ata."""
        tau0 = 1.0
        for u, v in self.G.edges():
            self.pheromone[self._edge_key(u, v)] = tau0

    def _get_pheromone(self, u: int, v: int) -> float:
        return self.pheromone.get(self._edge_key(u, v), 1.0)

    def _heuristic(self, u: int, v: int) -> float:
        """
        Yerel sezgisel bilgi (eta).
        Basit bir tanım: 1 / (edge_delay + 1).
        """
        edge_data = self.G.get_edge_data(u, v, {})
        delay = float(edge_data.get("delay_ms", 1.0))
        # Delay küçüldükçe eta artsın
        return 1.0 / (delay + 1.0)

    # -------------------------------------------------------------------------
    # Cost hesaplama (metrics.evaluate_path ile)
    # -------------------------------------------------------------------------

    def _path_cost(self, path: Optional[List[int]]) -> float:
        """
        Verilen path için TotalCost'u hesaplar.
        Path geçersiz / None ise inf döner.
        """

        if not path or len(path) < 2:
            return float("inf")

        metrics = evaluate_path(
            self.G,
            path,
            self.demand_B,
        )

        if not metrics.get("Feasible", False):
            return float("inf")

        return float(metrics["TotalCost"])

    # -------------------------------------------------------------------------
    # Çözüm üretimi (her karınca için path)
    # -------------------------------------------------------------------------

    def _build_path_for_ant(self) -> Optional[List[int]]:
        """
        Bir karıncanın source'tan destination'a kadar
        olası bir path oluşturmasını simüle eder.
        """

        current = self.source
        path = [current]
        visited = {current}
        steps = 0

        while current != self.destination and steps < self.max_steps:
            neighbors = [
                v for v in self.G.neighbors(current)
                if v not in visited
            ]

            if not neighbors:
                # Çıkış yok, karınca "sıkıştı"
                return None

            # Olasılık hesabı
            probabilities = []
            total_score = 0.0
            for v in neighbors:
                tau = self._get_pheromone(current, v) ** self.alpha
                eta = self._heuristic(current, v) ** self.beta
                score = tau * eta
                probabilities.append((v, score))
                total_score += score

            if total_score <= 0.0:
                # Çok uç durumda, tamamen rastgele seçim
                next_node = self.rng.choice(neighbors)
            else:
                # Rulet tekeri seçimi
                r = self.rng.random() * total_score
                cumulative = 0.0
                next_node = neighbors[-1]
                for v, score in probabilities:
                    cumulative += score
                    if r <= cumulative:
                        next_node = v
                        break

            path.append(next_node)
            visited.add(next_node)
            current = next_node
            steps += 1

        if current != self.destination:
            # Maks adım sayısına ulaştı ama hedefe varamadı
            return None

        return path

    # -------------------------------------------------------------------------
    # Pheromone güncelleme ve ana ACO döngüsü
    # -------------------------------------------------------------------------

    def _evaporate_pheromones(self) -> None:
        """Tüm kenarlarda feromon buharlaştır."""
        for key in list(self.pheromone.keys()):
            self.pheromone[key] *= (1.0 - self.rho)

    def _deposit_pheromones(self, path: List[int], cost: float) -> None:
        """En iyi path üzerinde feromon arttır."""
        if not path or cost <= 0.0 or math.isinf(cost):
            return
        delta = self.q / cost
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            key = self._edge_key(u, v)
            self.pheromone[key] = self.pheromone.get(key, 0.0) + delta

    def run(self) -> Tuple[Optional[List[int]], float]:
        """
        Ana ACO döngüsü:
        - Her iterasyonda n_ants adet karınca path üretir.
        - Her iterasyonda en iyi path'e göre feromon güncellenir.
        - Global en iyi path ve cost döndürülür.
        """
        for _ in range(self.n_iterations):
            iteration_best_path: Optional[List[int]] = None
            iteration_best_cost: float = float("inf")

            # Karıncaların path üretmesi
            for _ant in range(self.n_ants):
                path = self._build_path_for_ant()
                cost = self._path_cost(path)

                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_path = path

            # Eğer bu iterasyonda kullanılabilir bir path bulunamadıysa skip
            if iteration_best_path is None or math.isinf(iteration_best_cost):
                continue

            # Global en iyi çözümü güncelle
            if iteration_best_cost < self.best_cost:
                self.best_cost = iteration_best_cost
                self.best_path = iteration_best_path

            # Feromon güncellemesi
            self._evaporate_pheromones()
            self._deposit_pheromones(iteration_best_path, iteration_best_cost)

        return self.best_path, self.best_cost


def run_aco(
    G: nx.Graph,
    source: int,
    destination: int,
    demand_B: float,
    n_ants: Optional[int] = None,
    n_iterations: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Optional[List[int]], float]:
    """
    Dışarıdan kullanılacak basit wrapper.
    Ağırlıklar metrics/config tarafında ayarlı; burada sadece
    ACO parametreleri (karınca, iterasyon, seed) kontrol ediliyor.
    """

    if n_ants is None:
        n_ants = ACO_N_ANTS
    if n_iterations is None:
        n_iterations = ACO_N_ITERATIONS

    aco = AntColonyRouting(
        G=G,
        source=source,
        destination=destination,
        demand_B=demand_B,
        n_ants=n_ants,
        n_iterations=n_iterations,
        seed=seed,
        **kwargs,
    )
    return aco.run()
