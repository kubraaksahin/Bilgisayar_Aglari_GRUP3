import networkx as nx
import math
# Bu sınıf gecikme, güvenilirlik ve kaynak maliyeti üzerinden rotaları puanlar.
class RouteEvaluator:
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def calculate_total_delay(self, path):  #Rotanın uçtan uca toplam gecikmesini hesaplar.
        total_delay = 0
        if not path: return float('inf') 
        
        source = path[0]
        destination = path[-1]

        for i in range(len(path) - 1):    #bağlantı gecikmelerini hesaplar.
            u, v = path[i], path[i+1]
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                total_delay += edge_data.get('delay_ms', 0)

        for node in path:                #düğüm gecikmelerini hesaplar.
            if node != source and node != destination:
                if node in self.graph.nodes:
                    node_data = self.graph.nodes[node]
                    total_delay += node_data.get('s_ms', 0) 
        #toplam gecikme=bağlantı gecikmeleri + ara düğüm işlem gecikmeleri
        return total_delay

    def calculate_reliability_cost(self, path):    #Rotanın güvenilirlik maliyetini hesaplar.
        total_reliability_cost = 0
        for i in range(len(path) - 1):    #Bağlantı güvenilirlik maliyetini hesaplar
            u, v = path[i], path[i+1]
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                r_link = edge_data.get('r_link', 0.99) 
                if r_link > 0:
                    total_reliability_cost += -math.log(r_link)
                else: return float('inf') 

        for node in path:    #Düğüm güvenirlik maliyetini hesaplar.
            if node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                r_node = node_data.get('r_node', 0.99) 
                if r_node > 0:
                    total_reliability_cost += -math.log(r_node)
                else: return float('inf')

        return total_reliability_cost

    def calculate_resource_cost(self, path):    #Rotanın ağ kaynak kullanım maliyetini hesaplar.
        total_resource_cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                bw = edge_data.get('capacity_mbps', 1) 
                if bw > 0:
                    total_resource_cost += 1000.0 / bw
                else: return float('inf')
        return total_resource_cost

    def calculate_total_fitness(self, path, w_delay=0.33, w_rel=0.33, w_res=0.34):     #Rotanın toplam maliyetini hesaplar. Algoritmaların amacı bu değeri minimize etmektir.
        if not path or len(path) < 2:
            return {'total_cost': float('inf')} 

        delay = self.calculate_total_delay(path)
        rel_cost = self.calculate_reliability_cost(path)
        res_cost = self.calculate_resource_cost(path)

        total_cost = (w_delay * delay) + (w_rel * rel_cost) + (w_res * res_cost)
        
        return {
            'total_cost': total_cost,
            'delay': delay,
            'reliability_cost': rel_cost,
            'resource_cost': res_cost
        }
