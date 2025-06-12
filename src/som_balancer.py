import numpy as np
from minisom import MiniSom
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque, defaultdict
import time

@dataclass
class Request:
    id: int
    cpu_demand: float
    memory_demand: float
    arrival_time: float = field(default_factory=time.time)
    
    def get_features(self) -> np.ndarray:
        return np.array([self.cpu_demand, self.memory_demand])
    
    @property
    def size_category(self) -> str:
        total_demand = self.cpu_demand + self.memory_demand
        if total_demand < 0.4:
            return 'small'
        elif total_demand < 1.0:
            return 'medium'
        else:
            return 'large'

@dataclass
class Server:
    id: int
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    total_processing_time: float = 0.0
    handled_request_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def utilization(self) -> float:
        return (self.cpu_usage + self.memory_usage) / 2.0
    
    @property
    def avg_response_time(self) -> float:
        return self.total_processing_time / max(1, self.request_count)
    
    def can_handle(self, req: Request) -> bool:
        return (self.cpu_usage + req.cpu_demand <= 1.0 and
                self.memory_usage + req.memory_demand <= 1.0)
    
    def allocate(self, req: Request):
        self.cpu_usage += req.cpu_demand
        self.memory_usage += req.memory_demand
        self.request_count += 1
        self.handled_request_types[req.size_category] += 1

class SOMLoadBalancer:
    def __init__(self, 
                 servers: List[Server], 
                 som_size: int = 10,
                 initial_learning_rate: float = 0.9,
                 min_learning_rate: float = 0.01,
                 decay_constant: float = 1000,
                 sigma: float = 2.0,
                 retrain_interval: int = 100,
                 window_size: int = 200,
                 enable_specialization: bool = True):
        
        self.servers = servers
        self.som_size = som_size
        self.enable_specialization = enable_specialization
        
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_constant = decay_constant
        self.current_iteration = 0
        self.learning_rate_history = []
        
        self.retrain_interval = retrain_interval
        self.window_size = window_size
        self.training_data = deque(maxlen=window_size)
        
        current_lr = self._get_current_learning_rate()
        self.som = MiniSom(som_size, som_size, 2,
                          sigma=sigma,
                          learning_rate=current_lr,
                          neighborhood_function='gaussian',
                          random_seed=42)
        
        self.hits = {}
        self.request_count = 0
        self.allocation_history = []
        
        self.bmu_request_types = defaultdict(lambda: defaultdict(int))
        
        n_servers = len(servers)
        self.server_groups = {
            'small': list(range(0, n_servers // 3)),
            'medium': list(range(n_servers // 3, 2 * n_servers // 3)),
            'large': list(range(2 * n_servers // 3, n_servers))
        }
        
        self._initial_training()
    
    def _initial_training(self, n_samples: int = 100):
        data = []
        data.extend(np.random.beta(2, 5, (n_samples//3, 2)))
        data.extend(np.random.beta(5, 5, (n_samples//3, 2)))
        data.extend(np.random.beta(5, 2, (n_samples//3, 2)))
        
        data = np.array(data)
        self.som.train_random(data, 500)
        self.current_iteration += 500
    
    def _get_current_learning_rate(self) -> float:
        lr = self.initial_learning_rate * np.exp(-self.current_iteration / self.decay_constant)
        lr = max(lr, self.min_learning_rate)
        self.learning_rate_history.append((self.current_iteration, lr))
        return lr
    
    def select_server(self, request: Request) -> Optional[Server]:
        features = request.get_features()
        
        bmu = self.som.winner(features)
        self.hits[bmu] = self.hits.get(bmu, 0) + 1
        
        self.bmu_request_types[bmu][request.size_category] += 1
        
        server = self._map_bmu_to_server(bmu, request)
        
        if server:
            self.allocation_history.append({
                'request_id': request.id,
                'server_id': server.id,
                'bmu': bmu,
                'request_type': request.size_category,
                'timestamp': time.time()
            })
            
            self.training_data.append(features)
            self.request_count += 1
            
            if self.request_count % self.retrain_interval == 0:
                self._retrain()
        
        return server
    
    def _map_bmu_to_server(self, bmu: Tuple[int, int], request: Request) -> Optional[Server]:
        available = [s for s in self.servers if s.can_handle(request)]
        if not available:
            return None
        
        if self.enable_specialization:
            request_type = request.size_category
            preferred_servers = [s for s in available if s.id in self.server_groups[request_type]]
            candidates = preferred_servers if preferred_servers else available
        else:
            candidates = available
        
        scores = []
        for server in candidates:
            util_score = 1.0 - server.utilization
            
            specialization_score = 0.0
            if server.request_count > 0:
                same_type_ratio = server.handled_request_types[request.size_category] / server.request_count
                specialization_score = same_type_ratio * 0.2
            
            grid_x, grid_y = bmu
            if grid_x < self.som_size // 3:
                bmu_type = 'small'
            elif grid_x < 2 * self.som_size // 3:
                bmu_type = 'medium'
            else:
                bmu_type = 'large'
            
            bmu_match_score = 0.1 if server.id in self.server_groups[bmu_type] else 0.0
            
            total_score = util_score + specialization_score + bmu_match_score
            scores.append((total_score, server))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]
    
    def _retrain(self):
        if len(self.training_data) < 10:
            return
        
        current_lr = self._get_current_learning_rate()
        self.som.learning_rate = current_lr
        
        data = np.array(self.training_data)
        self.som.train_batch(data, num_iteration=50)
        self.current_iteration += 50
    
    def get_metrics(self) -> Dict:
        server_utils = [s.utilization for s in self.servers]
        
        specialization_stats = {}
        for category in ['small', 'medium', 'large']:
            servers_in_group = self.server_groups[category]
            total_requests = sum(self.servers[sid].request_count for sid in servers_in_group)
            category_requests = sum(self.servers[sid].handled_request_types[category] for sid in servers_in_group)
            specialization_stats[category] = {
                'total_handled': total_requests,
                'correct_type': category_requests,
                'specialization_rate': category_requests / max(1, total_requests) * 100
            }
        
        bmu_analysis = {}
        for bmu, types in self.bmu_request_types.items():
            dominant_type = max(types.items(), key=lambda x: x[1])[0] if types else 'unknown'
            bmu_analysis[str(bmu)] = {
                'dominant_type': dominant_type,
                'type_distribution': dict(types)
            }
        
        return {
            'request_count': self.request_count,
            'current_lr': self._get_current_learning_rate(),
            'active_neurons': len(self.hits),
            'total_neurons': self.som_size ** 2,
            'neuron_usage_rate': len(self.hits) / (self.som_size ** 2) * 100,
            'server_utilization': {
                'mean': np.mean(server_utils),
                'std': np.std(server_utils),
                'min': np.min(server_utils),
                'max': np.max(server_utils)
            },
            'load_balance_score': 1.0 - np.std(server_utils),
            'most_active_neuron': max(self.hits.items(), key=lambda x: x[1]) if self.hits else None,
            'specialization_stats': specialization_stats,
            'bmu_analysis': bmu_analysis
        }
    
    def get_weights(self) -> np.ndarray:
        return self.som.get_weights()
    
    def get_hits(self) -> Dict[Tuple[int, int], int]:
        return self.hits
    
    def get_learning_history(self) -> List[Tuple[int, float]]:
        return self.learning_rate_history.copy()
    
    def print_debug_info(self):
        print("\n=== SOM Load Balancer Debug Info ===")
        print(f"Total Requests: {self.request_count}")
        
        print("\n--- BMU to Request Type Mapping ---")
        for bmu, types in sorted(self.bmu_request_types.items())[:5]:
            print(f"BMU {bmu}: {dict(types)}")
        
        print("\n--- Server Specialization ---")
        for category, servers in self.server_groups.items():
            print(f"\n{category.upper()} Servers (IDs {servers[0]}-{servers[-1]}):")
            for sid in servers[:3]:
                server = self.servers[sid]
                if server.request_count > 0:
                    print(f"  Server {sid}: {dict(server.handled_request_types)}")
        
        print("\n--- Success Metrics ---")
        metrics = self.get_metrics()
        for category, stats in metrics['specialization_stats'].items():
            print(f"{category}: {stats['specialization_rate']:.1f}% specialization")