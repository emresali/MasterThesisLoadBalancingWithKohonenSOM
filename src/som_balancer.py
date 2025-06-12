"""
Simplified SOM Load Balancer with Adaptive Learning Rate
"""
import numpy as np
from minisom import MiniSom
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

@dataclass
class Request:
    id: int
    cpu_demand: float  # 0-1
    memory_demand: float  # 0-1
   
    def get_features(self) -> np.ndarray:
        return np.array([self.cpu_demand, self.memory_demand])

@dataclass
class Server:
    id: int
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
   
    @property
    def utilization(self) -> float:
        return (self.cpu_usage + self.memory_usage) / 2.0
   
    def can_handle(self, req: Request) -> bool:
        return (self.cpu_usage + req.cpu_demand <= 1.0 and
                self.memory_usage + req.memory_demand <= 1.0)
   
    def allocate(self, req: Request):
        self.cpu_usage += req.cpu_demand
        self.memory_usage += req.memory_demand

class SOMLoadBalancer:
    def __init__(self, servers: List[Server], som_size=10, 
                 # NEU: Adaptive Learning Parameter
                 initial_learning_rate=0.9,
                 min_learning_rate=0.01, 
                 decay_constant=1000,
                 sigma=2.0):
        self.servers = servers
        
        # NEU: Adaptive Learning Rate Parameter
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_constant = decay_constant
        self.current_iteration = 0
        self.learning_rate_history = []
        
        # Initialize SOM with initial learning rate
        current_lr = self._get_current_learning_rate()
        self.som = MiniSom(som_size, som_size, 2, 
                          sigma=sigma,
                          learning_rate=current_lr,  # NEU: Adaptive Rate
                          random_seed=42)
        
        self.hits = {}  # Track neuron activations
        self.request_count = 0  # NEU: Für periodisches Retraining
        self.training_data = []  # NEU: Sammle Daten für Retraining
       
        # Initial training
        data = np.random.rand(100, 2)
        self.som.train_random(data, 500)
        self.current_iteration += 500  # NEU: Update iteration counter
   
    def _get_current_learning_rate(self) -> float:
        """NEU: Berechnet adaptive Lernrate"""
        # Exponentieller Decay: α(t) = α₀ * exp(-t/λ)
        lr = self.initial_learning_rate * np.exp(-self.current_iteration / self.decay_constant)
        lr = max(lr, self.min_learning_rate)
        
        # Speichere für Monitoring
        self.learning_rate_history.append((self.current_iteration, lr))
        return lr
    
    def select_server(self, request: Request) -> Optional[Server]:
        features = request.get_features()
        bmu = self.som.winner(features)
       
        # Track hits
        self.hits[bmu] = self.hits.get(bmu, 0) + 1
       
        # Find best available server
        available = [s for s in self.servers if s.can_handle(request)]
        if not available:
            return None
        
        # NEU: Sammle Training Data
        self.training_data.append(features)
        self.request_count += 1
        
        # NEU: Periodisches Retraining mit adaptiver Rate
        if self.request_count % 100 == 0:  # Alle 100 Requests
            self._retrain()
           
        # Simple selection: least utilized
        return min(available, key=lambda s: s.utilization)
    
    def _retrain(self):
        """NEU: Retrain mit adaptiver Lernrate"""
        if len(self.training_data) > 10:
            # Update learning rate
            current_lr = self._get_current_learning_rate()
            self.som.learning_rate = current_lr
            
            # Retrain mit letzten Daten
            data = np.array(self.training_data[-200:])  # Letzte 200 Samples
            self.som.train_batch(data, num_iteration=50)
            self.current_iteration += 50
   
    def get_weights(self) -> np.ndarray:
        return self.som.get_weights()
   
    def get_hits(self) -> Dict[Tuple[int, int], int]:
        return self.hits
    
    def get_current_lr(self) -> float:
        """NEU: Aktuelle Lernrate abrufen"""
        return self._get_current_learning_rate()
    
    def get_learning_history(self) -> List[Tuple[int, float]]:
        """NEU: Learning Rate History für externe Visualisierung"""
        return self.learning_rate_history.copy()