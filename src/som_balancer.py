import numpy as np
from minisom import MiniSom
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import defaultdict, deque
import time

@dataclass
class Request:
    id: int
    cpu_demand: float
    memory_demand: float
    processing_time: float  # Expected processing time in ms
    arrival_time: float = field(default_factory=time.time)
    
    def get_features(self) -> np.ndarray:
        return np.array([self.cpu_demand, self.memory_demand])

@dataclass 
class VirtualMachine:
    id: int
    som_position: Tuple[int, int]
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Queue and response time tracking
    request_queue: deque = field(default_factory=deque)
    processing_requests: List[Request] = field(default_factory=list)
    total_response_time: float = 0.0
    completed_requests: int = 0
    
    # Neighbor awareness
    activation_level: float = 0.0  # How active this VM is due to neighbor load
    neighbor_load_factor: float = 0.0  # Influence from overloaded neighbors
    
    @property
    def utilization(self) -> float:
        return (self.cpu_usage + self.memory_usage) / 2.0
    
    @property
    def queue_length(self) -> int:
        return len(self.request_queue)
    
    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / max(1, self.completed_requests)
    
    def is_overloaded(self) -> bool:
        return self.utilization > 0.8 or self.queue_length > 5
    
    def can_accept(self, req: Request) -> bool:
        # Consider both resources and queue length
        future_cpu = self.cpu_usage + req.cpu_demand
        future_mem = self.memory_usage + req.memory_demand
        return future_cpu <= 1.0 and future_mem <= 1.0 and self.queue_length < 10

class SOMLoadBalancer:
    def __init__(self, 
                 som_size: int = 8,
                 initial_learning_rate: float = 0.9,
                 sigma: float = 2.0,
                 neighbor_influence: float = 0.3,
                 load_spread_factor: float = 0.5):
        
        self.som_size = som_size
        self.neighbor_influence = neighbor_influence
        self.load_spread_factor = load_spread_factor
        
        # Initialize SOM
        self.som = MiniSom(som_size, som_size, 2,
                          sigma=sigma,
                          learning_rate=initial_learning_rate,
                          neighborhood_function='gaussian',
                          random_seed=42)
        
        # Create VMs
        self.vms = {}
        vm_id = 0
        for i in range(som_size):
            for j in range(som_size):
                self.vms[(i, j)] = VirtualMachine(
                    id=vm_id,
                    som_position=(i, j)
                )
                vm_id += 1
        
        self.time_step = 0
        self.load_history = []
        self.response_time_history = []
        self.load_shift_events = []
        
        self._initial_training()
    
    def _initial_training(self):
        data = np.random.rand(100, 2)
        self.som.train_random(data, 500)
    
    def _get_neighbors(self, pos: Tuple[int, int], radius: int = 1) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.som_size and 0 <= ny < self.som_size:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _update_neighbor_activation(self):
        """Update activation levels based on neighbor overload"""
        new_activations = {}
        
        for pos, vm in self.vms.items():
            # Check neighbor load
            neighbors = self._get_neighbors(pos)
            overloaded_neighbors = 0
            total_neighbor_load = 0
            
            for n_pos in neighbors:
                neighbor = self.vms[n_pos]
                if neighbor.is_overloaded():
                    overloaded_neighbors += 1
                total_neighbor_load += neighbor.utilization
            
            # Calculate activation based on neighbor stress
            if neighbors:
                avg_neighbor_load = total_neighbor_load / len(neighbors)
                vm.neighbor_load_factor = avg_neighbor_load
                
                # Activation increases with overloaded neighbors
                activation_increase = (overloaded_neighbors / len(neighbors)) * self.neighbor_influence
                vm.activation_level = min(1.0, vm.activation_level * 0.9 + activation_increase)
            
            new_activations[pos] = vm.activation_level
        
        return new_activations
    
    def select_vm(self, request: Request) -> Optional[VirtualMachine]:
        features = request.get_features()
        bmu_pos = self.som.winner(features)
        bmu_vm = self.vms[bmu_pos]
        
        # Update neighbor activations
        self._update_neighbor_activation()
        
        # If BMU is overloaded, actively spread to neighbors
        if bmu_vm.is_overloaded():
            # Get neighbors sorted by activation level and current load
            neighbors = self._get_neighbors(bmu_pos, radius=2)
            candidates = []
            
            for n_pos in neighbors:
                neighbor_vm = self.vms[n_pos]
                if neighbor_vm.can_accept(request):
                    # Score based on activation level and current load
                    score = (1 - neighbor_vm.utilization) + neighbor_vm.activation_level * self.load_spread_factor
                    candidates.append((score, neighbor_vm, n_pos))
            
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                selected_vm = candidates[0][1]
                
                # Record load shift event
                self.load_shift_events.append({
                    'time': self.time_step,
                    'from_pos': bmu_pos,
                    'to_pos': candidates[0][2],
                    'reason': 'overload_spreading'
                })
                
                return selected_vm
        
        # Normal selection if BMU not overloaded
        elif bmu_vm.can_accept(request):
            return bmu_vm
        
        # Fallback: search in expanding radius
        for radius in range(1, self.som_size // 2):
            neighbors = self._get_neighbors(bmu_pos, radius)
            
            best_vm = None
            best_score = -1
            
            for n_pos in neighbors:
                neighbor_vm = self.vms[n_pos]
                if neighbor_vm.can_accept(request):
                    score = 1 - neighbor_vm.utilization
                    if score > best_score:
                        best_score = score
                        best_vm = neighbor_vm
            
            if best_vm:
                return best_vm
        
        return None
    
    def process_request(self, request: Request) -> Dict:
        start_time = time.time()
        
        vm = self.select_vm(request)
        
        if vm:
            # Add to queue
            vm.request_queue.append(request)
            
            # Simulate processing
            vm.cpu_usage += request.cpu_demand
            vm.memory_usage += request.memory_demand
            
            # Calculate response time (queue wait + processing)
            queue_wait = vm.queue_length * 10  # 10ms per queued request
            response_time = queue_wait + request.processing_time
            
            vm.total_response_time += response_time
            vm.completed_requests += 1
            
            return {
                'success': True,
                'vm_id': vm.id,
                'vm_pos': vm.som_position,
                'response_time': response_time,
                'queue_length': vm.queue_length,
                'utilization': vm.utilization
            }
        
        return {'success': False}
    
    def simulate_processing(self):
        """Simulate request completion and resource release"""
        for vm in self.vms.values():
            if vm.request_queue:
                # Process some requests
                processed = min(3, len(vm.request_queue))
                for _ in range(processed):
                    if vm.request_queue:
                        req = vm.request_queue.popleft()
                        # Release resources
                        vm.cpu_usage = max(0, vm.cpu_usage - req.cpu_demand * 0.8)
                        vm.memory_usage = max(0, vm.memory_usage - req.memory_demand * 0.8)
    
    def record_state(self):
        """Record current system state for visualization"""
        vm_states = []
        
        for pos, vm in self.vms.items():
            vm_states.append({
                'pos': pos,
                'utilization': vm.utilization,
                'queue_length': vm.queue_length,
                'activation_level': vm.activation_level,
                'avg_response_time': vm.avg_response_time,
                'is_overloaded': vm.is_overloaded()
            })
        
        self.load_history.append({
            'time_step': self.time_step,
            'vm_states': vm_states,
            'avg_utilization': np.mean([s['utilization'] for s in vm_states]),
            'max_queue_length': max([s['queue_length'] for s in vm_states]),
            'overloaded_vms': sum([s['is_overloaded'] for s in vm_states])
        })
        
        # Response time statistics
        response_times = [vm.avg_response_time for vm in self.vms.values() if vm.completed_requests > 0]
        if response_times:
            self.response_time_history.append({
                'time_step': self.time_step,
                'avg_response_time': np.mean(response_times),
                'p95_response_time': np.percentile(response_times, 95),
                'max_response_time': np.max(response_times)
            })
    
    def step(self):
        """Advance simulation by one time step"""
        self.simulate_processing()
        self.record_state()
        self.time_step += 1
    
    def get_load_shift_matrix(self) -> np.ndarray:
        """Get matrix showing load shifts between VMs"""
        n = self.som_size * self.som_size
        matrix = np.zeros((n, n))
        
        for event in self.load_shift_events:
            from_vm = self.vms[event['from_pos']]
            to_vm = self.vms[event['to_pos']]
            matrix[from_vm.id, to_vm.id] += 1
        
        return matrix
    
    def get_metrics(self) -> Dict:
        """Get current system metrics"""
        vm_utils = [vm.utilization for vm in self.vms.values()]
        
        # Communication/shift analysis
        total_shifts = len(self.load_shift_events)
        recent_shifts = len([e for e in self.load_shift_events 
                           if e['time'] >= self.time_step - 10])
        
        return {
            'time_step': self.time_step,
            'total_vms': len(self.vms),
            'vm_utilization': {
                'mean': np.mean(vm_utils),
                'std': np.std(vm_utils),
                'min': np.min(vm_utils),
                'max': np.max(vm_utils)
            },
            'load_balance_score': 1.0 - np.std(vm_utils),
            'overloaded_vms': sum(1 for vm in self.vms.values() if vm.is_overloaded()),
            'avg_activation_level': np.mean([vm.activation_level for vm in self.vms.values()]),
            'load_shifts': {
                'total': total_shifts,
                'recent': recent_shifts,
                'rate': recent_shifts / 10 if self.time_step >= 10 else 0
            },
            'response_times': {
                'avg': np.mean([vm.avg_response_time for vm in self.vms.values() if vm.completed_requests > 0]) if any(vm.completed_requests > 0 for vm in self.vms.values()) else 0,
                'max_queue': max(vm.queue_length for vm in self.vms.values())
            }
        }