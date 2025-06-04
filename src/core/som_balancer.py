"""
src/balancer/som_balancer.py
Self-Organizing Map based load balancer using MiniSom
"""
import numpy as np
import logging
from typing import List, Optional, Dict, Tuple
from .base_balancer import BaseLoadBalancer
from ..core.request import Request
from ..core.server import Server

try:
    from minisom import MiniSom
except ImportError:
    raise ImportError("MiniSom library is required. Install with: pip install minisom")

logger = logging.getLogger(__name__)

class SOMBalancer(BaseLoadBalancer):
    """Self-Organizing Map based load balancer using MiniSom library"""
    
    def __init__(self, servers: List[Server], som_size: int = 10, 
                 learning_rate: float = 0.5, sigma: float = 1.0, 
                 neighborhood_function: str = 'gaussian'):
        super().__init__(servers)
        
        # SOM parameters
        self.som_size = som_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        
        # Feature configuration: [cpu_util, memory_util, connection_ratio, response_time]
        self.n_features = 4
        
        # Initialize MiniSom
        self.som = MiniSom(
            x=som_size, 
            y=som_size, 
            input_len=self.n_features,
            learning_rate=learning_rate,
            sigma=sigma,
            neighborhood_function=neighborhood_function,
            random_seed=42  # For reproducible results
        )
        
        # Tracking for analysis and visualization
        self.server_neuron_map: Dict[str, Tuple[int, int]] = {}
        self.request_count = 0
        self.training_data = []
        self.training_interval = 100  # Retrain every N requests
        
        # Initialize with server data
        self._initial_training()
        
        logger.info(f"Initialized SOM Load Balancer with {som_size}x{som_size} grid using MiniSom")
    
    def get_algorithm_name(self) -> str:
        """Return the name of the algorithm"""
        return "Self-Organizing Map (SOM) - MiniSom"
    
    def _get_server_features(self, server: Server) -> np.ndarray:
        """Extract normalized feature vector from server state"""
        features = np.array([
            server.cpu_usage / 100.0,  # CPU utilization [0,1]
            server.memory_usage / 100.0,  # Memory utilization [0,1]
            server.active_connections / max(server.config['max_connections'], 1),  # Connection ratio [0,1]
            min(server.average_response_time / 1000.0, 1.0)  # Response time normalized [0,1]
        ])
        return np.clip(features, 0.0, 1.0)
    
    def _get_request_features(self, request: Request) -> np.ndarray:
        """Extract features from request characteristics"""
        # Simple mapping based on request properties
        size_factor = min(request.size_mb / 100.0, 1.0)  
        connection_factor = min(request.required_connections / 10.0, 1.0)
        
        # Estimated resource impact
        features = np.array([
            size_factor * 0.6,    # Estimated CPU impact
            size_factor * 0.8,    # Estimated memory impact  
            connection_factor,    # Connection requirement
            size_factor * 0.4     # Estimated response time impact
        ])
        return np.clip(features, 0.0, 1.0)
    
    def _initial_training(self):
        """Initial SOM training with current server states"""
        if not self.servers:
            return
            
        # Collect server features for initial training
        server_features = [self._get_server_features(server) for server in self.servers]
        
        # Need at least some data points for training
        if len(server_features) == 0:
            logger.warning("No servers available for initial training")
            return
            
        training_data = np.array(server_features)
        
        # Train SOM with server data
        self.som.train_random(training_data, num_iteration=100, verbose=False)
        logger.debug(f"Initial SOM training completed with {len(training_data)} servers")
    
    def select_server(self, request: Request) -> Optional[Server]:
        """Select server using SOM-based matching with MiniSom"""
        # Filter available servers
        available_servers = [
            s for s in self.get_healthy_servers() 
            if s.can_handle_request(request.size_mb, request.required_connections)
        ]
        
        if not available_servers:
            logger.warning(f"No server can handle request {request.id}")
            return None
        
        # Get request features
        request_features = self._get_request_features(request)
        
        # Find BMU for request using MiniSom
        request_bmu = self.som.winner(request_features)
        
        # Find best matching server based on distance to request BMU
        best_server = None
        min_score = float('inf')
        
        for server in available_servers:
            server_features = self._get_server_features(server)
            server_bmu = self.som.winner(server_features)
            
            # Update server-neuron mapping for visualization
            self.server_neuron_map[server.id] = server_bmu
            
            # Calculate distance in SOM grid space
            grid_distance = np.sqrt(
                (request_bmu[0] - server_bmu[0])**2 + 
                (request_bmu[1] - server_bmu[1])**2
            )
            
            # Combine grid distance with server load (lower is better)
            load_penalty = server.utilization_score  # 0-1 scale
            combined_score = grid_distance + 0.3 * load_penalty  # Tunable weight
            
            if combined_score < min_score:
                min_score = combined_score
                best_server = server
        
        # Periodic retraining with recent data
        self.request_count += 1
        if best_server:
            server_features = self._get_server_features(best_server)
            self.training_data.append(server_features)
            
            # Retrain periodically
            if self.request_count % self.training_interval == 0:
                self._retrain_som()
            
            logger.debug(f"SOM selected server {best_server.id} "
                        f"(BMU: {request_bmu}, score: {min_score:.3f})")
        
        return best_server
    
    def _retrain_som(self):
        """Retrain SOM with accumulated training data"""
        if len(self.training_data) > 10:  # Minimum data for retraining
            training_array = np.array(self.training_data[-200:])  # Use recent data
            self.som.train_batch(training_array, num_iteration=50, verbose=False)
            logger.debug(f"SOM retrained with {len(training_array)} samples")
    
    def get_som_state(self) -> Dict[str, Any]:
        """Get current SOM state for visualization"""
        activation_map = None
        if self.training_data:
            try:
                training_array = np.array(self.training_data)
                activation_map = self.som.activation_response(training_array)
            except Exception as e:
                logger.warning(f"Could not generate activation map: {e}")
        
        return {
            'weights': self.som.get_weights(),
            'activation_map': activation_map,
            'server_neuron_map': self.server_neuron_map.copy(),
            'som_size': self.som_size,
            'learning_rate': self.learning_rate,
            'sigma': self.sigma,
            'request_count': self.request_count
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get SOM-specific metrics"""
        metrics = super().get_metrics()
        
        som_metrics = {
            'som_algorithm': 'MiniSom',
            'som_size': self.som_size,
            'som_learning_rate': self.learning_rate,
            'som_sigma': self.sigma,
            'request_count': self.request_count,
            'training_data_size': len(self.training_data),
            'mapped_servers': len(self.server_neuron_map)
        }
        
        metrics.update(som_metrics)
        return metrics