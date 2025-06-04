"""
src/core/server.py
Server class representing a single server in the load balancing system
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from collections import deque
import numpy as np
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for a server instance"""
    cpu_cores: int = 8
    memory_gb: float = 32.0
    max_connections: int = 1000
    bandwidth_mbps: float = 1000.0
    location: str = "default"
    server_type: str = "general"  # general, cpu-optimized, memory-optimized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for compatibility"""
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'max_connections': self.max_connections,
            'bandwidth_mbps': self.bandwidth_mbps,
            'location': self.location,
            'server_type': self.server_type
        }


class Server:
    """Represents a single server in the load balancing cluster"""
    
    def __init__(self, server_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a server instance
        
        Args:
            server_id: Unique identifier for the server
            config: Server configuration as dict or ServerConfig
        """
        self.id = server_id
        
        # Handle both dict and ServerConfig input for compatibility
        if config is None:
            self.config = ServerConfig().to_dict()
        elif isinstance(config, dict):
            # Merge with defaults
            default_config = ServerConfig().to_dict()
            default_config.update(config)
            self.config = default_config
        elif isinstance(config, ServerConfig):
            self.config = config.to_dict()
        else:
            self.config = ServerConfig().to_dict()
        
        # Current metrics
        self.cpu_usage: float = np.random.uniform(10.0, 30.0)  # Start with some baseline load
        self.memory_usage: float = np.random.uniform(15.0, 35.0)
        self.active_connections: int = 0
        self.bandwidth_usage: float = 0.0  # Mbps
        
        # Performance metrics
        self.response_times: deque = deque(maxlen=100)  # Last 100 response times
        self.error_count: int = 0
        self.processed_requests: int = 0
        
        # State
        self.is_healthy: bool = True
        self.last_health_check: Optional[datetime] = None
        self.start_time: datetime = datetime.now()
        
        # Initialize with some baseline response times
        for _ in range(5):
            self.response_times.append(np.random.normal(100, 20))  # ~100ms baseline
        
        logger.debug(f"Initialized server {self.id} with config: {self.config}")
    
    def copy(self) -> 'Server':
        """Create a deep copy of the server for experiments"""
        new_server = Server(self.id, self.config.copy())
        new_server.cpu_usage = self.cpu_usage
        new_server.memory_usage = self.memory_usage
        new_server.active_connections = self.active_connections
        new_server.bandwidth_usage = self.bandwidth_usage
        new_server.response_times = copy.deepcopy(self.response_times)
        new_server.error_count = self.error_count
        new_server.processed_requests = self.processed_requests
        new_server.is_healthy = self.is_healthy
        return new_server
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate server uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in milliseconds"""
        if not self.response_times:
            return 100.0  # Default baseline
        return float(np.mean(self.response_times))
    
    @property
    def response_time_percentile_95(self) -> float:
        """Calculate 95th percentile response time"""
        if not self.response_times:
            return 150.0  # Default baseline
        return float(np.percentile(self.response_times, 95))
    
    @property
    def utilization_score(self) -> float:
        """
        Calculate overall utilization score (0-1)
        Weighted combination of different metrics
        """
        weights = {
            'cpu': 0.3,
            'memory': 0.3,
            'connections': 0.2,
            'bandwidth': 0.2
        }
        
        # Calculate normalized usage ratios
        connection_ratio = self.active_connections / max(self.config['max_connections'], 1)
        bandwidth_ratio = self.bandwidth_usage / max(self.config['bandwidth_mbps'], 1000)
        
        score = (
            weights['cpu'] * (self.cpu_usage / 100.0) +
            weights['memory'] * (self.memory_usage / 100.0) +
            weights['connections'] * connection_ratio +
            weights['bandwidth'] * bandwidth_ratio
        )
        
        return min(1.0, max(0.0, score))  # Ensure [0, 1] range
    
    def get_feature_vector(self) -> np.ndarray:
        """
        Get feature vector for SOM input
        
        Returns:
            Normalized feature vector [0, 1] range
        """
        features = np.array([
            self.cpu_usage / 100.0,  # CPU utilization
            self.memory_usage / 100.0,  # Memory utilization
            self.active_connections / max(self.config['max_connections'], 1),  # Connection ratio
            min(self.average_response_time / 1000.0, 1.0),  # Response time (capped at 1 second)
            self.bandwidth_usage / max(self.config['bandwidth_mbps'], 1000),  # Bandwidth ratio
            min(self.error_count / 100.0, 1.0),  # Error ratio (capped)
            self.utilization_score  # Overall utilization
        ])
        
        return np.clip(features, 0.0, 1.0)  # Ensure [0, 1] range
    
    def can_handle_request(self, request_size_mb: float, 
                          required_connections: int = 1) -> bool:
        """
        Check if server can handle a new request
        
        Args:
            request_size_mb: Size of request in MB
            required_connections: Number of connections needed
            
        Returns:
            True if server can handle the request
        """
        if not self.is_healthy:
            return False
            
        # Check CPU and memory limits (leave some headroom)
        if self.cpu_usage > 85 or self.memory_usage > 85:
            return False
            
        # Check connection capacity
        if self.active_connections + required_connections > self.config['max_connections']:
            return False
            
        # Check bandwidth capacity (use 80% as threshold)
        bandwidth_threshold = self.config['bandwidth_mbps'] * 0.8
        if self.bandwidth_usage + request_size_mb > bandwidth_threshold:
            return False
        
        return True
    
    def allocate_request(self, request_id: str, size_mb: float, 
                        connections: int = 1) -> Dict[str, Any]:
        """
        Allocate resources for a request
        
        Args:
            request_id: Unique request identifier
            size_mb: Size of request in MB
            connections: Number of connections to allocate
            
        Returns:
            Allocation result dictionary
        """
        if not self.can_handle_request(size_mb, connections):
            logger.debug(f"Server {self.id} cannot handle request {request_id}")
            return {
                'success': False,
                'reason': 'Insufficient resources',
                'request_id': request_id,
                'server_id': self.id
            }
        
        # Allocate resources
        self.active_connections += connections
        self.bandwidth_usage += size_mb
        
        # Simulate realistic CPU and memory impact based on request size
        base_cpu_impact = 2.0 + (size_mb / 10.0)  # Larger requests = more CPU
        base_memory_impact = 1.0 + (size_mb / 20.0)  # Larger requests = more memory
        
        # Add some variability
        cpu_impact = max(0.5, np.random.normal(base_cpu_impact, base_cpu_impact * 0.3))
        memory_impact = max(0.3, np.random.normal(base_memory_impact, base_memory_impact * 0.2))
        
        self.cpu_usage = min(100.0, self.cpu_usage + cpu_impact)
        self.memory_usage = min(100.0, self.memory_usage + memory_impact)
        
        logger.debug(f"Server {self.id} allocated request {request_id} "
                    f"(CPU: +{cpu_impact:.1f}%, Memory: +{memory_impact:.1f}%)")
        
        return {
            'success': True,
            'server_id': self.id,
            'request_id': request_id,
            'allocated_at': datetime.now(),
            'cpu_impact': cpu_impact,
            'memory_impact': memory_impact
        }
    
    def release_request(self, request_id: str, size_mb: float, 
                       connections: int = 1, response_time_ms: float = 0):
        """
        Release resources after request completion
        
        Args:
            request_id: Request identifier
            size_mb: Size of request in MB
            connections: Number of connections to release
            response_time_ms: Response time in milliseconds
        """
        # Release resources
        self.active_connections = max(0, self.active_connections - connections)
        self.bandwidth_usage = max(0.0, self.bandwidth_usage - size_mb)
        
        # Simulate resource release (gradual decrease)
        cpu_release = max(1.0, np.random.uniform(1.5, 4.0))
        memory_release = max(0.5, np.random.uniform(1.0, 3.0))
        
        self.cpu_usage = max(np.random.uniform(5.0, 15.0), self.cpu_usage - cpu_release)
        self.memory_usage = max(np.random.uniform(10.0, 20.0), self.memory_usage - memory_release)
        
        # Record performance metrics
        if response_time_ms > 0:
            self.response_times.append(response_time_ms)
        
        self.processed_requests += 1
        
        # Simulate occasional errors (very low probability)
        if np.random.random() < 0.001:  # 0.1% error rate
            self.error_count += 1
            
        logger.debug(f"Server {self.id} released request {request_id} "
                    f"(Response time: {response_time_ms:.1f}ms)")
    
    def health_check(self) -> bool:
        """
        Perform health check on the server
        
        Returns:
            True if server is healthy
        """
        self.last_health_check = datetime.now()
        
        # Check various health criteria
        checks = [
            self.cpu_usage < 95,
            self.memory_usage < 95,
            self.error_count < 100,  # Allow some errors
            self.average_response_time < 5000  # 5 seconds max
        ]
        
        old_health = self.is_healthy
        self.is_healthy = all(checks)
        
        if old_health != self.is_healthy:
            status = "healthy" if self.is_healthy else "unhealthy"
            logger.info(f"Server {self.id} health changed to {status}")
        
        return self.is_healthy
    
    def simulate_background_load(self):
        """Simulate background server activity"""
        # Small random fluctuations in CPU and memory
        self.cpu_usage += np.random.normal(0, 0.5)
        self.memory_usage += np.random.normal(0, 0.3)
        
        # Keep within reasonable bounds
        self.cpu_usage = np.clip(self.cpu_usage, 5.0, 100.0)
        self.memory_usage = np.clip(self.memory_usage, 10.0, 100.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current server metrics as dictionary"""
        return {
            'server_id': self.id,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'active_connections': self.active_connections,
            'max_connections': self.config['max_connections'],
            'bandwidth_usage': self.bandwidth_usage,
            'max_bandwidth': self.config['bandwidth_mbps'],
            'utilization_score': self.utilization_score,
            'average_response_time': self.average_response_time,
            'response_time_95p': self.response_time_percentile_95,
            'error_count': self.error_count,
            'processed_requests': self.processed_requests,
            'is_healthy': self.is_healthy,
            'uptime_seconds': self.uptime_seconds,
            'last_health_check': self.last_health_check
        }
    
    def __repr__(self) -> str:
        return (f"Server(id={self.id}, utilization={self.utilization_score:.2f}, "
                f"cpu={self.cpu_usage:.1f}%, memory={self.memory_usage:.1f}%, "
                f"connections={self.active_connections}/{self.config['max_connections']}, "
                f"healthy={self.is_healthy})")
    
    def __str__(self) -> str:
        return self.__repr__()