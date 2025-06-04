"""
src/core/request.py
Request class representing client requests
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Represents a client request to be load balanced"""
   
    # Basic properties
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    size_mb: float = 1.0
    required_connections: int = 1
    expected_duration_ms: float = 1000.0
    priority: str = "normal"  # low, normal, high, critical
   
    # Request type and characteristics
    request_type: str = "http"  # http, websocket, streaming, batch
    cpu_intensity: float = 1.0  # Relative CPU requirement (0.1 - 5.0)
    memory_intensity: float = 1.0  # Relative memory requirement (0.1 - 5.0)
   
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
   
    # Assignment
    assigned_server_id: Optional[str] = None
   
    # Metadata
    client_id: Optional[str] = None
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize request properties after initialization"""
        # Ensure positive values
        self.size_mb = max(0.1, self.size_mb)
        self.required_connections = max(1, self.required_connections)
        self.expected_duration_ms = max(10.0, self.expected_duration_ms)
        
        # Normalize intensity values
        self.cpu_intensity = max(0.1, min(5.0, self.cpu_intensity))
        self.memory_intensity = max(0.1, min(5.0, self.memory_intensity))
        
        # Validate priority
        valid_priorities = ["low", "normal", "high", "critical"]
        if self.priority not in valid_priorities:
            logger.warning(f"Invalid priority '{self.priority}', defaulting to 'normal'")
            self.priority = "normal"
        
        # Validate request type
        valid_types = ["http", "websocket", "streaming", "batch", "api", "upload", "download"]
        if self.request_type not in valid_types:
            logger.warning(f"Invalid request_type '{self.request_type}', defaulting to 'http'")
            self.request_type = "http"
   
    @property
    def is_assigned(self) -> bool:
        """Check if request is assigned to a server"""
        return self.assigned_server_id is not None
   
    @property
    def is_completed(self) -> bool:
        """Check if request is completed"""
        return self.completed_at is not None
    
    @property
    def is_active(self) -> bool:
        """Check if request is currently being processed"""
        return self.is_assigned and not self.is_completed
   
    @property
    def response_time_ms(self) -> Optional[float]:
        """Calculate response time if completed"""
        if not self.is_completed or not self.assigned_at:
            return None
        return (self.completed_at - self.assigned_at).total_seconds() * 1000
   
    @property
    def waiting_time_ms(self) -> float:
        """Calculate time spent waiting for assignment"""
        end_time = self.assigned_at or datetime.now()
        return (end_time - self.created_at).total_seconds() * 1000
    
    @property
    def total_time_ms(self) -> float:
        """Calculate total time from creation to completion (or now)"""
        end_time = self.completed_at or datetime.now()
        return (end_time - self.created_at).total_seconds() * 1000
    
    @property
    def priority_weight(self) -> float:
        """Get numeric weight for priority (for SOM features)"""
        priority_weights = {
            "low": 0.25,
            "normal": 0.5,
            "high": 0.75,
            "critical": 1.0
        }
        return priority_weights.get(self.priority, 0.5)
    
    @property
    def estimated_resource_impact(self) -> float:
        """Calculate estimated total resource impact for SOM"""
        # Combine size, connections, and intensity factors
        base_impact = (self.size_mb / 10.0) + (self.required_connections / 5.0)
        intensity_factor = (self.cpu_intensity + self.memory_intensity) / 2.0
        
        return min(10.0, base_impact * intensity_factor)
   
    def get_feature_vector(self) -> list:
        """
        Get feature vector for SOM input (request characteristics)
        
        Returns:
            Normalized feature vector [0, 1] range
        """
        features = [
            min(self.size_mb / 100.0, 1.0),  # Size factor (normalized by 100MB)
            min(self.required_connections / 10.0, 1.0),  # Connection factor (normalized by 10)
            min(self.expected_duration_ms / 10000.0, 1.0),  # Duration factor (normalized by 10s)
            self.priority_weight,  # Priority weight [0, 1]
            min(self.cpu_intensity / 5.0, 1.0),  # CPU intensity [0, 1]
            min(self.memory_intensity / 5.0, 1.0),  # Memory intensity [0, 1]
            min(self.estimated_resource_impact / 10.0, 1.0)  # Overall impact [0, 1]
        ]
        
        return features
   
    def assign_to_server(self, server_id: str):
        """Assign request to a server"""
        if self.is_assigned:
            logger.warning(f"Request {self.id} is already assigned to {self.assigned_server_id}")
            return
            
        self.assigned_server_id = server_id
        self.assigned_at = datetime.now()
        
        logger.debug(f"Request {self.id} assigned to server {server_id}")
   
    def mark_completed(self, actual_response_time_ms: Optional[float] = None):
        """
        Mark request as completed
        
        Args:
            actual_response_time_ms: Override calculated response time
        """
        if not self.is_assigned:
            logger.warning(f"Cannot complete unassigned request {self.id}")
            return
            
        if self.is_completed:
            logger.warning(f"Request {self.id} is already completed")
            return
            
        self.completed_at = datetime.now()
        
        response_time = actual_response_time_ms or self.response_time_ms
        logger.debug(f"Request {self.id} completed in {response_time:.1f}ms")
    
    def copy(self) -> 'Request':
        """Create a copy of the request for experiments"""
        # Use dataclass fields to create clean copy
        new_request = Request(
            id=self.id + "_copy",  # Unique ID for copy
            size_mb=self.size_mb,
            required_connections=self.required_connections,
            expected_duration_ms=self.expected_duration_ms,
            priority=self.priority,
            request_type=self.request_type,
            cpu_intensity=self.cpu_intensity,
            memory_intensity=self.memory_intensity,
            client_id=self.client_id,
            region=self.region,
            metadata=copy.deepcopy(self.metadata)
        )
        return new_request
    
    def reset_assignment(self):
        """Reset assignment status (for experiments)"""
        self.assigned_server_id = None
        self.assigned_at = None
        self.completed_at = None
   
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return {
            'id': self.id,
            'size_mb': self.size_mb,
            'required_connections': self.required_connections,
            'expected_duration_ms': self.expected_duration_ms,
            'priority': self.priority,
            'priority_weight': self.priority_weight,
            'request_type': self.request_type,
            'cpu_intensity': self.cpu_intensity,
            'memory_intensity': self.memory_intensity,
            'estimated_resource_impact': self.estimated_resource_impact,
            'created_at': self.created_at.isoformat(),
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'assigned_server_id': self.assigned_server_id,
            'response_time_ms': self.response_time_ms,
            'waiting_time_ms': self.waiting_time_ms,
            'total_time_ms': self.total_time_ms,
            'is_assigned': self.is_assigned,
            'is_completed': self.is_completed,
            'is_active': self.is_active,
            'client_id': self.client_id,
            'region': self.region,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Request':
        """Create Request from dictionary"""
        # Extract datetime fields
        created_at = datetime.fromisoformat(data['created_at'])
        assigned_at = datetime.fromisoformat(data['assigned_at']) if data.get('assigned_at') else None
        completed_at = datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None
        
        request = cls(
            id=data['id'],
            size_mb=data['size_mb'],
            required_connections=data['required_connections'],
            expected_duration_ms=data['expected_duration_ms'],
            priority=data['priority'],
            request_type=data['request_type'],
            cpu_intensity=data.get('cpu_intensity', 1.0),
            memory_intensity=data.get('memory_intensity', 1.0),
            created_at=created_at,
            assigned_server_id=data.get('assigned_server_id'),
            client_id=data.get('client_id'),
            region=data.get('region'),
            metadata=data.get('metadata', {})
        )
        
        # Set timestamps
        request.assigned_at = assigned_at
        request.completed_at = completed_at
        
        return request
    
    def __repr__(self) -> str:
        status = "completed" if self.is_completed else "active" if self.is_assigned else "pending"
        return (f"Request(id={self.id[:8]}..., size={self.size_mb}MB, "
                f"connections={self.required_connections}, priority={self.priority}, "
                f"status={status})")
    
    def __str__(self) -> str:
        return self.__repr__()