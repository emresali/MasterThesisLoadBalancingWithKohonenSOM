"""
Efficient heatmap utilities using seaborn and matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HeatmapUtils:
    """Utility functions for efficient heatmap creation"""
    
    @staticmethod
    def create_correlation_heatmap(data: np.ndarray, labels: list, 
                                 title: str = "Correlation Matrix") -> plt.Figure:
        """Create correlation heatmap using seaborn"""
        try:
            # Calculate correlation matrix efficiently
            corr_matrix = np.corrcoef(data.T)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use seaborn for efficient heatmap
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       fmt='.2f',
                       xticklabels=labels,
                       yticklabels=labels,
                       cmap='coolwarm',
                       center=0,
                       square=True,
                       ax=ax)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return None
    
    @staticmethod
    def create_distance_matrix_heatmap(weights: np.ndarray, 
                                     title: str = "U-Matrix") -> plt.Figure:
        """Create efficient distance matrix heatmap"""
        try:
            som_height, som_width, n_features = weights.shape
            
            # Calculate distance matrix efficiently using broadcasting
            u_matrix = np.zeros((som_height, som_width))
            
            for i in range(som_height):
                for j in range(som_width):
                    # Get neighbor indices
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < som_height and 0 <= nj < som_width and 
                                (di != 0 or dj != 0)):
                                neighbors.append((ni, nj))
                    
                    # Calculate average distance to neighbors
                    if neighbors:
                        distances = [np.linalg.norm(weights[i, j] - weights[ni, nj]) 
                                   for ni, nj in neighbors]
                        u_matrix[i, j] = np.mean(distances)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(u_matrix, 
                       cmap='viridis',
                       annot=False,
                       square=True,
                       cbar_kws={'label': 'Average Distance to Neighbors'},
                       ax=ax)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('SOM X')
            ax.set_ylabel('SOM Y')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distance matrix heatmap: {e}")
            return None
    
    @staticmethod
    def create_feature_heatmaps(weights: np.ndarray, feature_names: list) -> plt.Figure:
        """Create multiple feature heatmaps efficiently"""
        try:
            n_features = len(feature_names)
            cols = min(4, n_features)
            rows = (n_features + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Color palettes for different features
            colormaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys']
            
            for i, feature_name in enumerate(feature_names):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                cmap = colormaps[i % len(colormaps)]
                
                # Create heatmap for this feature
                sns.heatmap(weights[:, :, i],
                           annot=True,
                           fmt='.2f',
                           cmap=cmap,
                           square=True,
                           cbar_kws={'label': f'{feature_name} Value'},
                           ax=ax)
                
                ax.set_title(f'{feature_name} Feature Map', fontweight='bold')
                ax.set_xlabel('SOM X')
                ax.set_ylabel('SOM Y')
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature heatmaps: {e}")
            return None
    
    @staticmethod
    def create_interactive_heatmap(data: np.ndarray, 
                                 title: str = "Interactive Heatmap",
                                 highlight_cells: Optional[list] = None) -> plt.Figure:
        """Create interactive-style heatmap with highlights"""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create base heatmap
            im = sns.heatmap(data,
                           annot=True,
                           fmt='.3f',
                           cmap='viridis',
                           square=True,
                           linewidths=0.5,
                           ax=ax)
            
            # Add highlights if specified
            if highlight_cells:
                for cell in highlight_cells:
                    if len(cell) >= 2:
                        x, y = cell[0], cell[1]
                        color = cell[2] if len(cell) > 2 else 'red'
                        
                        # Add colored rectangle
                        rect = plt.Rectangle((x, y), 1, 1, 
                                           fill=False, 
                                           edgecolor=color, 
                                           linewidth=3)
                        ax.add_patch(rect)
                        
                        # Add marker
                        ax.plot(x + 0.5, y + 0.5, '*', 
                               color=color, markersize=15, 
                               markeredgecolor='white', markeredgewidth=1)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive heatmap: {e}")
            return None


class SOMMetrics:
    """Efficient SOM quality metrics calculation"""
    
    @staticmethod
    def calculate_quantization_error(som_weights: np.ndarray, 
                                   training_data: np.ndarray) -> float:
        """Calculate quantization error efficiently"""
        try:
            if len(training_data) == 0:
                return 0.0
            
            som_height, som_width, n_features = som_weights.shape
            total_error = 0.0
            
            for data_point in training_data:
                # Find BMU efficiently
                min_distance = float('inf')
                
                for i in range(som_height):
                    for j in range(som_width):
                        distance = np.linalg.norm(data_point - som_weights[i, j])
                        if distance < min_distance:
                            min_distance = distance
                
                total_error += min_distance
            
            return total_error / len(training_data)
            
        except Exception as e:
            logger.error(f"Error calculating quantization error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_topographic_error(som_weights: np.ndarray, 
                                  training_data: np.ndarray) -> float:
        """Calculate topographic error efficiently"""
        try:
            if len(training_data) < 2:
                return 0.0
            
            som_height, som_width, n_features = som_weights.shape
            topology_errors = 0
            
            # Sample data for efficiency
            sample_size = min(100, len(training_data))
            sample_indices = np.random.choice(len(training_data), sample_size, replace=False)
            
            for idx in sample_indices:
                data_point = training_data[idx]
                
                # Find two best matching units
                distances = []
                positions = []
                
                for i in range(som_height):
                    for j in range(som_width):
                        distance = np.linalg.norm(data_point - som_weights[i, j])
                        distances.append(distance)
                        positions.append((i, j))
                
                # Get two smallest distances
                sorted_indices = np.argsort(distances)
                bmu1_pos = positions[sorted_indices[0]]
                bmu2_pos = positions[sorted_indices[1]]
                
                # Check if BMUs are neighbors
                distance = np.sqrt((bmu1_pos[0] - bmu2_pos[0])**2 + 
                                 (bmu1_pos[1] - bmu2_pos[1])**2)
                
                if distance > 1.5:  # Not direct neighbors
                    topology_errors += 1
            
            return topology_errors / sample_size
            
        except Exception as e:
            logger.error(f"Error calculating topographic error: {e}")
            return 0.0