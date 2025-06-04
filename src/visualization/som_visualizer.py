"""
Efficient SOM Visualization using matplotlib + seaborn
Focus on heatmaps and neighborhood functions for Master Thesis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SOMVisualizer:
    """Efficient SOM visualizer for Master Thesis presentations"""
    
    def __init__(self, som_balancer):
        """Initialize with SOM balancer instance"""
        self.balancer = som_balancer
        self.som_size = som_balancer.som_size
        
        # Set efficient plotting style
        sns.set_theme(style="white")
        plt.rcParams['figure.dpi'] = 150
        
        logger.info(f"Initialized SOM Visualizer for {self.som_size}x{self.som_size} grid")
    
    def create_heatmap_analysis(self, output_dir: Optional[Path] = None, 
                               highlight_winner: Optional[Tuple[int, int]] = None) -> str:
        """
        Main heatmap visualization for thesis (what your professor wants!)
        """
        try:
            # Get SOM state efficiently
            som_state = self.balancer.get_som_state()
            weights = som_state['weights']
            server_map = som_state.get('server_neuron_map', {})
            
            # Create comprehensive figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('SOM Load Balancer: Heatmap Analysis', fontsize=16, fontweight='bold')
            
            # 1. U-Matrix (Distance Map) - Key for thesis!
            u_matrix = self._calculate_u_matrix(weights)
            sns.heatmap(u_matrix, annot=False, cmap='viridis', ax=axes[0, 0], 
                       cbar_kws={'label': 'Average Distance'})
            axes[0, 0].set_title('U-Matrix (Topology Map)\nDark = Similar Regions')
            self._add_winner_marker(axes[0, 0], highlight_winner)
            
            # 2. Feature Heatmaps - CPU
            sns.heatmap(weights[:, :, 0], annot=True, fmt='.2f', cmap='Reds', 
                       ax=axes[0, 1], cbar_kws={'label': 'CPU Feature'})
            axes[0, 1].set_title('CPU Utilization Feature')
            
            # 3. Feature Heatmaps - Memory  
            sns.heatmap(weights[:, :, 1], annot=True, fmt='.2f', cmap='Blues',
                       ax=axes[0, 2], cbar_kws={'label': 'Memory Feature'})
            axes[0, 2].set_title('Memory Utilization Feature')
            
            # 4. Neighborhood Function (Core SOM concept!)
            neighborhood = self._calculate_neighborhood_influence(highlight_winner)
            sns.heatmap(neighborhood, annot=True, fmt='.2f', cmap='hot', 
                       ax=axes[1, 0], cbar_kws={'label': 'Learning Influence'})
            axes[1, 0].set_title(f'Neighborhood Function (σ={self.balancer.sigma:.3f})')
            
            # 5. Server Assignment Map
            server_grid = self._create_server_assignment_grid(server_map)
            sns.heatmap(server_grid, annot=True, fmt='d', cmap='tab20', 
                       ax=axes[1, 1], cbar_kws={'label': 'Server ID'})
            axes[1, 1].set_title('Server-Neuron Assignments')
            
            # 6. Combined Feature Space
            combined = np.mean(weights, axis=2)  # Average all features
            sns.heatmap(combined, annot=True, fmt='.2f', cmap='plasma',
                       ax=axes[1, 2], cbar_kws={'label': 'Combined Features'})
            axes[1, 2].set_title('Combined Feature Space')
            self._add_winner_marker(axes[1, 2], highlight_winner)
            
            plt.tight_layout()
            
            # Save efficiently
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                filepath = output_dir / 'som_heatmap_analysis.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"SOM heatmap analysis saved to {filepath}")
                return str(filepath)
            else:
                plt.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Error creating SOM heatmap analysis: {e}")
            return None
    
    def create_neighborhood_visualization(self, winner_position: Tuple[int, int], 
                                        output_dir: Optional[Path] = None) -> str:
        """Focused neighborhood function visualization"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('SOM Neighborhood Function Analysis', fontsize=14, fontweight='bold')
            
            # 1. Neighborhood influence heatmap
            neighborhood = self._calculate_neighborhood_influence(winner_position)
            im1 = sns.heatmap(neighborhood, annot=True, fmt='.2f', cmap='hot', ax=axes[0])
            axes[0].set_title(f'Gaussian Neighborhood\n(σ={self.balancer.sigma:.3f})')
            
            # 2. 3D surface plot of neighborhood
            x, y = np.meshgrid(range(self.som_size), range(self.som_size))
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.plot_surface(x, y, neighborhood, cmap='hot', alpha=0.8)
            ax2.set_title('3D Neighborhood Surface')
            ax2.set_xlabel('SOM X')
            ax2.set_ylabel('SOM Y')
            ax2.set_zlabel('Influence')
            
            # 3. Cross-section plot
            center_x, center_y = winner_position
            cross_section = neighborhood[center_y, :]  # Horizontal slice
            axes[2].plot(range(self.som_size), cross_section, 'o-', linewidth=2, markersize=6)
            axes[2].axvline(center_x, color='red', linestyle='--', label='Winner Position')
            axes[2].set_xlabel('SOM X Position')
            axes[2].set_ylabel('Learning Influence')
            axes[2].set_title('Neighborhood Cross-Section')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
            
            plt.tight_layout()
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                filepath = output_dir / 'som_neighborhood_analysis.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Neighborhood visualization saved to {filepath}")
                return str(filepath)
            else:
                plt.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Error creating neighborhood visualization: {e}")
            return None
    
    def create_learning_evolution(self, metrics_history: list, 
                                output_dir: Optional[Path] = None) -> str:
        """Show SOM learning evolution efficiently"""
        try:
            if not metrics_history:
                return None
            
            # Extract data efficiently using pandas-like approach
            timestamps = [m.get('timestamp', 0) for m in metrics_history]
            learning_rates = [m.get('som_learning_rate', self.balancer.learning_rate) for m in metrics_history]
            sigmas = [m.get('som_sigma', self.balancer.sigma) for m in metrics_history]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('SOM Learning Evolution', fontsize=14, fontweight='bold')
            
            # 1. Parameter decay
            axes[0, 0].plot(timestamps, learning_rates, 'b-', linewidth=2, label='Learning Rate (η)')
            ax_twin = axes[0, 0].twinx()
            ax_twin.plot(timestamps, sigmas, 'r-', linewidth=2, label='Neighborhood (σ)')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Learning Rate', color='b')
            ax_twin.set_ylabel('Neighborhood Radius', color='r')
            axes[0, 0].set_title('Parameter Decay Over Time')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Performance metrics
            success_rates = [m.get('success_rate', 0) for m in metrics_history]
            axes[0, 1].plot(timestamps, success_rates, 'g-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_title('System Performance')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Load balance quality
            load_variances = []
            for m in metrics_history:
                servers = m.get('servers', [])
                if servers:
                    utils = [s.get('utilization_score', 0) for s in servers]
                    load_variances.append(np.var(utils))
                else:
                    load_variances.append(0)
            
            axes[1, 0].plot(timestamps, load_variances, 'orange', linewidth=2)
            axes[1, 0].fill_between(timestamps, 0, load_variances, alpha=0.3, color='orange')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Load Variance (Lower = Better)')
            axes[1, 0].set_title('Load Balance Quality')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Learning quality assessment
            axes[1, 1].axis('off')
            
            # Calculate final learning assessment
            final_lr = learning_rates[-1] if learning_rates else 0
            final_sigma = sigmas[-1] if sigmas else 0
            final_success = success_rates[-1] if success_rates else 0
            final_variance = load_variances[-1] if load_variances else 0
            
            assessment = self._assess_learning_quality(final_lr, final_sigma, final_success, final_variance)
            
            assessment_text = f"""
Learning Assessment:

Final Learning Rate: {final_lr:.4f}
Final Neighborhood: {final_sigma:.4f}
Success Rate: {final_success:.1f}%
Load Variance: {final_variance:.4f}

Overall Quality: {assessment}
            """
            axes[1, 1].text(0.1, 0.9, assessment_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Learning Quality Assessment')
            
            plt.tight_layout()
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                filepath = output_dir / 'som_learning_evolution.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Learning evolution plot saved to {filepath}")
                return str(filepath)
            else:
                plt.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Error creating learning evolution plot: {e}")
            return None
    
    def _calculate_u_matrix(self, weights: np.ndarray) -> np.ndarray:
        """Calculate U-Matrix efficiently using vectorized operations"""
        som_size = weights.shape[0]
        u_matrix = np.zeros((som_size, som_size))
        
        for i in range(som_size):
            for j in range(som_size):
                distances = []
                # Get neighbors efficiently
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < som_size and 0 <= nj < som_size and 
                            (di != 0 or dj != 0)):
                            dist = np.linalg.norm(weights[i, j] - weights[ni, nj])
                            distances.append(dist)
                
                u_matrix[i, j] = np.mean(distances) if distances else 0
        
        return u_matrix
    
    def _calculate_neighborhood_influence(self, winner_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        """Calculate Gaussian neighborhood function"""
        neighborhood = np.zeros((self.som_size, self.som_size))
        
        if winner_pos is None:
            # Use center as default
            winner_pos = (self.som_size // 2, self.som_size // 2)
        
        center_x, center_y = winner_pos
        
        for i in range(self.som_size):
            for j in range(self.som_size):
                # Calculate distance from winner
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                # Apply Gaussian function
                neighborhood[i, j] = np.exp(-dist**2 / (2 * self.balancer.sigma**2))
        
        return neighborhood
    
    def _create_server_assignment_grid(self, server_map: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """Create server assignment grid for heatmap"""
        grid = np.zeros((self.som_size, self.som_size))
        
        for i, (server_id, (x, y)) in enumerate(server_map.items()):
            if 0 <= x < self.som_size and 0 <= y < self.som_size:
                grid[y, x] = i + 1  # Server index (1-based for visualization)
        
        return grid
    
    def _add_winner_marker(self, ax, winner_pos: Optional[Tuple[int, int]]):
        """Add winner neuron marker to heatmap"""
        if winner_pos:
            ax.plot(winner_pos[0] + 0.5, winner_pos[1] + 0.5, 'w*', 
                   markersize=15, markeredgecolor='black', markeredgewidth=2)
    
    def _assess_learning_quality(self, lr: float, sigma: float, success: float, variance: float) -> str:
        """Simple learning quality assessment"""
        if success > 95 and variance < 0.01:
            return "Excellent"
        elif success > 90 and variance < 0.05:
            return "Good"
        elif success > 80 and variance < 0.1:
            return "Fair"
        else:
            return "Poor"