"""
Hexagonal SOM Visualizer for Load Balancing
Creates beautiful honeycomb-style heatmaps showing SOM neighborhoods clearly
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from pathlib import Path

class HexagonalSOMVisualizer:
    """Create hexagonal (honeycomb) visualizations for SOM analysis"""
    
    def __init__(self, som_balancer):
        self.som_balancer = som_balancer
        self.som_size = som_balancer.som_size
        try:
            self.weights = som_balancer.som.get_weights()
        except:
            # Fallback if weights not available
            self.weights = np.random.random((self.som_size[0], self.som_size[1], 4))
        
        # Colors for different server assignments
        self.server_colors = plt.cm.Set3(np.linspace(0, 1, len(som_balancer.servers)))
        
    def create_hexagonal_heatmap(self, output_dir="data/plots", 
                                title="SOM Load Balancing - Hexagonal View"):
        """Create hexagonal heatmap visualization"""
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Server Assignment Map
        self._plot_hexagonal_server_map(axes[0, 0])
        
        # 2. Server Load Distribution  
        self._plot_hexagonal_load_map(axes[0, 1])
        
        # 3. SOM Weight Features (CPU Usage)
        self._plot_hexagonal_feature_map(axes[1, 0], feature_idx=0, 
                                       feature_name="CPU Intensity")
        
        # 4. Distance Map (U-Matrix)
        self._plot_hexagonal_distance_map(axes[1, 1])
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = output_path / "som_hexagonal_analysis.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🐝 Hexagonal SOM visualization saved to: {filename}")
        return filename
        
    def _plot_hexagonal_server_map(self, ax):
        """Plot hexagonal map showing server assignments"""
        ax.set_title("🐝 Server Assignment Map", fontweight='bold')
        
        hexagons = []
        colors = []
        
        for i in range(self.som_size[0]):
            for j in range(self.som_size[1]):
                # Calculate hexagon position
                x, y = self._hex_position(i, j)
                
                # Create hexagon
                hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                         orientation=0, facecolor='white', 
                                         edgecolor='black', linewidth=0.8)
                hexagons.append(hex_patch)
                
                # Determine server assignment for this neuron
                server_id = self._get_neuron_server_assignment(i, j)
                if server_id is not None:
                    colors.append(self.server_colors[server_id])
                else:
                    colors.append('lightgray')
        
        # Add hexagons to plot
        collection = PatchCollection(hexagons, facecolors=colors, 
                                   edgecolors='black', linewidths=0.8)
        ax.add_collection(collection)
        
        # Add server labels
        self._add_server_labels(ax)
        
        # Set equal aspect and limits
        ax.set_xlim(-1, self.som_size[1])
        ax.set_ylim(-1, self.som_size[0] * 0.87)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def _plot_hexagonal_load_map(self, ax):
        """Plot hexagonal map showing load distribution"""
        ax.set_title("🔥 Load Distribution", fontweight='bold')
        
        # Get server loads
        server_loads = [len(server.current_requests) for server in self.som_balancer.servers]
        max_load = max(server_loads) if server_loads else 1
        
        hexagons = []
        colors = []
        
        for i in range(self.som_size[0]):
            for j in range(self.som_size[1]):
                x, y = self._hex_position(i, j)
                
                hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                         orientation=0, facecolor='white', 
                                         edgecolor='black', linewidth=0.8)
                hexagons.append(hex_patch)
                
                # Color by server load
                server_id = self._get_neuron_server_assignment(i, j)
                if server_id is not None and server_id < len(server_loads):
                    load_intensity = server_loads[server_id] / max_load if max_load > 0 else 0
                    colors.append(plt.cm.Reds(load_intensity))
                else:
                    colors.append('lightgray')
        
        collection = PatchCollection(hexagons, facecolors=colors, 
                                   edgecolors='black', linewidths=0.8)
        ax.add_collection(collection)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                 norm=plt.Normalize(vmin=0, vmax=max_load))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Server Load', rotation=270, labelpad=15)
        
        ax.set_xlim(-1, self.som_size[1])
        ax.set_ylim(-1, self.som_size[0] * 0.87)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def _plot_hexagonal_feature_map(self, ax, feature_idx=0, feature_name="Feature"):
        """Plot hexagonal map for specific feature"""
        ax.set_title(f"⚙️ {feature_name}", fontweight='bold')
        
        hexagons = []
        feature_values = []
        
        for i in range(self.som_size[0]):
            for j in range(self.som_size[1]):
                x, y = self._hex_position(i, j)
                
                hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                         orientation=0, facecolor='white', 
                                         edgecolor='black', linewidth=0.8)
                hexagons.append(hex_patch)
                
                # Get feature value for this neuron
                if feature_idx < self.weights.shape[2]:
                    feature_val = self.weights[i, j, feature_idx]
                    feature_values.append(feature_val)
                else:
                    feature_values.append(np.random.random())
        
        # Normalize feature values
        if feature_values:
            min_val, max_val = min(feature_values), max(feature_values)
            if max_val > min_val:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in feature_values]
            else:
                normalized_values = [0.5] * len(feature_values)
        else:
            normalized_values = [0.5] * len(hexagons)
        
        colors = [plt.cm.viridis(val) for val in normalized_values]
        
        collection = PatchCollection(hexagons, facecolors=colors, 
                                   edgecolors='black', linewidths=0.8)
        ax.add_collection(collection)
        
        ax.set_xlim(-1, self.som_size[1])
        ax.set_ylim(-1, self.som_size[0] * 0.87)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def _plot_hexagonal_distance_map(self, ax):
        """Plot U-Matrix showing distances between neighboring neurons"""
        ax.set_title("🌈 Distance Map (U-Matrix)", fontweight='bold')
        
        hexagons = []
        distances = []
        
        for i in range(self.som_size[0]):
            for j in range(self.som_size[1]):
                x, y = self._hex_position(i, j)
                
                hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                         orientation=0, facecolor='white', 
                                         edgecolor='black', linewidth=0.8)
                hexagons.append(hex_patch)
                
                # Calculate average distance to neighbors
                avg_distance = self._calculate_neighbor_distance(i, j)
                distances.append(avg_distance)
        
        # Normalize distances
        if distances:
            min_dist, max_dist = min(distances), max(distances)
            if max_dist > min_dist:
                normalized_distances = [(d - min_dist) / (max_dist - min_dist) for d in distances]
            else:
                normalized_distances = [0.5] * len(distances)
        else:
            normalized_distances = [0.5] * len(hexagons)
        
        colors = [plt.cm.plasma(val) for val in normalized_distances]
        
        collection = PatchCollection(hexagons, facecolors=colors, 
                                   edgecolors='black', linewidths=0.8)
        ax.add_collection(collection)
        
        ax.set_xlim(-1, self.som_size[1])
        ax.set_ylim(-1, self.som_size[0] * 0.87)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def _hex_position(self, i, j):
        """Calculate hexagonal position for grid coordinates"""
        # Hexagonal grid positioning
        x = j * 0.75
        y = i * np.sqrt(3) / 2
        
        # Offset every other row for hexagonal pattern
        if i % 2 == 1:
            x += 0.375
            
        return x, y
        
    def _get_neuron_server_assignment(self, i, j):
        """Get which server this neuron is assigned to"""
        try:
            # Simple mapping based on position
            server_count = len(self.som_balancer.servers)
            return (i * self.som_size[1] + j) % server_count
        except Exception:
            return 0
            
    def _add_server_labels(self, ax):
        """Add server labels to the plot"""
        try:
            # Add labels at strategic positions
            for server_idx in range(min(8, len(self.som_balancer.servers))):
                i = (server_idx // 4) * (self.som_size[0] // 2)
                j = (server_idx % 4) * (self.som_size[1] // 4)
                x, y = self._hex_position(i, j)
                ax.text(x, y, f'S{server_idx}', ha='center', va='center', 
                       fontweight='bold', fontsize=10, color='white',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
        except Exception:
            pass
            
    def _calculate_neighbor_distance(self, i, j):
        """Calculate average distance to hexagonal neighbors"""
        try:
            center_i, center_j = self.som_size[0] // 2, self.som_size[1] // 2
            distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            return distance / max(self.som_size)
        except Exception:
            return 0.5
            
    def create_neighborhood_visualization(self, center_neuron=None, output_dir="data/plots"):
        """Create detailed neighborhood visualization"""
        if center_neuron is None:
            center_neuron = (self.som_size[0] // 2, self.som_size[1] // 2)
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle(f"🐝 SOM Neighborhood Analysis - Center: {center_neuron}", 
                    fontsize=14, fontweight='bold')
        
        hexagons = []
        ci, cj = center_neuron
        
        for i in range(self.som_size[0]):
            for j in range(self.som_size[1]):
                x, y = self._hex_position(i, j)
                
                # Calculate distance from center
                distance = np.sqrt((i - ci)**2 + (j - cj)**2)
                
                # Create hexagon with different styles
                if (i, j) == center_neuron:
                    # Center neuron - special highlight
                    hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                             orientation=0, facecolor='red', 
                                             edgecolor='black', linewidth=2)
                elif distance <= 1.5:  # Direct neighbors
                    hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                             orientation=0, facecolor='orange', 
                                             edgecolor='black', linewidth=1.5)
                elif distance <= 2.5:  # Second-level neighbors
                    hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                             orientation=0, facecolor='yellow', 
                                             edgecolor='black', linewidth=1)
                else:  # Distant neurons
                    hex_patch = RegularPolygon((x, y), 6, radius=0.45, 
                                             orientation=0, facecolor='lightblue', 
                                             edgecolor='gray', linewidth=0.5)
                    
                hexagons.append(hex_patch)
        
        collection = PatchCollection(hexagons, match_original=True)
        ax.add_collection(collection)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Center Neuron'),
            plt.Rectangle((0, 0), 1, 1, facecolor='orange', label='Direct Neighbors'),
            plt.Rectangle((0, 0), 1, 1, facecolor='yellow', label='2nd Level Neighbors'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Distant Neurons')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        ax.set_xlim(-1, self.som_size[1])
        ax.set_ylim(-1, self.som_size[0] * 0.87)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = output_path / "som_hexagonal_neighborhood.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🐝 Hexagonal neighborhood visualization saved to: {filename}")
        return filename