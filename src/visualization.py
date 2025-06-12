import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import RegularPolygon
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class SOMVisualizer:
    def __init__(self, save_dir: str = "data/plots", dpi: int = 300):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
    def create_comprehensive_analysis(self, balancer, filename: str = "som_analysis.png"):
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        ax_hex = fig.add_subplot(gs[0:2, 0:2])
        self._plot_hexagonal_som(ax_hex, balancer)
        
        ax_lr = fig.add_subplot(gs[0, 2])
        self._plot_learning_rate(ax_lr, balancer)
        
        ax_util = fig.add_subplot(gs[1, 2])
        self._plot_server_utilization(ax_util, balancer)
        
        ax_neuron = fig.add_subplot(gs[2, 0])
        self._plot_neuron_distribution(ax_neuron, balancer)
        
        ax_balance = fig.add_subplot(gs[2, 1])
        self._plot_load_balance(ax_balance, balancer)
        
        ax_metrics = fig.add_subplot(gs[2, 2])
        self._plot_key_metrics(ax_metrics, balancer)
        
        save_path = self.save_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_hexagonal_som(self, ax, balancer):
        size = balancer.som_size
        weights = balancer.get_weights()
        hits = balancer.get_hits()
        
        u_matrix = self._calculate_umatrix(weights)
        
        for i in range(size):
            for j in range(size):
                x = j * 1.5
                y = i * np.sqrt(3)
                if i % 2 == 1:
                    x += 0.75
                
                val = u_matrix[i, j]
                normalized_val = (val - u_matrix.min()) / (u_matrix.max() - u_matrix.min() + 1e-8)
                
                hit_count = hits.get((i, j), 0)
                radius = 0.4 + 0.5 * min(hit_count / max(hits.values() or [1]), 1.0)
                
                hex = RegularPolygon((x, y), 6, radius=radius,
                                   facecolor=plt.cm.viridis(normalized_val),
                                   edgecolor='white', linewidth=0.5)
                ax.add_patch(hex)
                
                if hit_count > 0:
                    ax.text(x, y, str(hit_count), ha='center', va='center',
                           fontsize=6, color='white' if normalized_val > 0.5 else 'black')
        
        ax.set_xlim(-1, size * 1.5)
        ax.set_ylim(-1, size * np.sqrt(3))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('SOM Topology (U-Matrix + Hits)', fontsize=10, fontweight='bold')
    
    def _plot_learning_rate(self, ax, balancer):
        history = balancer.get_learning_history()
        if not history:
            return
        
        iterations, rates = zip(*history)
        ax.semilogy(iterations, rates, 'b-', linewidth=2)
        ax.axhline(y=balancer.min_learning_rate, color='r', linestyle='--', 
                   linewidth=1, alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=8)
        ax.set_ylabel('Learning Rate', fontsize=8)
        ax.set_title('Adaptive α(t)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)
    
    def _plot_server_utilization(self, ax, balancer):
        utils = [s.utilization * 100 for s in balancer.servers]
        server_ids = [s.id for s in balancer.servers]
        
        bars = ax.bar(server_ids, utils, color=sns.color_palette("husl", len(utils)))
        
        mean_util = np.mean(utils)
        ax.axhline(y=mean_util, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Server ID', fontsize=8)
        ax.set_ylabel('Utilization (%)', fontsize=8)
        ax.set_title('Server Load Distribution', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', labelsize=7)
        
        ax.text(0.95, 0.95, f'σ = {np.std(utils):.1f}%', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)
    
    def _plot_neuron_distribution(self, ax, balancer):
        hits_values = list(balancer.get_hits().values())
        if not hits_values:
            return
        
        ax.hist(hits_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Hit Count', fontsize=8)
        ax.set_ylabel('# Neurons', fontsize=8)
        ax.set_title('Neuron Activity', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', labelsize=7)
    
    def _plot_load_balance(self, ax, balancer):
        metrics = balancer.get_metrics()
        utils = [s.utilization for s in balancer.servers]
        
        # Load distribution over time would be more useful
        ax.boxplot(utils, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        ax.set_ylabel('Server Utilization', fontsize=8)
        ax.set_title('Load Distribution', fontsize=10, fontweight='bold')
        ax.set_xticklabels(['All Servers'])
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        ax.text(0.95, 0.95, f'Balance Score: {metrics["load_balance_score"]:.3f}\nStd Dev: {np.std(utils):.3f}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=7)
    
    def _plot_key_metrics(self, ax, balancer):
        metrics = balancer.get_metrics()
        
        ax.axis('off')
        text = f"""Performance Metrics:
        
Requests: {metrics['request_count']}
Active Neurons: {metrics['active_neurons']}/{metrics['total_neurons']}
Neuron Usage: {metrics['neuron_usage_rate']:.1f}%
Load Balance: {metrics['load_balance_score']:.3f}
Current α: {metrics['current_lr']:.4f}
        """
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _calculate_umatrix(self, weights):
        size = weights.shape[0]
        u_matrix = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                neighbors = []
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size:
                        dist = np.linalg.norm(weights[i,j] - weights[ni,nj])
                        neighbors.append(dist)
                u_matrix[i,j] = np.mean(neighbors) if neighbors else 0
        
        return u_matrix

    
    def create_comparison_matrix(self, results: Dict[str, Dict], filename: str = "comparison_matrix.png"):
        configs = list(results.keys())
        metrics = ['success_rate', 'load_std', 'neuron_usage', 'response_time']
        
        matrix = np.zeros((len(configs), len(metrics)))
        for i, config in enumerate(configs):
            for j, metric in enumerate(metrics):
                if metric in results[config]:
                    matrix[i, j] = results[config][metric]
        
        for j in range(len(metrics)):
            col = matrix[:, j]
            if col.max() > col.min():
                matrix[:, j] = (col - col.min()) / (col.max() - col.min())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=metrics, yticklabels=configs,
                    cbar_kws={'label': 'Normalized Score'})
        
        ax.set_title('Configuration Performance Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

def create_configuration_comparison(results: Dict[str, Dict[str, Any]], configs: List[Dict[str, Any]]):
    save_dir = Path("data/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SOM Configuration Comparison', fontsize=16, fontweight='bold')
    
    workloads = ['uniform', 'bursty', 'mixed']
    config_names = [c['name'] for c in configs]
    colors = ['blue', 'red', 'green', 'orange'][:len(configs)]
    
    def plot_bars(ax, metric, ylabel, title):
        x = np.arange(len(workloads))
        width = 0.8 / len(configs)
        
        for i, name in enumerate(config_names):
            values = [results[name][w][metric] for w in workloads]
            ax.bar(x + i*width, values, width, label=name, color=colors[i], alpha=0.7)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + width*(len(configs)-1)/2)
        ax.set_xticklabels(workloads)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plot_bars(axes[0, 0], 'success_rate', 'Success Rate (%)', 'Success Rate by Workload')
    
    ax = axes[0, 1]
    for i, name in enumerate(config_names):
        values = [results[name][w]['load_std'] for w in workloads]
        ax.plot(workloads, values, 'o-', label=name, color=colors[i], linewidth=2, markersize=8)
    ax.set_ylabel('Load STD')
    ax.set_title('Load Balance Quality (Lower is Better)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plot_bars(axes[0, 2], 'neuron_usage', 'Neuron Usage (%)', 'SOM Neuron Utilization')
    
    for idx, (workload, ylabel) in enumerate([('uniform', 'Load STD'), ('bursty', 'Average Load')]):
        ax = axes[1, idx]
        for i, name in enumerate(config_names):
            history = results[name][workload]['history']
            steps = [h['step'] for h in history]
            values = [h['std'] if idx == 0 else h['mean'] for h in history]
            ax.plot(steps, values, label=name, color=colors[i], linewidth=2)
        ax.set_xlabel('Request Number')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Load Evolution - {workload.capitalize()}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    ax.axis('off')
    
    scores = {}
    for name in config_names:
        score = sum(
            results[name][w]['success_rate'] * 0.4 + 
            results[name][w]['load_balance_score'] * 100 * 0.4 + 
            results[name][w]['neuron_usage'] * 0.2
            for w in workloads
        ) / len(workloads)
        scores[name] = score
    
    summary = "Configuration Summary:\n\n"
    for config in configs:
        name = config['name']
        summary += f"{name}:\n"
        summary += f"  Size: {config['som_size']}×{config['som_size']}, "
        summary += f"LR: {config.get('learning_rate', 'adaptive')}, σ: {config['sigma']}\n"
        summary += f"  Score: {scores[name]:.1f}/100\n\n"
    
    winner = max(scores, key=scores.get)
    summary += f"Best: {winner}"
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'configuration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quick_analysis(balancer, prefix: str = "analysis"):
    viz = SOMVisualizer()
    viz.create_comprehensive_analysis(balancer, f"{prefix}_results.png")