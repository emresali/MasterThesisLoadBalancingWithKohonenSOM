"""
Complete SOM Visualization Suite
- Basic visualizations (hexagonal, learning rate, analysis)
- Parameter testing visualizations
- Configuration comparison visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, List, Any, Optional

# === BASIC VISUALIZATIONS ===

def create_full_visualization(balancer, save_dir="data/plots"):
    """Create complete SOM visualization including learning rate"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    create_hexagonal_plot(balancer, save_dir / "som_hex.png")
    create_learning_rate_plot(balancer, save_dir / "learning_rate.png")
    create_analysis_plot(balancer, save_dir / "som_analysis.png")

def create_hexagonal_plot(balancer, save_path):
    """Create hexagonal SOM visualization"""
    size = balancer.som.get_weights().shape[0]
    weights = balancer.get_weights()
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
   
    # U-Matrix
    u_matrix = calculate_umatrix(weights)
    plot_hex_grid(ax1, size, u_matrix, "U-Matrix", cm.viridis)
   
    # Hits
    hits_matrix = np.zeros((size, size))
    for (i, j), count in balancer.get_hits().items():
        hits_matrix[i, j] = count
    plot_hex_grid(ax2, size, hits_matrix, "Neuron Hits", cm.hot)
   
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def create_learning_rate_plot(balancer, save_path):
    """Plot adaptive learning rate over time"""
    history = balancer.get_learning_history()
    if not history:
        return
    
    iterations, rates = zip(*history)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, rates, 'b-', linewidth=2, label='Learning Rate')
    ax.axhline(y=balancer.min_learning_rate, color='r', linestyle='--', 
               label=f'Min LR = {balancer.min_learning_rate}')
    
    # Markiere Retraining-Punkte
    retrain_points = [i for i, _ in history if i % 100 == 0 and i > 0]
    retrain_rates = [r for i, r in history if i % 100 == 0 and i > 0]
    if retrain_points:
        ax.scatter(retrain_points, retrain_rates, color='green', 
                  s=50, zorder=5, label='Retraining Points')
    
    ax.set_xlabel('Iterationen')
    ax.set_ylabel('Lernrate')
    ax.set_title('Adaptive Learning Rate Verlauf')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def create_analysis_plot(balancer, save_path):
    """Combined analysis plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. U-Matrix
    weights = balancer.get_weights()
    u_matrix = calculate_umatrix(weights)
    im1 = ax1.imshow(u_matrix, cmap='viridis')
    ax1.set_title('U-Matrix (Distance Map)')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Hit Map
    size = weights.shape[0]
    hits_matrix = np.zeros((size, size))
    for (i, j), count in balancer.get_hits().items():
        hits_matrix[i, j] = count
    im2 = ax2.imshow(hits_matrix, cmap='hot')
    ax2.set_title('Neuron Activation Frequency')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Learning Rate Curve
    history = balancer.get_learning_history()
    if history:
        iterations, rates = zip(*history)
        ax3.plot(iterations, rates, 'b-', linewidth=2)
        ax3.set_xlabel('Iterationen')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Adaptive Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # 4. SOM Statistics
    ax4.axis('off')
    stats_text = f"""
    SOM Statistics:
    
    Total Requests: {balancer.request_count}
    Current LR: {balancer.get_current_lr():.6f}
    Initial LR: {balancer.initial_learning_rate}
    Min LR: {balancer.min_learning_rate}
    Decay Constant: {balancer.decay_constant}
    Current Iteration: {balancer.current_iteration}
    
    Active Neurons: {len(balancer.get_hits())}
    Most Active: {max(balancer.get_hits().items(), key=lambda x: x[1]) if balancer.get_hits() else 'None'}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# === PARAMETER TESTING VISUALIZATIONS ===

def create_parameter_analysis_plot(test_results: Dict[str, List[Dict]], save_path: Optional[str] = None):
    """Create parameter analysis visualization from test results"""
    if save_path is None:
        save_path = 'data/plots/kohonen_parameters.png'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Learning rate effect
    if 'learning_rates' in test_results:
        rates, success, load_std = [], [], []
        for r in test_results['learning_rates']:
            rates.append(r['param'])
            success.append(r['success'])
            load_std.append(r['load_std'])
        
        ax1.plot(rates, success, 'o-', markersize=8, linewidth=2)
        ax1.set_xlabel('Learning Rate (η)')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Learning Rate Effect on Success')
        ax1.grid(True, alpha=0.3)
    
    # 2. Map size effect
    if 'map_sizes' in test_results:
        sizes, neurons, success = [], [], []
        for s in test_results['map_sizes']:
            sizes.append(f"{s['param']}×{s['param']}")
            neurons.append(s['active_neurons'] / (s['param']**2) * 100)
            success.append(s['success'])
        
        ax2.bar(sizes, neurons, alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(sizes, success, 'ro-', markersize=8)
        ax2.set_xlabel('Map Size')
        ax2.set_ylabel('Active Neurons (%)', color='b')
        ax2_twin.set_ylabel('Success Rate (%)', color='r')
        ax2.set_title('Map Size Effects')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Neighborhood radius effect
    if 'sigmas' in test_results:
        sigmas, success, load_std = [], [], []
        for s in test_results['sigmas']:
            sigmas.append(s['param'])
            success.append(s['success'])
            load_std.append(s['load_std'])
        
        ax3.plot(sigmas, load_std, 'go-', markersize=8, linewidth=2)
        ax3.set_xlabel('Neighborhood Radius (σ)')
        ax3.set_ylabel('Load Standard Deviation')
        ax3.set_title('Neighborhood Effect on Load Balance')
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()  # Lower is better
    
    # 4. Summary
    ax4.axis('off')
    summary = "Kohonen SOM Parameter Guidelines:\n\n"
    
    if 'learning_rates' in test_results:
        best_lr = max(test_results['learning_rates'], key=lambda x: x['success'])
        summary += f"• Best Learning Rate: η = {best_lr['param']}\n"
        summary += f"  (Success: {best_lr['success']:.1f}%)\n\n"
    
    if 'map_sizes' in test_results:
        best_size = max(test_results['map_sizes'], key=lambda x: x['success'])
        summary += f"• Best Map Size: {best_size['param']}×{best_size['param']}\n"
        summary += f"  (Success: {best_size['success']:.1f}%)\n\n"
    
    if 'sigmas' in test_results:
        best_sigma = min(test_results['sigmas'], key=lambda x: x['load_std'])
        summary += f"• Best Neighborhood: σ = {best_sigma['param']}\n"
        summary += f"  (Load STD: {best_sigma['load_std']:.3f})\n"
    
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes,
             verticalalignment='top', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Kohonen (SOM) Parameter Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# === COMPARISON VISUALIZATIONS ===

def create_comparison_plots(results: Dict[str, Dict[str, Any]], configs: List[Dict[str, Any]]):
    """Create comparison visualizations between different SOM configurations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SOM Configuration Comparison', fontsize=16, fontweight='bold')
    
    workloads = ['uniform', 'bursty', 'mixed']
    config_names = [c['name'] for c in configs]
    colors = ['blue', 'red', 'green', 'orange'][:len(configs)]
    
    # Helper function for bar plots
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
    
    # 1. Success Rate
    plot_bars(axes[0, 0], 'success_rate', 'Success Rate (%)', 'Success Rate by Workload')
    
    # 2. Load Balance Quality
    ax = axes[0, 1]
    for i, name in enumerate(config_names):
        values = [results[name][w]['load_std'] for w in workloads]
        ax.plot(workloads, values, 'o-', label=name, color=colors[i], linewidth=2, markersize=8)
    ax.set_ylabel('Load STD')
    ax.set_title('Load Balance Quality (Lower is Better)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Neuron Usage
    plot_bars(axes[0, 2], 'neuron_usage', 'Neuron Usage (%)', 'SOM Neuron Utilization')
    
    # 4-5. Load Evolution
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
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate scores
    scores = {}
    for name in config_names:
        score = sum(
            results[name][w]['success_rate'] * 0.4 + 
            results[name][w]['load_balance_score'] * 100 * 0.4 + 
            results[name][w]['neuron_usage'] * 0.2
            for w in workloads
        ) / len(workloads)
        scores[name] = score
    
    # Summary text
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
    save_path = Path("data/plots/comparison")
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'configuration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# === HELPER FUNCTIONS ===

def plot_hex_grid(ax, size, values, title, cmap):
    """Plot hexagonal grid"""
    vmin, vmax = values.min(), values.max() or 1
   
    for i in range(size):
        for j in range(size):
            x = j * 1.5
            y = i * np.sqrt(3)
            if i % 2 == 1:
                x += 0.75
           
            val = (values[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0
            hex = RegularPolygon((x, y), 6, radius=0.9,
                               facecolor=cmap(val),
                               edgecolor='black')
            ax.add_patch(hex)
           
            if values[i, j] > 0:
                ax.text(x, y, f'{values[i, j]:.1f}',
                       ha='center', va='center', fontsize=8)
   
    ax.set_xlim(-1, size * 1.5)
    ax.set_ylim(-1, size * np.sqrt(3))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)

def calculate_umatrix(weights):
    """Calculate U-Matrix"""
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