import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import sys

sys.path.append('.')

from src.som_balancer import SOMLoadBalancer, Server, Request
from src.visualization import SOMVisualizer

def load_optimal_params():
    with open('data/results/optimal_kohonen_params.json', 'r') as f:
        data = json.load(f)
    return data['recommendation']

def run_workload_test(workload_name, request_generator, n_requests=1000, n_servers=20):
    print(f"\nTesting workload: {workload_name}")
    print("-" * 50)
    
    params = load_optimal_params()
    print(f"Parameters: Size={params['som_size']}x{params['som_size']}, α={params['learning_rate']:.3f}, σ={params['sigma']:.3f}")
    
    servers = [Server(id=i) for i in range(n_servers)]
    balancer = SOMLoadBalancer(
        servers,
        som_size=params['som_size'],
        initial_learning_rate=params['learning_rate'],
        sigma=params['sigma']
    )
    
    success_count = 0
    response_times = []
    server_loads_history = []
    
    print(f"Running {n_requests} requests...")
    start_time = time.time()
    
    for i in range(n_requests):
        cpu, mem = request_generator(i)
        req = Request(i, cpu_demand=cpu, memory_demand=mem)
        
        req_start = time.time()
        server = balancer.select_server(req)
        
        if server:
            server.allocate(req)
            success_count += 1
            response_times.append((time.time() - req_start) * 1000)
            
            if np.random.random() < 0.3:
                release_factor = np.random.uniform(0.6, 0.9)
                server.cpu_usage = max(0, server.cpu_usage - req.cpu_demand * release_factor)
                server.memory_usage = max(0, server.memory_usage - req.memory_demand * release_factor)
        
        if (i + 1) % 100 == 0:
            loads = [s.utilization for s in servers]
            server_loads_history.append({
                'step': i + 1,
                'mean': np.mean(loads),
                'std': np.std(loads),
                'min': np.min(loads),
                'max': np.max(loads)
            })
            print(f"  Progress: {i+1}/{n_requests} - Success: {success_count}/{i+1} ({success_count/(i+1)*100:.1f}%)")
    
    duration = time.time() - start_time
    
    metrics = balancer.get_metrics()
    final_loads = [s.utilization for s in servers]
    
    results = {
        'workload': workload_name,
        'duration': duration,
        'success_rate': success_count / n_requests * 100,
        'total_requests': n_requests,
        'successful_requests': success_count,
        'avg_response_time': np.mean(response_times) if response_times else 0,
        'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
        'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
        'load_balance_score': metrics['load_balance_score'],
        'server_utilization': {
            'mean': np.mean(final_loads) * 100,
            'std': np.std(final_loads) * 100,
            'min': np.min(final_loads) * 100,
            'max': np.max(final_loads) * 100
        },
        'neuron_usage': metrics['neuron_usage_rate'],
        'active_neurons': metrics['active_neurons'],
        'history': server_loads_history
    }
    
    viz = SOMVisualizer()
    viz.create_comprehensive_analysis(balancer, f"{workload_name}_kohonen_test.png")
    
    print(f"\nResults for {workload_name}:")
    print(f"  Success rate: {results['success_rate']:.1f}%")
    print(f"  Response time: {results['avg_response_time']:.2f}ms (P95: {results['p95_response_time']:.2f}ms)")
    print(f"  Load balance score: {results['load_balance_score']:.3f}")
    print(f"  Server utilization: {results['server_utilization']['mean']:.1f}% ± {results['server_utilization']['std']:.1f}%")
    print(f"  Active neurons: {results['active_neurons']}/{params['som_size']**2} ({results['neuron_usage']:.1f}%)")
    
    return results

def workload_generators():
    def uniform(i):
        return np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4)
    
    def bursty(i):
        if i % 100 < 20:
            return np.random.uniform(0.6, 0.9), np.random.uniform(0.6, 0.9)
        else:
            return np.random.uniform(0.1, 0.3), np.random.uniform(0.1, 0.3)
    
    def hotspot(i):
        if np.random.random() < 0.8:
            return np.random.uniform(0.05, 0.2), np.random.uniform(0.05, 0.2)
        else:
            return np.random.uniform(0.5, 0.8), np.random.uniform(0.5, 0.8)
    
    def realistic(i):
        request_type = np.random.choice(['small', 'medium', 'large'], p=[0.6, 0.3, 0.1])
        if request_type == 'small':
            return np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15)
        elif request_type == 'medium':
            return np.random.uniform(0.2, 0.4), np.random.uniform(0.2, 0.4)
        else:
            return np.random.uniform(0.5, 0.7), np.random.uniform(0.5, 0.7)
    
    return {
        'uniform': uniform,
        'bursty': bursty,
        'hotspot': hotspot,
        'realistic': realistic
    }

def create_comparison_plot(all_results):
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    workloads = list(all_results.keys())
    
    success_rates = [all_results[w]['success_rate'] for w in workloads]
    ax1.bar(workloads, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate by Workload')
    ax1.set_ylim(0, 100)
    
    avg_rt = [all_results[w]['avg_response_time'] for w in workloads]
    p95_rt = [all_results[w]['p95_response_time'] for w in workloads]
    x = np.arange(len(workloads))
    ax2.bar(x - 0.2, avg_rt, 0.4, label='Average', alpha=0.8)
    ax2.bar(x + 0.2, p95_rt, 0.4, label='P95', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads)
    ax2.set_ylabel('Response Time (ms)')
    ax2.set_title('Response Time Comparison')
    ax2.legend()
    
    lb_scores = [all_results[w]['load_balance_score'] for w in workloads]
    ax3.plot(workloads, lb_scores, 'go-', linewidth=2, markersize=10)
    ax3.set_ylabel('Load Balance Score')
    ax3.set_title('Load Balancing Quality')
    ax3.set_ylim(0.8, 1.0)
    ax3.grid(True, alpha=0.3)
    
    neuron_usage = [all_results[w]['neuron_usage'] for w in workloads]
    ax4.bar(workloads, neuron_usage, color='purple', alpha=0.7)
    ax4.set_ylabel('Neuron Usage (%)')
    ax4.set_title('SOM Neuron Utilization')
    ax4.set_ylim(0, 100)
    
    plt.suptitle('Kohonen Maps Performance across Different Workloads', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/plots/workload_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plot saved: data/plots/workload_comparison.png")

# Direct execution
print("Kohonen Maps Workload Tests")
print("=" * 60)

try:
    if not Path('data/results/optimal_kohonen_params.json').exists():
        print("Error: No optimal parameters found.")
        print("Please run 'python experiments/optimize_parameters.py' first.")
    else:
        generators = workload_generators()
        
        all_results = {}
        for name, generator in generators.items():
            results = run_workload_test(name, generator, n_requests=1000)
            all_results[name] = results
        
        results_dir = Path("data/results")
        with open(results_dir / 'workload_test_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        create_comparison_plot(all_results)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        best_workload = max(all_results, key=lambda w: all_results[w]['success_rate'])
        worst_workload = min(all_results, key=lambda w: all_results[w]['success_rate'])
        
        print(f"\nBest performance: {best_workload} ({all_results[best_workload]['success_rate']:.1f}% success)")
        print(f"Most challenging: {worst_workload} ({all_results[worst_workload]['success_rate']:.1f}% success)")
        
        avg_lb_score = np.mean([r['load_balance_score'] for r in all_results.values()])
        avg_neuron_usage = np.mean([r['neuron_usage'] for r in all_results.values()])
        
        print(f"\nAverage load balance score: {avg_lb_score:.3f}")
        print(f"Average neuron usage: {avg_neuron_usage:.1f}%")
        
        print("\nAll tests completed successfully.")
        print(f"Results saved to: {results_dir / 'workload_test_results.json'}")
        print("Visualizations in: data/plots/")
        
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()