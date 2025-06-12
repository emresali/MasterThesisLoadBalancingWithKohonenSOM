import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import sys

sys.path.append('.')

from src.som_balancer import SOMLoadBalancer, Server, Request
from experiments.parameter_tuning import CompactParameterTuner

def quick_parameter_search():
    print("Kohonen Maps Parameter Optimization")
    print("-" * 40)
    
    param_ranges = {
        'som_size': (6, 15),
        'learning_rate': (0.1, 0.9),
        'sigma': (1.0, 4.0)
    }
    
    print("Search space:")
    print(f"  SOM size: {param_ranges['som_size'][0]}x{param_ranges['som_size'][0]} to {param_ranges['som_size'][1]}x{param_ranges['som_size'][1]}")
    print(f"  Learning rate: {param_ranges['learning_rate'][0]} to {param_ranges['learning_rate'][1]}")
    print(f"  Sigma: {param_ranges['sigma'][0]} to {param_ranges['sigma'][1]}")
    
    print("\nStarting parameter search...")
    tuner = CompactParameterTuner(
        param_ranges=param_ranges,
        n_initial_points=8,
        n_iterations=12
    )
    
    start_time = time.time()
    best = tuner.optimize()
    duration = time.time() - start_time
    
    print(f"\nOptimization completed in {duration:.1f} seconds")
    
    print("\nBest parameters found:")
    print(f"  SOM size: {best['som_size']}x{best['som_size']}")
    print(f"  Learning rate: {best['learning_rate']:.3f}")
    print(f"  Sigma: {best['sigma']:.3f}")
    print(f"  Score: {best['score']:.3f}")
    
    return best

def test_best_config(best_params):
    print("\nTesting best configuration")
    print("-" * 40)
    
    servers = [Server(id=i) for i in range(20)]
    balancer = SOMLoadBalancer(
        servers,
        som_size=int(best_params['som_size']),
        initial_learning_rate=best_params['learning_rate'],
        sigma=best_params['sigma']
    )
    
    n_requests = 500
    success = 0
    
    print(f"Sending {n_requests} requests...")
    for i in range(n_requests):
        if i % 100 < 20:
            cpu = np.random.uniform(0.3, 0.6)
            mem = np.random.uniform(0.3, 0.6)
        else:
            cpu = np.random.uniform(0.1, 0.3)
            mem = np.random.uniform(0.1, 0.3)
        
        req = Request(i, cpu_demand=cpu, memory_demand=mem)
        server = balancer.select_server(req)
        
        if server:
            server.allocate(req)
            success += 1
            
            if np.random.random() < 0.25:
                server.cpu_usage = max(0, server.cpu_usage - req.cpu_demand * 0.7)
                server.memory_usage = max(0, server.memory_usage - req.memory_demand * 0.7)
        
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_requests} processed...")
    
    metrics = balancer.get_metrics()
    
    print(f"\nResults with optimal parameters:")
    print(f"  Success rate: {success}/{n_requests} ({success/n_requests*100:.1f}%)")
    print(f"  Load balance score: {metrics['load_balance_score']:.3f}")
    print(f"  Active neurons: {metrics['active_neurons']}/{metrics['total_neurons']} ({metrics['neuron_usage_rate']:.1f}%)")
    print(f"  Server utilization: {metrics['server_utilization']['mean']*100:.1f}% ± {metrics['server_utilization']['std']*100:.1f}%")
    
    from src.visualization import SOMVisualizer
    viz = SOMVisualizer()
    viz.create_comprehensive_analysis(balancer, "optimal_config_test.png")
    print(f"\nVisualization saved: data/plots/optimal_config_test.png")
    
    return metrics

def save_results(best_params, test_metrics):
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'optimization': {
            'best_params': best_params,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'validation': {
            'success_rate': test_metrics['server_utilization']['mean'] * 100,
            'load_balance_score': test_metrics['load_balance_score'],
            'neuron_usage_rate': test_metrics['neuron_usage_rate']
        },
        'recommendation': {
            'som_size': int(best_params['som_size']),
            'learning_rate': float(best_params['learning_rate']),
            'sigma': float(best_params['sigma'])
        }
    }
    
    with open(results_dir / 'optimal_kohonen_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved in: data/results/optimal_kohonen_params.json")

# Direct execution
try:
    best_params = quick_parameter_search()
    test_metrics = test_best_config(best_params)
    save_results(best_params, test_metrics)
    
    print("\nParameter optimization completed successfully.")
    print("Next step: Run workload tests with the optimal parameters.")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()