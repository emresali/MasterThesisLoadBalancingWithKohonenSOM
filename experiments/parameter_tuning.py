"""
SOM Parameter Tuning
"""
import numpy as np
import pandas as pd
from itertools import product
import json
from pathlib import Path
from src.som_balancer import SOMLoadBalancer, Server, Request


class ParameterTuning:
    def __init__(self):
        self.param_grid = {
            'som_size': [6, 8, 10],
            'learning_rate': [0.1, 0.5, 0.9],
            'sigma': [1.0, 2.0, 3.0]
        }
        self.results = []
    
    def evaluate_config(self, size, lr, sigma):
        """Test one configuration"""
        servers = [Server(id=i) for i in range(10)]
        balancer = SOMLoadBalancer(servers, som_size=size, 
                                 learning_rate=lr, sigma=sigma)
        
        # Generate requests
        success = 0
        for i in range(500):
            req = Request(i, 
                         cpu_demand=np.random.uniform(0.1, 0.5),
                         memory_demand=np.random.uniform(0.1, 0.5))
            server = balancer.select_server(req)
            if server:
                server.allocate(req)
                success += 1
        
        # Metrics
        loads = [s.utilization for s in servers]
        return {
            'som_size': size,
            'learning_rate': lr,
            'sigma': sigma,
            'success_rate': success / 500 * 100,
            'load_std': np.std(loads),
            'avg_load': np.mean(loads)
        }
    
    def run(self):
        """Run parameter sweep"""
        for size, lr, sigma in product(*self.param_grid.values()):
            print(f"Testing: size={size}, lr={lr}, sigma={sigma}")
            metrics = self.evaluate_config(size, lr, sigma)
            self.results.append(metrics)
        
        # Save results
        Path("data/results").mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv("data/results/parameter_tuning.csv", index=False)
        
        # Find best
        best = df.loc[df['success_rate'].idxmax()]
        with open("data/results/best_params.json", 'w') as f:
            json.dump(best.to_dict(), f, indent=2)
        
        print(f"\nBest params: size={best['som_size']}, "
              f"lr={best['learning_rate']}, sigma={best['sigma']}")
        
        return best