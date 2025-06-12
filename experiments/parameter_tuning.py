import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class CompactParameterTuner:
    def __init__(self, 
                 param_ranges: Optional[Dict] = None,
                 n_initial_points: int = 10,
                 n_iterations: int = 20):
        
        self.param_ranges = param_ranges or {
            'som_size': (4, 16),
            'learning_rate': (0.01, 1.0),
            'sigma': (0.5, 4.0)
        }
        
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
    
    def objective_function(self, params: np.ndarray) -> float:
        som_size = int(params[0])
        learning_rate = params[1]
        sigma = params[2]
        
        from src.som_balancer import SOMLoadBalancer, Server, Request
        
        servers = [Server(id=i) for i in range(20)]
        balancer = SOMLoadBalancer(
            servers,
            som_size=som_size,
            initial_learning_rate=learning_rate,
            sigma=sigma
        )
        
        n_requests = 200
        success = 0
        
        for i in range(n_requests):
            req = Request(
                i,
                cpu_demand=np.random.beta(2, 5),
                memory_demand=np.random.beta(2, 5)
            )
            
            server = balancer.select_server(req)
            if server:
                server.allocate(req)
                success += 1
                
                if np.random.random() < 0.3:
                    server.cpu_usage *= 0.7
                    server.memory_usage *= 0.7
        
        metrics = balancer.get_metrics()
        
        score = (
            0.5 * (success / n_requests) +
            0.3 * metrics['load_balance_score'] +
            0.2 * (metrics['neuron_usage_rate'] / 100)
        )
        
        return score
    
    def optimize(self) -> Dict:
        X_init = self._sample_random_params(self.n_initial_points)
        y_init = []
        
        print("Initial exploration...")
        for i, params in enumerate(X_init):
            score = self.objective_function(params)
            y_init.append(score)
            self._update_best(params, score)
            print(f"  Sample {i+1}/{self.n_initial_points}: score = {score:.3f}")
        
        X = X_init
        y = np.array(y_init)
        
        print("\nOptimization phase...")
        for i in range(self.n_iterations):
            self.gp.fit(X, y)
            
            next_params = self._get_next_params(X, y)
            score = self.objective_function(next_params)
            
            X = np.vstack([X, next_params])
            y = np.append(y, score)
            self._update_best(next_params, score)
            
            print(f"  Iteration {i+1}/{self.n_iterations}: score = {score:.3f} (best = {self.best_score:.3f})")
        
        self._save_results()
        return self._format_best_params()
    
    def _sample_random_params(self, n_samples: int) -> np.ndarray:
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=n_samples)
        
        scaled = np.zeros_like(samples)
        for i, (low, high) in enumerate(self.param_ranges.values()):
            scaled[:, i] = low + samples[:, i] * (high - low)
        
        return scaled
    
    def _get_next_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        candidates = self._sample_random_params(1000)
        
        mu, sigma = self.gp.predict(candidates, return_std=True)
        
        y_best = y.max()
        with np.errstate(divide='warn'):
            Z = (mu - y_best) / sigma
            ei = sigma * (Z * stats.norm.cdf(Z) + stats.norm.pdf(Z))
            ei[sigma == 0.0] = 0.0
        
        best_idx = np.argmax(ei)
        return candidates[best_idx]
    
    def _update_best(self, params: np.ndarray, score: float):
        self.results.append({
            'som_size': int(params[0]),
            'learning_rate': params[1],
            'sigma': params[2],
            'score': score
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
    
    def _format_best_params(self) -> Dict:
        return {
            'som_size': int(self.best_params[0]),
            'learning_rate': float(self.best_params[1]),
            'sigma': float(self.best_params[2]),
            'score': float(self.best_score)
        }
    
    def _save_results(self):
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        df.to_csv(results_dir / "tuning_history.csv", index=False)
        
        with open(results_dir / "best_params.json", 'w') as f:
            json.dump(self._format_best_params(), f, indent=2)
        
        summary = {
            'n_evaluations': len(self.results),
            'best_score': float(self.best_score),
            'score_improvement': float(self.best_score - self.results[0]['score']),
            'convergence_iteration': next(
                i for i, r in enumerate(self.results) 
                if r['score'] == self.best_score
            )
        }
        
        with open(results_dir / "tuning_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

class GridSearchTuner:
    def __init__(self, param_grid: Optional[Dict] = None):
        self.param_grid = param_grid or {
            'som_size': [6, 8, 10, 12],
            'learning_rate': [0.1, 0.3, 0.5, 0.7],
            'sigma': [1.0, 1.5, 2.0, 2.5]
        }
    
    def search(self, eval_func) -> pd.DataFrame:
        from itertools import product
        
        results = []
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        total = np.prod([len(v) for v in param_values])
        print(f"Grid search: {total} configurations")
        
        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))
            
            param_array = np.array([
                params['som_size'],
                params['learning_rate'],
                params['sigma']
            ])
            
            score = eval_func(param_array)
            params['score'] = score
            results.append(params)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total}")
        
        df = pd.DataFrame(results)
        df.to_csv("data/results/grid_search_results.csv", index=False)
        
        return df