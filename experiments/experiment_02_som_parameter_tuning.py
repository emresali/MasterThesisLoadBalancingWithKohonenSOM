"""
Experiment 2: SOM Parameter Tuning
Test different SOM parameters to find optimal configuration for load balancing

Author: Emre Sali
Master Thesis: Dynamic Load Balancing with SOMs
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from itertools import product
import json
import yaml

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import required modules
try:
    from core.som_balancer import SOMBalancer
    from core.server import Server
    from core.request import Request
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Using fallback implementations...")
    
    # Fallback implementations
    from minisom import MiniSom
    
    class SOMBalancer:
        def __init__(self, servers, som_size=(5, 5), learning_rate=0.5, sigma=1.0, random_seed=None):
            self.servers = servers
            self.som_size = som_size
            self.learning_rate = learning_rate
            self.sigma = sigma
            self.input_dim = 2  # CPU and Memory demands
            
            # Initialize SOM
            self.som = MiniSom(som_size[0], som_size[1], self.input_dim,
                              sigma=sigma, learning_rate=learning_rate, 
                              random_seed=random_seed)
            
            # Map servers to SOM positions
            self.server_positions = self._map_servers_to_som()
            
        def _map_servers_to_som(self):
            """Map servers to SOM grid positions"""
            positions = {}
            total_positions = self.som_size[0] * self.som_size[1]
            servers_per_position = max(1, len(self.servers) // total_positions)
            
            position_idx = 0
            for i, server in enumerate(self.servers):
                x = position_idx // self.som_size[1]
                y = position_idx % self.som_size[1]
                positions[server.id] = (x, y)
                
                if (i + 1) % servers_per_position == 0:
                    position_idx = (position_idx + 1) % total_positions
                    
            return positions
        
        def train(self, training_data):
            """Train the SOM with request patterns"""
            self.som.train(training_data, num_iteration=1000)
            
        def process_request(self, request):
            """Process a request and return the best server"""
            if hasattr(request, 'cpu_demand'):
                input_vector = [request.cpu_demand, request.memory_demand]
            else:
                input_vector = [request.get('cpu', 0.1), request.get('memory', 0.1)]
            
            # Find best matching unit (BMU)
            winner = self.som.winner(input_vector)
            
            # Find server closest to this position
            best_server = self._find_best_server_for_position(winner)
            return best_server
        
        def _find_best_server_for_position(self, som_position):
            """Find the best available server for a SOM position"""
            min_distance = float('inf')
            best_server = 0
            
            for server_id, server_pos in self.server_positions.items():
                distance = np.sqrt((som_position[0] - server_pos[0])**2 + 
                                 (som_position[1] - server_pos[1])**2)
                
                # Consider server load in selection
                server = self.servers[server_id]
                load_factor = (server.cpu_usage / server.cpu_capacity + 
                             server.memory_usage / server.memory_capacity) / 2
                
                # Penalize heavily loaded servers
                adjusted_distance = distance + load_factor * 2
                
                if adjusted_distance < min_distance:
                    min_distance = adjusted_distance
                    best_server = server_id
                    
            return best_server
    
    class WorkloadGenerator:
        def __init__(self, pattern='mixed'):
            self.pattern = pattern
            
        def generate_requests(self, num_requests):
            requests = []
            for i in range(num_requests):
                if self.pattern == 'bursty':
                    # Bursty workload
                    if i % 100 < 20:  # 20% burst period
                        cpu_demand = np.random.exponential(0.8)
                        memory_demand = np.random.exponential(0.8)
                    else:
                        cpu_demand = np.random.exponential(0.2)
                        memory_demand = np.random.exponential(0.2)
                elif self.pattern == 'periodic':
                    # Periodic workload
                    phase = (i / 100) * 2 * np.pi
                    base_load = 0.3 + 0.3 * np.sin(phase)
                    cpu_demand = base_load + np.random.normal(0, 0.1)
                    memory_demand = base_load + np.random.normal(0, 0.1)
                else:
                    # Mixed/random workload
                    cpu_demand = np.random.exponential(0.3)
                    memory_demand = np.random.exponential(0.3)
                
                # Clip values to reasonable ranges
                cpu_demand = np.clip(cpu_demand, 0.01, 1.0)
                memory_demand = np.clip(memory_demand, 0.01, 1.0)
                
                requests.append(Request(cpu_demand, memory_demand))
            return requests
    
    class Server:
        def __init__(self, server_id, cpu_capacity=1.0, memory_capacity=1.0):
            self.id = server_id
            self.cpu_capacity = cpu_capacity
            self.memory_capacity = memory_capacity
            self.cpu_usage = 0.0
            self.memory_usage = 0.0
    
    class Request:
        def __init__(self, cpu_demand, memory_demand):
            self.cpu_demand = cpu_demand
            self.memory_demand = memory_demand


class SOMParameterTuningExperiment:
    def __init__(self, config_path=None):
        """Initialize the SOM parameter tuning experiment"""
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config = self.load_config(config_path)
        self.results = []
        self.best_parameters = None
        
        # Use config values or defaults
        self.num_servers = self.config.get('servers', {}).get('count', 8)
        self.num_requests = self.config.get('experiments', {}).get('tuning_requests', 2000)
        
        # Output directory from config
        base_output_dir = self.config.get('output', {}).get('results_dir', 'data/results')
        self.output_dir = Path(base_output_dir) / "som_parameter_tuning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameter grid from config
        param_config = self.config.get('experiments', {}).get('som_parameter_grid', {})
        self.parameter_grid = {
            'som_size': [(s, s) for s in param_config.get('som_size', [3, 4, 5, 6, 8])],
            'learning_rate': param_config.get('learning_rate', [0.1, 0.3, 0.5, 0.7, 0.9]),
            'sigma': param_config.get('sigma', [0.5, 1.0, 1.5, 2.0, 2.5]),
            'workload_pattern': ['mixed', 'bursty', 'periodic']
        }
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
        
    def setup_servers(self):
        """Create server instances with different capacities"""
        servers = []
        for i in range(self.num_servers):
            # Create heterogeneous servers
            cpu_capacity = np.random.uniform(0.7, 1.3)
            memory_capacity = np.random.uniform(0.7, 1.3)
            server = Server(i, cpu_capacity, memory_capacity)
            servers.append(server)
        return servers
    
    def evaluate_som_configuration(self, som_size, learning_rate, sigma, workload_pattern):
        """Evaluate a specific SOM configuration"""
        print(f"    Testing: size={som_size}, lr={learning_rate:.1f}, sigma={sigma:.1f}, pattern={workload_pattern}")
        
        # Setup
        servers = self.setup_servers()
        workload_gen = WorkloadGenerator(pattern=workload_pattern)
        requests = workload_gen.generate_requests(self.num_requests)
        
        # Create and train SOM balancer
        som_balancer = SOMBalancer(servers, som_size=som_size, 
                                 learning_rate=learning_rate, sigma=sigma,
                                 random_seed=42)  # Fixed seed for reproducibility
        
        # Generate training data from first 20% of requests
        training_size = int(0.2 * len(requests))
        training_data = []
        for req in requests[:training_size]:
            training_data.append([req.cpu_demand, req.memory_demand])
        
        # Train SOM
        som_balancer.train(np.array(training_data))
        
        # Test on remaining requests
        test_requests = requests[training_size:]
        
        return self.run_load_balancing_simulation(som_balancer, test_requests, servers)
    
    def run_load_balancing_simulation(self, balancer, requests, servers):
        """Run load balancing simulation and collect metrics"""
        # Reset server states
        for server in servers:
            server.cpu_usage = 0.0
            server.memory_usage = 0.0
        
        metrics = {
            'response_times': [],
            'server_utilizations': [],
            'load_variance': [],
            'rejected_requests': 0,
            'load_distribution': [0] * len(servers)
        }
        
        start_time = time.time()
        
        for i, request in enumerate(requests):
            try:
                server_id = balancer.process_request(request)
                server = servers[server_id]
                
                # Track load distribution
                metrics['load_distribution'][server_id] += 1
                
                # Check if server can handle the request
                if (server.cpu_usage + request.cpu_demand <= server.cpu_capacity and
                    server.memory_usage + request.memory_demand <= server.memory_capacity):
                    
                    # Accept request
                    server.cpu_usage += request.cpu_demand
                    server.memory_usage += request.memory_demand
                    
                    # Calculate response time
                    load_factor = (server.cpu_usage / server.cpu_capacity + 
                                 server.memory_usage / server.memory_capacity) / 2
                    response_time = 0.05 + load_factor * 0.3  # 50ms to 350ms
                    metrics['response_times'].append(response_time)
                    
                else:
                    # Reject request
                    metrics['rejected_requests'] += 1
                
                # Simulate request completion (reduce server load over time)
                if i % 20 == 0:  # Every 20 requests
                    for s in servers:
                        s.cpu_usage = max(0, s.cpu_usage - 0.03)
                        s.memory_usage = max(0, s.memory_usage - 0.03)
                
                # Record metrics every 100 requests
                if i % 100 == 0:
                    cpu_utils = [s.cpu_usage / s.cpu_capacity for s in servers]
                    memory_utils = [s.memory_usage / s.memory_capacity for s in servers]
                    all_utils = cpu_utils + memory_utils
                    
                    metrics['server_utilizations'].append(np.mean(all_utils))
                    metrics['load_variance'].append(np.var(all_utils))
                
            except Exception as e:
                print(f"      Error processing request {i}: {e}")
                metrics['rejected_requests'] += 1
        
        execution_time = time.time() - start_time
        
        # Calculate performance metrics
        avg_response_time = np.mean(metrics['response_times']) if metrics['response_times'] else float('inf')
        avg_utilization = np.mean(metrics['server_utilizations']) if metrics['server_utilizations'] else 0
        load_variance = np.mean(metrics['load_variance']) if metrics['load_variance'] else float('inf')
        rejection_rate = metrics['rejected_requests'] / len(requests)
        
        # Calculate load balancing quality (lower variance = better)
        load_std = np.std(metrics['load_distribution'])
        load_balance_score = 1.0 / (1.0 + load_std)  # Higher score = better balance
        
        # Calculate overall performance score
        performance_score = (
            (1.0 / (1.0 + avg_response_time)) * 0.3 +  # Response time (30%)
            avg_utilization * 0.2 +                    # Utilization (20%)
            load_balance_score * 0.3 +                 # Load balance (30%)
            (1.0 - rejection_rate) * 0.2               # Acceptance rate (20%)
        )
        
        return {
            'avg_response_time': avg_response_time,
            'avg_utilization': avg_utilization,
            'load_variance': load_variance,
            'rejection_rate': rejection_rate,
            'load_balance_score': load_balance_score,
            'performance_score': performance_score,
            'execution_time': execution_time,
            'load_distribution': metrics['load_distribution']
        }
    
    def run_parameter_sweep(self):
        """Run parameter sweep across all combinations"""
        print("🚀 Starting SOM Parameter Sweep...")
        
        total_combinations = (len(self.parameter_grid['som_size']) * 
                            len(self.parameter_grid['learning_rate']) * 
                            len(self.parameter_grid['sigma']) * 
                            len(self.parameter_grid['workload_pattern']))
        
        print(f"📊 Testing {total_combinations} parameter combinations...")
        
        combination_count = 0
        
        for som_size, learning_rate, sigma, workload_pattern in product(
            self.parameter_grid['som_size'],
            self.parameter_grid['learning_rate'], 
            self.parameter_grid['sigma'],
            self.parameter_grid['workload_pattern']
        ):
            combination_count += 1
            print(f"  🔬 Combination {combination_count}/{total_combinations}")
            
            try:
                # Run evaluation
                metrics = self.evaluate_som_configuration(
                    som_size, learning_rate, sigma, workload_pattern
                )
                
                # Store results
                result = {
                    'som_size': som_size,
                    'learning_rate': learning_rate,
                    'sigma': sigma,
                    'workload_pattern': workload_pattern,
                    **metrics
                }
                
                self.results.append(result)
                
                print(f"      ✅ Score: {metrics['performance_score']:.3f}, "
                      f"Response: {metrics['avg_response_time']:.3f}s, "
                      f"Balance: {metrics['load_balance_score']:.3f}")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        print(f"✅ Parameter sweep completed! Tested {len(self.results)} configurations.")
    
    def find_best_parameters(self):
        """Find the best performing parameter combination"""
        if not self.results:
            return None
        
        # Sort by performance score
        sorted_results = sorted(self.results, key=lambda x: x['performance_score'], reverse=True)
        
        self.best_parameters = sorted_results[0]
        
        print("\n🏆 Best Parameter Configuration:")
        print(f"   SOM Size: {self.best_parameters['som_size']}")
        print(f"   Learning Rate: {self.best_parameters['learning_rate']:.1f}")
        print(f"   Sigma: {self.best_parameters['sigma']:.1f}")
        print(f"   Workload Pattern: {self.best_parameters['workload_pattern']}")
        print(f"   Performance Score: {self.best_parameters['performance_score']:.3f}")
        print(f"   Average Response Time: {self.best_parameters['avg_response_time']:.3f}s")
        print(f"   Load Balance Score: {self.best_parameters['load_balance_score']:.3f}")
        
        return self.best_parameters
    
    def create_visualizations(self):
        """Create visualizations of parameter tuning results"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SOM Parameter Tuning Results', fontsize=16)
        
        # 1. Performance Score vs SOM Size
        som_size_performance = df.groupby('som_size')['performance_score'].mean()
        som_sizes_str = [f"{x[0]}x{x[1]}" for x in som_size_performance.index]
        axes[0, 0].bar(som_sizes_str, som_size_performance.values)
        axes[0, 0].set_title('Performance vs SOM Size')
        axes[0, 0].set_ylabel('Performance Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Performance Score vs Learning Rate
        lr_performance = df.groupby('learning_rate')['performance_score'].mean()
        axes[0, 1].plot(lr_performance.index, lr_performance.values, 'o-')
        axes[0, 1].set_title('Performance vs Learning Rate')
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Performance Score')
        
        # 3. Performance Score vs Sigma
        sigma_performance = df.groupby('sigma')['performance_score'].mean()
        axes[0, 2].plot(sigma_performance.index, sigma_performance.values, 'o-')
        axes[0, 2].set_title('Performance vs Sigma')
        axes[0, 2].set_xlabel('Sigma')
        axes[0, 2].set_ylabel('Performance Score')
        
        # 4. Response Time vs Load Balance Score
        axes[1, 0].scatter(df['load_balance_score'], df['avg_response_time'], alpha=0.6)
        axes[1, 0].set_title('Response Time vs Load Balance')
        axes[1, 0].set_xlabel('Load Balance Score')
        axes[1, 0].set_ylabel('Avg Response Time (s)')
        
        # 5. Performance by Workload Pattern
        pattern_performance = df.groupby('workload_pattern')['performance_score'].mean()
        axes[1, 1].bar(pattern_performance.index, pattern_performance.values)
        axes[1, 1].set_title('Performance by Workload Pattern')
        axes[1, 1].set_ylabel('Performance Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Heatmap of SOM Size vs Learning Rate
        pivot_table = df.pivot_table(values='performance_score', 
                                   index='som_size', 
                                   columns='learning_rate', 
                                   aggfunc='mean')
        im = axes[1, 2].imshow(pivot_table.values, cmap='viridis', aspect='auto')
        axes[1, 2].set_title('Performance Heatmap')
        axes[1, 2].set_xlabel('Learning Rate')
        axes[1, 2].set_ylabel('SOM Size')
        
        # Set tick labels for heatmap
        axes[1, 2].set_xticks(range(len(pivot_table.columns)))
        axes[1, 2].set_xticklabels(pivot_table.columns)
        axes[1, 2].set_yticks(range(len(pivot_table.index)))
        axes[1, 2].set_yticklabels([f"{x[0]}x{x[1]}" for x in pivot_table.index])
        
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_tuning_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {self.output_dir}")
    
    def save_results(self):
        """Save detailed results to files"""
        # Save all results as CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "som_parameter_results.csv", index=False)
        
        # Save best parameters
        if self.best_parameters:
            with open(self.output_dir / "best_parameters.json", 'w') as f:
                json.dump(self.best_parameters, f, indent=2, default=str)
        
        # Save summary statistics
        summary = {
            'total_configurations_tested': len(self.results),
            'best_performance_score': max([r['performance_score'] for r in self.results]),
            'average_performance_score': np.mean([r['performance_score'] for r in self.results]),
            'parameter_ranges': self.parameter_grid
        }
        
        with open(self.output_dir / "experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {self.output_dir}")
    
    def execute(self):
        """Execute the complete SOM parameter tuning experiment"""
        print("🚀 Starting SOM Parameter Tuning Experiment")
        print(f"📊 Servers: {self.num_servers}, Requests per test: {self.num_requests}")
        
        try:
            # Run parameter sweep
            self.run_parameter_sweep()
            
            # Find best parameters
            best_params = self.find_best_parameters()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_results()
            
            return {
                'best_parameters': best_params,
                'all_results': self.results,
                'summary': {
                    'total_tests': len(self.results),
                    'best_score': best_params['performance_score'] if best_params else 0
                }
            }
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# For standalone testing
if __name__ == "__main__":
    experiment = SOMParameterTuningExperiment()
    results = experiment.execute()
    if results:
        print("✅ SOM Parameter Tuning completed successfully!")
    else:
        print("❌ Experiment failed!")