"""
Main experiment runner
"""
import numpy as np
from pathlib import Path
from src.som_balancer import SOMLoadBalancer, Server, Request
from src.visualization import create_hexagonal_plot
from experiments.parameter_tuning import ParameterTuning


def test_basic():
    """Quick functionality test"""
    print("Testing basic SOM...")
    
    servers = [Server(id=i) for i in range(8)]
    balancer = SOMLoadBalancer(servers)
    
    # Test requests
    for i in range(50):
        req = Request(i, 
                     cpu_demand=np.random.uniform(0.1, 0.4),
                     memory_demand=np.random.uniform(0.1, 0.4))
        server = balancer.select_server(req)
        if server:
            server.allocate(req)
    
    # Create visualization
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    create_hexagonal_plot(balancer, "data/plots/test_som.png")
    print("Test complete! Check data/plots/test_som.png")


def run_parameter_tuning():
    """Find best SOM parameters"""
    print("\nRunning parameter tuning...")
    tuning = ParameterTuning()
    best = tuning.run()
    return best


def test_workloads(size=10, lr=0.5, sigma=2.0):
    """Test different workload patterns"""
    print("\nTesting workload patterns...")
    
    for name in ['uniform', 'bursty', 'mixed']:
        print(f"\nTesting {name} workload...")
        servers = [Server(id=i) for i in range(10)]
        balancer = SOMLoadBalancer(servers, som_size=size, 
                                 learning_rate=lr, sigma=sigma)
        
        success = 0
        for i in range(300):
            # Generate different patterns
            if name == 'uniform':
                cpu = np.random.uniform(0.1, 0.5)
                mem = np.random.uniform(0.1, 0.5)
            elif name == 'bursty':
                if i % 50 < 10:  # 20% burst
                    cpu = np.random.uniform(0.6, 0.9)
                    mem = np.random.uniform(0.6, 0.9)
                else:
                    cpu = np.random.uniform(0.1, 0.3)
                    mem = np.random.uniform(0.1, 0.3)
            else:  # mixed
                cpu = np.random.choice([0.1, 0.5, 0.8])
                mem = np.random.choice([0.1, 0.5, 0.8])
            
            req = Request(i, cpu_demand=cpu, memory_demand=mem)
            
            server = balancer.select_server(req)
            if server:
                server.allocate(req)
                success += 1
                
                # Sometimes release resources (simulate completed requests)
                if np.random.random() < 0.3:  # 30% chance
                    server.cpu_usage = max(0, server.cpu_usage - req.cpu_demand)
                    server.memory_usage = max(0, server.memory_usage - req.memory_demand)
        
        print(f"  Success rate: {success/300*100:.1f}%")
        
        # Analyze neuron usage
        total_hits = sum(balancer.get_hits().values())
        active_neurons = len([h for h in balancer.get_hits().values() if h > 0])
        total_neurons = balancer.som.get_weights().shape[0] ** 2
        print(f"  Active neurons: {active_neurons}/{total_neurons} ({active_neurons/total_neurons*100:.1f}%)")
        print(f"  Average hits per active neuron: {total_hits/active_neurons if active_neurons > 0 else 0:.1f}")
        
        create_hexagonal_plot(balancer, f"data/plots/som_{name}.png")


def main():
    print("=== SOM Load Balancing Experiments ===\n")
    
    # 1. Basic test
    test_basic()
    
    # 2. Parameter tuning
    best = run_parameter_tuning()
    
    # 3. Test workloads with best params
    test_workloads(
        size=int(best['som_size']),
        lr=best['learning_rate'],
        sigma=best['sigma']
    )
    
    print("\nAll experiments complete!")
    print("Results in: data/results/")
    print("Plots in: data/plots/")


if __name__ == "__main__":
    main()