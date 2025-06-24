from config.config_loader import load_config, get_balancer_params
from src.som_balancer import SOMLoadBalancer, Request
import numpy as np

# 1. Load configuration from YAML file
config = load_config("config/config.yaml")
params = get_balancer_params(config)

# 2. Initialize the SOM-based load balancer with config parameters
balancer = SOMLoadBalancer(**params)

# 3. Generate sample workload and simulate requests
for i in range(100):
    req = Request(
        id=i,
        cpu_demand=np.random.uniform(0.1, 0.3),
        memory_demand=np.random.uniform(0.1, 0.3),
        processing_time=np.random.uniform(20, 40)
    )
    result = balancer.process_request(req)
    balancer.step()
    print(f"Step {i}:", result)

# 4. Optional: Print final system metrics
print("Final metrics:", balancer.get_metrics())