import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_balancer_params(config: dict) -> dict:
    return {
        'som_size': config['vms']['scenarios'][config['vms']['default_scenario']]['som_size'],
        'initial_learning_rate': config['som_parameters']['initial_learning_rate'],
        'sigma': config['som_parameters']['sigma'],
        'neighbor_influence': config['som_parameters']['neighbor_influence'],
        'load_spread_factor': config['som_parameters']['load_spread_factor'],
        'max_queue_length': config['response_time']['queue']['max_length'],
        'queue_penalty_ms': config['response_time']['queue']['processing_penalty_per_item']
    }
