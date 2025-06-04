"""
Master experiment runner
Run individual experiments or full experiment suite
"""
import sys
import argparse
from pathlib import Path
import os

# Add src and experiments to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
experiments_path = current_dir / "experiments"

# Add paths to sys.path if they exist
if src_path.exists():
    sys.path.insert(0, str(src_path))
if experiments_path.exists():
    sys.path.insert(0, str(experiments_path))

# Also add current directory
sys.path.insert(0, str(current_dir))

def run_experiment_01():
    """Run basic algorithm comparison"""
    try:
        # Try different import approaches
        try:
            from experiment_01_basic_comparison import BasicComparisonExperiment
        except ImportError:
            from experiments.experiment_01_basic_comparison import BasicComparisonExperiment
        
        print("🚀 Running Experiment 1: Basic Algorithm Comparison")
        experiment = BasicComparisonExperiment()
        results = experiment.execute()
        
        if results:
            print("✅ Experiment 1 completed successfully!")
            return True
        else:
            print("❌ Experiment 1 failed!")
            return False
    except Exception as e:
        print(f"❌ Error running Experiment 1: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")  # Show first few paths
        return False


def run_experiment_02():
    """Run SOM parameter tuning"""
    try:
        try:
            from experiment_02_som_parameter_tuning import SOMParameterTuningExperiment
        except ImportError:
            from experiments.experiment_02_som_parameter_tuning import SOMParameterTuningExperiment
        
        print("🚀 Running Experiment 2: SOM Parameter Tuning")
        experiment = SOMParameterTuningExperiment()
        results = experiment.execute()
        
        if results and 'best_parameters' in results:
            print("✅ Experiment 2 completed successfully!")
            best = results['best_parameters']
            print(f"🎯 Best parameters: size={best['som_size']}, lr={best['learning_rate']:.2f}, sigma={best['sigma']:.2f}")
            return True
        else:
            print("❌ Experiment 2 failed!")
            return False
    except Exception as e:
        print(f"❌ Error running Experiment 2: {e}")
        return False


def run_experiment_03():
    """Run SOM visualization"""
    try:
        try:
            from experiment_03_som_visualization import SOMVisualizationExperiment
        except ImportError:
            from experiments.experiment_03_som_visualization import SOMVisualizationExperiment
        
        print("🚀 Running Experiment 3: SOM Visualization")
        experiment = SOMVisualizationExperiment()
        results = experiment.execute()
        
        if results:
            print("✅ Experiment 3 completed successfully!")
            print(f"📊 Created visualizations for {len(results)} SOM configurations")
            return True
        else:
            print("❌ Experiment 3 failed!")
            return False
    except Exception as e:
        print(f"❌ Error running Experiment 3: {e}")
        return False


def run_all_experiments():
    """Run all experiments in sequence"""
    print("🎓 Master Thesis: Dynamic Load Balancing with SOMs")
    print("🚀 Running full experiment suite...")
    
    experiments = [
        ("Basic Algorithm Comparison", run_experiment_01),
        ("SOM Parameter Tuning", run_experiment_02),
        ("SOM Visualization", run_experiment_03),
    ]
    
    successful = 0
    total = len(experiments)
    
    for name, experiment_func in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        if experiment_func():
            successful += 1
        
        print(f"Progress: {successful}/{total} experiments completed")
    
    print(f"\n🏁 Experiment suite completed!")
    print(f"✅ {successful}/{total} experiments successful")
    
    if successful == total:
        print("🎉 All experiments completed successfully!")
        print("📁 Results saved to: data/experiments/")
    else:
        print(f"⚠️  {total - successful} experiments failed")
    
    return successful == total


def list_experiments():
    """List available experiments"""
    experiments = [
        ("01", "basic", "Basic Algorithm Comparison"),
        ("02", "tuning", "SOM Parameter Tuning"),
        ("03", "viz", "SOM Visualization"),
        ("all", "all", "Run all experiments")
    ]
    
    print("Available experiments:")
    for short, alias, description in experiments:
        print(f"  {short:3s} | {alias:8s} | {description}")


def check_environment():
    """Check if the environment is set up correctly"""
    print("🔍 Checking environment setup...")
    
    # Check directories
    current_dir = Path(__file__).parent
    required_dirs = ["src", "experiments", "data"]
    
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            print(f"✅ Directory '{dir_name}' found")
        else:
            print(f"⚠️  Directory '{dir_name}' missing")
    
    # Check experiment files
    experiments_dir = current_dir / "experiments"
    if experiments_dir.exists():
        experiment_files = [
            "experiment_01_basic_comparison.py",
            "experiment_02_som_parameter_tuning.py", 
            "experiment_03_som_visualization.py"
        ]
        
        for file_name in experiment_files:
            file_path = experiments_dir / file_name
            if file_path.exists():
                print(f"✅ Experiment file '{file_name}' found")
            else:
                print(f"❌ Experiment file '{file_name}' missing")
    
    # Check Python path
    print(f"\n📁 Current working directory: {os.getcwd()}")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📦 Python path includes:")
    for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
        print(f"   {i}: {path}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Run Master Thesis Experiments")
    parser.add_argument('experiment', nargs='?', 
                       help='Experiment to run (01, 02, 03, basic, tuning, viz, all)')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    parser.add_argument('--check', action='store_true', help='Check environment setup')
    
    args = parser.parse_args()
    
    if args.check:
        check_environment()
        return
    
    if args.list:
        list_experiments()
        return
    
    if not args.experiment:
        print("🎓 Master Thesis: Dynamic Load Balancing with SOMs")
        print("\nUsage:")
        print("  python run_experiments.py 01       # Basic algorithm comparison")
        print("  python run_experiments.py 02       # SOM parameter tuning")
        print("  python run_experiments.py 03       # SOM visualization")
        print("  python run_experiments.py all      # Run all experiments")
        print("  python run_experiments.py --list   # List all experiments")
        print("  python run_experiments.py --check  # Check environment")
        print("\nAliases:")
        print("  basic = 01, tuning = 02, viz = 03")
        return
    
    experiment = args.experiment.lower()
    
    # Create data directory if it doesn't exist
    Path("data/experiments").mkdir(parents=True, exist_ok=True)
    
    # Run selected experiment
    if experiment in ['01', 'basic']:
        success = run_experiment_01()
    elif experiment in ['02', 'tuning']:
        success = run_experiment_02()
    elif experiment in ['03', 'viz', 'visualization']:
        success = run_experiment_03()
    elif experiment == 'all':
        success = run_all_experiments()
    else:
        print(f"❌ Unknown experiment: {experiment}")
        print("Use --list to see available experiments")
        return
    
    if success:
        print(f"\n🎉 Successfully completed!")
    else:
        print(f"\n💥 Something went wrong!")


if __name__ == "__main__":
    main()