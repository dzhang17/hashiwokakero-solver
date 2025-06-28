#!/usr/bin/env python3
"""
Main experiment runner for Hashiwokakero solver evaluation
"""
import argparse
import yaml
from pathlib import Path
import logging
from datetime import datetime
import sys
import importlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class ExperimentRunner:
    """Main class for running experiments"""
    
    def __init__(self, config_path='experiment_framework/config/experiment_config.yaml'):
        """Initialize experiment runner with configuration"""
        self.config_path = Path(config_path)
        self.config = self.load_config(self.config_path)
        self.results_dir = Path(self.config.get('results_dir', 'results'))
        self.setup_logging()
        
        # Map experiment names to classes - will be loaded dynamically
        self.experiment_classes = {}
        self.load_experiment_classes()
        
    def load_experiment_classes(self):
        """Dynamically load experiment classes"""
        experiment_modules = {
            'feasibility_study': 'FeasibilityStudy',
            'scalability_test': 'ScalabilityTest', 
            'algorithm_comparison': 'AlgorithmComparison',
            'parameter_sensitivity': 'ParameterSensitivity',
            'difficulty_analysis': 'DifficultyAnalysis'
        }
        
        for exp_name, class_name in experiment_modules.items():
            try:
                # Try to import the module
                module = importlib.import_module(f'experiment_framework.experiments.{exp_name}')
                exp_class = getattr(module, class_name)
                self.experiment_classes[exp_name] = exp_class
                self.logger.info(f"Loaded experiment class: {exp_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load experiment {exp_name}: {e}")
                
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        if not config_path.exists():
            # Try relative to current directory
            config_path = Path.cwd() / config_path
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.results_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'experiment_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_experiment(self, experiment_name, experiment_config):
        """Run a single experiment"""
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        if experiment_name not in self.experiment_classes:
            self.logger.error(f"Unknown experiment: {experiment_name}")
            return
            
        try:
            # Create experiment instance
            experiment_class = self.experiment_classes[experiment_name]
            # Pass only the specific experiment config, not the whole config
            experiment = experiment_class(experiment_config, self.results_dir)
            
            # Run the experiment
            results = experiment.run()
            
            # Save results
            if hasattr(experiment, 'save_results'):
                experiment.save_results(results)
            else:
                self.logger.warning("Experiment does not have save_results method")
            
            self.logger.info(f"Completed experiment: {experiment_name}")
            
        except Exception as e:
            self.logger.error(f"Error running experiment {experiment_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def run_all(self):
        """Run all experiments specified in configuration"""
        experiments = self.config.get('experiments', {})
        
        if not experiments:
            self.logger.warning("No experiments configured")
            return
            
        for exp_name, exp_config in experiments.items():
            if exp_config.get('enabled', True):
                self.run_experiment(exp_name, exp_config)
            else:
                self.logger.info(f"Skipping disabled experiment: {exp_name}")
                
    def run_selected(self, experiment_names):
        """Run selected experiments"""
        experiments = self.config.get('experiments', {})
        
        for exp_name in experiment_names:
            if exp_name in experiments:
                exp_config = experiments[exp_name]
                self.run_experiment(exp_name, exp_config)
            else:
                self.logger.warning(f"Experiment not found in config: {exp_name}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run Hashiwokakero solver experiments')
    parser.add_argument(
        '--config',
        type=str,
        default='experiment_framework/config/experiment_config.yaml',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        help='Specific experiments to run (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        runner = ExperimentRunner(args.config)
        
        if args.experiments:
            runner.run_selected(args.experiments)
        else:
            runner.run_all()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()