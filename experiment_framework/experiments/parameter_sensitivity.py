# experiments/parameter_sensitivity.py
import numpy as np
from typing import List, Dict
import itertools

from experiment_framework.experiments.base_experiment import BaseExperiment

class ParameterSensitivity(BaseExperiment):
    def run(self) -> List[Dict]:
        """Run parameter sensitivity analysis"""
        
        # Config is already the parameter_sensitivity config
        config = self.config
        sizes = config['sizes']
        instances_per_size = config['instances_per_size']
        base_time_limit = config.get('timeout', 300)
        
        # Run LNS parameter sensitivity
        if 'lns' in config['parameters']:
            self.logger.info("Running LNS parameter sensitivity...")
            self.run_lns_sensitivity(sizes[0], instances_per_size, 
                                   config['parameters']['lns'], base_time_limit)
        
        # Run ILP parameter sensitivity
        if 'ilp' in config['parameters']:
            self.logger.info("Running ILP parameter sensitivity...")
            self.run_ilp_sensitivity(sizes[0], instances_per_size,
                                   config['parameters']['ilp'], base_time_limit)
        
        # Analyze sensitivity
        self.analyze_sensitivity()
        
        return self.results
    
    def run_lns_sensitivity(self, size: int, num_instances: int, 
                           parameters: Dict, time_limit: float):
        """Test LNS parameter sensitivity"""
        
        # Select test instances
        instances = self.select_instances(size, num_instances)
        
        # Create parameter combinations
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        # Use Latin Hypercube or grid search
        if len(param_names) <= 2:
            # Full grid for 1-2 parameters
            combinations = list(itertools.product(*param_values))
        else:
            # Sample parameter space for 3+ parameters
            combinations = self.sample_parameter_space(parameters, n_samples=20)
        
        self.logger.info(f"Testing {len(combinations)} parameter combinations")
        
        for combo_idx, param_combo in enumerate(combinations):
            # Create parameter dict
            param_dict = dict(zip(param_names, param_combo))
            self.logger.info(f"Testing combination {combo_idx + 1}: {param_dict}")
            
            for instance_path in instances:
                puzzle = self.load_puzzle(instance_path)
                
                # Create LNS config with these parameters
                lns_config = {
                    'time_limit': time_limit,
                    'verbose': False,
                    **param_dict
                }
                
                # Run test
                from src.solvers import LNSSolver, LNSSolverConfig
                config = LNSSolverConfig(**lns_config)
                solver = LNSSolver(config)
                
                import time
                start_time = time.time()
                
                try:
                    result = solver.solve(puzzle)
                    solve_time = time.time() - start_time
                    
                    metrics = {
                        'experiment': 'parameter_sensitivity',
                        'solver': 'lns',
                        'size': size,
                        'instance': instance_path.name,
                        'parameters': param_dict,
                        'success': result.success,
                        'solve_time': solve_time,
                        'iterations': result.iterations,
                        'final_objective': result.stats.get('final_objective', None) if result.stats else None
                    }
                    
                    # Add LNS-specific stats
                    if result.stats:
                        metrics['improvements'] = result.stats.get('improvements', 0)
                        metrics['cache_hit_rate'] = result.stats.get('cache_hit_rate', 0)
                    
                    self.results.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Error with parameters {param_dict}: {e}")
                    self.results.append({
                        'experiment': 'parameter_sensitivity',
                        'solver': 'lns',
                        'parameters': param_dict,
                        'error': str(e),
                        'success': False
                    })
    
    def run_ilp_sensitivity(self, size: int, num_instances: int,
                           parameters: Dict, time_limit: float):
        """Test ILP parameter sensitivity"""
        
        instances = self.select_instances(size, num_instances)
        
        # For ILP, parameters are usually discrete
        combinations = list(itertools.product(*parameters.values()))
        param_names = list(parameters.keys())
        
        for param_combo in combinations:
            param_dict = dict(zip(param_names, param_combo))
            
            for instance_path in instances:
                puzzle = self.load_puzzle(instance_path)
                
                # Create ILP config
                ilp_config = {
                    'time_limit': time_limit,
                    'verbose': False,
                    'solver_name': 'cbc',
                    **param_dict
                }
                
                # Run test
                from src.solvers import ILPSolver, ILPSolverConfig
                config = ILPSolverConfig(**ilp_config)
                solver = ILPSolver(config)
                
                import time
                start_time = time.time()
                
                try:
                    result = solver.solve(puzzle)
                    solve_time = time.time() - start_time
                    
                    metrics = {
                        'experiment': 'parameter_sensitivity',
                        'solver': 'ilp',
                        'size': size,
                        'instance': instance_path.name,
                        'parameters': param_dict,
                        'success': result.success,
                        'solve_time': solve_time,
                        'iterations': result.iterations
                    }
                    
                    # Add ILP-specific stats
                    if result.stats:
                        metrics['objective_value'] = result.stats.get('objective_value', None)
                        metrics['preprocessing'] = result.stats.get('preprocessing', {})
                    
                    self.results.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Error with parameters {param_dict}: {e}")
    
    def sample_parameter_space(self, parameters: Dict, n_samples: int) -> List:
        """Sample parameter space using Latin Hypercube Sampling"""
        from scipy.stats import qmc
        
        param_names = list(parameters.keys())
        param_ranges = []
        
        for name, values in parameters.items():
            if isinstance(values[0], (int, float)):
                param_ranges.append([min(values), max(values)])
            else:
                # For discrete parameters, use indices
                param_ranges.append([0, len(values) - 1])
        
        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=len(param_names))
        sample = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        scaled_samples = []
        for i, point in enumerate(sample):
            scaled_point = []
            for j, (param_name, value) in enumerate(zip(param_names, point)):
                param_range = param_ranges[j]
                if isinstance(parameters[param_name][0], (int, float)):
                    # Continuous parameter
                    scaled_value = param_range[0] + value * (param_range[1] - param_range[0])
                    if isinstance(parameters[param_name][0], int):
                        scaled_value = int(round(scaled_value))
                    scaled_point.append(scaled_value)
                else:
                    # Discrete parameter
                    idx = int(value * len(parameters[param_name]))
                    idx = min(idx, len(parameters[param_name]) - 1)
                    scaled_point.append(parameters[param_name][idx])
            
            scaled_samples.append(tuple(scaled_point))
        
        return scaled_samples
    
    def analyze_sensitivity(self):
        """Analyze parameter sensitivity results"""
        if not self.results:
            return
        
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        # Separate by solver
        for solver in df['solver'].unique():
            solver_df = df[df['solver'] == solver]
            
            if len(solver_df) == 0:
                continue
            
            # Extract parameter columns
            param_df = pd.json_normalize(solver_df['parameters'])
            param_names = param_df.columns.tolist()
            
            # Combine with metrics
            analysis_df = pd.concat([
                solver_df[['success', 'solve_time', 'iterations']].reset_index(drop=True),
                param_df
            ], axis=1)
            
            # Calculate correlation between parameters and success
            correlations = {}
            for param in param_names:
                if analysis_df[param].dtype in ['float64', 'int64']:
                    correlations[param] = {
                        'success_corr': analysis_df[param].corr(analysis_df['success']),
                        'time_corr': analysis_df[param].corr(analysis_df['solve_time'])
                    }
            
            # Save analysis
            analysis_df.to_csv(self.output_dir / f'{solver}_sensitivity_raw.csv', index=False)
            
            # Save correlations
            import json
            with open(self.output_dir / f'{solver}_correlations.json', 'w') as f:
                json.dump(correlations, f, indent=2)
            
            # Find best parameter combination
            best_idx = analysis_df.groupby(param_names)['success'].mean().idxmax()
            with open(self.output_dir / f'{solver}_best_params.txt', 'w') as f:
                f.write(f"Best parameters for {solver}:\n")
                if isinstance(best_idx, tuple):
                    for param, value in zip(param_names, best_idx):
                        f.write(f"  {param}: {value}\n")
                else:
                    f.write(f"  {param_names[0]}: {best_idx}\n")