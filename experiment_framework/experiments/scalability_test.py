# experiments/scalability_test.py
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

from experiment_framework.experiments.base_experiment import BaseExperiment

class ScalabilityTest(BaseExperiment):
    def run(self) -> List[Dict]:
        """Run scalability test across different problem sizes"""
        
        # Config is already passed correctly
        config = self.config
        sizes = config['sizes']
        instances_per_size = config['instances_per_size']
        # Use timeout for all sizes
        time_limit = config.get('timeout', 600)
        
        # Ensure lists have same length
        if isinstance(instances_per_size, int):
            instances_per_size = [instances_per_size] * len(sizes)
        if isinstance(time_limits, (int, float)):
            time_limits = [time_limits] * len(sizes)
        
        # Select solvers to test
        solvers_to_test = ['ilp_optimized', 'lns_tuned']
        
        for i, size in enumerate(sizes):
            self.logger.info(f"\nTesting scalability for {size} islands...")
            
            # Select representative instances
            instances = self.select_representative_instances(
                size, instances_per_size[i]
            )
            
            for j, instance_path in enumerate(instances):
                self.logger.info(f"Testing instance {j+1}/{len(instances)}: {instance_path.name}")
                
                # Load puzzle
                puzzle = self.load_puzzle(instance_path)
                
                # Test each solver
                for solver_name in solvers_to_test:
                    self.logger.info(f"  Running {solver_name}...")
                    
                    result = self.run_single_test(
                        puzzle, solver_name, time_limits[i]
                    )
                    
                    # Add metadata
                    result.update({
                        'experiment': 'scalability',
                        'size': size,
                        'instance': instance_path.name,
                        'time_limit': time_limits[i],
                        'instance_path': str(instance_path)
                    })
                    
                    self.results.append(result)
                    
                    # Log result
                    status = "SUCCESS" if result['success'] else "FAILED"
                    self.logger.info(f"    {status} in {result['solve_time']:.2f}s")
        
        # Generate scalability analysis
        self.analyze_scalability()
        
        return self.results
    
    def select_representative_instances(self, size: int, count: int) -> List[Path]:
        """Select representative instances covering different difficulties"""
        
        dataset_path = Path(self.config['dataset_paths'][size])
        all_files = list(dataset_path.glob("*.has"))
        
        # Group by density and obstacles
        groups = {}
        for file in all_files:
            parts = file.stem.split('_')
            if len(parts) >= 5:
                density = int(parts[3])
                obstacles = int(parts[4])
                key = (density, obstacles)
                
                if key not in groups:
                    groups[key] = []
                groups[key].append(file)
        
        # Select instances from each group
        selected = []
        instances_per_group = max(1, count // len(groups))
        
        for key, files in sorted(groups.items()):
            # Take up to instances_per_group from each group
            import random
            random.seed(42)
            group_selection = random.sample(
                files, 
                min(instances_per_group, len(files))
            )
            selected.extend(group_selection)
        
        # If we need more, add randomly
        if len(selected) < count:
            remaining = list(set(all_files) - set(selected))
            random.seed(42)
            additional = random.sample(
                remaining, 
                min(count - len(selected), len(remaining))
            )
            selected.extend(additional)
        
        # If we have too many, trim
        if len(selected) > count:
            random.seed(42)
            selected = random.sample(selected, count)
        
        return sorted(selected)
    
    def analyze_scalability(self):
        """Analyze scalability results"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Group by size and solver
        scalability_stats = df.groupby(['size', 'solver']).agg({
            'success': ['count', 'sum', 'mean'],
            'solve_time': ['mean', 'median', 'std'],
            'valid_solution': 'mean'
        }).round(3)
        
        # Save scalability analysis
        scalability_stats.to_csv(self.output_dir / 'scalability_stats.csv')
        
        # Calculate timeout rates
        timeout_stats = df.groupby(['size', 'solver'])['timed_out'].agg(['sum', 'mean'])
        timeout_stats.to_csv(self.output_dir / 'timeout_stats.csv')