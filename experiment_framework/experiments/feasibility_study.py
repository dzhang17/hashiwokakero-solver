# experiments/feasibility_study.py
from typing import List, Dict
from pathlib import Path
import random
import pandas as pd
from experiment_framework.experiments.base_experiment import BaseExperiment

class FeasibilityStudy(BaseExperiment):
    """
    Quick feasibility study to verify all solvers work correctly
    and establish baseline performance expectations.
    """
    
    def run(self) -> List[Dict]:
        """Run feasibility study on a small subset of instances"""
        
        # Config is already passed correctly
        config = self.config
        
        # Get config values with defaults
        sizes = config.get('sizes', [100, 200])
        instances_per_size = config.get('instances_per_size', 5)
        time_limit = config.get('timeout', 300)
        
        # Default solvers if not specified
        solvers = config.get('solvers', {'ilp': {}, 'lns': {}})
        
        # Get config values with defaults
        sizes = config.get('sizes', [100, 200])
        instances_per_size = config.get('instances_per_size', 5)
        time_limit = config.get('timeout', 300)
        
        # Default solvers if not specified
        solvers = config.get('solvers', {'ilp': {}, 'lns': {}})
        
        # Get config values with defaults
        sizes = config.get('sizes', [100, 200])
        instances_per_size = config.get('instances_per_size', 5)
        time_limit = config.get('timeout', 300)
        
        # Default solvers if not specified
        solvers = config.get('solvers', {'ilp': {}, 'lns': {}})
        sizes = config['sizes']
        instances_per_size = config['instances_per_size']
        time_limit = config.get('timeout', 300)
        
        self.logger.info("Starting feasibility study...")
        self.logger.info(f"Testing on {sizes} with {instances_per_size} instances each")
        
        # Test each solver on diverse instances
        for size in sizes:
            self.logger.info(f"\nTesting {size}-island instances...")
            
            # Select diverse instances for testing
            test_instances = self.select_diverse_instances(size, instances_per_size)
            
            for i, instance_path in enumerate(test_instances):
                self.logger.info(f"\nInstance {i+1}/{len(test_instances)}: {instance_path.name}")
                
                # Parse instance metadata
                metadata = self.parse_instance_metadata(instance_path)
                self.logger.info(f"  Density: {metadata.get('density', 'N/A')}%, "
                               f"Obstacles: {metadata.get('obstacles', 'N/A')}%")
                
                # Load puzzle
                try:
                    puzzle = self.load_puzzle(instance_path)
                    self.logger.info(f"  Loaded puzzle: {len(puzzle.islands)} islands, "
                                   f"grid size {puzzle.width}x{puzzle.height}")
                except Exception as e:
                    self.logger.error(f"  Failed to load puzzle: {e}")
                    continue
                
                # Test each solver
                for solver_name in solvers.keys():
                    self.logger.info(f"  Testing {solver_name}...")
                    
                    result = self.run_single_test(puzzle, solver_name, time_limit)
                    
                    # Add metadata
                    result.update({
                        'experiment': 'feasibility',
                        'size': size,
                        'instance': instance_path.name,
                        'instance_path': str(instance_path),
                        **metadata
                    })
                    
                    self.results.append(result)
                    
                    # Log result summary
                    if result['success']:
                        self.logger.info(f"    ✓ SUCCESS in {result['solve_time']:.2f}s "
                                       f"({result['iterations']} iterations)")
                        if not result.get('valid_solution', True):
                            self.logger.warning("    ⚠ Solution validation failed!")
                    else:
                        self.logger.info(f"    ✗ FAILED after {result['solve_time']:.2f}s")
                        if 'error' in result:
                            self.logger.error(f"    Error: {result['error']}")
        
        # Analyze feasibility results
        self.analyze_feasibility_results()
        
        return self.results
    
    def select_diverse_instances(self, size: int, count: int) -> List[Path]:
        """Select diverse instances covering different characteristics"""
        
        dataset_path = Path(self.config['dataset_paths'][size])
        all_files = list(dataset_path.glob("*.has"))
        
        if len(all_files) <= count:
            return sorted(all_files)
        
        # Group by characteristics
        grouped = self.group_instances_by_characteristics(all_files)
        
        # Select from each group
        selected = []
        instances_per_group = max(1, count // len(grouped))
        
        for group_key, group_files in grouped.items():
            # Random sample from each group
            random.seed(42)  # For reproducibility
            sample_size = min(instances_per_group, len(group_files))
            selected.extend(random.sample(group_files, sample_size))
        
        # If we need more, add randomly
        if len(selected) < count:
            remaining = list(set(all_files) - set(selected))
            random.seed(42)
            additional = random.sample(remaining, min(count - len(selected), len(remaining)))
            selected.extend(additional)
        
        return sorted(selected[:count])
    
    def group_instances_by_characteristics(self, files: List[Path]) -> Dict[tuple, List[Path]]:
        """Group instances by their characteristics"""
        
        groups = {}
        
        for file in files:
            metadata = self.parse_instance_metadata(file)
            
            # Create group key based on density and obstacles
            density = metadata.get('density', 0)
            obstacles = metadata.get('obstacles', 0)
            
            # Categorize density
            if density <= 25:
                density_cat = 'low'
            elif density <= 50:
                density_cat = 'medium'
            else:
                density_cat = 'high'
            
            # Categorize obstacles
            if obstacles == 0:
                obstacle_cat = 'none'
            elif obstacles <= 5:
                obstacle_cat = 'few'
            else:
                obstacle_cat = 'many'
            
            key = (density_cat, obstacle_cat)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(file)
        
        return groups
    
    def parse_instance_metadata(self, file_path: Path) -> Dict:
        """Parse metadata from instance filename"""
        
        import re
        
        # Expected format: Hs_GG_NNN_DD_OO_III.has
        match = re.match(
            r'Hs_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.has',
            file_path.name
        )
        
        if match:
            return {
                'grid_size': int(match.group(1)),
                'num_islands': int(match.group(2)),
                'density': int(match.group(3)),
                'obstacles': int(match.group(4)),
                'instance_id': int(match.group(5))
            }
        
        return {}
    
    def analyze_feasibility_results(self):
        """Analyze and summarize feasibility study results"""
        
        if not self.results:
            return
        
        import pandas as pd
        
        df = pd.DataFrame(self.results)
        
        # Overall summary
        summary = {
            'total_tests': len(df),
            'total_failures': len(df[df['success'] == False]),
            'solvers_tested': list(df['solver'].unique()),
            'sizes_tested': list(df['size'].unique())
        }
        
        # Per-solver summary
        solver_summary = df.groupby('solver').agg({
            'success': ['count', 'sum', 'mean'],
            'solve_time': ['mean', 'std', 'min', 'max'],
            'valid_solution': lambda x: x[x.index[x == True]].count() if 'valid_solution' in df else 0
        }).round(3)
        
        # Save summary
        with open(self.output_dir / 'feasibility_summary.txt', 'w') as f:
            f.write("FEASIBILITY STUDY SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total tests run: {summary['total_tests']}\n")
            f.write(f"Total failures: {summary['total_failures']}\n")
            f.write(f"Solvers tested: {', '.join(summary['solvers_tested'])}\n")
            f.write(f"Problem sizes: {', '.join(map(str, summary['sizes_tested']))}\n\n")
            
            f.write("SOLVER PERFORMANCE:\n")
            f.write("-" * 50 + "\n")
            
            for solver in df['solver'].unique():
                solver_df = df[df['solver'] == solver]
                success_rate = solver_df['success'].mean() * 100
                avg_time = solver_df[solver_df['success'] == True]['solve_time'].mean()
                
                f.write(f"\n{solver}:\n")
                f.write(f"  Success rate: {success_rate:.1f}%\n")
                f.write(f"  Average solve time: {avg_time:.2f}s\n")
                
                # Check for any errors
                errors = solver_df[solver_df['success'] == False]
                if len(errors) > 0:
                    f.write(f"  Failures: {len(errors)}\n")
                    if 'error' in errors.columns:
                        unique_errors = errors['error'].dropna().unique()
                        for error in unique_errors[:3]:  # Show first 3 unique errors
                            f.write(f"    - {error[:100]}...\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("RECOMMENDATIONS:\n")
            
            # Generate recommendations
            all_success = df.groupby('solver')['success'].mean()
            best_solver = all_success.idxmax()
            worst_solver = all_success.idxmin()
            
            f.write(f"- Best performing solver: {best_solver} "
                   f"({all_success[best_solver]*100:.1f}% success rate)\n")
            
            if all_success[worst_solver] < 0.5:
                f.write(f"- {worst_solver} has low success rate "
                       f"({all_success[worst_solver]*100:.1f}%) - check configuration\n")
            
            # Check for systematic failures
            if 'density' in df.columns:
                density_success = df.groupby('density')['success'].mean()
                difficult_densities = density_success[density_success < 0.5].index.tolist()
                if difficult_densities:
                    f.write(f"- Difficult density levels: {difficult_densities}\n")
        
        # Save detailed results
        solver_summary.to_csv(self.output_dir / 'feasibility_solver_stats.csv')
        
        # Create quick visualization
        self.create_feasibility_plots(df)
    
    def create_feasibility_plots(self, df: pd.DataFrame):
        """Create simple feasibility study plots"""