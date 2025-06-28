#!/usr/bin/env python3
"""
Fixed algorithm comparison experiment
"""
from experiment_framework.experiments.base_experiment import BaseExperiment
import time
from pathlib import Path
import logging

class AlgorithmComparison(BaseExperiment):
    """Compare different algorithms on Hashiwokakero puzzles"""
    
    def _read_puzzle_file(self, filename):
        """Read puzzle from .has file - same method as solve.py"""
        from src.core.puzzle import Puzzle
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = lines[0].strip().split()
        rows, cols, num_islands = int(header[0]), int(header[1]), int(header[2])
        
        # Create puzzle
        puzzle = Puzzle(rows, cols)
        
        # Parse grid and add islands
        island_id = 0
        for row in range(rows):
            values = list(map(int, lines[row + 1].strip().split()))
            for col in range(cols):
                if values[col] > 0:
                    puzzle.add_island(row, col, values[col])
                    island_id += 1
        
        return puzzle
    
    def run(self):
        """Run algorithm comparison experiment"""
        self.logger.info("Running algorithm comparison...")
        
        # Get configuration
        timeout = self.config.get('timeout', 300)
        num_instances = self.config.get('instances_per_size', 10)
        algorithms = self.config.get('algorithms', ['ilp', 'lns'])
        
        results = {
            'experiment': 'algorithm_comparison',
            'timestamp': time.time(),
            'config': self.config,
            'instances': []
        }
        
        # Load test instances
        dataset_path = Path(self.config.get('dataset_paths', {}).get(100, 'dataset/100'))
        if not dataset_path.exists():
            self.logger.error(f"Dataset path not found: {dataset_path}")
            return results
            
        # Get .has files based on configuration
        has_files = list(dataset_path.glob("*.has"))[:num_instances]
        self.logger.info(f"Found {len(has_files)} instances to test")
        
        # Test each instance with each algorithm
        for has_file in has_files:
            self.logger.info(f"Testing instance: {has_file.name}")
            
            instance_result = {
                'file': has_file.name,
                'algorithms': {}
            }
            
            # Load puzzle
            try:
                from src.core.puzzle import Puzzle
                # Use the same loading method as lns_test.py
                puzzle = self._read_puzzle_file(has_file)
                self.logger.info(f"  Puzzle: {puzzle.width}x{puzzle.height}, {len(puzzle.islands)} islands")
                
                # Test each algorithm
                for algo in algorithms:
                    self.logger.info(f"  Testing {algo}...")
                    
                    start_time = time.time()
                    try:
                        # Create solver with timeout configuration
                        if algo == 'ilp':
                            from src.solvers.ilp_solver import ILPSolver, ILPSolverConfig
                            config = ILPSolverConfig(
                                solver_name='cbc',
                                use_lazy_constraints=True,    # Same as solve.py
                                use_preprocessing=True,        # Same as solve.py
                                time_limit=timeout,
                                debug_mode=False,             # Set to True for debugging
                                verbose=False,
                                solver_options={
                                    'timelimit': timeout,
                                    'ratioGap': 0.01,
                                }
                            )
                            solver = ILPSolver(config)
                        elif algo == 'lns':
                            from src.solvers.lns_solver import LNSSolver, LNSSolverConfig
                            config = LNSSolverConfig(
                                time_limit=timeout,
                                initial_destroy_rate=0.25,   # Same as lns_test.py
                                min_destroy_rate=0.1,
                                max_destroy_rate=0.5,
                                destroy_rate_increase=1.1,
                                destroy_rate_decrease=0.95,
                                repair_time_limit=5.0,
                                use_warm_start=True,
                                accept_worse_solutions=True,
                                initial_temperature=10.0,
                                cooling_rate=0.97,
                                max_iterations_without_improvement=80,
                                min_temperature=0.01,
                                track_statistics=True,
                                verbose=False
                            )
                            solver = LNSSolver(config)
                        else:
                            self.logger.warning(f"Unknown algorithm: {algo}")
                            continue
                        
                        # Solve the puzzle
                        result = solver.solve(puzzle.copy())
                        solve_time = time.time() - start_time
                        
                        # Debug logging
                        self.logger.debug(f"    Solver returned: {type(result)}")
                        if hasattr(result, '__dict__'):
                            self.logger.debug(f"    Result attributes: {result.__dict__.keys()}")
                        
                        # Parse result - handle SolverResult object properly
                        if hasattr(result, 'success'):
                            # It's a SolverResult object
                            solved = result.success
                            solution = result.solution
                            stats = result.stats if hasattr(result, 'stats') else {}
                            message = result.message if hasattr(result, 'message') else ''
                            
                            self.logger.debug(f"    SolverResult - success: {solved}, has solution: {solution is not None}")
                        else:
                            # Legacy format - assume it's a Puzzle object
                            solution = result
                            solved = solution is not None
                            stats = {}
                            message = ''
                            self.logger.debug(f"    Legacy format - solution is None: {solution is None}")
                        
                        # Validate solution if solved
                        if solved and solution:
                            from src.core.validator import PuzzleValidator
                            validation = PuzzleValidator.validate_solution(solution)
                            actual_solved = validation.is_valid
                            
                            self.logger.debug(f"    Validation result: {actual_solved}")
                            if not actual_solved:
                                self.logger.warning(f"    Solution marked as success but validation failed!")
                                self.logger.warning(f"    Validation errors: {validation.errors}")
                                # Log more details about the solution
                                self.logger.debug(f"    Solution has {len(solution.bridges)} bridges")
                                self.logger.debug(f"    Solution has {len(solution.islands)} islands")
                                for island in solution.islands[:3]:  # First 3 islands
                                    current = solution.get_island_bridges(island.id)
                                    self.logger.debug(f"    Island {island.id}: required={island.required_bridges}, current={current}")
                        else:
                            actual_solved = False
                            validation = None
                        
                        instance_result['algorithms'][algo] = {
                            'solved': actual_solved,  # Use actual validation result
                            'time': solve_time,
                            'bridges': len(solution.bridges) if solution and hasattr(solution, 'bridges') else 0,
                            'stats': stats,
                            'message': message,
                            'validation_errors': validation.errors if validation and not validation.is_valid else []
                        }
                        
                        if actual_solved:
                            self.logger.info(f"    ✓ Solved in {solve_time:.2f}s with {len(solution.bridges)} bridges")
                        else:
                            self.logger.info(f"    ✗ Failed to solve in {solve_time:.2f}s")
                            if message:
                                self.logger.info(f"    Message: {message}")
                        
                    except Exception as e:
                        self.logger.error(f"    Error: {e}")
                        import traceback
                        self.logger.debug(traceback.format_exc())
                        instance_result['algorithms'][algo] = {
                            'solved': False,
                            'time': time.time() - start_time,
                            'error': str(e)
                        }
                
            except Exception as e:
                self.logger.error(f"  Failed to load puzzle: {e}")
                instance_result['error'] = str(e)
                
            results['instances'].append(instance_result)
        
        # Show summary
        self._print_summary(results, algorithms)
        
        return results
    
    def _print_summary(self, results, algorithms):
        """Print a nice summary of results"""
        self.logger.info("\n" + "="*70)
        self.logger.info("SUMMARY:")
        self.logger.info("="*70)
        
        # Summary table header
        self.logger.info(f"{'Algorithm':<10} | {'Solved':<12} | {'Success Rate':<12} | {'Avg Time':<10}")
        self.logger.info("-"*70)
        
        for algo in algorithms:
            instances = results.get('instances', [])
            algo_results = [inst.get('algorithms', {}).get(algo, {}) for inst in instances]
            
            solved_count = sum(1 for r in algo_results if r.get('solved', False))
            total_count = len([r for r in algo_results if 'solved' in r])
            
            if total_count > 0:
                success_rate = solved_count / total_count
                avg_time = sum(r.get('time', 0) for r in algo_results if 'time' in r) / total_count
            else:
                success_rate = 0
                avg_time = 0
            
            self.logger.info(f"{algo.upper():<10} | {solved_count}/{total_count:<10} | {success_rate:>10.1%} | {avg_time:>8.2f}s")
        
        self.logger.info("="*70)
        
        # Print any errors
        error_count = 0
        for inst in results.get('instances', []):
            for algo in algorithms:
                algo_result = inst.get('algorithms', {}).get(algo, {})
                if 'error' in algo_result:
                    error_count += 1
                    
        if error_count > 0:
            self.logger.warning(f"\nTotal errors encountered: {error_count}")
        
    def analyze_results(self, results):
        """Analyze the results for reporting"""
        analysis = {
            'total_instances': len(results.get('instances', [])),
            'algorithms': {},
            'comparison': {}
        }
        
        algorithms = self.config.get('algorithms', ['ilp', 'lns'])
        
        # Aggregate results by algorithm
        for algo in algorithms:
            solved = 0
            total_time = 0
            count = 0
            errors = 0
            
            for instance in results.get('instances', []):
                if algo in instance.get('algorithms', {}):
                    algo_result = instance['algorithms'][algo]
                    if algo_result.get('solved', False):
                        solved += 1
                    total_time += algo_result.get('time', 0)
                    count += 1
                    if 'error' in algo_result:
                        errors += 1
            
            analysis['algorithms'][algo] = {
                'solved': solved,
                'total': count,
                'success_rate': solved / count if count > 0 else 0,
                'avg_time': total_time / count if count > 0 else 0,
                'avg_time_solved': total_time / solved if solved > 0 else 0,
                'errors': errors
            }
        
        # Compare algorithms pairwise
        if len(algorithms) >= 2:
            for i, algo1 in enumerate(algorithms):
                for algo2 in algorithms[i+1:]:
                    comparison_key = f"{algo1}_vs_{algo2}"
                    both_solved = 0
                    algo1_only = 0
                    algo2_only = 0
                    neither = 0
                    
                    for instance in results.get('instances', []):
                        algo1_solved = instance.get('algorithms', {}).get(algo1, {}).get('solved', False)
                        algo2_solved = instance.get('algorithms', {}).get(algo2, {}).get('solved', False)
                        
                        if algo1_solved and algo2_solved:
                            both_solved += 1
                        elif algo1_solved and not algo2_solved:
                            algo1_only += 1
                        elif not algo1_solved and algo2_solved:
                            algo2_only += 1
                        else:
                            neither += 1
                    
                    analysis['comparison'][comparison_key] = {
                        'both_solved': both_solved,
                        f'{algo1}_only': algo1_only,
                        f'{algo2}_only': algo2_only,
                        'neither_solved': neither
                    }
        
        return analysis
    
    def save_results(self, results):
        """Save results to files"""
        # Save raw results
        super().save_results(results)
        
        # Also save a summary CSV for easy analysis
        import csv
        summary_file = self.results_dir / 'algorithm_comparison_summary.csv'
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            algorithms = self.config.get('algorithms', ['ilp', 'lns'])
            header = ['Instance'] + [f'{algo}_solved' for algo in algorithms] + [f'{algo}_time' for algo in algorithms]
            writer.writerow(header)
            
            # Data rows
            for inst in results.get('instances', []):
                row = [inst['file']]
                for algo in algorithms:
                    algo_result = inst.get('algorithms', {}).get(algo, {})
                    row.append('Yes' if algo_result.get('solved', False) else 'No')
                for algo in algorithms:
                    algo_result = inst.get('algorithms', {}).get(algo, {})
                    row.append(f"{algo_result.get('time', 0):.2f}")
                writer.writerow(row)
        
        self.logger.info(f"Summary saved to: {summary_file}")