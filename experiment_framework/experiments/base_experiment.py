# experiments/base_experiment.py
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
import sys
sys.path.append('..')

from src.solvers import get_solver, SolverConfig, ILPSolverConfig, LNSSolverConfig
from src.core.puzzle import Puzzle
from experiment_framework.utils.logger import setup_logger

class BaseExperiment(ABC):
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            self.__class__.__name__,
            self.output_dir / 'experiment.log'
        )
        
        self.results = []
        
    @abstractmethod
    def run(self) -> List[Dict]:
        """Run the experiment and return results"""
        pass
    
    def load_puzzle(self, file_path: Path) -> Puzzle:
        """Load a puzzle from .has file"""
        return Puzzle.load_from_has(file_path)
    
    def create_solver(self, solver_name: str, solver_config: Dict):
        """Create a solver instance with given configuration"""
        solver_info = self.config['solvers'][solver_name]
        
        if solver_info['class'] == 'ILPSolver':
            config = ILPSolverConfig(**solver_config)
        elif solver_info['class'] == 'LNSSolver':
            config = LNSSolverConfig(**solver_config)
        else:
            config = SolverConfig(**solver_config)
        
        # Get solver class name (remove 'Solver' suffix if present)
        solver_type = solver_info['class'].replace('Solver', '').lower()
        return get_solver(solver_type, config)
    
    def run_single_test(self, puzzle: Puzzle, solver_name: str, 
                       time_limit: float) -> Dict:
        """Run a single solver test"""
        
        # Get solver configuration
        solver_config = self.config['solvers'][solver_name]['config'].copy()
        solver_config['time_limit'] = time_limit
        solver_config['verbose'] = False
        
        # Create solver
        solver = self.create_solver(solver_name, solver_config)
        
        # Run solver
        start_time = time.time()
        
        try:
            result = solver.solve(puzzle)
            solve_time = time.time() - start_time
            
            # Collect metrics
            metrics = {
                'solver': solver_name,
                'success': result.success,
                'solve_time': solve_time,
                'iterations': result.iterations,
                'memory_used': result.memory_used,
                'message': result.message,
                'timed_out': solve_time >= time_limit * 0.99
            }
            
            # Add solver-specific stats
            if result.stats:
                metrics['stats'] = result.stats
            
            # Validate solution if successful
            if result.success and result.solution:
                from src.core.validator import PuzzleValidator
                validation = PuzzleValidator.validate_solution(result.solution)
                metrics['valid_solution'] = validation.is_valid
                if not validation.is_valid:
                    metrics['validation_errors'] = validation.errors
            else:
                metrics['valid_solution'] = False
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error running {solver_name}: {str(e)}")
            return {
                'solver': solver_name,
                'success': False,
                'solve_time': time.time() - start_time,
                'error': str(e),
                'timed_out': False,
                'valid_solution': False
            }
    
    def select_instances(self, size: int, count: int, 
                        criteria: Dict = None) -> List[Path]:
        """Select instances based on size and optional criteria"""
        
        dataset_path = Path(self.config['dataset_paths'][size])
        all_files = list(dataset_path.glob("*.has"))
        
        if criteria:
            # Filter based on criteria
            filtered_files = []
            for file in all_files:
                if self.matches_criteria(file, criteria):
                    filtered_files.append(file)
            all_files = filtered_files
        
        # Sample if needed
        if count < len(all_files):
            import random
            random.seed(42)  # For reproducibility
            selected = random.sample(all_files, count)
        else:
            selected = all_files
        
        return sorted(selected)
    
    def matches_criteria(self, file_path: Path, criteria: Dict) -> bool:
        """Check if file matches given criteria"""
        # Parse filename
        parts = file_path.stem.split('_')
        if len(parts) < 6:
            return False
        
        file_density = int(parts[3])
        file_obstacles = int(parts[4])
        
        if 'density' in criteria and file_density != criteria['density']:
            return False
        if 'obstacles' in criteria and file_obstacles != criteria['obstacles']:
            return False
        
        return True
    
    def save_results(self, results=None):
        """Save results to CSV and JSON"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = self.output_dir / 'results.csv'
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")
        
        # Save as JSON
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary statistics
        self.save_summary_stats(df)
    
    def save_summary_stats(self, df: pd.DataFrame):
        """Save summary statistics"""
        summary = {
            'total_tests': len(df),
            'by_solver': df.groupby('solver').agg({
                'success': ['count', 'sum', 'mean'],
                'solve_time': ['mean', 'median', 'std', 'min', 'max'],
                'valid_solution': ['sum', 'mean']
            }).to_dict()
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)