"""
Benchmark system for comparing solver performance.
"""

import time
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import multiprocessing as mp
from functools import partial
import traceback
from tqdm import tqdm

from ..core.puzzle import Puzzle, Difficulty
from ..core.validator import PuzzleValidator
from ..core.utils import setup_logger, memory_usage, DifficultyEstimator
from ..solvers import get_solver, SolverConfig
from ..generators.puzzle_generator import PuzzleGenerator, PuzzleGeneratorConfig


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test"""
    puzzle_id: str
    algorithm: str
    success: bool
    solve_time: float
    iterations: int
    memory_mb: float
    
    # Puzzle characteristics
    width: int
    height: int
    num_islands: int
    difficulty: str
    
    # Solution quality
    is_valid: bool = False
    error_message: str = ""
    
    # Additional metrics
    timestamp: str = ""
    extra_stats: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        if self.extra_stats is None:
            result['extra_stats'] = {}
        return result


class BenchmarkConfig:
    """Configuration for benchmark tests"""
    
    def __init__(self, **kwargs):
        # Test parameters
        self.algorithms: List[str] = kwargs.get('algorithms', ['ilp', 'sa', 'hybrid'])
        self.sizes: List[Tuple[int, int]] = kwargs.get('sizes', [
            (7, 7), (10, 10), (15, 15), (20, 20), (25, 25)
        ])
        self.difficulties: List[Difficulty] = kwargs.get('difficulties', [
            Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD
        ])
        self.puzzles_per_config: int = kwargs.get('puzzles_per_config', 10)
        
        # Solver parameters
        self.solver_configs: Dict[str, SolverConfig] = kwargs.get('solver_configs', {})
        self.time_limit: float = kwargs.get('time_limit', 60.0)
        
        # Execution parameters
        self.parallel: bool = kwargs.get('parallel', True)
        self.num_workers: int = kwargs.get('num_workers', mp.cpu_count() - 1)
        self.save_solutions: bool = kwargs.get('save_solutions', False)
        
        # Output parameters
        self.output_dir: Path = Path(kwargs.get('output_dir', 'results/benchmarks'))
        self.save_intermediate: bool = kwargs.get('save_intermediate', True)


class Benchmark:
    """Run benchmarks on Hashiwokakero solvers"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        self.generator = PuzzleGenerator(PuzzleGeneratorConfig(
            ensure_unique=True,
            solver_time_limit=10.0
        ))
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
    def run(self) -> pd.DataFrame:
        """
        Run complete benchmark suite.
        
        Returns:
            DataFrame with all benchmark results
        """
        self.logger.info("Starting benchmark suite")
        start_time = time.time()
        
        # Generate or load test puzzles
        test_puzzles = self._prepare_test_puzzles()
        self.logger.info(f"Prepared {len(test_puzzles)} test puzzles")
        
        # Run benchmarks
        if self.config.parallel:
            self._run_parallel(test_puzzles)
        else:
            self._run_sequential(test_puzzles)
            
        # Convert results to DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.output_dir / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save detailed JSON
        json_file = self.config.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'config': {
                    'algorithms': self.config.algorithms,
                    'sizes': self.config.sizes,
                    'difficulties': [d.value for d in self.config.difficulties],
                    'puzzles_per_config': self.config.puzzles_per_config,
                    'time_limit': self.config.time_limit
                },
                'results': [r.to_dict() for r in self.results],
                'summary': self._compute_summary(results_df)
            }, f, indent=2)
            
        total_time = time.time() - start_time
        self.logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        self.logger.info(f"Results saved to {results_file}")
        
        return results_df
        
    def _prepare_test_puzzles(self) -> List[Tuple[str, Puzzle]]:
        """Generate or load test puzzles"""
        puzzles = []
        puzzle_dir = self.config.output_dir / "test_puzzles"
        puzzle_dir.mkdir(exist_ok=True)
        
        for width, height in self.config.sizes:
            for difficulty in self.config.difficulties:
                # Check if puzzles already exist
                existing_puzzles = list(puzzle_dir.glob(
                    f"{difficulty.value}_{width}x{height}_*.json"
                ))
                
                if len(existing_puzzles) >= self.config.puzzles_per_config:
                    # Load existing puzzles
                    for i in range(self.config.puzzles_per_config):
                        puzzle = Puzzle.load(existing_puzzles[i])
                        puzzle_id = existing_puzzles[i].stem
                        puzzles.append((puzzle_id, puzzle))
                else:
                    # Generate new puzzles
                    self.logger.info(f"Generating {width}x{height} {difficulty.value} puzzles")
                    
                    for i in range(self.config.puzzles_per_config):
                        puzzle = self.generator.generate(
                            width, height, difficulty, 
                            strategy='random' if i % 2 == 0 else 'solution_based'
                        )
                        
                        if puzzle:
                            puzzle_id = f"{difficulty.value}_{width}x{height}_{i:04d}"
                            puzzle_file = puzzle_dir / f"{puzzle_id}.json"
                            puzzle.save(puzzle_file)
                            puzzles.append((puzzle_id, puzzle))
                            
        return puzzles
        
    def _run_sequential(self, test_puzzles: List[Tuple[str, Puzzle]]):
        """Run benchmarks sequentially"""
        total_tests = len(test_puzzles) * len(self.config.algorithms)
        
        with tqdm(total=total_tests, desc="Running benchmarks") as pbar:
            for puzzle_id, puzzle in test_puzzles:
                for algorithm in self.config.algorithms:
                    result = self._run_single_test(puzzle_id, puzzle, algorithm)
                    self.results.append(result)
                    pbar.update(1)
                    
                    # Save intermediate results
                    if self.config.save_intermediate and len(self.results) % 50 == 0:
                        self._save_intermediate_results()
                        
    def _run_parallel(self, test_puzzles: List[Tuple[str, Puzzle]]):
        """Run benchmarks in parallel"""
        # Prepare all test cases
        test_cases = [
            (puzzle_id, puzzle, algorithm)
            for puzzle_id, puzzle in test_puzzles
            for algorithm in self.config.algorithms
        ]
        
        self.logger.info(f"Running {len(test_cases)} tests with {self.config.num_workers} workers")
        
        # Create worker pool
        with mp.Pool(processes=self.config.num_workers) as pool:
            # Use partial to pass the method
            worker_func = partial(run_single_test_wrapper, self.config)
            
            # Run tests with progress bar
            with tqdm(total=len(test_cases), desc="Running benchmarks") as pbar:
                for result in pool.imap_unordered(worker_func, test_cases):
                    self.results.append(result)
                    pbar.update(1)
                    
                    # Save intermediate results
                    if self.config.save_intermediate and len(self.results) % 50 == 0:
                        self._save_intermediate_results()
                        
    def _run_single_test(self, puzzle_id: str, puzzle: Puzzle, 
                        algorithm: str) -> BenchmarkResult:
        """Run a single benchmark test"""
        # Get solver configuration
        if algorithm in self.config.solver_configs:
            solver_config = self.config.solver_configs[algorithm]
        else:
            # Use default configuration
            solver_config = SolverConfig(
                time_limit=self.config.time_limit,
                verbose=False
            )
            
        # Initialize result
        result = BenchmarkResult(
            puzzle_id=puzzle_id,
            algorithm=algorithm,
            success=False,
            solve_time=0.0,
            iterations=0,
            memory_mb=0.0,
            width=puzzle.width,
            height=puzzle.height,
            num_islands=len(puzzle.islands),
            difficulty=DifficultyEstimator.estimate_difficulty(puzzle).value,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # Create solver
            solver = get_solver(algorithm, solver_config)
            
            # Measure initial memory
            initial_memory = memory_usage()
            
            # Run solver
            solver_result = solver.solve(puzzle)
            
            # Update result
            result.success = solver_result.success
            result.solve_time = solver_result.solve_time
            result.iterations = solver_result.iterations
            result.memory_mb = memory_usage() - initial_memory
            
            # Validate solution if successful
            if solver_result.success and solver_result.solution:
                validation = PuzzleValidator.validate_solution(solver_result.solution)
                result.is_valid = validation.is_valid
                if not validation.is_valid:
                    result.error_message = "; ".join(validation.errors)
                    
                # Save solution if requested
                if self.config.save_solutions:
                    solution_dir = self.config.output_dir / "solutions" / algorithm
                    solution_dir.mkdir(parents=True, exist_ok=True)
                    solution_file = solution_dir / f"{puzzle_id}.json"
                    solver_result.solution.save(solution_file)
                    
            # Add extra statistics
            result.extra_stats = solver_result.stats
            
        except Exception as e:
            result.error_message = f"Exception: {str(e)}"
            self.logger.error(f"Error in {algorithm} on {puzzle_id}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
        return result
        
    def _save_intermediate_results(self):
        """Save intermediate results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = self.config.output_dir / f"intermediate_{timestamp}.json"
        
        with open(temp_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f)
            
    def _compute_summary(self, results_df: pd.DataFrame) -> dict:
        """Compute summary statistics"""
        summary = {}
        
        # Overall statistics
        summary['total_tests'] = len(results_df)
        summary['successful_tests'] = results_df['success'].sum()
        summary['success_rate'] = results_df['success'].mean()
        
        # Per algorithm statistics
        summary['by_algorithm'] = {}
        for algorithm in self.config.algorithms:
            alg_data = results_df[results_df['algorithm'] == algorithm]
            
            summary['by_algorithm'][algorithm] = {
                'success_rate': alg_data['success'].mean(),
                'avg_time': alg_data['solve_time'].mean(),
                'median_time': alg_data['solve_time'].median(),
                'avg_memory': alg_data['memory_mb'].mean(),
                'valid_solutions': alg_data['is_valid'].sum(),
                'total_tests': len(alg_data)
            }
            
        # Per difficulty statistics
        summary['by_difficulty'] = {}
        for difficulty in ['easy', 'medium', 'hard', 'expert']:
            diff_data = results_df[results_df['difficulty'] == difficulty]
            if len(diff_data) > 0:
                summary['by_difficulty'][difficulty] = {
                    'success_rate': diff_data['success'].mean(),
                    'avg_time': diff_data['solve_time'].mean()
                }
                
        # Per size statistics
        summary['by_size'] = {}
        for size in results_df['num_islands'].unique():
            size_data = results_df[results_df['num_islands'] == size]
            summary['by_size'][int(size)] = {
                'success_rate': size_data['success'].mean(),
                'avg_time': size_data['solve_time'].mean()
            }
            
        return summary


def run_single_test_wrapper(config: BenchmarkConfig, 
                           test_case: Tuple[str, Puzzle, str]) -> BenchmarkResult:
    """Wrapper function for parallel execution"""
    puzzle_id, puzzle, algorithm = test_case
    
    # Create a temporary Benchmark instance for the test
    benchmark = Benchmark(config)
    return benchmark._run_single_test(puzzle_id, puzzle, algorithm)


class BenchmarkAnalyzer:
    """Analyze benchmark results"""
    
    def __init__(self, results_file: Path):
        """Load benchmark results from file"""
        self.results_df = pd.read_csv(results_file)
        self.logger = setup_logger(self.__class__.__name__)
        
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics by algorithm"""
        summary = self.results_df.groupby('algorithm').agg({
            'success': ['count', 'sum', 'mean'],
            'solve_time': ['mean', 'median', 'std', 'min', 'max'],
            'memory_mb': ['mean', 'max'],
            'iterations': ['mean', 'max']
        }).round(3)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        
        return summary
        
    def get_performance_by_size(self) -> pd.DataFrame:
        """Analyze performance by problem size"""
        return self.results_df.groupby(['algorithm', 'num_islands']).agg({
            'success': 'mean',
            'solve_time': 'mean',
            'memory_mb': 'mean'
        }).round(3)
        
    def get_performance_by_difficulty(self) -> pd.DataFrame:
        """Analyze performance by difficulty"""
        return self.results_df.groupby(['algorithm', 'difficulty']).agg({
            'success': 'mean',
            'solve_time': 'mean'
        }).round(3)
        
    def find_best_algorithm(self, metric: str = 'solve_time',
                          constraints: Optional[dict] = None) -> str:
        """
        Find best algorithm based on metric and constraints.
        
        Args:
            metric: Metric to optimize ('solve_time', 'success', 'memory_mb')
            constraints: Optional constraints dict
            
        Returns:
            Best algorithm name
        """
        data = self.results_df.copy()
        
        # Apply constraints
        if constraints:
            for col, value in constraints.items():
                if col in data.columns:
                    data = data[data[col] == value]
                    
        # Calculate performance
        if metric == 'solve_time':
            # For time, lower is better
            performance = data.groupby('algorithm')[metric].mean()
            best = performance.idxmin()
        else:
            # For success rate, higher is better
            performance = data.groupby('algorithm')[metric].mean()
            best = performance.idxmax()
            
        return best