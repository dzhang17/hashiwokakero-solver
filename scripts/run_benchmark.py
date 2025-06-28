#!/usr/bin/env python3
"""
Script to run comprehensive benchmarks on Hashiwokakero solvers.

Usage:
    python scripts/run_benchmark.py --algorithms ilp sa hybrid --sizes 10x10 15x15 20x20
    python scripts/run_benchmark.py --quick  # Quick test with small puzzles
    python scripts/run_benchmark.py --full   # Full benchmark suite
"""

import click
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.puzzle import Difficulty
from src.analysis.benchmark import Benchmark, BenchmarkConfig
from src.analysis.report_generator import ReportGenerator
from src.visualization.performance_viz import PerformanceVisualizer
from src.solvers import ILPSolverConfig, SASolverConfig, HybridSolverConfig


# Predefined benchmark suites
BENCHMARK_SUITES = {
    'quick': {
        'algorithms': ['greedy', 'ilp', 'sa'],
        'sizes': [(7, 7), (10, 10)],
        'difficulties': [Difficulty.EASY, Difficulty.MEDIUM],
        'puzzles_per_config': 5,
        'time_limit': 30.0
    },
    'standard': {
        'algorithms': ['ilp', 'sa', 'hybrid'],
        'sizes': [(10, 10), (15, 15), (20, 20)],
        'difficulties': [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD],
        'puzzles_per_config': 10,
        'time_limit': 60.0
    },
    'full': {
        'algorithms': ['ilp', 'sa', 'hybrid', 'adaptive'],
        'sizes': [(7, 7), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30)],
        'difficulties': [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXPERT],
        'puzzles_per_config': 20,
        'time_limit': 120.0
    },
    'scalability': {
        'algorithms': ['ilp', 'sa', 'hybrid'],
        'sizes': [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        'difficulties': [Difficulty.MEDIUM],
        'puzzles_per_config': 10,
        'time_limit': 180.0
    }
}


@click.command()
@click.option('--suite', type=click.Choice(['quick', 'standard', 'full', 'scalability']),
              help='Use predefined benchmark suite')
@click.option('--algorithms', '-a', multiple=True,
              type=click.Choice(['greedy', 'ilp', 'sa', 'hybrid', 'adaptive']),
              help='Algorithms to benchmark')
@click.option('--sizes', '-s', multiple=True,
              help='Puzzle sizes (format: WIDTHxHEIGHT)')
@click.option('--difficulties', '-d', multiple=True,
              type=click.Choice(['easy', 'medium', 'hard', 'expert']),
              help='Difficulty levels to test')
@click.option('--puzzles-per-config', '-n', type=int, default=10,
              help='Number of puzzles per configuration')
@click.option('--time-limit', '-t', type=float, default=60.0,
              help='Time limit per puzzle in seconds')
@click.option('--parallel/--sequential', default=True,
              help='Run benchmarks in parallel')
@click.option('--workers', '-w', type=int, default=None,
              help='Number of parallel workers')
@click.option('--output-dir', '-o', type=click.Path(), default='results/benchmarks',
              help='Output directory for results')
@click.option('--generate-report', '-r', is_flag=True,
              help='Generate Excel report after benchmark')
@click.option('--visualize', '-v', is_flag=True,
              help='Create performance visualizations')
@click.option('--save-solutions', is_flag=True,
              help='Save all solutions')
def main(suite, algorithms, sizes, difficulties, puzzles_per_config, time_limit,
         parallel, workers, output_dir, generate_report, visualize, save_solutions):
    """Run comprehensive benchmarks on Hashiwokakero solvers."""
    
    click.echo("="*60)
    click.echo("Hashiwokakero Solver Benchmark System")
    click.echo("="*60)
    
    # Determine configuration
    if suite:
        # Use predefined suite
        suite_config = BENCHMARK_SUITES[suite]
        click.echo(f"\nUsing {suite} benchmark suite:")
        click.echo(f"  Algorithms: {', '.join(suite_config['algorithms'])}")
        click.echo(f"  Sizes: {suite_config['sizes']}")
        click.echo(f"  Difficulties: {[d.value for d in suite_config['difficulties']]}")
        click.echo(f"  Puzzles per config: {suite_config['puzzles_per_config']}")
        click.echo(f"  Time limit: {suite_config['time_limit']}s")
        
        config_dict = suite_config.copy()
        
    else:
        # Use custom configuration
        if not algorithms:
            click.echo("Error: Specify algorithms or use a predefined suite")
            sys.exit(1)
            
        # Parse sizes
        parsed_sizes = []
        if sizes:
            for size in sizes:
                try:
                    width, height = map(int, size.split('x'))
                    parsed_sizes.append((width, height))
                except ValueError:
                    click.echo(f"Error: Invalid size format '{size}' (use WIDTHxHEIGHT)")
                    sys.exit(1)
        else:
            parsed_sizes = [(10, 10), (15, 15), (20, 20)]
            
        # Parse difficulties
        if difficulties:
            parsed_difficulties = [Difficulty[d.upper()] for d in difficulties]
        else:
            parsed_difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
            
        config_dict = {
            'algorithms': list(algorithms),
            'sizes': parsed_sizes,
            'difficulties': parsed_difficulties,
            'puzzles_per_config': puzzles_per_config,
            'time_limit': time_limit
        }
        
    # Add execution parameters
    config_dict.update({
        'parallel': parallel,
        'num_workers': workers,
        'output_dir': output_dir,
        'save_solutions': save_solutions
    })
    
    # Create solver configurations
    solver_configs = {
        'ilp': ILPSolverConfig(
            time_limit=config_dict['time_limit'],
            solver_options={
                'timelimit': config_dict['time_limit'],
                'threads': 4,
                'ratio': 0.01
            }
        ),
        'sa': SASolverConfig(
            time_limit=config_dict['time_limit'],
            max_iterations=100000,
            initial_temperature=100.0,
            cooling_rate=0.95,
            adaptive_parameters=True
        ),
        'hybrid': HybridSolverConfig(
            time_limit=config_dict['time_limit'],
            strategy='ilp_first',
            ilp_time_fraction=0.7
        ),
        'adaptive': HybridSolverConfig(
            time_limit=config_dict['time_limit'],
            strategy='adaptive'
        )
    }
    
    config_dict['solver_configs'] = solver_configs
    
    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(**config_dict)
    
    # Calculate total tests
    total_tests = (len(config_dict['algorithms']) * 
                  len(config_dict['sizes']) * 
                  len(config_dict['difficulties']) * 
                  config_dict['puzzles_per_config'])
    
    click.echo(f"\nTotal tests to run: {total_tests}")
    
    if not click.confirm("\nProceed with benchmark?"):
        click.echo("Benchmark cancelled.")
        sys.exit(0)
        
    # Run benchmark
    click.echo("\nStarting benchmark...\n")
    
    benchmark = Benchmark(benchmark_config)
    results_df = benchmark.run()
    
    click.echo(f"\nBenchmark completed! Results summary:")
    click.echo(f"  Total tests: {len(results_df)}")
    click.echo(f"  Successful: {results_df['success'].sum()}")
    click.echo(f"  Failed: {(~results_df['success']).sum()}")
    click.echo(f"  Success rate: {results_df['success'].mean() * 100:.1f}%")
    
    # Display algorithm summary
    click.echo("\nAlgorithm Performance:")
    for algorithm in config_dict['algorithms']:
        alg_data = results_df[results_df['algorithm'] == algorithm]
        click.echo(f"  {algorithm}:")
        click.echo(f"    Success rate: {alg_data['success'].mean() * 100:.1f}%")
        click.echo(f"    Avg time: {alg_data['solve_time'].mean():.3f}s")
        click.echo(f"    Median time: {alg_data['solve_time'].median():.3f}s")
        
    # Generate report if requested
    if generate_report:
        click.echo("\nGenerating Excel report...")
        
        # Find latest results file
        results_files = list(Path(output_dir).glob("benchmark_results_*.csv"))
        if not results_files:
            click.echo("Error: No results file found")
            sys.exit(1)
            
        latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
        
        # Generate report
        report_gen = ReportGenerator(latest_results, Path(output_dir) / "reports")
        report_path = report_gen.generate_excel_report()
        
        click.echo(f"Report generated: {report_path}")
        
        # Also generate LaTeX report
        latex_path = report_gen.generate_latex_report()
        click.echo(f"LaTeX report generated: {latex_path}")
        
    # Create visualizations if requested
    if visualize:
        click.echo("\nCreating performance visualizations...")
        
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        viz = PerformanceVisualizer()
        
        # Create various plots
        plots = [
            ('time_comparison', viz.plot_time_comparison),
            ('scalability', viz.plot_scalability),
            ('success_rate', viz.plot_success_rate),
            ('memory_usage', viz.plot_memory_usage)
        ]
        
        for plot_name, plot_func in plots:
            try:
                fig = plot_func(results_df, save_path=viz_dir / f"{plot_name}.png")
                if fig:
                    click.echo(f"  Created {plot_name} plot")
            except Exception as e:
                click.echo(f"  Error creating {plot_name} plot: {e}")
                
        # Create comprehensive report
        try:
            fig = viz.create_comprehensive_report(
                results_df, 
                save_path=viz_dir / "comprehensive_report.png"
            )
            click.echo("  Created comprehensive report")
        except Exception as e:
            click.echo(f"  Error creating comprehensive report: {e}")
            
    click.echo("\n" + "="*60)
    click.echo("Benchmark complete!")
    click.echo("="*60)


if __name__ == '__main__':
    main()