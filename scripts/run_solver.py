#!/usr/bin/env python3
"""
Script to solve a single Hashiwokakero puzzle.

Usage:
    python scripts/run_solver.py puzzle.json --algorithm ilp --visualize
    python scripts/run_solver.py --generate 10x10 --difficulty medium --algorithm hybrid
"""

import click
import sys
from pathlib import Path
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.puzzle import Puzzle, Difficulty
from src.core.validator import PuzzleValidator
from src.core.utils import PuzzleConverter, setup_logger
from src.solvers import get_solver, SolverConfig, ILPSolverConfig, SASolverConfig, HybridSolverConfig
from src.generators.puzzle_generator import PuzzleGenerator, PuzzleGeneratorConfig
from src.visualization.static_viz import PuzzleVisualizer


@click.command()
@click.argument('puzzle_file', required=False, type=click.Path())
@click.option('--algorithm', '-a', type=click.Choice(['greedy', 'ilp', 'sa', 'hybrid', 'adaptive']), 
              default='ilp', help='Solving algorithm to use')
@click.option('--time-limit', '-t', type=float, default=60.0, 
              help='Time limit in seconds')
@click.option('--visualize', '-v', is_flag=True, 
              help='Visualize the puzzle and solution')
@click.option('--save-solution', '-s', type=click.Path(), 
              help='Save solution to file')
@click.option('--generate', '-g', type=str, 
              help='Generate puzzle instead (format: WIDTHxHEIGHT)')
@click.option('--difficulty', '-d', 
              type=click.Choice(['easy', 'medium', 'hard', 'expert']), 
              default='medium', help='Difficulty for generated puzzle')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--output-dir', '-o', type=click.Path(), default='results/solutions',
              help='Output directory for visualizations')
def main(puzzle_file, algorithm, time_limit, visualize, save_solution, 
         generate, difficulty, verbose, output_dir):
    """Solve a Hashiwokakero puzzle using specified algorithm."""
    
    # Setup
    logger = setup_logger("PuzzleSolver", level="DEBUG" if verbose else "INFO")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or generate puzzle
    if generate:
        # Parse dimensions
        try:
            width, height = map(int, generate.split('x'))
        except ValueError:
            click.echo("Error: Generate format should be WIDTHxHEIGHT (e.g., 10x10)")
            sys.exit(1)
            
        # Generate puzzle
        logger.info(f"Generating {width}x{height} {difficulty} puzzle...")
        generator = PuzzleGenerator(PuzzleGeneratorConfig(
            ensure_unique=True,
            solver_time_limit=10.0
        ))
        
        puzzle = generator.generate(
            width, height, 
            Difficulty[difficulty.upper()],
            strategy='solution_based'
        )
        
        if not puzzle:
            click.echo("Error: Failed to generate valid puzzle")
            sys.exit(1)
            
        # Save generated puzzle
        puzzle_path = output_path / f"generated_{width}x{height}_{difficulty}.json"
        puzzle.save(puzzle_path)
        logger.info(f"Generated puzzle saved to {puzzle_path}")
        
    elif puzzle_file:
        # Load puzzle from file
        puzzle_path = Path(puzzle_file)
        if not puzzle_path.exists():
            click.echo(f"Error: Puzzle file '{puzzle_file}' not found")
            sys.exit(1)
            
        try:
            puzzle = Puzzle.load(puzzle_path)
            logger.info(f"Loaded puzzle from {puzzle_path}")
        except Exception as e:
            click.echo(f"Error loading puzzle: {e}")
            sys.exit(1)
    else:
        click.echo("Error: Either provide a puzzle file or use --generate")
        sys.exit(1)
        
    # Display puzzle info
    logger.info(f"Puzzle: {puzzle.width}x{puzzle.height} with {len(puzzle.islands)} islands")
    
    # Validate puzzle
    validation = PuzzleValidator.validate_puzzle_structure(puzzle)
    if not validation:
        click.echo(f"Error: Invalid puzzle - {'; '.join(validation.errors)}")
        sys.exit(1)
        
    # Create solver with appropriate configuration
    if algorithm == 'ilp':
        config = ILPSolverConfig(
            time_limit=time_limit,
            verbose=verbose,
            solver_options={'timelimit': time_limit, 'threads': 4}
        )
    elif algorithm == 'sa':
        config = SASolverConfig(
            time_limit=time_limit,
            verbose=verbose,
            max_iterations=50000,
            initial_temperature=100.0,
            cooling_rate=0.95
        )
    elif algorithm in ['hybrid', 'adaptive']:
        config = HybridSolverConfig(
            time_limit=time_limit,
            verbose=verbose,
            strategy='ilp_first' if algorithm == 'hybrid' else 'adaptive'
        )
    else:  # greedy
        config = SolverConfig(time_limit=time_limit, verbose=verbose)
        
    # Create and run solver
    logger.info(f"Starting {algorithm} solver...")
    solver = get_solver(algorithm, config)
    
    # Add progress callback if verbose
    if verbose:
        def progress_callback(iteration, solution, stats):
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: {stats}")
                
        if hasattr(solver, 'add_progress_callback'):
            solver.add_progress_callback(progress_callback)
            
    # Solve puzzle
    start_time = time.time()
    result = solver.solve(puzzle)
    solve_time = time.time() - start_time
    
    # Display results
    click.echo("\n" + "="*50)
    click.echo(f"Algorithm: {algorithm}")
    click.echo(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
    click.echo(f"Time: {solve_time:.3f} seconds")
    click.echo(f"Iterations: {result.iterations}")
    click.echo(f"Memory: {result.memory_used:.1f} MB")
    
    if result.message:
        click.echo(f"Message: {result.message}")
        
    if result.stats:
        click.echo(f"Additional stats: {json.dumps(result.stats, indent=2)}")
        
    click.echo("="*50 + "\n")
    
    # Validate and save solution if successful
    if result.success and result.solution:
        # Validate solution
        sol_validation = PuzzleValidator.validate_solution(result.solution)
        
        if sol_validation:
            click.echo("✓ Solution is valid!")
            
            # Display solution as text
            if verbose:
                click.echo("\nSolution:")
                click.echo(PuzzleConverter.to_string(result.solution, show_bridges=True))
                
            # Save solution if requested
            if save_solution:
                save_path = Path(save_solution)
                result.solution.save(save_path)
                click.echo(f"\nSolution saved to {save_path}")
                
        else:
            click.echo("✗ Solution is invalid!")
            click.echo(f"Errors: {'; '.join(sol_validation.errors)}")
            
    else:
        click.echo("No valid solution found.")
        
    # Visualize if requested
    if visualize:
        viz = PuzzleVisualizer(figsize=(10, 10))
        
        # Original puzzle
        puzzle_img = output_path / f"puzzle_{algorithm}.png"
        viz.visualize(
            puzzle,
            show_solution=False,
            title=f"Original Puzzle ({puzzle.width}x{puzzle.height})",
            save_path=puzzle_img,
            show_plot=True
        )
        
        # Solution
        if result.success and result.solution:
            solution_img = output_path / f"solution_{algorithm}.png"
            viz.visualize(
                result.solution,
                show_solution=True,
                title=f"Solution by {algorithm.upper()} ({solve_time:.2f}s)",
                save_path=solution_img,
                show_plot=True
            )
            
            # Comparison
            comparison_img = output_path / f"comparison_{algorithm}.png"
            viz.create_comparison_plot(
                [puzzle, result.solution],
                ["Original Puzzle", f"{algorithm.upper()} Solution"],
                save_path=comparison_img
            )
            
            click.echo(f"\nVisualizations saved to {output_path}")


if __name__ == '__main__':
    main()