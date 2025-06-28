#!/usr/bin/env python3
"""
Script to generate Hashiwokakero puzzles.

Usage:
    python scripts/generate_puzzles.py --count 10 --size 15x15 --difficulty medium
    python scripts/generate_puzzles.py --batch easy:5x5:10 medium:10x10:10 hard:15x15:5
    python scripts/generate_puzzles.py --test-suite  # Generate standard test suite
"""

import click
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.puzzle import Puzzle, Difficulty
from src.core.utils import PuzzleConverter, save_puzzle_batch
from src.generators.puzzle_generator import PuzzleGenerator, PuzzleGeneratorConfig
from src.visualization.static_viz import PuzzleVisualizer


# Standard test suite configuration
TEST_SUITE = {
    'small': [
        (Difficulty.EASY, (5, 5), 20),
        (Difficulty.EASY, (7, 7), 20),
        (Difficulty.MEDIUM, (7, 7), 20),
        (Difficulty.MEDIUM, (10, 10), 20),
    ],
    'medium': [
        (Difficulty.MEDIUM, (15, 15), 15),
        (Difficulty.HARD, (15, 15), 15),
        (Difficulty.MEDIUM, (20, 20), 15),
        (Difficulty.HARD, (20, 20), 15),
    ],
    'large': [
        (Difficulty.HARD, (25, 25), 10),
        (Difficulty.EXPERT, (25, 25), 10),
        (Difficulty.HARD, (30, 30), 10),
        (Difficulty.EXPERT, (30, 30), 10),
    ],
    'extra_large': [
        (Difficulty.EXPERT, (40, 40), 5),
        (Difficulty.EXPERT, (50, 50), 5),
    ]
}


@click.command()
@click.option('--count', '-n', type=int, default=10,
              help='Number of puzzles to generate')
@click.option('--size', '-s', type=str, default='10x10',
              help='Puzzle size (format: WIDTHxHEIGHT)')
@click.option('--difficulty', '-d', 
              type=click.Choice(['easy', 'medium', 'hard', 'expert']),
              default='medium', help='Puzzle difficulty')
@click.option('--batch', '-b', multiple=True,
              help='Batch generation (format: DIFFICULTY:WIDTHxHEIGHT:COUNT)')
@click.option('--test-suite', is_flag=True,
              help='Generate standard test suite')
@click.option('--output-dir', '-o', type=click.Path(), default='data/puzzles',
              help='Output directory for puzzles')
@click.option('--strategy', type=click.Choice(['random', 'solution_based', 'pattern', 'mixed']),
              default='mixed', help='Generation strategy')
@click.option('--ensure-unique/--no-ensure-unique', default=True,
              help='Ensure puzzles have unique solutions')
@click.option('--visualize', '-v', is_flag=True,
              help='Create visualizations of generated puzzles')
@click.option('--create-sheet', is_flag=True,
              help='Create printable puzzle sheet')
@click.option('--seed', type=int, default=None,
              help='Random seed for reproducibility')
def main(count, size, difficulty, batch, test_suite, output_dir, strategy,
         ensure_unique, visualize, create_sheet, seed):
    """Generate Hashiwokakero puzzles with various configurations."""
    
    click.echo("="*60)
    click.echo("Hashiwokakero Puzzle Generator")
    click.echo("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create generator
    generator_config = PuzzleGeneratorConfig(
        ensure_unique=ensure_unique,
        max_attempts=100,
        solver_time_limit=10.0,
        random_seed=seed
    )
    generator = PuzzleGenerator(generator_config)
    
    # Determine what to generate
    generation_tasks = []
    
    if test_suite:
        # Generate standard test suite
        click.echo("\nGenerating standard test suite...")
        for category, configs in TEST_SUITE.items():
            click.echo(f"\n{category.upper()} puzzles:")
            for diff, (width, height), num in configs:
                generation_tasks.append((diff, width, height, num))
                click.echo(f"  - {num} {diff.value} puzzles at {width}x{height}")
                
    elif batch:
        # Parse batch specifications
        click.echo("\nBatch generation:")
        for spec in batch:
            try:
                parts = spec.split(':')
                if len(parts) != 3:
                    raise ValueError("Invalid format")
                    
                diff_str, size_str, count_str = parts
                diff = Difficulty[diff_str.upper()]
                width, height = map(int, size_str.split('x'))
                num = int(count_str)
                
                generation_tasks.append((diff, width, height, num))
                click.echo(f"  - {num} {diff.value} puzzles at {width}x{height}")
                
            except Exception as e:
                click.echo(f"Error parsing batch spec '{spec}': {e}")
                click.echo("Format should be DIFFICULTY:WIDTHxHEIGHT:COUNT")
                sys.exit(1)
                
    else:
        # Single configuration
        try:
            width, height = map(int, size.split('x'))
        except ValueError:
            click.echo(f"Error: Invalid size format '{size}' (use WIDTHxHEIGHT)")
            sys.exit(1)
            
        diff = Difficulty[difficulty.upper()]
        generation_tasks.append((diff, width, height, count))
        click.echo(f"\nGenerating {count} {difficulty} puzzles at {width}x{height}")
        
    # Calculate total puzzles
    total_puzzles = sum(num for _, _, _, num in generation_tasks)
    click.echo(f"\nTotal puzzles to generate: {total_puzzles}")
    
    if not click.confirm("\nProceed with generation?"):
        click.echo("Generation cancelled.")
        sys.exit(0)
        
    # Generate puzzles
    all_puzzles = []
    generated_count = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with click.progressbar(length=total_puzzles, label='Generating puzzles') as bar:
        for diff, width, height, num in generation_tasks:
            # Create subdirectory for this configuration
            config_dir = output_path / f"{diff.value}_{width}x{height}"
            config_dir.mkdir(exist_ok=True)
            
            puzzles_for_config = []
            
            for i in range(num):
                # Determine strategy
                if strategy == 'mixed':
                    # Alternate between strategies
                    strategies = ['random', 'solution_based', 'pattern']
                    current_strategy = strategies[i % len(strategies)]
                else:
                    current_strategy = strategy
                    
                # Generate puzzle
                puzzle = generator.generate(width, height, diff, current_strategy)
                
                if puzzle:
                    # Save puzzle
                    puzzle_id = f"{diff.value}_{width}x{height}_{timestamp}_{i:04d}"
                    puzzle_path = config_dir / f"{puzzle_id}.json"
                    puzzle.save(puzzle_path)
                    
                    puzzles_for_config.append(puzzle)
                    all_puzzles.append((puzzle_id, puzzle))
                    generated_count += 1
                    
                bar.update(1)
                
            # Save batch info
            if puzzles_for_config:
                batch_info = {
                    'difficulty': diff.value,
                    'width': width,
                    'height': height,
                    'count': len(puzzles_for_config),
                    'timestamp': timestamp,
                    'strategy': strategy,
                    'ensure_unique': ensure_unique
                }
                
                info_path = config_dir / f"batch_info_{timestamp}.json"
                with open(info_path, 'w') as f:
                    json.dump(batch_info, f, indent=2)
                    
    click.echo(f"\n\nGeneration complete!")
    click.echo(f"Successfully generated {generated_count}/{total_puzzles} puzzles")
    click.echo(f"Puzzles saved to: {output_path}")
    
    # Create visualizations if requested
    if visualize or create_sheet:
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        viz = PuzzleVisualizer()
        
        if visualize:
            click.echo("\nCreating visualizations...")
            
            # Create sample visualizations (first 5 puzzles of each difficulty)
            samples_by_diff = {}
            for puzzle_id, puzzle in all_puzzles[:20]:  # Limit to first 20
                diff_value = puzzle_id.split('_')[0]
                if diff_value not in samples_by_diff:
                    samples_by_diff[diff_value] = []
                if len(samples_by_diff[diff_value]) < 5:
                    samples_by_diff[diff_value].append(puzzle)
                    
            # Create difficulty showcase
            showcase_path = viz_dir / f"difficulty_showcase_{timestamp}.png"
            viz.create_difficulty_showcase(samples_by_diff, showcase_path)
            click.echo(f"Difficulty showcase saved to: {showcase_path}")
            
        if create_sheet:
            click.echo("\nCreating puzzle sheets...")
            
            # Group puzzles by configuration
            sheets_created = 0
            
            for diff, width, height, _ in generation_tasks:
                config_puzzles = [
                    puzzle for pid, puzzle in all_puzzles
                    if pid.startswith(f"{diff.value}_{width}x{height}")
                ]
                
                if config_puzzles:
                    # Create sheets with 6 puzzles each (2x3 grid)
                    puzzles_per_sheet = 6
                    
                    for sheet_num, i in enumerate(range(0, len(config_puzzles), puzzles_per_sheet)):
                        sheet_puzzles = config_puzzles[i:i+puzzles_per_sheet]
                        
                        if sheet_puzzles:
                            sheet_path = viz_dir / f"puzzle_sheet_{diff.value}_{width}x{height}_{sheet_num+1}.png"
                            viz.create_puzzle_sheet(
                                sheet_puzzles,
                                rows=2,
                                cols=3,
                                save_path=sheet_path
                            )
                            sheets_created += 1
                            
            click.echo(f"Created {sheets_created} puzzle sheets in: {viz_dir}")
            
    # Create summary report
    summary = {
        'timestamp': timestamp,
        'total_generated': generated_count,
        'configurations': [
            {
                'difficulty': diff.value,
                'size': f"{width}x{height}",
                'requested': num,
                'generated': sum(1 for pid, _ in all_puzzles 
                               if pid.startswith(f"{diff.value}_{width}x{height}"))
            }
            for diff, width, height, num in generation_tasks
        ],
        'generator_config': {
            'ensure_unique': ensure_unique,
            'strategy': strategy,
            'seed': seed
        }
    }
    
    summary_path = output_path / f"generation_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    click.echo(f"\nGeneration summary saved to: {summary_path}")
    
    # Display sample puzzles in terminal
    if generated_count > 0 and click.confirm("\nDisplay sample puzzle?"):
        sample_id, sample_puzzle = all_puzzles[0]
        click.echo(f"\nSample puzzle ({sample_id}):")
        click.echo(PuzzleConverter.to_string(sample_puzzle))
        
        # Show statistics
        from src.core.validator import PuzzleValidator
        stats = PuzzleValidator.get_puzzle_statistics(sample_puzzle)
        click.echo(f"\nPuzzle statistics:")
        click.echo(f"  Islands: {stats['num_islands']}")
        click.echo(f"  Average bridges per island: {stats['avg_bridges_per_island']:.1f}")
        click.echo(f"  Density: {stats['density']:.2%}")
        click.echo(f"  Degree distribution: {stats['degree_distribution']}")


if __name__ == '__main__':
    main()