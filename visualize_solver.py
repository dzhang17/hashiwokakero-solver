#!/usr/bin/env python3
"""
Hashiwokakero Solver and Visualizer
Solve and visualize Hashiwokakero puzzles using ILP or LNS solver

Usage:
    python visualize_solver.py <puzzle_file> [--solver {ilp|lns|both}] [--time-limit TIME] [--save]
    
Examples:
    python visualize_solver.py dataset/100/Hs_16_100_25_00_001.has
    python visualize_solver.py dataset/200/Hs_24_200_50_05_010.has --solver lns
    python visualize_solver.py dataset/400/Hs_34_400_75_15_020.has --solver both --time-limit 300
"""

import sys
import os
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.puzzle import Puzzle
from src.core.validator import PuzzleValidator
from src.solvers.ilp_solver import ILPSolver, ILPSolverConfig
from src.solvers.lns_solver import LargeNeighborhoodSearchSolver, LNSSolverConfig


def read_puzzle_file(filename):
    """Read puzzle from .has file format"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split()
    rows, cols, num_islands = int(header[0]), int(header[1]), int(header[2])
    
    # Create puzzle
    puzzle = Puzzle(rows, cols)
    
    # Parse grid and add islands
    for row in range(rows):
        values = list(map(int, lines[row + 1].strip().split()))
        for col in range(cols):
            if values[col] > 0:
                puzzle.add_island(row, col, values[col])
    
    return puzzle


def visualize_solution(puzzle, solution=None, title="Hashiwokakero Puzzle", 
                      solver_stats=None, filename=None, show_grid=True):
    """
    Visualize puzzle and solution with enhanced graphics
    """
    # Create figure with stats panel if solution exists
    if solution and solver_stats:
        fig = plt.figure(figsize=(16, 10))
        ax_main = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        ax_stats = plt.subplot2grid((1, 3), (0, 2))
    else:
        fig, ax_main = plt.subplots(figsize=(12, 10))
        ax_stats = None
    
    # Setup main plot
    ax_main.set_xlim(-0.5, puzzle.width - 0.5)
    ax_main.set_ylim(-0.5, puzzle.height - 0.5)
    ax_main.set_aspect('equal')
    ax_main.invert_yaxis()
    ax_main.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Grid
    if show_grid:
        for i in range(puzzle.width + 1):
            ax_main.axvline(i - 0.5, color='lightgray', alpha=0.3, linewidth=0.5)
        for i in range(puzzle.height + 1):
            ax_main.axhline(i - 0.5, color='lightgray', alpha=0.3, linewidth=0.5)
    
    ax_main.set_xticks(range(puzzle.width))
    ax_main.set_yticks(range(puzzle.height))
    ax_main.set_xlabel('Column', fontsize=12)
    ax_main.set_ylabel('Row', fontsize=12)
    
    # Draw bridges if solution exists
    if solution and solution.bridges:
        # Group bridges by type for better visualization
        single_bridges = []
        double_bridges = []
        
        for bridge in solution.bridges:
            if bridge.count == 1:
                single_bridges.append(bridge)
            else:
                double_bridges.append(bridge)
        
        # Draw single bridges
        for bridge in single_bridges:
            island1 = solution._id_to_island[bridge.island1_id]
            island2 = solution._id_to_island[bridge.island2_id]
            
            if island1.row == island2.row:  # Horizontal
                y = island1.row
                x1, x2 = sorted([island1.col, island2.col])
                ax_main.plot([x1, x2], [y, y], 'steelblue', linewidth=3, 
                           solid_capstyle='round', alpha=0.8, zorder=1)
            else:  # Vertical
                x = island1.col
                y1, y2 = sorted([island1.row, island2.row])
                ax_main.plot([x, x], [y1, y2], 'steelblue', linewidth=3, 
                           solid_capstyle='round', alpha=0.8, zorder=1)
        
        # Draw double bridges
        for bridge in double_bridges:
            island1 = solution._id_to_island[bridge.island1_id]
            island2 = solution._id_to_island[bridge.island2_id]
            
            if island1.row == island2.row:  # Horizontal
                y = island1.row
                x1, x2 = sorted([island1.col, island2.col])
                ax_main.plot([x1, x2], [y - 0.08, y - 0.08], 'darkblue', 
                           linewidth=2.5, solid_capstyle='round', alpha=0.8, zorder=1)
                ax_main.plot([x1, x2], [y + 0.08, y + 0.08], 'darkblue', 
                           linewidth=2.5, solid_capstyle='round', alpha=0.8, zorder=1)
            else:  # Vertical
                x = island1.col
                y1, y2 = sorted([island1.row, island2.row])
                ax_main.plot([x - 0.08, x - 0.08], [y1, y2], 'darkblue', 
                           linewidth=2.5, solid_capstyle='round', alpha=0.8, zorder=1)
                ax_main.plot([x + 0.08, x + 0.08], [y1, y2], 'darkblue', 
                           linewidth=2.5, solid_capstyle='round', alpha=0.8, zorder=1)
    
    # Draw islands
    islands_to_draw = solution.islands if solution else puzzle.islands
    
    for island in islands_to_draw:
        # Check if island requirements are satisfied
        satisfied = False
        current_bridges = 0
        
        if solution:
            current_bridges = solution.get_island_bridges(island.id)
            satisfied = (current_bridges == island.required_bridges)
        
        # Color based on satisfaction
        if solution:
            if satisfied:
                face_color = '#2ECC71'  # Green
                edge_color = '#27AE60'
                text_color = 'white'
            else:
                face_color = '#E74C3C'  # Red
                edge_color = '#C0392B'
                text_color = 'white'
        else:
            face_color = 'white'
            edge_color = 'black'
            text_color = 'black'
        
        # Draw island circle
        circle = patches.Circle((island.col, island.row), 0.4, 
                              facecolor=face_color, edgecolor=edge_color, 
                              linewidth=2.5, zorder=2)
        ax_main.add_patch(circle)
        
        # Add number
        ax_main.text(island.col, island.row, str(island.required_bridges),
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=text_color, zorder=3)
        
        # Add small text showing current bridges if not satisfied
        if solution and not satisfied:
            ax_main.text(island.col + 0.5, island.row - 0.5, f'({current_bridges})',
                        ha='center', va='center', fontsize=9, color='red',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Statistics panel
    if ax_stats and solver_stats:
        ax_stats.axis('off')
        
        # Title
        stats_title = "Solver Statistics"
        ax_stats.text(0.5, 0.95, stats_title, transform=ax_stats.transAxes,
                     fontsize=14, fontweight='bold', ha='center')
        
        # Prepare statistics text
        stats_text = ""
        
        # Basic stats
        if 'solver_name' in solver_stats:
            stats_text += f"Solver: {solver_stats['solver_name']}\n"
        if 'solver_time' in solver_stats:
            stats_text += f"Time: {solver_stats['solver_time']:.2f}s\n"
        if 'iterations' in solver_stats:
            stats_text += f"Iterations: {solver_stats['iterations']}\n"
        if 'improvements' in solver_stats:
            stats_text += f"Improvements: {solver_stats['improvements']}\n"
        
        stats_text += "\n"
        
        # Solution quality
        if solution:
            total_bridges = sum(b.count for b in solution.bridges)
            total_required = sum(i.required_bridges for i in solution.islands) // 2
            satisfied_islands = sum(1 for i in solution.islands 
                                  if solution.get_island_bridges(i.id) == i.required_bridges)
            
            stats_text += f"Islands: {len(solution.islands)}\n"
            stats_text += f"Satisfied: {satisfied_islands}/{len(solution.islands)} "
            stats_text += f"({satisfied_islands/len(solution.islands)*100:.1f}%)\n"
            stats_text += f"Total Bridges: {total_bridges}/{total_required}\n"
            stats_text += f"Connections: {len(solution.bridges)}\n"
            
            # Single vs double bridges
            single_count = sum(1 for b in solution.bridges if b.count == 1)
            double_count = sum(1 for b in solution.bridges if b.count == 2)
            stats_text += f"\nBridge Types:\n"
            stats_text += f"Single: {single_count}\n"
            stats_text += f"Double: {double_count}\n"
        
        # Additional solver-specific stats
        if 'objective_value' in solver_stats:
            stats_text += f"\nObjective: {solver_stats['objective_value']:.2f}\n"
        if 'final_destroy_rate' in solver_stats:
            stats_text += f"Destroy Rate: {solver_stats['final_destroy_rate']:.3f}\n"
        
        # Display stats
        ax_stats.text(0.1, 0.85, stats_text, transform=ax_stats.transAxes,
                     fontfamily='monospace', fontsize=11, verticalalignment='top')
        
        # Add validation status
        if solution:
            validation = PuzzleValidator.validate_solution(solution)
            if validation.is_valid:
                status_text = "✓ Valid Solution"
                status_color = 'green'
            else:
                status_text = "✗ Invalid Solution"
                status_color = 'red'
            
            ax_stats.text(0.5, 0.1, status_text, transform=ax_stats.transAxes,
                         fontsize=16, fontweight='bold', ha='center',
                         color=status_color,
                         bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved visualization to {filename}")
    
    plt.show()


def solve_with_ilp(puzzle, time_limit=300):
    """Solve puzzle using ILP solver"""
    print("\n" + "="*60)
    print("INTEGER LINEAR PROGRAMMING (ILP) SOLVER")
    print("="*60)
    
    config = ILPSolverConfig(
        solver_name='cbc',
        time_limit=time_limit,
        use_lazy_constraints=True,
        use_preprocessing=True,
        verbose=True,
        debug_mode=False,
        solver_options={
            'timelimit': time_limit,
            'ratioGap': 0.01,
        }
    )
    
    solver = ILPSolver(config)
    
    print(f"Solving {puzzle.width}x{puzzle.height} puzzle with {len(puzzle.islands)} islands...")
    print(f"Time limit: {time_limit}s")
    
    start_time = time.time()
    result = solver.solve(puzzle)
    solve_time = time.time() - start_time
    
    # Prepare stats
    stats = {
        'solver_name': 'ILP (CBC)',
        'solver_time': solve_time,
    }
    
    if result.stats:
        stats.update(result.stats)
    
    print(f"\nSolved in {solve_time:.2f}s")
    
    if result.success:
        print("✅ Solution found!")
        validation = PuzzleValidator.validate_solution(result.solution)
        if validation.is_valid:
            print("✅ Valid solution!")
        else:
            print("❌ Invalid solution:")
            for error in validation.errors:
                print(f"  - {error}")
    else:
        print(f"❌ Failed: {result.message}")
    
    return result, stats


def solve_with_lns(puzzle, time_limit=300):
    """Solve puzzle using LNS solver"""
    print("\n" + "="*60)
    print("LARGE NEIGHBORHOOD SEARCH (LNS) SOLVER")
    print("="*60)
    
    config = LNSSolverConfig(
        time_limit=time_limit,
        initial_destroy_rate=0.25,
        min_destroy_rate=0.1,
        max_destroy_rate=0.5,
        destroy_rate_increase=1.1,
        destroy_rate_decrease=0.95,
        repair_time_limit=5.0,
        use_warm_start=True,
        use_parallel_repair=True,
        accept_worse_solutions=True,
        initial_temperature=10.0,
        cooling_rate=0.97,
        max_iterations_without_improvement=100,
        track_statistics=True,
        verbose=True
    )
    
    solver = LargeNeighborhoodSearchSolver(config)
    
    print(f"Solving {puzzle.width}x{puzzle.height} puzzle with {len(puzzle.islands)} islands...")
    print(f"Time limit: {time_limit}s")
    
    start_time = time.time()
    result = solver.solve(puzzle)
    solve_time = time.time() - start_time
    
    # Prepare stats
    stats = {
        'solver_name': 'LNS',
        'solver_time': solve_time,
    }
    
    if result.stats:
        stats.update(result.stats)
    
    print(f"\nSolved in {solve_time:.2f}s")
    
    if result.success:
        print("✅ Solution found!")
        validation = PuzzleValidator.validate_solution(result.solution)
        if validation.is_valid:
            print("✅ Valid solution!")
        else:
            print("❌ Invalid solution:")
            for error in validation.errors:
                print(f"  - {error}")
        
        if result.stats:
            print(f"\nIterations: {result.stats.get('iterations', 'N/A')}")
            print(f"Improvements: {result.stats.get('improvements', 'N/A')}")
    else:
        print(f"❌ Failed: {result.message}")
    
    return result, stats


def compare_solvers(puzzle, time_limit=300):
    """Compare both solvers on the same puzzle"""
    print("\n" + "="*60)
    print("COMPARING ILP AND LNS SOLVERS")
    print("="*60)
    
    results = []
    
    # Solve with ILP
    ilp_result, ilp_stats = solve_with_ilp(puzzle, time_limit)
    results.append(('ILP', ilp_result, ilp_stats))
    
    # Solve with LNS
    lns_result, lns_stats = solve_with_lns(puzzle, time_limit)
    results.append(('LNS', lns_result, lns_stats))
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Solver':<10} {'Success':<10} {'Time (s)':<12} {'Valid':<10} {'Objective':<10}")
    print("-"*60)
    
    for name, result, stats in results:
        success = "Yes" if result.success else "No"
        time_str = f"{stats['solver_time']:.2f}"
        
        if result.success:
            validation = PuzzleValidator.validate_solution(result.solution)
            valid = "Yes" if validation.is_valid else "No"
            obj = stats.get('objective_value', stats.get('final_objective', 'N/A'))
            if isinstance(obj, (int, float)):
                obj_str = f"{obj:.2f}"
            else:
                obj_str = str(obj)
        else:
            valid = "N/A"
            obj_str = "N/A"
        
        print(f"{name:<10} {success:<10} {time_str:<12} {valid:<10} {obj_str:<10}")
    
    return results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Solve and visualize Hashiwokakero puzzles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dataset/100/Hs_16_100_25_00_001.has
  %(prog)s dataset/200/Hs_24_200_50_05_010.has --solver lns
  %(prog)s dataset/400/Hs_34_400_75_15_020.has --solver both --time-limit 300 --save
        """
    )
    
    parser.add_argument('puzzle_file', help='Path to the puzzle file (.has format)')
    parser.add_argument('--solver', choices=['ilp', 'lns', 'both'], default='ilp',
                       help='Which solver to use (default: ilp)')
    parser.add_argument('--time-limit', type=int, default=300,
                       help='Time limit in seconds (default: 300)')
    parser.add_argument('--save', action='store_true',
                       help='Save visualizations to files')
    parser.add_argument('--no-grid', action='store_true',
                       help='Hide grid lines in visualization')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.puzzle_file):
        print(f"Error: File '{args.puzzle_file}' not found!")
        return 1
    
    # Read puzzle
    try:
        puzzle = read_puzzle_file(args.puzzle_file)
    except Exception as e:
        print(f"Error reading puzzle file: {e}")
        return 1
    
    # Get puzzle name for saving
    puzzle_name = Path(args.puzzle_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print puzzle info
    print(f"\nPuzzle: {args.puzzle_file}")
    print(f"Size: {puzzle.width}x{puzzle.height}")
    print(f"Islands: {len(puzzle.islands)}")
    print(f"Total required bridges: {sum(island.required_bridges for island in puzzle.islands) // 2}")
    
    # Visualize original puzzle
    original_filename = None
    if args.save:
        original_filename = f"{puzzle_name}_original_{timestamp}.png"
    
    print("\nVisualizing original puzzle...")
    visualize_solution(
        puzzle, 
        solution=None,
        title=f"Original Puzzle: {puzzle_name}",
        filename=original_filename,
        show_grid=not args.no_grid
    )
    
    # Solve based on solver choice
    if args.solver == 'both':
        results = compare_solvers(puzzle, args.time_limit)
        
        # Visualize both solutions
        for name, result, stats in results:
            if result.success:
                solution_filename = None
                if args.save:
                    solution_filename = f"{puzzle_name}_{name.lower()}_solution_{timestamp}.png"
                
                visualize_solution(
                    puzzle,
                    solution=result.solution,
                    title=f"{name} Solution: {puzzle_name}",
                    solver_stats=stats,
                    filename=solution_filename,
                    show_grid=not args.no_grid
                )
    
    else:
        # Single solver
        if args.solver == 'ilp':
            result, stats = solve_with_ilp(puzzle, args.time_limit)
        else:  # lns
            result, stats = solve_with_lns(puzzle, args.time_limit)
        
        if result.success:
            solution_filename = None
            if args.save:
                solution_filename = f"{puzzle_name}_{args.solver}_solution_{timestamp}.png"
            
            print(f"\nVisualizing {args.solver.upper()} solution...")
            visualize_solution(
                puzzle,
                solution=result.solution,
                title=f"{args.solver.upper()} Solution: {puzzle_name}",
                solver_stats=stats,
                filename=solution_filename,
                show_grid=not args.no_grid
            )
        else:
            print("\nNo solution to visualize.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())