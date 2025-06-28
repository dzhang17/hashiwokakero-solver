#!/usr/bin/env python3
"""
Generate all figures and tables needed for the thesis
Compares ILP and LNS algorithms across all dataset sizes (100, 200, 300, 400)
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import csv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.puzzle import Puzzle
from src.solvers.ilp_solver import ILPSolver, ILPSolverConfig
from src.solvers.lns_solver import LNSSolver, LNSSolverConfig
from src.core.validator import PuzzleValidator


def read_puzzle_file(filename):
    """Read puzzle from .has file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split()
    rows, cols, num_islands = int(header[0]), int(header[1]), int(header[2])
    
    puzzle = Puzzle(rows, cols)
    
    for row in range(rows):
        values = list(map(int, lines[row + 1].strip().split()))
        for col in range(cols):
            if values[col] > 0:
                puzzle.add_island(row, col, values[col])
    
    return puzzle


def solve_with_ilp(puzzle, timeout=300):
    """Solve puzzle with ILP"""
    config = ILPSolverConfig(
        solver_name='cbc',
        use_lazy_constraints=True,
        use_preprocessing=True,
        time_limit=timeout,
        verbose=False,
        solver_options={
            'timelimit': timeout,
            'ratioGap': 0.01,
        }
    )
    
    solver = ILPSolver(config)
    start_time = time.time()
    result = solver.solve(puzzle.copy())
    solve_time = time.time() - start_time
    
    return {
        'time': solve_time,
        'success': result.success,
        'solution': result.solution if result.success else None,
        'stats': result.stats if hasattr(result, 'stats') else {}
    }


def solve_with_lns(puzzle, timeout=300):
    """Solve puzzle with LNS"""
    config = LNSSolverConfig(
        time_limit=timeout,
        initial_destroy_rate=0.25,
        min_destroy_rate=0.1,
        max_destroy_rate=0.5,
        repair_time_limit=5.0,
        use_warm_start=True,
        verbose=False
    )
    
    solver = LNSSolver(config)
    start_time = time.time()
    result = solver.solve(puzzle.copy())
    solve_time = time.time() - start_time
    
    return {
        'time': solve_time,
        'success': result.success,
        'solution': result.solution if result.success else None,
        'stats': result.stats if hasattr(result, 'stats') else {}
    }


def run_complete_benchmark():
    """Run benchmark on all datasets"""
    sizes = [100, 200, 300, 400]
    results = []
    
    print("="*80)
    print("RUNNING COMPLETE BENCHMARK FOR THESIS")
    print("="*80)
    
    for size in sizes:
        dataset_path = Path(f"dataset/{size}")
        if not dataset_path.exists():
            print(f"Warning: Dataset path {dataset_path} not found!")
            continue
        
        # Get all .has files
        has_files = sorted(dataset_path.glob("*.has"))
        print(f"\nProcessing {size}-island puzzles: {len(has_files)} files")
        
        for i, has_file in enumerate(has_files):
            print(f"  [{i+1}/{len(has_files)}] {has_file.name}...", end='', flush=True)
            
            try:
                # Load puzzle
                puzzle = read_puzzle_file(has_file)
                
                # Solve with ILP
                ilp_result = solve_with_ilp(puzzle, timeout=300)
                
                # Solve with LNS
                lns_result = solve_with_lns(puzzle, timeout=300)
                
                # Record results
                results.append({
                    'size': size,
                    'file': has_file.name,
                    'islands': len(puzzle.islands),
                    'ilp_time': ilp_result['time'],
                    'ilp_success': ilp_result['success'],
                    'lns_time': lns_result['time'],
                    'lns_success': lns_result['success'],
                })
                
                print(f" ILP: {ilp_result['time']:.2f}s ({ilp_result['success']}), "
                      f"LNS: {lns_result['time']:.2f}s ({lns_result['success']})")
                
            except Exception as e:
                print(f" ERROR: {e}")
                results.append({
                    'size': size,
                    'file': has_file.name,
                    'islands': 0,
                    'ilp_time': None,
                    'ilp_success': False,
                    'lns_time': None,
                    'lns_success': False,
                })
    
    return pd.DataFrame(results)


def generate_all_figures(df):
    """Generate all figures for thesis"""
    print("\n" + "="*80)
    print("GENERATING THESIS FIGURES")
    print("="*80)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = Path("thesis_figures")
    output_dir.mkdir(exist_ok=True)
    
    # Figure 1: Success Rate Comparison
    fig1_success_rate_comparison(df, output_dir)
    
    # Figure 2: Average Solving Time by Problem Size
    fig2_avg_time_by_size(df, output_dir)
    
    # Figure 3: Time Distribution Box Plots
    fig3_time_distribution(df, output_dir)
    
    # Figure 4: Speedup Analysis
    fig4_speedup_analysis(df, output_dir)
    
    # Figure 5: Scalability Analysis
    fig5_scalability_analysis(df, output_dir)
    
    # Figure 6: Performance Profile
    fig6_performance_profile(df, output_dir)
    
    # Figure 7: Success vs Time Trade-off
    fig7_success_time_tradeoff(df, output_dir)
    
    # Figure 8: Detailed Comparison Table
    fig8_detailed_comparison_table(df, output_dir)
    
    # Generate LaTeX tables
    generate_latex_tables(df, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")


def fig1_success_rate_comparison(df, output_dir):
    """Figure 1: Success rate comparison bar chart"""
    plt.figure(figsize=(10, 6))
    
    # Calculate success rates
    success_rates = df.groupby('size').agg({
        'ilp_success': 'mean',
        'lns_success': 'mean'
    }) * 100
    
    # Create bar plot
    x = np.arange(len(success_rates))
    width = 0.35
    
    plt.bar(x - width/2, success_rates['ilp_success'], width, label='ILP', color='steelblue', alpha=0.8)
    plt.bar(x + width/2, success_rates['lns_success'], width, label='LNS', color='darkorange', alpha=0.8)
    
    plt.xlabel('Problem Size (Number of Islands)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Success Rate Comparison: ILP vs LNS', fontsize=14, fontweight='bold')
    plt.xticks(x, success_rates.index)
    plt.legend(fontsize=12)
    plt.ylim(0, 105)
    
    # Add value labels
    for i, (ilp, lns) in enumerate(success_rates.values):
        plt.text(i - width/2, ilp + 1, f'{ilp:.1f}%', ha='center', va='bottom')
        plt.text(i + width/2, lns + 1, f'{lns:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Success Rate Comparison")


def fig2_avg_time_by_size(df, output_dir):
    """Figure 2: Average solving time by problem size"""
    plt.figure(figsize=(10, 6))
    
    # Calculate average times (only for successful solutions)
    avg_times = df[df['ilp_success'] | df['lns_success']].groupby('size').agg({
        'ilp_time': lambda x: x[df.loc[x.index, 'ilp_success']].mean(),
        'lns_time': lambda x: x[df.loc[x.index, 'lns_success']].mean()
    })
    
    # Create line plot
    sizes = avg_times.index
    plt.plot(sizes, avg_times['ilp_time'], 'o-', label='ILP', markersize=10, linewidth=2, color='steelblue')
    plt.plot(sizes, avg_times['lns_time'], 's-', label='LNS', markersize=10, linewidth=2, color='darkorange')
    
    plt.xlabel('Problem Size (Number of Islands)', fontsize=12)
    plt.ylabel('Average Solving Time (seconds)', fontsize=12)
    plt.title('Average Solving Time vs Problem Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Log scale for y-axis if needed
    if avg_times.max().max() / avg_times.min().min() > 100:
        plt.yscale('log')
        plt.ylabel('Average Solving Time (seconds, log scale)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_avg_time_by_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Average Solving Time by Size")


def fig3_time_distribution(df, output_dir):
    """Figure 3: Time distribution box plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    sizes = [100, 200, 300, 400]
    
    for i, size in enumerate(sizes):
        size_df = df[df['size'] == size]
        
        # Prepare data for box plot
        ilp_times = size_df[size_df['ilp_success']]['ilp_time'].values
        lns_times = size_df[size_df['lns_success']]['lns_time'].values
        
        data_to_plot = []
        labels = []
        
        if len(ilp_times) > 0:
            data_to_plot.append(ilp_times)
            labels.append('ILP')
        
        if len(lns_times) > 0:
            data_to_plot.append(lns_times)
            labels.append('LNS')
        
        if data_to_plot:
            bp = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['steelblue', 'darkorange']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            axes[i].set_ylabel('Time (seconds)')
            axes[i].set_title(f'{size} Islands')
            axes[i].grid(True, alpha=0.3)
            
            # Log scale if needed
            if len(data_to_plot) > 0 and max(max(d) for d in data_to_plot) / min(min(d) for d in data_to_plot) > 100:
                axes[i].set_yscale('log')
    
    plt.suptitle('Solving Time Distribution by Problem Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Time Distribution Box Plots")


def fig4_speedup_analysis(df, output_dir):
    """Figure 4: Speedup analysis (ILP time / LNS time)"""
    plt.figure(figsize=(10, 6))
    
    # Calculate speedup for instances solved by both
    both_solved = df[df['ilp_success'] & df['lns_success']].copy()
    both_solved['speedup'] = both_solved['ilp_time'] / both_solved['lns_time']
    
    # Group by size
    speedup_stats = both_solved.groupby('size')['speedup'].agg(['mean', 'std', 'count'])
    
    # Bar plot with error bars
    x = np.arange(len(speedup_stats))
    plt.bar(x, speedup_stats['mean'], yerr=speedup_stats['std'], 
            capsize=5, color='green', alpha=0.7)
    
    # Add horizontal line at speedup = 1
    plt.axhline(y=1, color='red', linestyle='--', label='Equal Performance')
    
    plt.xlabel('Problem Size (Number of Islands)', fontsize=12)
    plt.ylabel('Speedup Factor (ILP time / LNS time)', fontsize=12)
    plt.title('LNS Speedup over ILP', fontsize=14, fontweight='bold')
    plt.xticks(x, speedup_stats.index)
    plt.legend()
    
    # Add value labels
    for i, (mean, std, count) in enumerate(speedup_stats.values):
        plt.text(i, mean + std + 0.1, f'{mean:.2f}x\n(n={count})', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Speedup Analysis")


def fig5_scalability_analysis(df, output_dir):
    """Figure 5: Scalability analysis with trend lines"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    sizes = sorted(df['size'].unique())
    ilp_avg_times = []
    lns_avg_times = []
    
    for size in sizes:
        size_df = df[df['size'] == size]
        ilp_avg = size_df[size_df['ilp_success']]['ilp_time'].mean()
        lns_avg = size_df[size_df['lns_success']]['lns_time'].mean()
        ilp_avg_times.append(ilp_avg)
        lns_avg_times.append(lns_avg)
    
    # Plot data points
    plt.scatter(sizes, ilp_avg_times, s=100, color='steelblue', label='ILP', zorder=3)
    plt.scatter(sizes, lns_avg_times, s=100, color='darkorange', label='LNS', zorder=3)
    
    # Fit polynomial trends
    if len(sizes) >= 3:
        # ILP trend (potentially exponential)
        z_ilp = np.polyfit(sizes, np.log(ilp_avg_times), 1)
        p_ilp = np.poly1d(z_ilp)
        x_smooth = np.linspace(min(sizes), max(sizes), 100)
        plt.plot(x_smooth, np.exp(p_ilp(x_smooth)), '--', color='steelblue', alpha=0.8, label='ILP Trend')
        
        # LNS trend (potentially polynomial)
        z_lns = np.polyfit(sizes, lns_avg_times, 2)
        p_lns = np.poly1d(z_lns)
        plt.plot(x_smooth, p_lns(x_smooth), '--', color='darkorange', alpha=0.8, label='LNS Trend')
    
    plt.xlabel('Problem Size (Number of Islands)', fontsize=12)
    plt.ylabel('Average Solving Time (seconds)', fontsize=12)
    plt.title('Scalability Analysis: Algorithm Performance vs Problem Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Scalability Analysis")


def fig6_performance_profile(df, output_dir):
    """Figure 6: Performance profile"""
    plt.figure(figsize=(10, 6))
    
    # Calculate performance ratios
    both_solved = df[df['ilp_success'] & df['lns_success']].copy()
    
    if len(both_solved) > 0:
        # Calculate ratios
        both_solved['ilp_ratio'] = both_solved['ilp_time'] / both_solved[['ilp_time', 'lns_time']].min(axis=1)
        both_solved['lns_ratio'] = both_solved['lns_time'] / both_solved[['ilp_time', 'lns_time']].min(axis=1)
        
        # Create performance profile
        tau_values = np.logspace(0, 2, 100)  # From 1 to 100
        ilp_profile = []
        lns_profile = []
        
        for tau in tau_values:
            ilp_profile.append((both_solved['ilp_ratio'] <= tau).mean())
            lns_profile.append((both_solved['lns_ratio'] <= tau).mean())
        
        plt.plot(tau_values, ilp_profile, label='ILP', linewidth=2, color='steelblue')
        plt.plot(tau_values, lns_profile, label='LNS', linewidth=2, color='darkorange')
        
        plt.xlabel('Performance Ratio τ', fontsize=12)
        plt.ylabel('Fraction of Problems Solved', fontsize=12)
        plt.title('Performance Profile: ILP vs LNS', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.xlim(1, 100)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_performance_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Performance Profile")


def fig7_success_time_tradeoff(df, output_dir):
    """Figure 7: Success rate vs average time trade-off"""
    plt.figure(figsize=(10, 6))
    
    # Calculate metrics by size
    metrics = []
    for size in sorted(df['size'].unique()):
        size_df = df[df['size'] == size]
        
        ilp_success_rate = size_df['ilp_success'].mean() * 100
        ilp_avg_time = size_df[size_df['ilp_success']]['ilp_time'].mean()
        
        lns_success_rate = size_df['lns_success'].mean() * 100
        lns_avg_time = size_df[size_df['lns_success']]['lns_time'].mean()
        
        metrics.append({
            'size': size,
            'ilp_success': ilp_success_rate,
            'ilp_time': ilp_avg_time,
            'lns_success': lns_success_rate,
            'lns_time': lns_avg_time
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create scatter plot
    sizes = metrics_df['size'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    for i, size in enumerate(sizes):
        row = metrics_df[metrics_df['size'] == size].iloc[0]
        plt.scatter(row['ilp_time'], row['ilp_success'], s=200, c=[colors[i]], 
                   marker='o', edgecolors='black', linewidth=2, label=f'ILP-{size}')
        plt.scatter(row['lns_time'], row['lns_success'], s=200, c=[colors[i]], 
                   marker='s', edgecolors='black', linewidth=2, label=f'LNS-{size}')
    
    plt.xlabel('Average Solving Time (seconds)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Trade-off: Success Rate vs Solving Time', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Create custom legend
    ilp_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                           markersize=10, markeredgecolor='black', markeredgewidth=2, label='ILP')
    lns_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                           markersize=10, markeredgecolor='black', markeredgewidth=2, label='LNS')
    plt.legend(handles=[ilp_marker, lns_marker], loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_success_time_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Success vs Time Trade-off")


def fig8_detailed_comparison_table(df, output_dir):
    """Figure 8: Detailed comparison table as heatmap"""
    # Prepare summary statistics
    summary_data = []
    
    for size in sorted(df['size'].unique()):
        size_df = df[df['size'] == size]
        
        summary_data.append({
            'Size': size,
            'ILP Success Rate': f"{size_df['ilp_success'].mean() * 100:.1f}%",
            'LNS Success Rate': f"{size_df['lns_success'].mean() * 100:.1f}%",
            'ILP Avg Time': f"{size_df[size_df['ilp_success']]['ilp_time'].mean():.2f}s",
            'LNS Avg Time': f"{size_df[size_df['lns_success']]['lns_time'].mean():.2f}s",
            'Speedup': f"{size_df[size_df['ilp_success'] & size_df['lns_success']]['ilp_time'].mean() / size_df[size_df['ilp_success'] & size_df['lns_success']]['lns_time'].mean():.2f}x",
            'Total Instances': len(size_df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Detailed Performance Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_detailed_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Detailed Comparison Table")


def generate_latex_tables(df, output_dir):
    """Generate LaTeX tables for thesis"""
    # Summary statistics table
    latex_summary = "\\begin{table}[htbp]\n"
    latex_summary += "\\centering\n"
    latex_summary += "\\caption{Performance Comparison of ILP and LNS Algorithms}\n"
    latex_summary += "\\label{tab:performance_comparison}\n"
    latex_summary += "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    latex_summary += "\\hline\n"
    latex_summary += "Size & \\multicolumn{2}{c|}{Success Rate (\\%)} & \\multicolumn{2}{c|}{Avg. Time (s)} & Speedup & Instances \\\\\n"
    latex_summary += "\\cline{2-5}\n"
    latex_summary += " & ILP & LNS & ILP & LNS & & \\\\\n"
    latex_summary += "\\hline\n"
    
    for size in sorted(df['size'].unique()):
        size_df = df[df['size'] == size]
        ilp_success = size_df['ilp_success'].mean() * 100
        lns_success = size_df['lns_success'].mean() * 100
        ilp_time = size_df[size_df['ilp_success']]['ilp_time'].mean()
        lns_time = size_df[size_df['lns_success']]['lns_time'].mean()
        
        both_solved = size_df[size_df['ilp_success'] & size_df['lns_success']]
        if len(both_solved) > 0:
            speedup = both_solved['ilp_time'].mean() / both_solved['lns_time'].mean()
        else:
            speedup = 0
        
        latex_summary += f"{size} & {ilp_success:.1f} & {lns_success:.1f} & "
        latex_summary += f"{ilp_time:.2f} & {lns_time:.2f} & {speedup:.2f}x & {len(size_df)} \\\\\n"
    
    latex_summary += "\\hline\n"
    latex_summary += "\\end{tabular}\n"
    latex_summary += "\\end{table}\n"
    
    # Save LaTeX table
    with open(output_dir / 'latex_tables.tex', 'w') as f:
        f.write(latex_summary)
    
    print("✓ LaTeX Tables Generated")


def save_detailed_results(df, output_dir):
    """Save detailed results to CSV"""
    df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # Also save summary statistics
    summary = []
    for size in sorted(df['size'].unique()):
        size_df = df[df['size'] == size]
        summary.append({
            'size': size,
            'total_instances': len(size_df),
            'ilp_success_count': size_df['ilp_success'].sum(),
            'ilp_success_rate': size_df['ilp_success'].mean(),
            'lns_success_count': size_df['lns_success'].sum(),
            'lns_success_rate': size_df['lns_success'].mean(),
            'ilp_avg_time': size_df[size_df['ilp_success']]['ilp_time'].mean(),
            'ilp_std_time': size_df[size_df['ilp_success']]['ilp_time'].std(),
            'lns_avg_time': size_df[size_df['lns_success']]['lns_time'].mean(),
            'lns_std_time': size_df[size_df['lns_success']]['lns_time'].std(),
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    
    print("✓ Detailed Results Saved")


def main():
    """Main function"""
    print("THESIS FIGURE GENERATION SCRIPT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run complete benchmark
    results_df = run_complete_benchmark()
    
    # Generate all figures
    generate_all_figures(results_df)
    
    # Save detailed results
    output_dir = Path("thesis_figures")
    save_detailed_results(results_df, output_dir)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Print summary statistics
    for size in sorted(results_df['size'].unique()):
        size_df = results_df[results_df['size'] == size]
        print(f"\n{size} Islands:")
        print(f"  Total instances: {len(size_df)}")
        print(f"  ILP success rate: {size_df['ilp_success'].mean() * 100:.1f}%")
        print(f"  LNS success rate: {size_df['lns_success'].mean() * 100:.1f}%")
        
        if size_df['ilp_success'].any():
            print(f"  ILP avg time: {size_df[size_df['ilp_success']]['ilp_time'].mean():.2f}s")
        if size_df['lns_success'].any():
            print(f"  LNS avg time: {size_df[size_df['lns_success']]['lns_time'].mean():.2f}s")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll results saved to: thesis_figures/")


if __name__ == "__main__":
    main()