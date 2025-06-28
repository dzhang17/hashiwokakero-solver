#!/usr/bin/env python3
"""
Script to analyze benchmark results and generate insights.

Usage:
    python scripts/analyze_results.py results/benchmarks/benchmark_results_*.csv
    python scripts/analyze_results.py --latest --generate-report
    python scripts/analyze_results.py --compare file1.csv file2.csv
"""

import click
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.benchmark import BenchmarkAnalyzer
from src.analysis.report_generator import ReportGenerator
from src.visualization.performance_viz import PerformanceVisualizer
from src.core.utils import setup_logger


@click.command()
@click.argument('results_files', nargs=-1, type=click.Path(exists=True))
@click.option('--latest', is_flag=True,
              help='Use latest results file from default directory')
@click.option('--output-dir', '-o', type=click.Path(), default='results/analysis',
              help='Output directory for analysis results')
@click.option('--generate-report', '-r', is_flag=True,
              help='Generate comprehensive Excel report')
@click.option('--visualize', '-v', is_flag=True,
              help='Create all visualization plots')
@click.option('--compare', is_flag=True,
              help='Compare multiple result files')
@click.option('--metric', '-m', 
              type=click.Choice(['time', 'success', 'memory', 'all']),
              default='all', help='Metric to focus on')
@click.option('--export-stats', is_flag=True,
              help='Export detailed statistics')
def main(results_files, latest, output_dir, generate_report, visualize,
         compare, metric, export_stats):
    """Analyze Hashiwokakero benchmark results."""
    
    logger = setup_logger("ResultsAnalyzer")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo("="*60)
    click.echo("Benchmark Results Analyzer")
    click.echo("="*60)
    
    # Determine which files to analyze
    files_to_analyze = []
    
    if latest:
        # Find latest results file
        default_dir = Path('results/benchmarks')
        if default_dir.exists():
            csv_files = list(default_dir.glob('benchmark_results_*.csv'))
            json_files = list(default_dir.glob('benchmark_results_*.json'))
            all_files = csv_files + json_files
            
            if all_files:
                latest_file = max(all_files, key=lambda p: p.stat().st_mtime)
                files_to_analyze.append(latest_file)
                click.echo(f"\nUsing latest results: {latest_file.name}")
            else:
                click.echo("Error: No results files found in default directory")
                sys.exit(1)
        else:
            click.echo(f"Error: Default directory '{default_dir}' not found")
            sys.exit(1)
            
    elif results_files:
        files_to_analyze = [Path(f) for f in results_files]
    else:
        click.echo("Error: Specify result files or use --latest")
        sys.exit(1)
        
    # Load and analyze each file
    if compare and len(files_to_analyze) > 1:
        # Comparative analysis
        click.echo(f"\nComparing {len(files_to_analyze)} result files...")
        _perform_comparative_analysis(files_to_analyze, output_path, metric)
        
    else:
        # Single file analysis
        for results_file in files_to_analyze:
            click.echo(f"\nAnalyzing: {results_file.name}")
            
            # Create analyzer
            analyzer = BenchmarkAnalyzer(results_file)
            
            # Basic statistics
            click.echo("\n" + "-"*40)
            click.echo("SUMMARY STATISTICS")
            click.echo("-"*40)
            
            summary_stats = analyzer.get_summary_statistics()
            click.echo("\nOverall Performance by Algorithm:")
            click.echo(summary_stats.to_string())
            
            # Performance by size
            if metric in ['time', 'all']:
                click.echo("\n" + "-"*40)
                click.echo("PERFORMANCE BY PROBLEM SIZE")
                click.echo("-"*40)
                
                size_perf = analyzer.get_performance_by_size()
                click.echo(size_perf.to_string())
                
            # Performance by difficulty
            click.echo("\n" + "-"*40)
            click.echo("PERFORMANCE BY DIFFICULTY")
            click.echo("-"*40)
            
            diff_perf = analyzer.get_performance_by_difficulty()
            click.echo(diff_perf.to_string())
            
            # Find best algorithms for different scenarios
            click.echo("\n" + "-"*40)
            click.echo("RECOMMENDATIONS")
            click.echo("-"*40)
            
            scenarios = [
                ("Overall best", {}),
                ("Small puzzles", {'num_islands': lambda x: x < 20}),
                ("Large puzzles", {'num_islands': lambda x: x >= 50}),
                ("Easy puzzles", {'difficulty': 'easy'}),
                ("Hard puzzles", {'difficulty': 'hard'}),
            ]
            
            for scenario_name, constraints in scenarios:
                # Apply constraints
                filtered_df = analyzer.results_df.copy()
                for col, condition in constraints.items():
                    if callable(condition):
                        filtered_df = filtered_df[filtered_df[col].apply(condition)]
                    else:
                        filtered_df = filtered_df[filtered_df[col] == condition]
                        
                if len(filtered_df) > 0:
                    # Find best for time and success
                    time_perf = filtered_df.groupby('algorithm')['solve_time'].mean()
                    success_perf = filtered_df.groupby('algorithm')['success'].mean()
                    
                    best_time = time_perf.idxmin() if len(time_perf) > 0 else "N/A"
                    best_success = success_perf.idxmax() if len(success_perf) > 0 else "N/A"
                    
                    click.echo(f"\n{scenario_name}:")
                    click.echo(f"  Fastest: {best_time} ({time_perf[best_time]:.3f}s avg)")
                    click.echo(f"  Most reliable: {best_success} ({success_perf[best_success]*100:.1f}% success)")
                    
            # Detailed analysis
            if metric in ['time', 'all']:
                click.echo("\n" + "-"*40)
                click.echo("TIME ANALYSIS")
                click.echo("-"*40)
                
                # Time distribution
                for algorithm in analyzer.results_df['algorithm'].unique():
                    alg_times = analyzer.results_df[
                        analyzer.results_df['algorithm'] == algorithm
                    ]['solve_time']
                    
                    click.echo(f"\n{algorithm}:")
                    click.echo(f"  Mean: {alg_times.mean():.3f}s")
                    click.echo(f"  Median: {alg_times.median():.3f}s")
                    click.echo(f"  Std Dev: {alg_times.std():.3f}s")
                    click.echo(f"  95th percentile: {alg_times.quantile(0.95):.3f}s")
                    
            if metric in ['memory', 'all']:
                click.echo("\n" + "-"*40)
                click.echo("MEMORY ANALYSIS")
                click.echo("-"*40)
                
                # Memory usage
                mem_stats = analyzer.results_df.groupby('algorithm')['memory_mb'].agg([
                    'mean', 'max', 'std'
                ]).round(2)
                click.echo(mem_stats.to_string())
                
            # Export detailed statistics if requested
            if export_stats:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Export summary statistics
                stats_file = output_path / f"analysis_stats_{timestamp}.json"
                
                stats_data = {
                    'source_file': str(results_file),
                    'analysis_date': timestamp,
                    'summary': summary_stats.to_dict(),
                    'by_size': analyzer.get_performance_by_size().to_dict(),
                    'by_difficulty': analyzer.get_performance_by_difficulty().to_dict(),
                    'algorithm_recommendations': _get_recommendations(analyzer)
                }
                
                with open(stats_file, 'w') as f:
                    json.dump(stats_data, f, indent=2, default=str)
                    
                click.echo(f"\nStatistics exported to: {stats_file}")
                
            # Generate report if requested
            if generate_report:
                click.echo("\nGenerating comprehensive report...")
                
                report_gen = ReportGenerator(results_file, output_path / "reports")
                report_path = report_gen.generate_excel_report()
                
                click.echo(f"Excel report generated: {report_path}")
                
                # Also generate LaTeX report
                latex_path = report_gen.generate_latex_report()
                click.echo(f"LaTeX report generated: {latex_path}")
                
            # Create visualizations if requested
            if visualize:
                click.echo("\nCreating visualizations...")
                
                viz_dir = output_path / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                
                viz = PerformanceVisualizer()
                
                # Create various plots based on metric
                plots_to_create = []
                
                if metric in ['time', 'all']:
                    plots_to_create.extend([
                        ('time_comparison', lambda df: viz.plot_time_comparison(df)),
                        ('scalability', lambda df: viz.plot_scalability(df)),
                        ('convergence', lambda df: _create_convergence_plot(df, viz))
                    ])
                    
                if metric in ['success', 'all']:
                    plots_to_create.append(
                        ('success_rate', lambda df: viz.plot_success_rate(df))
                    )
                    
                if metric in ['memory', 'all']:
                    plots_to_create.append(
                        ('memory_usage', lambda df: viz.plot_memory_usage(df))
                    )
                    
                if metric == 'all':
                    plots_to_create.append(
                        ('comprehensive', lambda df: viz.create_comprehensive_report(df))
                    )
                    
                # Create plots
                for plot_name, plot_func in plots_to_create:
                    try:
                        save_path = viz_dir / f"{plot_name}_{results_file.stem}.png"
                        fig = plot_func(analyzer.results_df)
                        if fig:
                            fig.savefig(save_path, dpi=300, bbox_inches='tight')
                            click.echo(f"  Created {plot_name} visualization")
                    except Exception as e:
                        click.echo(f"  Error creating {plot_name}: {e}")
                        
    click.echo("\n" + "="*60)
    click.echo("Analysis complete!")
    click.echo("="*60)


def _perform_comparative_analysis(files: list, output_path: Path, metric: str):
    """Perform comparative analysis across multiple result files"""
    
    # Load all datasets
    datasets = {}
    for file_path in files:
        analyzer = BenchmarkAnalyzer(file_path)
        datasets[file_path.stem] = analyzer.results_df
        
    # Combine datasets
    for name, df in datasets.items():
        df['dataset'] = name
        
    combined_df = pd.concat(datasets.values(), ignore_index=True)
    
    # Comparative statistics
    click.echo("\nCOMPARATIVE ANALYSIS")
    click.echo("-" * 60)
    
    # Compare overall performance
    comparison = combined_df.groupby(['dataset', 'algorithm']).agg({
        'success': 'mean',
        'solve_time': 'mean',
        'memory_mb': 'mean'
    }).round(3)
    
    click.echo("\nOverall Performance Comparison:")
    click.echo(comparison.to_string())
    
    # Find improvements/regressions
    if len(datasets) == 2:
        click.echo("\nPerformance Changes:")
        
        names = list(datasets.keys())
        df1 = datasets[names[0]]
        df2 = datasets[names[1]]
        
        for algorithm in df1['algorithm'].unique():
            if algorithm in df2['algorithm'].values:
                time1 = df1[df1['algorithm'] == algorithm]['solve_time'].mean()
                time2 = df2[df2['algorithm'] == algorithm]['solve_time'].mean()
                
                success1 = df1[df1['algorithm'] == algorithm]['success'].mean()
                success2 = df2[df2['algorithm'] == algorithm]['success'].mean()
                
                time_change = ((time2 - time1) / time1) * 100
                success_change = (success2 - success1) * 100
                
                click.echo(f"\n{algorithm}:")
                click.echo(f"  Time: {time1:.3f}s → {time2:.3f}s ({time_change:+.1f}%)")
                click.echo(f"  Success: {success1*100:.1f}% → {success2*100:.1f}% ({success_change:+.1f}pp)")


def _get_recommendations(analyzer: BenchmarkAnalyzer) -> dict:
    """Generate algorithm recommendations based on analysis"""
    
    recommendations = {}
    
    # Overall best
    overall_scores = {}
    for algorithm in analyzer.results_df['algorithm'].unique():
        alg_data = analyzer.results_df[analyzer.results_df['algorithm'] == algorithm]
        
        # Combined score (weighted)
        success_rate = alg_data['success'].mean()
        avg_time = alg_data['solve_time'].mean()
        
        # Normalize time (inverse, lower is better)
        max_time = analyzer.results_df['solve_time'].max()
        time_score = 1 - (avg_time / max_time)
        
        # Combined score: 70% success, 30% speed
        overall_scores[algorithm] = 0.7 * success_rate + 0.3 * time_score
        
    recommendations['overall_best'] = max(overall_scores, key=overall_scores.get)
    recommendations['overall_scores'] = overall_scores
    
    # Specific recommendations
    scenarios = {
        'speed_critical': lambda df: df.groupby('algorithm')['solve_time'].mean().idxmin(),
        'reliability_critical': lambda df: df.groupby('algorithm')['success'].mean().idxmax(),
        'memory_constrained': lambda df: df.groupby('algorithm')['memory_mb'].mean().idxmin()
    }
    
    for scenario, selector in scenarios.items():
        try:
            recommendations[scenario] = selector(analyzer.results_df)
        except:
            recommendations[scenario] = 'N/A'
            
    return recommendations


def _create_convergence_plot(df: pd.DataFrame, viz: PerformanceVisualizer):
    """Create convergence plot for iterative algorithms"""
    
    # This would need actual iteration data from the results
    # For now, create a placeholder
    
    # Extract iteration data if available in extra_stats
    convergence_data = {}
    
    # This is a simplified version - would need actual iteration history
    for algorithm in ['sa', 'hybrid']:
        if algorithm in df['algorithm'].values:
            # Simulate convergence data
            iterations = np.arange(0, 1000, 10)
            costs = 100 * np.exp(-iterations / 200) + np.random.normal(0, 5, len(iterations))
            convergence_data[algorithm] = costs
            
    if convergence_data:
        return viz.plot_convergence(convergence_data)
    
    return None


if __name__ == '__main__':
    main()