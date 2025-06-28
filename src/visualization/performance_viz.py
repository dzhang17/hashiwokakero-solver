"""
Performance visualization for solver benchmarks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


class PerformanceVisualizer:
    """Visualize solver performance metrics"""
    
    def __init__(self, style: str = 'whitegrid', 
                 palette: str = 'husl',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize performance visualizer.
        
        Args:
            style: Seaborn style
            palette: Color palette
            figsize: Default figure size
            dpi: DPI for saved figures
        """
        sns.set_style(style)
        self.palette = sns.color_palette(palette)
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_time_comparison(self, results_df: pd.DataFrame,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot solving time comparison across algorithms.
        
        Args:
            results_df: DataFrame with columns: algorithm, size, time
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Box plot of solving times
        sns.boxplot(data=results_df, x='algorithm', y='time', 
                   hue='size', ax=ax, palette=self.palette)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Solving Time (seconds)', fontsize=12)
        ax.set_title('Solving Time Comparison by Algorithm and Problem Size', 
                    fontsize=14, pad=20)
        
        # Log scale for better visualization
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Adjust legend
        ax.legend(title='Grid Size', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def plot_scalability(self, results_df: pd.DataFrame,
                        save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot algorithm scalability with problem size.
        
        Args:
            results_df: DataFrame with columns: algorithm, num_islands, time
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Group by algorithm and plot mean time vs problem size
        for algorithm in results_df['algorithm'].unique():
            data = results_df[results_df['algorithm'] == algorithm]
            grouped = data.groupby('num_islands')['time'].agg(['mean', 'std'])
            
            ax.plot(grouped.index, grouped['mean'], 
                   marker='o', label=algorithm, linewidth=2)
            
            # Add error bars
            ax.fill_between(grouped.index,
                          grouped['mean'] - grouped['std'],
                          grouped['mean'] + grouped['std'],
                          alpha=0.2)
            
        ax.set_xlabel('Number of Islands', fontsize=12)
        ax.set_ylabel('Average Solving Time (seconds)', fontsize=12)
        ax.set_title('Algorithm Scalability Analysis', fontsize=14, pad=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def plot_success_rate(self, results_df: pd.DataFrame,
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot success rate by algorithm and difficulty.
        
        Args:
            results_df: DataFrame with columns: algorithm, difficulty, success
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate success rates
        success_rates = results_df.groupby(['algorithm', 'difficulty'])['success'].mean() * 100
        success_rates = success_rates.unstack()
        
        # Create grouped bar plot
        success_rates.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate by Algorithm and Difficulty', fontsize=14, pad=20)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
            
        ax.legend(title='Difficulty', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def plot_memory_usage(self, results_df: pd.DataFrame,
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot memory usage comparison.
        
        Args:
            results_df: DataFrame with columns: algorithm, size, memory_mb
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Pivot data for heatmap
        pivot_data = results_df.pivot_table(
            values='memory_mb',
            index='size',
            columns='algorithm',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Memory Usage (MB)'})
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Problem Size', fontsize=12)
        ax.set_title('Memory Usage Heatmap', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def plot_radar_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create radar chart comparing algorithms across multiple metrics.
        
        Args:
            metrics_dict: Dict of {algorithm: {metric: value}}
            save_path: Optional path to save figure
        """
        # Prepare data
        algorithms = list(metrics_dict.keys())
        metrics = list(next(iter(metrics_dict.values())).keys())
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data for each algorithm
        for i, algorithm in enumerate(algorithms):
            values = [metrics_dict[algorithm][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            # Normalize values to 0-1 range for better comparison
            max_vals = [max(metrics_dict[alg][metric] for alg in algorithms) 
                       for metric in metrics]
            normalized_values = [v/m if m > 0 else 0 for v, m in zip(values[:-1], max_vals)]
            normalized_values += normalized_values[:1]
            
            ax.plot(angles, normalized_values, 'o-', linewidth=2, 
                   label=algorithm, color=self.palette[i])
            ax.fill(angles, normalized_values, alpha=0.25, color=self.palette[i])
            
        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['20%', '40%', '60%', '80%'], size=8)
        
        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.set_title('Algorithm Performance Comparison\n(Normalized Metrics)', 
                    size=14, pad=30)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def plot_convergence(self, iterations_data: Dict[str, List[float]],
                        save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot convergence curves for iterative algorithms.
        
        Args:
            iterations_data: Dict of {algorithm: [cost_values]}
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for algorithm, costs in iterations_data.items():
            iterations = range(len(costs))
            ax.plot(iterations, costs, label=algorithm, linewidth=2)
            
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cost/Objective Value', fontsize=12)
        ax.set_title('Algorithm Convergence Comparison', fontsize=14, pad=20)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def create_comprehensive_report(self, results_df: pd.DataFrame,
                                   save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a comprehensive performance report with multiple plots.
        
        Args:
            results_df: DataFrame with all performance metrics
            save_path: Optional path to save figure
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 20))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Time comparison box plot
        ax1 = fig.add_subplot(gs[0, 0])
        algorithms = results_df['algorithm'].unique()
        for i, alg in enumerate(algorithms):
            data = results_df[results_df['algorithm'] == alg]['time']
            positions = [i]
            bp = ax1.boxplot(data, positions=positions, widths=0.6,
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(self.palette[i])
            
        ax1.set_xticklabels(algorithms)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Solving Time Distribution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Success rate bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        success_rates = results_df.groupby('algorithm')['success'].mean() * 100
        bars = ax2.bar(success_rates.index, success_rates.values)
        for i, bar in enumerate(bars):
            bar.set_color(self.palette[i])
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Overall Success Rate')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Scalability plot
        ax3 = fig.add_subplot(gs[1, :])
        for algorithm in algorithms:
            data = results_df[results_df['algorithm'] == algorithm]
            if 'num_islands' in data.columns:
                grouped = data.groupby('num_islands')['time'].mean()
                ax3.plot(grouped.index, grouped.values, 
                        marker='o', label=algorithm, linewidth=2)
        ax3.set_xlabel('Number of Islands')
        ax3.set_ylabel('Average Time (seconds)')
        ax3.set_title('Scalability Analysis')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Difficulty comparison heatmap
        ax4 = fig.add_subplot(gs[2, :])
        if 'difficulty' in results_df.columns:
            pivot = results_df.pivot_table(
                values='time',
                index='difficulty',
                columns='algorithm',
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                       ax=ax4, cbar_kws={'label': 'Avg Time (s)'})
            ax4.set_title('Average Solving Time by Difficulty')
            
        plt.suptitle('Comprehensive Performance Report', fontsize=16, y=0.995)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def plot_algorithm_timeline(self, timeline_data: List[Dict[str, Any]],
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a timeline visualization of algorithm execution.
        
        Args:
            timeline_data: List of dicts with keys: algorithm, start, end, phase
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(timeline_data)
        algorithms = df['algorithm'].unique()
        
        # Create Gantt chart
        for i, algorithm in enumerate(algorithms):
            alg_data = df[df['algorithm'] == algorithm]
            
            for _, row in alg_data.iterrows():
                duration = row['end'] - row['start']
                color = self.palette[hash(row['phase']) % len(self.palette)]
                
                ax.barh(i, duration, left=row['start'], height=0.6,
                       label=row['phase'] if row['phase'] not in ax.get_legend_handles_labels()[1] else '',
                       color=color, alpha=0.8)
                
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels(algorithms)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_title('Algorithm Execution Timeline', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig