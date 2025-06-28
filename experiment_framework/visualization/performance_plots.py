# visualization/performance_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class PerformancePlotter:
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 10)
        
    def plot_performance_comparison(self, df: pd.DataFrame, output_path: Path):
        """Create comprehensive performance comparison plots"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Success rate comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_success_rates(df, ax1)
        
        # 2. Time comparison (violin plot)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_time_violin(df, ax2)
        
        # 3. Performance profile
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_performance_profile(df, ax3)
        
        # 4. Scatter plot: time vs size
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_time_vs_size(df, ax4)
        
        # 5. Heatmap: solver vs instance difficulty
        ax5 = fig.add_subplot(gs[1, 1:])
        self.plot_difficulty_heatmap(df, ax5)
        
        # 6. Convergence comparison (if available)
        ax6 = fig.add_subplot(gs[2, :2])
        self.plot_convergence(df, ax6)
        
        # 7. Parameter impact (if available)
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_parameter_impact(df, ax7)
        
        plt.suptitle('Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_success_rates(self, df, ax):
        """Plot success rates by solver"""
        success_rates = df.groupby('solver')['success'].mean() * 100
        
        bars = ax.bar(success_rates.index, success_rates.values, 
                      color=self.colors[:len(success_rates)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rates by Solver')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
    
    def plot_time_violin(self, df, ax):
        """Plot solve time distribution using violin plot"""
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) > 0:
            # Log transform for better visualization
            successful_df['log_time'] = np.log10(successful_df['solve_time'] + 0.1)
            
            sns.violinplot(data=successful_df, x='solver', y='log_time', 
                          ax=ax, palette=self.colors)
            
            ax.set_ylabel('Log10(Solve Time + 0.1)')
            ax.set_title('Solve Time Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No successful solves', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_performance_profile(self, df, ax):
        """Plot performance profiles"""
        # Implementation from result_analyzer.py
        pass
    
    def plot_time_vs_size(self, df, ax):
        """Plot solve time vs problem size"""
        if 'size' in df.columns:
            for solver in df['solver'].unique():
                solver_df = df[(df['solver'] == solver) & (df['success'] == True)]
                if len(solver_df) > 0:
                    ax.scatter(solver_df['size'], solver_df['solve_time'],
                             label=solver, alpha=0.6, s=50)
            
            ax.set_xlabel('Problem Size (islands)')
            ax.set_ylabel('Solve Time (seconds)')
            ax.set_title('Solve Time vs Problem Size')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_difficulty_heatmap(self, df, ax):
        """Plot heatmap of performance vs instance difficulty"""
        if 'density' in df.columns and 'obstacles' in df.columns:
            # Create pivot table
            for solver in df['solver'].unique()[:1]:  # Just show one solver
                solver_df = df[df['solver'] == solver]
                
                pivot = solver_df.pivot_table(
                    values='success',
                    index='obstacles',
                    columns='density',
                    aggfunc='mean'
                ) * 100
                
                sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
                          ax=ax, vmin=0, vmax=100)
                
                ax.set_title(f'Success Rate (%) by Density and Obstacles - {solver}')
                ax.set_xlabel('Density (%)')
                ax.set_ylabel('Obstacles (%)')
    
    def plot_convergence(self, df, ax):
        """Plot convergence behavior"""
        # This would require iteration data
        ax.text(0.5, 0.5, 'Convergence data not available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Convergence Behavior')
    
    def plot_parameter_impact(self, df, ax):
        """Plot parameter sensitivity"""
        if 'parameters' in df.columns:
            # Extract parameter data
            pass
        else:
            ax.text(0.5, 0.5, 'Parameter data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Parameter Impact')