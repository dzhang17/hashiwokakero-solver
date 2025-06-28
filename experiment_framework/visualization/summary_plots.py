# visualization/summary_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

class SummaryPlotter:
    """Create summary visualizations for the paper"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication-quality figures
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)
        self.colors = sns.color_palette("husl", 10)
        
    def create_all_summary_plots(self, results_df: pd.DataFrame):
        """Create all summary plots for the paper"""
        
        self.logger.info("Creating summary plots...")
        
        # Main comparison figure
        self.create_main_comparison_figure(results_df)
        
        # Scalability figure
        self.create_scalability_figure(results_df)
        
        # Algorithm variant comparison
        self.create_variant_comparison_figure(results_df)
        
        # Difficulty analysis figure
        self.create_difficulty_analysis_figure(results_df)
        
        # Parameter sensitivity figure
        self.create_parameter_sensitivity_figure(results_df)
        
        # Final summary dashboard
        self.create_summary_dashboard(results_df)
        
    def create_main_comparison_figure(self, df: pd.DataFrame):
        """Create the main comparison figure for the paper"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Performance profiles (main result)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self.plot_performance_profiles(df, ax1)
        
        # 2. Success rates by size
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_success_by_size(df, ax2)
        
        # 3. Time distribution
        ax3 = fig.add_subplot(gs[1, 2])
        self.plot_time_distribution(df, ax3)
        
        # 4. Statistical significance
        ax4 = fig.add_subplot(gs[2, :])
        self.plot_statistical_summary(df, ax4)
        
        plt.suptitle('Comprehensive Algorithm Comparison', fontsize=16, fontweight='bold')
        plt.savefig(self.output_dir / 'main_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'main_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_scalability_figure(self, df: pd.DataFrame):
        """Create scalability analysis figure"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Filter to main solvers
        main_solvers = ['ilp_full', 'lns_aggressive']
        scalability_df = df[df['solver'].isin(main_solvers)]
        
        # 1. Solve time scaling
        ax1 = axes[0, 0]
        for solver in main_solvers:
            solver_data = scalability_df[scalability_df['solver'] == solver]
            
            stats = solver_data.groupby('size')['solve_time'].agg(['mean', 'std', 'count'])
            sizes = stats.index
            means = stats['mean']
            stds = stats['std']
            
            # Use error bars
            ax1.errorbar(sizes, means, yerr=stds, 
                        label=self.format_solver_name(solver),
                        marker='o', capsize=5, linewidth=2)
        
        ax1.set_xlabel('Problem Size (islands)')
        ax1.set_ylabel('Solve Time (seconds)')
        ax1.set_title('(a) Solve Time Scalability')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Success rate scaling
        ax2 = axes[0, 1]
        for solver in main_solvers:
            solver_data = scalability_df[scalability_df['solver'] == solver]
            success_rates = solver_data.groupby('size')['success'].mean() * 100
            
            ax2.plot(success_rates.index, success_rates.values,
                    label=self.format_solver_name(solver),
                    marker='s', linewidth=2)
        
        ax2.set_xlabel('Problem Size (islands)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('(b) Success Rate vs Problem Size')
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory usage (if available)
        ax3 = axes[1, 0]
        if 'memory_used' in df.columns:
            for solver in main_solvers:
                solver_data = scalability_df[scalability_df['solver'] == solver]
                memory_stats = solver_data.groupby('size')['memory_used'].agg(['mean', 'std'])
                
                ax3.errorbar(memory_stats.index, memory_stats['mean'],
                           yerr=memory_stats['std'],
                           label=self.format_solver_name(solver),
                           marker='^', capsize=5, linewidth=2)
            
            ax3.set_xlabel('Problem Size (islands)')
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('(c) Memory Usage Scaling')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Memory data not available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('(c) Memory Usage')
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency metric (success/time ratio)
        ax4 = axes[1, 1]
        for solver in main_solvers:
            solver_data = scalability_df[scalability_df['solver'] == solver]
            
            efficiency = []
            sizes = []
            
            for size in sorted(solver_data['size'].unique()):
                size_data = solver_data[solver_data['size'] == size]
                success_rate = size_data['success'].mean()
                avg_time = size_data['solve_time'].mean()
                
                if avg_time > 0:
                    eff = success_rate / np.log10(avg_time + 1)
                    efficiency.append(eff)
                    sizes.append(size)
            
            ax4.plot(sizes, efficiency,
                    label=self.format_solver_name(solver),
                    marker='d', linewidth=2)
        
        ax4.set_xlabel('Problem Size (islands)')
        ax4.set_ylabel('Efficiency (success rate / log(time))')
        ax4.set_title('(d) Solving Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Scalability Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scalability_analysis.pdf', dpi=300)
        plt.savefig(self.output_dir / 'scalability_analysis.png', dpi=300)
        plt.close()
    
    def create_variant_comparison_figure(self, df: pd.DataFrame):
        """Create figure comparing algorithm variants"""
        
        # Filter to variant comparison data
        variant_df = df[df['experiment'] == 'algorithm_comparison']
        
        if len(variant_df) == 0:
            print("No variant comparison data found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. ILP variants comparison
        ax1 = axes[0, 0]
        ilp_variants = [col for col in variant_df['variant'].unique() if 'ILP' in col]
        if ilp_variants:
            ilp_data = variant_df[variant_df['variant'].isin(ilp_variants)]
            
            success_rates = ilp_data.groupby('variant')['success'].mean() * 100
            times = ilp_data.groupby('variant')['solve_time'].median()
            
            x = np.arange(len(ilp_variants))
            width = 0.35
            
            ax1_twin = ax1.twinx()
            
            bars1 = ax1.bar(x - width/2, success_rates, width, 
                           label='Success Rate', color='skyblue')
            bars2 = ax1_twin.bar(x + width/2, times, width,
                               label='Median Time', color='coral')
            
            ax1.set_xlabel('ILP Variant')
            ax1.set_ylabel('Success Rate (%)', color='skyblue')
            ax1_twin.set_ylabel('Median Time (s)', color='coral')
            ax1.set_title('(a) ILP Variant Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels([self.format_variant_name(v) for v in ilp_variants],
                               rotation=45, ha='right')
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. LNS variants comparison
        ax2 = axes[0, 1]
        lns_variants = [col for col in variant_df['variant'].unique() if 'LNS' in col]
        if lns_variants:
            lns_data = variant_df[variant_df['variant'].isin(lns_variants)]
            
            success_rates = lns_data.groupby('variant')['success'].mean() * 100
            times = lns_data.groupby('variant')['solve_time'].median()
            
            x = np.arange(len(lns_variants))
            
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar(x - width/2, success_rates, width,
                           label='Success Rate', color='lightgreen')
            bars2 = ax2_twin.bar(x + width/2, times, width,
                               label='Median Time', color='salmon')
            
            ax2.set_xlabel('LNS Variant')
            ax2.set_ylabel('Success Rate (%)', color='lightgreen')
            ax2_twin.set_ylabel('Median Time (s)', color='salmon')
            ax2.set_title('(b) LNS Variant Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels([self.format_variant_name(v) for v in lns_variants],
                               rotation=45, ha='right')
            
            for bar in bars1:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Best variant selection matrix
        ax3 = axes[1, 0]
        self.plot_variant_selection_matrix(variant_df, ax3)
        
        # 4. Variant performance radar chart
        ax4 = axes[1, 1]
        self.plot_variant_radar(variant_df, ax4)
        
        plt.suptitle('Algorithm Variant Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'variant_comparison.pdf', dpi=300)
        plt.savefig(self.output_dir / 'variant_comparison.png', dpi=300)
        plt.close()
    
    def create_difficulty_analysis_figure(self, df: pd.DataFrame):
        """Create difficulty analysis figure"""
        
        difficulty_df = df[df['experiment'] == 'difficulty_analysis']
        
        if len(difficulty_df) == 0:
            print("No difficulty analysis data found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Success rate by density
        ax1 = axes[0, 0]
        if 'density' in difficulty_df.columns:
            for solver in difficulty_df['solver'].unique():
                solver_data = difficulty_df[difficulty_df['solver'] == solver]
                density_success = solver_data.groupby('density')['success'].mean() * 100
                
                ax1.plot(density_success.index, density_success.values,
                        label=self.format_solver_name(solver),
                        marker='o', linewidth=2)
            
            ax1.set_xlabel('Density (%)')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_title('(a) Success vs Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Success rate by obstacles
        ax2 = axes[0, 1]
        if 'obstacles' in difficulty_df.columns:
            for solver in difficulty_df['solver'].unique():
                solver_data = difficulty_df[difficulty_df['solver'] == solver]
                obstacle_success = solver_data.groupby('obstacles')['success'].mean() * 100
                
                ax2.plot(obstacle_success.index, obstacle_success.values,
                        label=self.format_solver_name(solver),
                        marker='s', linewidth=2)
            
            ax2.set_xlabel('Obstacles (%)')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('(b) Success vs Obstacles')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Combined difficulty heatmap
        ax3 = axes[0, 2]
        if 'density' in difficulty_df.columns and 'obstacles' in difficulty_df.columns:
            # Use first solver for heatmap
            solver = difficulty_df['solver'].iloc[0]
            solver_data = difficulty_df[difficulty_df['solver'] == solver]
            
            pivot = solver_data.pivot_table(
                values='success',
                index='obstacles',
                columns='density',
                aggfunc='mean'
            ) * 100
            
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
                       ax=ax3, vmin=0, vmax=100)
            ax3.set_title(f'(c) Success Heatmap - {self.format_solver_name(solver)}')
        
        # 4. Time vs difficulty scatter
        ax4 = axes[1, 0]
        if 'total_bridges' in difficulty_df.columns:
            for solver in difficulty_df['solver'].unique():
                solver_data = difficulty_df[(difficulty_df['solver'] == solver) & 
                                          (difficulty_df['success'] == True)]
                
                ax4.scatter(solver_data['total_bridges'], solver_data['solve_time'],
                          label=self.format_solver_name(solver),
                          alpha=0.6, s=30)
            
            ax4.set_xlabel('Total Bridges Required')
            ax4.set_ylabel('Solve Time (s)')
            ax4.set_title('(d) Complexity vs Time')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Feature importance
        ax5 = axes[1, 1]
        if Path(self.output_dir.parent / 'difficulty_analysis' / 'feature_importance.csv').exists():
            importance_df = pd.read_csv(
                self.output_dir.parent / 'difficulty_analysis' / 'feature_importance.csv'
            )
            
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            ax5.barh(importance_df['feature'], importance_df['importance'])
            ax5.set_xlabel('Importance Score')
            ax5.set_title('(e) Difficulty Factor Importance')
        else:
            ax5.text(0.5, 0.5, 'Feature importance\nnot available',
                    ha='center', va='center', transform=ax5.transAxes)
        
        # 6. Group comparison
        ax6 = axes[1, 2]
        group_performance = difficulty_df.groupby(['group', 'solver'])['success'].mean() * 100
        group_performance.unstack().plot(kind='bar', ax=ax6)
        ax6.set_xlabel('Instance Group')
        ax6.set_ylabel('Success Rate (%)')
        ax6.set_title('(f) Performance by Instance Group')
        ax6.legend(title='Solver', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Instance Difficulty Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_analysis.pdf', dpi=300)
        plt.savefig(self.output_dir / 'difficulty_analysis.png', dpi=300)
        plt.close()
    
    def create_parameter_sensitivity_figure(self, df: pd.DataFrame):
        """Create parameter sensitivity analysis figure"""
        
        param_df = df[df['experiment'] == 'parameter_sensitivity']
        
        if len(param_df) == 0:
            print("No parameter sensitivity data found")
            return
        
        # Separate by solver type
        ilp_params = param_df[param_df['solver'] == 'ilp']
        lns_params = param_df[param_df['solver'] == 'lns']
        
        if len(lns_params) > 0:
            self.create_lns_parameter_figure(lns_params)
        
        if len(ilp_params) > 0:
            self.create_ilp_parameter_figure(ilp_params)
    
    def create_lns_parameter_figure(self, lns_df: pd.DataFrame):
        """Create LNS parameter sensitivity figure"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract parameters from the dataframe
        param_columns = []
        if 'parameters' in lns_df.columns:
            # Parameters are stored as dict
            param_df = pd.json_normalize(lns_df['parameters'])
            param_columns = param_df.columns.tolist()
        
        # 1. Destroy rate impact
        ax1 = axes[0, 0]
        if 'destroy_rate' in param_columns:
            destroy_rates = param_df['destroy_rate'].unique()
            
            for rate in sorted(destroy_rates):
                mask = param_df['destroy_rate'] == rate
                subset = lns_df[mask]
                
                success_rate = subset['success'].mean() * 100
                avg_time = subset['solve_time'].mean()
                
                ax1.scatter(rate, success_rate, s=100, label=f'{rate:.1f}')
            
            ax1.set_xlabel('Destroy Rate')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_title('(a) Impact of Destroy Rate')
            ax1.grid(True, alpha=0.3)
        
        # 2. Cooling rate impact
        ax2 = axes[0, 1]
        if 'cooling_rate' in param_columns:
            cooling_rates = param_df['cooling_rate'].unique()
            
            results = []
            for rate in sorted(cooling_rates):
                mask = param_df['cooling_rate'] == rate
                subset = lns_df[mask]
                
                results.append({
                    'cooling_rate': rate,
                    'success_rate': subset['success'].mean() * 100,
                    'avg_time': subset['solve_time'].mean()
                })
            
            results_df = pd.DataFrame(results)
            
            ax2_twin = ax2.twinx()
            
            ax2.plot(results_df['cooling_rate'], results_df['success_rate'],
                    'b-o', label='Success Rate')
            ax2_twin.plot(results_df['cooling_rate'], results_df['avg_time'],
                         'r-s', label='Avg Time')
            
            ax2.set_xlabel('Cooling Rate')
            ax2.set_ylabel('Success Rate (%)', color='b')
            ax2_twin.set_ylabel('Average Time (s)', color='r')
            ax2.set_title('(b) Impact of Cooling Rate')
            ax2.grid(True, alpha=0.3)
        
        # 3. Repair time limit impact
        ax3 = axes[1, 0]
        if 'repair_time' in param_columns:
            repair_times = param_df['repair_time'].unique()
            
            for time_limit in sorted(repair_times):
                mask = param_df['repair_time'] == time_limit
                subset = lns_df[mask]
                
                success_rate = subset['success'].mean() * 100
                total_time = subset['solve_time'].mean()
                
                ax3.scatter(time_limit, success_rate, s=100,
                          label=f'{time_limit}s')
            
            ax3.set_xlabel('Repair Time Limit (s)')
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('(c) Impact of Repair Time Limit')
            ax3.grid(True, alpha=0.3)
        
        # 4. 3D parameter interaction
        ax4 = axes[1, 1]
        if len(param_columns) >= 2:
            # Create interaction plot
            param1 = param_columns[0]
            param2 = param_columns[1] if len(param_columns) > 1 else param_columns[0]
            
            pivot = pd.pivot_table(
                pd.concat([lns_df[['success']], param_df[[param1, param2]]], axis=1),
                values='success',
                index=param1,
                columns=param2,
                aggfunc='mean'
            ) * 100
            
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis',
                       ax=ax4, cbar_kws={'label': 'Success Rate (%)'})
            ax4.set_title(f'(d) {param1} vs {param2} Interaction')
        
        plt.suptitle('LNS Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lns_parameter_sensitivity.pdf', dpi=300)
        plt.savefig(self.output_dir / 'lns_parameter_sensitivity.png', dpi=300)
        plt.close()
    
    def create_summary_dashboard(self, df: pd.DataFrame):
        """Create a comprehensive summary dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Hashiwokakero Solver Comparison - Summary Dashboard',
                    fontsize=18, fontweight='bold')
        
        # 1. Overall winner (text box)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_overall_winner(df, ax1)
        
        # 2. Key metrics table
        ax2 = fig.add_subplot(gs[0, 1:3])
        self.plot_key_metrics_table(df, ax2)
        
        # 3. Recommendation matrix
        ax3 = fig.add_subplot(gs[0, 3])
        self.plot_recommendation_matrix(df, ax3)
        
        # 4. Performance profiles
        ax4 = fig.add_subplot(gs[1:3, :2])
        self.plot_performance_profiles(df, ax4)
        
        # 5. Scalability summary
        ax5 = fig.add_subplot(gs[1, 2:])
        self.plot_scalability_summary(df, ax5)
        
        # 6. Time distribution
        ax6 = fig.add_subplot(gs[2, 2])
        self.plot_time_distribution_summary(df, ax6)
        
        # 7. Success rate by category
        ax7 = fig.add_subplot(gs[2, 3])
        self.plot_success_by_category(df, ax7)
        
        # 8. Statistical significance summary
        ax8 = fig.add_subplot(gs[3, :])
        self.plot_significance_summary(df, ax8)
        
        plt.savefig(self.output_dir / 'summary_dashboard.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Helper methods
    def format_solver_name(self, solver: str) -> str:
        """Format solver name for display"""
        name_map = {
            'ilp_basic': 'ILP-Basic',
            'ilp_full': 'ILP-Full',
            'ilp_optimized': 'ILP-Opt',
            'lns_basic': 'LNS-Basic',
            'lns_adaptive': 'LNS-Adaptive',
            'lns_aggressive': 'LNS-Aggr',
            'lns_tuned': 'LNS-Tuned'
        }
        return name_map.get(solver, solver)
    
    def format_variant_name(self, variant: str) -> str:
        """Format variant name for display"""
        # Simplify variant names
        if 'pre1_lazy1' in variant:
            return 'Full'
        elif 'pre1_lazy0' in variant:
            return 'Preprocess'
        elif 'pre0_lazy1' in variant:
            return 'Lazy'
        elif 'pre0_lazy0' in variant:
            return 'Basic'
        elif 'adapt1_par1' in variant:
            return 'Full'
        elif 'adapt1_par0' in variant:
            return 'Adaptive'
        elif 'adapt0_par1' in variant:
            return 'Parallel'
        elif 'adapt0_par0' in variant:
            return 'Basic'
        return variant
    
    def plot_performance_profiles(self, df, ax):
        """Plot performance profiles (tau plot)"""
        
        # Implementation similar to result_analyzer.py
        # but with better styling for publication
        pass
    
    def plot_variant_selection_matrix(self, df, ax):
        """Plot variant selection matrix"""
        
        # Create a matrix showing best variant for different scenarios
        pass
    
    def plot_variant_radar(self, df, ax):
        """Plot radar chart comparing variants"""
        
        # Radar chart implementation
        pass