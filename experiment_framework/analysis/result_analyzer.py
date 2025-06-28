# analysis/result_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from scipy import stats

class ResultAnalyzer:
    def __init__(self, results: List[Dict], output_dir: Path):
        self.df = pd.DataFrame(results)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def generate_all_analyses(self):
        """Generate all analysis plots and tables"""
        
        print("Generating performance profiles...")
        self.generate_performance_profiles()
        
        print("Generating scalability curves...")
        self.plot_scalability_curves()
        
        print("Analyzing success rates...")
        self.analyze_success_rates()
        
        print("Analyzing time distributions...")
        self.analyze_time_distributions()
        
        print("Performing statistical tests...")
        self.perform_statistical_tests()
        
        print("Generating summary tables...")
        self.generate_summary_tables()
        
        print("Creating final report...")
        self.create_final_report()
    
    def generate_performance_profiles(self):
        """Generate performance profile plots"""
        
        # Filter to main comparison experiments
        comparison_df = self.df[self.df['experiment'] == 'algorithm_comparison']
        
        if len(comparison_df) == 0:
            print("No algorithm comparison data found")
            return
        
        # Get unique solvers
        solvers = comparison_df['solver'].unique()
        
        # Create pivot table of solve times
        pivot = comparison_df.pivot_table(
            values='solve_time',
            index='instance',
            columns='solver'
        )
        
        # Calculate performance ratios
        min_times = pivot.min(axis=1)
        ratios = pivot.div(min_times, axis=0)
        
        # Create performance profile plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        max_ratio = 10  # Maximum ratio to plot
        tau_values = np.logspace(0, np.log10(max_ratio), 100)
        
        for solver in solvers:
            if solver in ratios.columns:
                # Calculate cumulative distribution
                solver_ratios = ratios[solver].dropna()
                proportions = []
                
                for tau in tau_values:
                    prop = (solver_ratios <= tau).sum() / len(solver_ratios)
                    proportions.append(prop)
                
                ax.plot(tau_values, proportions, label=solver, linewidth=2)
        
        ax.set_xlabel('Performance Ratio (τ)', fontsize=12)
        ax.set_ylabel('P(solver ≤ τ × best)', fontsize=12)
        ax.set_title('Performance Profiles', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(1, max_ratio)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_profiles.png', dpi=300)
        plt.close()
    
    def plot_scalability_curves(self):
        """Plot scalability analysis"""
        
        scalability_df = self.df[self.df['experiment'] == 'scalability']
        
        if len(scalability_df) == 0:
            print("No scalability data found")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Solve time vs problem size
        ax1 = axes[0, 0]
        for solver in scalability_df['solver'].unique():
            solver_data = scalability_df[scalability_df['solver'] == solver]
            
            # Calculate statistics
            stats_data = solver_data.groupby('size')['solve_time'].agg(['mean', 'std'])
            sizes = stats_data.index
            means = stats_data['mean']
            stds = stats_data['std']
            
            ax1.errorbar(sizes, means, yerr=stds, label=solver, 
                        marker='o', capsize=5, linewidth=2)
        
        ax1.set_xlabel('Problem Size (islands)')
        ax1.set_ylabel('Solve Time (seconds)')
        ax1.set_title('Solve Time Scalability')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Success rate vs problem size
        ax2 = axes[0, 1]
        for solver in scalability_df['solver'].unique():
            solver_data = scalability_df[scalability_df['solver'] == solver]
            success_rates = solver_data.groupby('size')['success'].mean() * 100
            
            ax2.plot(success_rates.index, success_rates.values, 
                    label=solver, marker='s', linewidth=2)
        
        ax2.set_xlabel('Problem Size (islands)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate vs Problem Size')
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Timeout rate vs problem size
        ax3 = axes[1, 0]
        for solver in scalability_df['solver'].unique():
            solver_data = scalability_df[scalability_df['solver'] == solver]
            timeout_rates = solver_data.groupby('size')['timed_out'].mean() * 100
            
            ax3.plot(timeout_rates.index, timeout_rates.values,
                    label=solver, marker='^', linewidth=2)
        
        ax3.set_xlabel('Problem Size (islands)')
        ax3.set_ylabel('Timeout Rate (%)')
        ax3.set_title('Timeout Rate vs Problem Size')
        ax3.set_ylim(0, 105)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Memory usage vs problem size (if available)
        ax4 = axes[1, 1]
        if 'memory_used' in scalability_df.columns:
            for solver in scalability_df['solver'].unique():
                solver_data = scalability_df[scalability_df['solver'] == solver]
                memory_stats = solver_data.groupby('size')['memory_used'].agg(['mean', 'std'])
                
                ax4.errorbar(memory_stats.index, memory_stats['mean'], 
                           yerr=memory_stats['std'], label=solver,
                           marker='d', capsize=5, linewidth=2)
            
            ax4.set_xlabel('Problem Size (islands)')
            ax4.set_ylabel('Memory Usage (MB)')
            ax4.set_title('Memory Usage Scalability')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Memory usage data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Memory Usage Scalability')
        
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scalability_analysis.png', dpi=300)
        plt.close()
    
    def analyze_success_rates(self):
        """Analyze success rates across different experiments"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall success rates by solver
        ax1 = axes[0, 0]
        success_by_solver = self.df.groupby('solver')['success'].agg(['sum', 'count'])
        success_by_solver['rate'] = success_by_solver['sum'] / success_by_solver['count'] * 100
        
        bars = ax1.bar(success_by_solver.index, success_by_solver['rate'])
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Overall Success Rates by Solver')
        ax1.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Success rates by experiment type
        ax2 = axes[0, 1]
        if 'experiment' in self.df.columns:
            exp_solver_success = self.df.groupby(['experiment', 'solver'])['success'].mean() * 100
            exp_solver_success.unstack().plot(kind='bar', ax=ax2)
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Success Rates by Experiment Type')
            ax2.legend(title='Solver', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Success rate distribution
        ax3 = axes[1, 0]
        for solver in self.df['solver'].unique():
            solver_data = self.df[self.df['solver'] == solver]
            ax3.hist(solver_data['success'].astype(int), bins=2, alpha=0.6, 
                    label=solver, density=True)
        
        ax3.set_xlabel('Success (0=Failed, 1=Success)')
        ax3.set_ylabel('Density')
        ax3.set_title('Success Distribution by Solver')
        ax3.legend()
        ax3.set_xticks([0, 1])
        
        # 4. Success vs solve time scatter
        ax4 = axes[1, 1]
        for solver in self.df['solver'].unique():
            solver_data = self.df[self.df['solver'] == solver]
            successful = solver_data[solver_data['success'] == True]
            
            if len(successful) > 0:
                ax4.scatter(successful['solve_time'], 
                          successful['success'].astype(int) + np.random.uniform(-0.05, 0.05, len(successful)),
                          alpha=0.6, label=solver, s=50)
        
        ax4.set_xlabel('Solve Time (seconds)')
        ax4.set_ylabel('Success')
        ax4.set_title('Success vs Solve Time')
        ax4.set_xscale('log')
        ax4.set_ylim(-0.1, 1.1)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_analysis.png', dpi=300)
        plt.close()
    
    def analyze_time_distributions(self):
        """Analyze solve time distributions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Filter successful solves only
        successful_df = self.df[self.df['success'] == True]
        
        # 1. Time distribution by solver (histogram)
        ax1 = axes[0, 0]
        for solver in successful_df['solver'].unique():
            solver_times = successful_df[successful_df['solver'] == solver]['solve_time']
            if len(solver_times) > 0:
                ax1.hist(np.log10(solver_times + 0.01), bins=30, alpha=0.6, 
                        label=solver, density=True)
        
        ax1.set_xlabel('Log10(Solve Time + 0.01)')
        ax1.set_ylabel('Density')
        ax1.set_title('Solve Time Distribution (Log Scale)')
        ax1.legend()
        
        # 2. Box plot of solve times
        ax2 = axes[0, 1]
        solve_time_data = []
        labels = []
        
        for solver in successful_df['solver'].unique():
            solver_times = successful_df[successful_df['solver'] == solver]['solve_time']
            if len(solver_times) > 0:
                solve_time_data.append(solver_times)
                labels.append(solver)
        
        if solve_time_data:
            box_plot = ax2.boxplot(solve_time_data, labels=labels, showmeans=True)
            ax2.set_ylabel('Solve Time (seconds)')
            ax2.set_title('Solve Time Box Plot')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # 3. Time percentiles
        ax3 = axes[1, 0]
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        
        for solver in successful_df['solver'].unique():
            solver_times = successful_df[successful_df['solver'] == solver]['solve_time']
            if len(solver_times) > 0:
                percs = np.percentile(solver_times, percentiles)
                ax3.plot(percentiles, percs, marker='o', label=solver)
        
        ax3.set_xlabel('Percentile')
        ax3.set_ylabel('Solve Time (seconds)')
        ax3.set_title('Solve Time Percentiles')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        for solver in successful_df['solver'].unique():
            solver_times = successful_df[successful_df['solver'] == solver]['solve_time']
            if len(solver_times) > 0:
                sorted_times = np.sort(solver_times)
                cum_prob = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
                ax4.plot(sorted_times, cum_prob, label=solver, linewidth=2)
        
        ax4.set_xlabel('Solve Time (seconds)')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution of Solve Times')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_distribution_analysis.png', dpi=300)
        plt.close()
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests"""
        
        results = {}
        
        # Get unique solver pairs
        solvers = self.df['solver'].unique()
        
        # Pairwise comparisons
        from itertools import combinations
        
        for solver1, solver2 in combinations(solvers, 2):
            solver1_data = self.df[self.df['solver'] == solver1]
            solver2_data = self.df[self.df['solver'] == solver2]
            
            # Match instances for paired tests
            common_instances = set(solver1_data['instance']) & set(solver2_data['instance'])
            
            if len(common_instances) > 0:
                # Get matched data
                s1_matched = solver1_data[solver1_data['instance'].isin(common_instances)].sort_values('instance')
                s2_matched = solver2_data[solver2_data['instance'].isin(common_instances)].sort_values('instance')
                
                # Success rate comparison (McNemar test)
                s1_success = s1_matched['success'].values
                s2_success = s2_matched['success'].values
                
                # Create contingency table
                both_success = np.sum((s1_success == 1) & (s2_success == 1))
                s1_only = np.sum((s1_success == 1) & (s2_success == 0))
                s2_only = np.sum((s1_success == 0) & (s2_success == 1))
                neither = np.sum((s1_success == 0) & (s2_success == 0))
                
                # McNemar test
                from statsmodels.stats.contingency_tables import mcnemar
                table = [[both_success, s1_only], [s2_only, neither]]
                mcnemar_result = mcnemar(table, exact=False)
                
                # Solve time comparison (Wilcoxon signed-rank test)
                s1_times = s1_matched['solve_time'].values
                s2_times = s2_matched['solve_time'].values
                
                # Only compare where both succeeded
                both_success_mask = (s1_success == 1) & (s2_success == 1)
                if np.sum(both_success_mask) > 0:
                    wilcoxon_result = stats.wilcoxon(
                        s1_times[both_success_mask], 
                        s2_times[both_success_mask]
                    )
                else:
                    wilcoxon_result = None
                
                results[f'{solver1}_vs_{solver2}'] = {
                    'n_instances': len(common_instances),
                    'success_rate_1': np.mean(s1_success),
                    'success_rate_2': np.mean(s2_success),
                    'mcnemar_statistic': mcnemar_result.statistic,
                    'mcnemar_pvalue': mcnemar_result.pvalue,
                    'median_time_1': np.median(s1_times[s1_success == 1]) if np.sum(s1_success) > 0 else np.nan,
                    'median_time_2': np.median(s2_times[s2_success == 1]) if np.sum(s2_success) > 0 else np.nan,
                    'wilcoxon_statistic': wilcoxon_result.statistic if wilcoxon_result else np.nan,
                    'wilcoxon_pvalue': wilcoxon_result.pvalue if wilcoxon_result else np.nan
                }
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(self.output_dir / 'statistical_tests.csv')
        
        # Create summary report
        with open(self.output_dir / 'statistical_summary.txt', 'w') as f:
            f.write("Statistical Test Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for comparison, data in results.items():
                f.write(f"\n{comparison}:\n")
                f.write(f"  Success rates: {data['success_rate_1']:.3f} vs {data['success_rate_2']:.3f}\n")
                f.write(f"  McNemar p-value: {data['mcnemar_pvalue']:.4f}")
                
                if data['mcnemar_pvalue'] < 0.05:
                    f.write(" (SIGNIFICANT)\n")
                else:
                    f.write(" (not significant)\n")
                
                if not np.isnan(data['wilcoxon_pvalue']):
                    f.write(f"  Median times: {data['median_time_1']:.2f}s vs {data['median_time_2']:.2f}s\n")
                    f.write(f"  Wilcoxon p-value: {data['wilcoxon_pvalue']:.4f}")
                    
                    if data['wilcoxon_pvalue'] < 0.05:
                        f.write(" (SIGNIFICANT)\n")
                    else:
                        f.write(" (not significant)\n")
    
    def generate_summary_tables(self):
        """Generate summary tables for the paper"""
        
        # Table 1: Overall performance summary
        summary_stats = self.df.groupby('solver').agg({
            'success': ['count', 'sum', lambda x: np.mean(x) * 100],
            'solve_time': ['mean', 'median', 'std'],
            'timed_out': lambda x: np.mean(x) * 100
        }).round(2)
        
        summary_stats.columns = ['Total_Tests', 'Successful', 'Success_Rate_%',
                                'Mean_Time', 'Median_Time', 'Std_Time', 'Timeout_Rate_%']
        
        summary_stats.to_csv(self.output_dir / 'overall_summary.csv')
        
        # Table 2: Performance by problem size
        if 'size' in self.df.columns:
            size_summary = self.df.groupby(['size', 'solver']).agg({
                'success': lambda x: np.mean(x) * 100,
                'solve_time': 'median',
                'timed_out': lambda x: np.mean(x) * 100
            }).round(2)
            
            size_summary.columns = ['Success_Rate_%', 'Median_Time', 'Timeout_Rate_%']
            size_summary.to_csv(self.output_dir / 'size_summary.csv')
        
        # Table 3: Best configurations
        if 'variant' in self.df.columns:
            variant_summary = self.df.groupby('variant').agg({
                'success': lambda x: np.mean(x) * 100,
                'solve_time': 'median',
                'solver_type': 'first'
            }).round(2)
            
            variant_summary = variant_summary.sort_values('success', ascending=False)
            variant_summary.to_csv(self.output_dir / 'variant_summary.csv')
        
        # Create LaTeX tables
        self.create_latex_tables(summary_stats, size_summary if 'size' in self.df.columns else None)
    
    def create_latex_tables(self, overall_summary, size_summary):
        """Create LaTeX formatted tables"""
        
        with open(self.output_dir / 'latex_tables.tex', 'w') as f:
            # Overall summary table
            f.write("% Overall Performance Summary\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Overall Performance Summary}\n")
            f.write("\\label{tab:overall_summary}\n")
            f.write("\\begin{tabular}{lrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Solver & Tests & Success Rate (\\%) & Mean Time (s) & Median Time (s) & Timeout Rate (\\%) \\\\\n")
            f.write("\\midrule\n")
            
            for solver, row in overall_summary.iterrows():
                f.write(f"{solver} & {int(row['Total_Tests'])} & {row['Success_Rate_%']:.1f} & ")
                f.write(f"{row['Mean_Time']:.2f} & {row['Median_Time']:.2f} & {row['Timeout_Rate_%']:.1f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Size summary table (if available)
            if size_summary is not None:
                f.write("% Performance by Problem Size\n")
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Performance by Problem Size}\n")
                f.write("\\label{tab:size_summary}\n")
                f.write("\\begin{tabular}{llrrr}\n")
                f.write("\\toprule\n")
                f.write("Size & Solver & Success Rate (\\%) & Median Time (s) & Timeout Rate (\\%) \\\\\n")
                f.write("\\midrule\n")
                
                for (size, solver), row in size_summary.iterrows():
                    f.write(f"{size} & {solver} & {row['Success_Rate_%']:.1f} & ")
                    f.write(f"{row['Median_Time']:.2f} & {row['Timeout_Rate_%']:.1f} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
    
    def create_final_report(self):
        """Create a comprehensive final report"""
        
        with open(self.output_dir / 'final_report.md', 'w') as f:
            f.write("# Hashiwokakero Solver Experiment Results\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("## Summary\n\n")
            
            # Key findings
            f.write("### Key Findings\n\n")
            
            # Best solver overall
            success_by_solver = self.df.groupby('solver')['success'].mean()
            best_solver = success_by_solver.idxmax()
            f.write(f"- **Best overall solver**: {best_solver} ")
            f.write(f"(success rate: {success_by_solver[best_solver]*100:.1f}%)\n")
            
            # Scalability winner
            if 'size' in self.df.columns and len(self.df['size'].unique()) > 1:
                large_instances = self.df[self.df['size'] == self.df['size'].max()]
                if len(large_instances) > 0:
                    large_success = large_instances.groupby('solver')['success'].mean()
                    best_large = large_success.idxmax()
                    f.write(f"- **Best for large instances**: {best_large} ")
                    f.write(f"(success rate on {self.df['size'].max()}-island: {large_success[best_large]*100:.1f}%)\n")
            
            # Speed comparison
            successful_df = self.df[self.df['success'] == True]
            if len(successful_df) > 0:
                median_times = successful_df.groupby('solver')['solve_time'].median()
                fastest = median_times.idxmin()
                f.write(f"- **Fastest solver**: {fastest} ")
                f.write(f"(median time: {median_times[fastest]:.2f}s)\n")
            
            f.write("\n### Detailed Results\n\n")
            
            # Include summary statistics
            f.write("#### Overall Performance\n\n")
            summary_stats = self.df.groupby('solver').agg({
                'success': ['count', lambda x: np.mean(x) * 100],
                'solve_time': ['mean', 'median'],
            }).round(2)
            
            f.write(summary_stats.to_markdown())
            f.write("\n\n")
            
            # Statistical significance
            if Path(self.output_dir / 'statistical_tests.csv').exists():
                f.write("#### Statistical Significance\n\n")
                stat_df = pd.read_csv(self.output_dir / 'statistical_tests.csv', index_col=0)
                
                significant_comparisons = stat_df[
                    (stat_df['mcnemar_pvalue'] < 0.05) | 
                    (stat_df['wilcoxon_pvalue'] < 0.05)
                ]
                
                if len(significant_comparisons) > 0:
                    f.write("Significant differences found:\n\n")
                    for comparison, row in significant_comparisons.iterrows():
                        f.write(f"- **{comparison}**: ")
                        if row['mcnemar_pvalue'] < 0.05:
                            f.write(f"Success rates differ (p={row['mcnemar_pvalue']:.4f}) ")
                        if row['wilcoxon_pvalue'] < 0.05:
                            f.write(f"Solve times differ (p={row['wilcoxon_pvalue']:.4f})")
                        f.write("\n")
                else:
                    f.write("No statistically significant differences found.\n")
            
            f.write("\n### Recommendations\n\n")
            
            # Generate recommendations based on results
            f.write("Based on the experimental results:\n\n")
            
            if 'size' in self.df.columns:
                # Size-based recommendations
                for size in sorted(self.df['size'].unique()):
                    size_data = self.df[self.df['size'] == size]
                    size_success = size_data.groupby('solver')['success'].mean()
                    best_for_size = size_success.idxmax()
                    f.write(f"- For {size}-island problems: use **{best_for_size}**\n")
            
            f.write("\n### Plots Generated\n\n")
            f.write("The following visualizations have been created:\n\n")
            
            for plot_file in self.output_dir.glob("*.png"):
                f.write(f"- {plot_file.name}\n")