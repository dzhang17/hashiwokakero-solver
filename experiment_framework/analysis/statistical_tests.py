# analysis/statistical_tests.py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalyzer:
    def __init__(self, results_df: pd.DataFrame, alpha: float = 0.05):
        self.df = results_df
        self.alpha = alpha
        
    def run_all_tests(self):
        """Run all statistical tests"""
        
        results = {
            'pairwise_success': self.pairwise_success_tests(),
            'pairwise_time': self.pairwise_time_tests(),
            'anova': self.anova_tests(),
            'effect_sizes': self.calculate_effect_sizes(),
            'correlation': self.correlation_analysis()
        }
        
        return results
    
    def pairwise_success_tests(self):
        """Pairwise comparison of success rates using McNemar test"""
        
        solvers = self.df['solver'].unique()
        results = {}
        
        for i, solver1 in enumerate(solvers):
            for solver2 in solvers[i+1:]:
                # Get matched instances
                s1_data = self.df[self.df['solver'] == solver1]
                s2_data = self.df[self.df['solver'] == solver2]
                
                # Match by instance
                merged = pd.merge(
                    s1_data[['instance', 'success']],
                    s2_data[['instance', 'success']],
                    on='instance',
                    suffixes=('_1', '_2')
                )
                
                if len(merged) > 0:
                    # Create contingency table
                    both_success = ((merged['success_1'] == True) & 
                                  (merged['success_2'] == True)).sum()
                    s1_only = ((merged['success_1'] == True) & 
                              (merged['success_2'] == False)).sum()
                    s2_only = ((merged['success_1'] == False) & 
                              (merged['success_2'] == True)).sum()
                    neither = ((merged['success_1'] == False) & 
                              (merged['success_2'] == False)).sum()
                    
                    table = [[both_success, s1_only], [s2_only, neither]]
                    
                    # McNemar test
                    result = mcnemar(table, exact=False, correction=True)
                    
                    results[f'{solver1}_vs_{solver2}'] = {
                        'statistic': result.statistic,
                        'pvalue': result.pvalue,
                        'contingency_table': table,
                        'n_matched': len(merged)
                    }
        
        return results
    
    def pairwise_time_tests(self):
        """Pairwise comparison of solve times"""
        
        solvers = self.df['solver'].unique()
        results = {}
        
        for i, solver1 in enumerate(solvers):
            for solver2 in solvers[i+1:]:
                # Get matched successful instances
                s1_data = self.df[(self.df['solver'] == solver1) & 
                                 (self.df['success'] == True)]
                s2_data = self.df[(self.df['solver'] == solver2) & 
                                 (self.df['success'] == True)]
                
                # Match by instance
                merged = pd.merge(
                    s1_data[['instance', 'solve_time']],
                    s2_data[['instance', 'solve_time']],
                    on='instance',
                    suffixes=('_1', '_2')
                )
                
                if len(merged) > 0:
                    times1 = merged['solve_time_1'].values
                    times2 = merged['solve_time_2'].values
                    
                    # Wilcoxon signed-rank test
                    wilcoxon_result = stats.wilcoxon(times1, times2)
                    
                    # Mann-Whitney U test (independent samples)
                    mannwhitney_result = stats.mannwhitneyu(
                        s1_data['solve_time'], 
                        s2_data['solve_time']
                    )
                    
                    results[f'{solver1}_vs_{solver2}'] = {
                        'wilcoxon_statistic': wilcoxon_result.statistic,
                        'wilcoxon_pvalue': wilcoxon_result.pvalue,
                        'mannwhitney_statistic': mannwhitney_result.statistic,
                        'mannwhitney_pvalue': mannwhitney_result.pvalue,
                        'n_paired': len(merged),
                        'median_diff': np.median(times1) - np.median(times2)
                    }
        
        return results
    
    def anova_tests(self):
        """ANOVA tests for multiple group comparisons"""
        
        results = {}
        
        # Kruskal-Wallis test for solve times
        solver_groups = []
        for solver in self.df['solver'].unique():
            solver_times = self.df[(self.df['solver'] == solver) & 
                                  (self.df['success'] == True)]['solve_time']
            if len(solver_times) > 0:
                solver_groups.append(solver_times.values)
        
        if len(solver_groups) > 2:
            kw_result = stats.kruskal(*solver_groups)
            results['kruskal_wallis'] = {
                'statistic': kw_result.statistic,
                'pvalue': kw_result.pvalue,
                'n_groups': len(solver_groups)
            }
        
        # Chi-square test for success rates
        contingency = pd.crosstab(self.df['solver'], self.df['success'])
        chi2_result = stats.chi2_contingency(contingency)
        
        results['chi_square'] = {
            'statistic': chi2_result[0],
            'pvalue': chi2_result[1],
            'dof': chi2_result[2],
            'expected': chi2_result[3]
        }
        
        return results
    
    def calculate_effect_sizes(self):
        """Calculate effect sizes for comparisons"""
        
        results = {}
        solvers = self.df['solver'].unique()
        
        for i, solver1 in enumerate(solvers):
            for solver2 in solvers[i+1:]:
                s1_times = self.df[(self.df['solver'] == solver1) & 
                                  (self.df['success'] == True)]['solve_time']
                s2_times = self.df[(self.df['solver'] == solver2) & 
                                  (self.df['success'] == True)]['solve_time']
                
                if len(s1_times) > 0 and len(s2_times) > 0:
                    # Cohen's d
                    pooled_std = np.sqrt(((len(s1_times)-1)*s1_times.std()**2 + 
                                         (len(s2_times)-1)*s2_times.std()**2) / 
                                        (len(s1_times) + len(s2_times) - 2))
                    
                    cohens_d = (s1_times.mean() - s2_times.mean()) / pooled_std
                    
                    # Cliff's delta (non-parametric effect size)
                    cliffs_delta = self.calculate_cliffs_delta(s1_times, s2_times)
                    
                    results[f'{solver1}_vs_{solver2}'] = {
                        'cohens_d': cohens_d,
                        'cliffs_delta': cliffs_delta,
                        'interpretation': self.interpret_effect_size(cohens_d)
                    }
        
        return results
    
    def calculate_cliffs_delta(self, x, y):
        """Calculate Cliff's delta effect size"""
        nx = len(x)
        ny = len(y)
        
        greater = 0
        less = 0
        
        for xi in x:
            for yi in y:
                if xi > yi:
                    greater += 1
                elif xi < yi:
                    less += 1
        
        return (greater - less) / (nx * ny)
    
    def interpret_effect_size(self, d):
        """Interpret Cohen's d effect size"""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    def correlation_analysis(self):
        """Analyze correlations between features and performance"""
        
        # Extract numeric features
        numeric_cols = ['solve_time', 'iterations', 'memory_used']
        if 'size' in self.df.columns:
            numeric_cols.append('size')
        
        # Add instance features if available
        instance_features = ['density', 'obstacles', 'num_islands']
        for feature in instance_features:
            if feature in self.df.columns:
                numeric_cols.append(feature)
        
        # Calculate correlations
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find significant correlations
        significant_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                r = corr_matrix.iloc[i, j]
                
                # Test significance
                n = len(self.df)
                t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                
                if p_value < self.alpha:
                    significant_corrs.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': r,
                        'pvalue': p_value
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'significant_correlations': significant_corrs
        }