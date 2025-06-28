# experiments/difficulty_analysis.py
from typing import List, Dict
import pandas as pd
import numpy as np
from experiment_framework.experiments.base_experiment import BaseExperiment

class DifficultyAnalysis(BaseExperiment):
    def run(self) -> List[Dict]:
        """Analyze which instance features affect solving difficulty"""
        
        # Config is already the difficulty_analysis config
        config = self.config
        sizes = config['sizes']
        instances_per_group = config['instances_per_group']
        time_limit = config.get('timeout', 300)
        groups = config.get('difficulty_groups', {})
        
        for size in sizes:
            self.logger.info(f"\nAnalyzing difficulty factors for {size}-island instances...")
            
            for group_name, group_config in groups.items():
                # group_name is already from the dict key
                self.logger.info(f"\nTesting group: {group_name}")
                
                # Select instances matching criteria
                criteria = {k: v for k, v in group_config.items() if k != 'name'}
                instances = self.select_instances(size, instances_per_group, criteria)
                
                if not instances:
                    self.logger.warning(f"No instances found for group {group_name}")
                    continue
                
                self.logger.info(f"Found {len(instances)} instances for {group_name}")
                
                # Test each instance with both solvers
                for instance_path in instances:
                    puzzle = self.load_puzzle(instance_path)
                    
                    # Calculate puzzle statistics
                    puzzle_stats = self.calculate_puzzle_stats(puzzle)
                    
                    # Test with ILP
                    ilp_result = self.run_single_test(
                        puzzle, 'ilp_optimized', time_limit
                    )
                    
                    ilp_result.update({
                        'experiment': 'difficulty_analysis',
                        'size': size,
                        'group': group_name,
                        'instance': instance_path.name,
                        **criteria,
                        **puzzle_stats
                    })
                    
                    self.results.append(ilp_result)
                    
                    # Test with LNS
                    lns_result = self.run_single_test(
                        puzzle, 'lns_tuned', time_limit
                    )
                    
                    lns_result.update({
                        'experiment': 'difficulty_analysis',
                        'size': size,
                        'group': group_name,
                        'instance': instance_path.name,
                        **criteria,
                        **puzzle_stats
                    })
                    
                    self.results.append(lns_result)
        
        # Analyze difficulty factors
        self.analyze_difficulty_factors()
        
        return self.results
    
    def calculate_puzzle_stats(self, puzzle) -> Dict:
        """Calculate statistics about puzzle structure"""
        
        island_degrees = [island.required_bridges for island in puzzle.islands]
        
        # Calculate connectivity metrics
        num_islands = len(puzzle.islands)
        total_bridges_required = sum(island_degrees) // 2
        
        # Estimate maximum possible connections
        # This is approximate - actual max depends on grid layout
        grid_area = puzzle.width * puzzle.height
        max_possible_connections = 0
        
        for i, island1 in enumerate(puzzle.islands):
            for j, island2 in enumerate(puzzle.islands[i+1:], i+1):
                # Check if connection is possible (same row or column)
                if island1.row == island2.row or island1.col == island2.col:
                    max_possible_connections += 1
        
        connectivity_ratio = total_bridges_required / max(1, max_possible_connections)
        
        # Calculate degree distribution metrics
        degree_variance = np.var(island_degrees) if island_degrees else 0
        degree_skewness = self.calculate_skewness(island_degrees)
        
        # Count islands by degree
        degree_counts = pd.Series(island_degrees).value_counts().to_dict()
        
        return {
            'num_islands': num_islands,
            'total_bridges': total_bridges_required,
            'avg_degree': np.mean(island_degrees) if island_degrees else 0,
            'max_degree': max(island_degrees) if island_degrees else 0,
            'min_degree': min(island_degrees) if island_degrees else 0,
            'degree_variance': degree_variance,
            'degree_skewness': degree_skewness,
            'connectivity_ratio': connectivity_ratio,
            'islands_degree_1': degree_counts.get(1, 0),
            'islands_degree_2': degree_counts.get(2, 0),
            'islands_degree_high': sum(v for k, v in degree_counts.items() if k >= 6),
            'grid_fill_ratio': num_islands / grid_area
        }
    
    def calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of a distribution"""
        if not data or len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        n = len(data)
        skewness = (n / ((n-1) * (n-2))) * sum(((x - mean) / std) ** 3 for x in data)
        return skewness
    
    def analyze_difficulty_factors(self):
        """Analyze which factors correlate with difficulty"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Define difficulty metrics
        df['difficulty_score'] = (
            (1 - df['success'].astype(int)) * 100 +  # Failed = 100 points
            df['solve_time'] / df['solve_time'].max() * 50 +  # Normalized time
            (df['timed_out'].astype(int) * 50)  # Timeout = 50 points
        )
        
        # Correlation analysis
        numeric_features = [
            'density', 'obstacles', 'num_islands', 'total_bridges',
            'avg_degree', 'degree_variance', 'connectivity_ratio',
            'grid_fill_ratio'
        ]
        
        # Filter to existing columns
        numeric_features = [f for f in numeric_features if f in df.columns]
        
        correlations = df[numeric_features + ['difficulty_score']].corr()['difficulty_score']
        correlations = correlations.drop('difficulty_score').sort_values(ascending=False)
        
        # Save correlation analysis
        correlations.to_csv(self.output_dir / 'difficulty_correlations.csv')
        
        # Group analysis
        group_stats = df.groupby(['solver', 'group']).agg({
            'success': 'mean',
            'solve_time': 'mean',
            'difficulty_score': 'mean'
        }).round(3)
        
        group_stats.to_csv(self.output_dir / 'group_difficulty_stats.csv')
        
        # Feature importance using simple regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        feature_cols = [f for f in numeric_features if f in df.columns]
        X = df[feature_cols].fillna(0)
        y = df['difficulty_score']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit random forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        # Save summary report
        with open(self.output_dir / 'difficulty_summary.txt', 'w') as f:
            f.write("Difficulty Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Top difficulty correlations:\n")
            for feature, corr in correlations.head(5).items():
                f.write(f"  {feature}: {corr:.3f}\n")
            
            f.write("\nMost difficult groups:\n")
            difficult_groups = df.groupby('group')['difficulty_score'].mean().sort_values(ascending=False)
            for group, score in difficult_groups.head(5).items():
                f.write(f"  {group}: {score:.1f}\n")
            
            f.write("\nFeature importance (Random Forest):\n")
            for _, row in importance.head(5).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.3f}\n")