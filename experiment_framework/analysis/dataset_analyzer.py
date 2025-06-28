# analysis/dataset_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import re

class DatasetAnalyzer:
    def __init__(self, dataset_paths: Dict[int, str]):
        self.dataset_paths = {
            size: Path(path) for size, path in dataset_paths.items()
        }
        self.results = []
        
    def analyze_all_datasets(self) -> pd.DataFrame:
        """Analyze characteristics of all datasets"""
        
        for size, path in self.dataset_paths.items():
            print(f"\nAnalyzing {size}-island dataset...")
            
            # Get all .has files
            files = list(path.glob("*.has"))
            print(f"Found {len(files)} instances")
            
            for file in files:
                instance_info = self.analyze_instance(file, size)
                if instance_info:
                    self.results.append(instance_info)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Generate visualizations
        self.generate_characteristic_plots(df)
        
        return df
    
    def analyze_instance(self, file_path: Path, expected_size: int) -> Dict:
        """Analyze a single instance file"""
        
        # Parse filename: Hs_GG_NNN_DD_OO_III.has
        match = re.match(
            r'Hs_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.has', 
            file_path.name
        )
        
        if not match:
            print(f"Warning: Could not parse filename {file_path.name}")
            return None
        
        grid_size = int(match.group(1))
        num_islands = int(match.group(2))
        density = int(match.group(3))
        obstacles = int(match.group(4))
        instance_id = int(match.group(5))
        
        # Load and analyze puzzle structure
        puzzle_stats = self.load_puzzle_stats(file_path)
        
        return {
            'size_category': expected_size,
            'grid_size': grid_size,
            'num_islands': num_islands,
            'density': density,
            'obstacles': obstacles,
            'instance_id': instance_id,
            'filename': file_path.name,
            'file_path': str(file_path),
            **puzzle_stats
        }
    
    def load_puzzle_stats(self, file_path: Path) -> Dict:
        """Load puzzle and calculate statistics"""
        try:
            # Import your puzzle loader
            import sys
            sys.path.append('..')
            from src.core.puzzle import Puzzle
            
            puzzle = Puzzle.load_from_has(file_path)
            
            # Calculate statistics
            island_degrees = [island.required_bridges for island in puzzle.islands]
            
            return {
                'actual_islands': len(puzzle.islands),
                'grid_width': puzzle.width,
                'grid_height': puzzle.height,
                'min_degree': min(island_degrees) if island_degrees else 0,
                'max_degree': max(island_degrees) if island_degrees else 0,
                'avg_degree': np.mean(island_degrees) if island_degrees else 0,
                'std_degree': np.std(island_degrees) if island_degrees else 0,
                'total_bridges': sum(island_degrees) // 2
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {
                'actual_islands': 0,
                'grid_width': 0,
                'grid_height': 0,
                'min_degree': 0,
                'max_degree': 0,
                'avg_degree': 0,
                'std_degree': 0,
                'total_bridges': 0
            }
    
    def generate_characteristic_plots(self, df: pd.DataFrame):
        """Generate visualization of dataset characteristics"""
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Instance distribution by size
        ax1 = plt.subplot(3, 3, 1)
        df['size_category'].value_counts().sort_index().plot(
            kind='bar', ax=ax1, color='skyblue'
        )
        ax1.set_title('Instances per Size Category')
        ax1.set_xlabel('Number of Islands')
        ax1.set_ylabel('Count')
        
        # 2. Density distribution
        ax2 = plt.subplot(3, 3, 2)
        for size in sorted(df['size_category'].unique()):
            subset = df[df['size_category'] == size]
            density_counts = subset['density'].value_counts().sort_index()
            ax2.plot(density_counts.index, density_counts.values, 
                    marker='o', label=f'{size} islands')
        ax2.set_title('Density Distribution by Size')
        ax2.set_xlabel('Density (%)')
        ax2.set_ylabel('Count')
        ax2.legend()
        
        # 3. Obstacle distribution
        ax3 = plt.subplot(3, 3, 3)
        for size in sorted(df['size_category'].unique()):
            subset = df[df['size_category'] == size]
            obstacle_counts = subset['obstacles'].value_counts().sort_index()
            ax3.plot(obstacle_counts.index, obstacle_counts.values, 
                    marker='s', label=f'{size} islands')
        ax3.set_title('Obstacle Distribution by Size')
        ax3.set_xlabel('Obstacles (%)')
        ax3.set_ylabel('Count')
        ax3.legend()
        
        # 4. Average degree distribution
        ax4 = plt.subplot(3, 3, 4)
        df.boxplot(column='avg_degree', by='size_category', ax=ax4)
        ax4.set_title('Average Island Degree by Size')
        ax4.set_xlabel('Size Category')
        ax4.set_ylabel('Average Degree')
        
        # 5. Total bridges vs islands
        ax5 = plt.subplot(3, 3, 5)
        colors = df['density'].values
        scatter = ax5.scatter(df['num_islands'], df['total_bridges'], 
                            c=colors, cmap='viridis', alpha=0.6)
        ax5.set_title('Total Bridges vs Number of Islands')
        ax5.set_xlabel('Number of Islands')
        ax5.set_ylabel('Total Bridges')
        plt.colorbar(scatter, ax=ax5, label='Density (%)')
        
        # 6. Grid size distribution
        ax6 = plt.subplot(3, 3, 6)
        df.groupby(['size_category', 'grid_size']).size().unstack().plot(
            kind='bar', ax=ax6, stacked=True
        )
        ax6.set_title('Grid Size Distribution')
        ax6.set_xlabel('Size Category')
        ax6.set_ylabel('Count')
        ax6.legend(title='Grid Size')
        
        # 7. Degree statistics
        ax7 = plt.subplot(3, 3, 7)
        degree_stats = df.groupby('size_category')[['min_degree', 'max_degree']].mean()
        degree_stats.plot(kind='bar', ax=ax7)
        ax7.set_title('Average Min/Max Degree by Size')
        ax7.set_xlabel('Size Category')
        ax7.set_ylabel('Degree')
        ax7.legend(['Min Degree', 'Max Degree'])
        
        # 8. Density vs Difficulty proxy (std of degrees)
        ax8 = plt.subplot(3, 3, 8)
        for size in sorted(df['size_category'].unique()):
            subset = df[df['size_category'] == size]
            density_groups = subset.groupby('density')['std_degree'].mean()
            ax8.plot(density_groups.index, density_groups.values, 
                    marker='o', label=f'{size} islands')
        ax8.set_title('Degree Std Dev vs Density')
        ax8.set_xlabel('Density (%)')
        ax8.set_ylabel('Std Dev of Island Degrees')
        ax8.legend()
        
        # 9. Summary statistics table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        summary_stats = df.groupby('size_category').agg({
            'num_islands': 'count',
            'total_bridges': 'mean',
            'avg_degree': 'mean',
            'grid_size': lambda x: x.mode()[0] if len(x) > 0 else 0
        }).round(2)
        
        summary_stats.columns = ['Count', 'Avg Bridges', 'Avg Degree', 'Grid Size']
        table = ax9.table(cellText=summary_stats.values,
                         colLabels=summary_stats.columns,
                         rowLabels=summary_stats.index,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax9.set_title('Summary Statistics by Size Category')
        
        plt.tight_layout()
        plt.savefig('dataset_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional detailed plots
        self.generate_detailed_plots(df)
    
    def generate_detailed_plots(self, df: pd.DataFrame):
        """Generate additional detailed analysis plots"""
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['num_islands', 'density', 'obstacles', 'total_bridges',
                       'avg_degree', 'std_degree', 'min_degree', 'max_degree']
        
        # Filter columns that exist in the dataframe
        existing_cols = [col for col in numeric_cols if col in df.columns]
        
        correlation_matrix = df[existing_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('feature_correlation.png', dpi=300)
        plt.close()