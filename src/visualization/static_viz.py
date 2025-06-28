"""
Static visualization for Hashiwokakero puzzles.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from ..core.puzzle import Puzzle, Island, Bridge


class PuzzleVisualizer:
    """Visualize Hashiwokakero puzzles"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10), dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size in inches
            dpi: Dots per inch for saved images
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Visual parameters
        self.island_radius = 0.3
        self.bridge_width = 0.05
        self.double_bridge_gap = 0.1
        self.grid_color = '#E0E0E0'
        self.island_color = '#2E86AB'
        self.bridge_color = '#424874'
        self.number_color = 'white'
        self.background_color = '#F7F7F7'
        
    def visualize(self, puzzle: Puzzle, 
                 show_solution: bool = True,
                 show_grid: bool = True,
                 title: Optional[str] = None,
                 save_path: Optional[Path] = None,
                 show_plot: bool = True) -> plt.Figure:
        """
        Create visualization of puzzle.
        
        Args:
            puzzle: The puzzle to visualize
            show_solution: Whether to show bridges
            show_grid: Whether to show grid lines
            title: Optional title for the plot
            save_path: Optional path to save the image
            show_plot: Whether to display the plot
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor(self.background_color)
        ax.set_facecolor(self.background_color)
        
        # Set up the plot
        ax.set_xlim(-0.5, puzzle.width - 0.5)
        ax.set_ylim(-0.5, puzzle.height - 0.5)
        ax.set_aspect('equal')
        
        # Invert y-axis to match typical puzzle orientation
        ax.invert_yaxis()
        
        # Draw grid if requested
        if show_grid:
            self._draw_grid(ax, puzzle.width, puzzle.height)
            
        # Draw bridges first (so they appear behind islands)
        if show_solution and puzzle.bridges:
            self._draw_bridges(ax, puzzle)
            
        # Draw islands
        self._draw_islands(ax, puzzle)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=16, pad=20)
            
        # Save if requested
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.background_color)
            
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def _draw_grid(self, ax, width: int, height: int):
        """Draw background grid"""
        # Vertical lines
        for x in range(width):
            ax.axvline(x, color=self.grid_color, linewidth=0.5, alpha=0.5)
            
        # Horizontal lines
        for y in range(height):
            ax.axhline(y, color=self.grid_color, linewidth=0.5, alpha=0.5)
            
    def _draw_islands(self, ax, puzzle: Puzzle):
        """Draw all islands"""
        for island in puzzle.islands:
            # Draw circle
            circle = plt.Circle((island.col, island.row), self.island_radius,
                              color=self.island_color, zorder=2)
            ax.add_patch(circle)
            
            # Draw number
            ax.text(island.col, island.row, str(island.required_bridges),
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color=self.number_color, zorder=3)
            
            # Add subtle shadow
            shadow = plt.Circle((island.col + 0.02, island.row + 0.02), 
                              self.island_radius,
                              color='gray', alpha=0.3, zorder=1)
            ax.add_patch(shadow)
            
    def _draw_bridges(self, ax, puzzle: Puzzle):
        """Draw all bridges"""
        for bridge in puzzle.bridges:
            island1 = puzzle._id_to_island[bridge.island1_id]
            island2 = puzzle._id_to_island[bridge.island2_id]
            
            if bridge.count == 1:
                self._draw_single_bridge(ax, island1, island2)
            else:  # count == 2
                self._draw_double_bridge(ax, island1, island2)
                
    def _draw_single_bridge(self, ax, island1: Island, island2: Island):
        """Draw a single bridge between two islands"""
        # Calculate endpoints (stop at island edges)
        dx = island2.col - island1.col
        dy = island2.row - island1.row
        length = np.sqrt(dx**2 + dy**2)
        
        # Unit vector
        ux = dx / length
        uy = dy / length
        
        # Start and end points (offset by island radius)
        x1 = island1.col + ux * self.island_radius
        y1 = island1.row + uy * self.island_radius
        x2 = island2.col - ux * self.island_radius
        y2 = island2.row - uy * self.island_radius
        
        # Draw line
        line = plt.Line2D([x1, x2], [y1, y2], 
                         color=self.bridge_color,
                         linewidth=self.bridge_width * 40,
                         solid_capstyle='round',
                         zorder=0)
        ax.add_line(line)
        
    def _draw_double_bridge(self, ax, island1: Island, island2: Island):
        """Draw a double bridge between two islands"""
        # Calculate endpoints and perpendicular offset
        dx = island2.col - island1.col
        dy = island2.row - island1.row
        length = np.sqrt(dx**2 + dy**2)
        
        # Unit vectors
        ux = dx / length
        uy = dy / length
        
        # Perpendicular unit vector
        px = -uy
        py = ux
        
        # Offset for double bridge
        offset = self.double_bridge_gap / 2
        
        # Draw two parallel bridges
        for sign in [-1, 1]:
            x1 = island1.col + ux * self.island_radius + sign * px * offset
            y1 = island1.row + uy * self.island_radius + sign * py * offset
            x2 = island2.col - ux * self.island_radius + sign * px * offset
            y2 = island2.row - uy * self.island_radius + sign * py * offset
            
            line = plt.Line2D([x1, x2], [y1, y2],
                             color=self.bridge_color,
                             linewidth=self.bridge_width * 30,
                             solid_capstyle='round',
                             zorder=0)
            ax.add_line(line)
            
    def create_comparison_plot(self, puzzles: List[Puzzle], 
                             titles: List[str],
                             save_path: Optional[Path] = None) -> plt.Figure:
        """Create side-by-side comparison of multiple puzzles"""
        n_puzzles = len(puzzles)
        fig, axes = plt.subplots(1, n_puzzles, figsize=(5 * n_puzzles, 5))
        
        if n_puzzles == 1:
            axes = [axes]
            
        for ax, puzzle, title in zip(axes, puzzles, titles):
            ax.set_xlim(-0.5, puzzle.width - 0.5)
            ax.set_ylim(-0.5, puzzle.height - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Draw puzzle
            self._draw_bridges(ax, puzzle)
            self._draw_islands(ax, puzzle)
            
            # Style
            ax.set_title(title, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
                
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
        
    def create_puzzle_sheet(self, puzzles: List[Puzzle],
                          rows: int, cols: int,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """Create a sheet of multiple puzzles (for printing)"""
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        # Flatten axes array for easier iteration
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
            
        puzzle_idx = 0
        
        for i in range(rows):
            for j in range(cols):
                ax = axes[i][j]
                
                if puzzle_idx < len(puzzles):
                    puzzle = puzzles[puzzle_idx]
                    
                    # Set up axis
                    ax.set_xlim(-0.5, puzzle.width - 0.5)
                    ax.set_ylim(-0.5, puzzle.height - 0.5)
                    ax.set_aspect('equal')
                    ax.invert_yaxis()
                    
                    # Draw puzzle (without solution)
                    self._draw_islands(ax, puzzle)
                    
                    # Add puzzle number
                    ax.text(0.02, 0.98, f"#{puzzle_idx + 1}",
                           transform=ax.transAxes,
                           ha='left', va='top',
                           fontsize=10)
                    
                    puzzle_idx += 1
                else:
                    ax.set_visible(False)
                    
                # Remove axes
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig


class AnimatedVisualizer(PuzzleVisualizer):
    """Create animated visualizations (saves as static frames)"""
    
    def visualize_solving_process(self, puzzle: Puzzle,
                                 solution_steps: List[Puzzle],
                                 save_dir: Path,
                                 fps: int = 2):
        """
        Create frames showing the solving process.
        
        Args:
            puzzle: Original puzzle
            solution_steps: List of intermediate solutions
            save_dir: Directory to save frames
            fps: Frames per second (for naming)
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create frame for each step
        for i, step in enumerate(solution_steps):
            save_path = save_dir / f"frame_{i:04d}.png"
            
            # Calculate completion percentage
            total_required = sum(island.required_bridges for island in puzzle.islands)
            current_bridges = sum(step.get_island_bridges(island.id) 
                                for island in step.islands)
            completion = current_bridges / total_required * 100 if total_required > 0 else 0
            
            title = f"Step {i+1}/{len(solution_steps)} - {completion:.1f}% Complete"
            
            self.visualize(step, show_solution=True, show_grid=True,
                         title=title, save_path=save_path, show_plot=False)
            
    def create_difficulty_showcase(self, puzzles_by_difficulty: dict,
                                  save_path: Path):
        """Create a showcase of puzzles at different difficulties"""
        difficulties = ['easy', 'medium', 'hard', 'expert']
        n_difficulties = len(difficulties)
        
        fig, axes = plt.subplots(2, n_difficulties, figsize=(n_difficulties * 4, 8))
        
        for col, difficulty in enumerate(difficulties):
            if difficulty in puzzles_by_difficulty and puzzles_by_difficulty[difficulty]:
                puzzle = puzzles_by_difficulty[difficulty][0]
                
                # Top row: puzzle
                ax_puzzle = axes[0, col]
                ax_puzzle.set_xlim(-0.5, puzzle.width - 0.5)
                ax_puzzle.set_ylim(-0.5, puzzle.height - 0.5)
                ax_puzzle.set_aspect('equal')
                ax_puzzle.invert_yaxis()
                
                self._draw_islands(ax_puzzle, puzzle)
                ax_puzzle.set_title(f"{difficulty.capitalize()}\n{puzzle.width}x{puzzle.height}",
                                  fontsize=12)
                
                # Bottom row: solution
                ax_solution = axes[1, col]
                ax_solution.set_xlim(-0.5, puzzle.width - 0.5)
                ax_solution.set_ylim(-0.5, puzzle.height - 0.5)
                ax_solution.set_aspect('equal')
                ax_solution.invert_yaxis()
                
                self._draw_bridges(ax_solution, puzzle)
                self._draw_islands(ax_solution, puzzle)
                ax_solution.set_title("Solution", fontsize=10)
                
                # Clean up axes
                for ax in [ax_puzzle, ax_solution]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig