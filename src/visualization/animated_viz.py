"""
Animated visualization for Hashiwokakero solving process.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import json

from ..core.puzzle import Puzzle, Island, Bridge
from .static_viz import PuzzleVisualizer


class AnimatedSolver:
    """Create animated visualization of solving process"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10), dpi: int = 100):
        """
        Initialize animated solver.
        
        Args:
            figsize: Figure size
            dpi: Dots per inch for output
        """
        self.figsize = figsize
        self.dpi = dpi
        self.viz = PuzzleVisualizer(figsize=figsize, dpi=dpi)
        
    def animate_solution_steps(self, puzzle: Puzzle, 
                             solution_steps: List[Dict[str, Any]],
                             output_path: Path,
                             fps: int = 2,
                             show_stats: bool = True):
        """
        Create animation of solution steps.
        
        Args:
            puzzle: Original puzzle
            solution_steps: List of solution step dictionaries
            output_path: Path to save animation (mp4 or gif)
            fps: Frames per second
            show_stats: Whether to show statistics panel
        """
        fig, (ax_main, ax_stats) = plt.subplots(
            1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]),
            gridspec_kw={'width_ratios': [2, 1]} if show_stats else {'width_ratios': [1, 0]}
        )
        
        if not show_stats:
            ax_stats.set_visible(False)
            
        # Set up main axis
        ax_main.set_xlim(-0.5, puzzle.width - 0.5)
        ax_main.set_ylim(-0.5, puzzle.height - 0.5)
        ax_main.set_aspect('equal')
        ax_main.invert_yaxis()
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        
        # Draw static elements (islands)
        self._draw_static_elements(ax_main, puzzle)
        
        # Initialize bridge lines
        bridge_lines = {}
        
        # Statistics text
        if show_stats:
            ax_stats.set_xlim(0, 1)
            ax_stats.set_ylim(0, 1)
            ax_stats.axis('off')
            
            stats_text = ax_stats.text(
                0.1, 0.9, '', fontsize=12, 
                verticalalignment='top',
                fontfamily='monospace'
            )
            
        # Animation function
        def animate(frame):
            if frame >= len(solution_steps):
                return
                
            step = solution_steps[frame]
            current_puzzle = step.get('puzzle')
            stats = step.get('stats', {})
            
            # Clear existing bridges
            for line in bridge_lines.values():
                line.remove()
            bridge_lines.clear()
            
            # Draw current bridges
            if current_puzzle:
                for bridge in current_puzzle.bridges:
                    line = self._draw_bridge(ax_main, current_puzzle, bridge)
                    if line:
                        bridge_lines[f"{bridge.island1_id}_{bridge.island2_id}"] = line
                        
            # Update title
            title = f"Step {frame + 1}/{len(solution_steps)}"
            if 'algorithm' in stats:
                title += f" - {stats['algorithm']}"
            ax_main.set_title(title, fontsize=14)
            
            # Update statistics
            if show_stats and stats_text:
                stats_str = self._format_stats(stats, current_puzzle, puzzle)
                stats_text.set_text(stats_str)
                
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(solution_steps),
            interval=1000/fps, blit=False
        )
        
        # Save animation
        if output_path.suffix == '.gif':
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            
        anim.save(str(output_path), writer=writer)
        plt.close(fig)
        
        print(f"Animation saved to {output_path}")
        
    def create_algorithm_comparison(self, puzzle: Puzzle,
                                  algorithm_results: Dict[str, List[Dict]],
                                  output_path: Path,
                                  fps: int = 2):
        """
        Create side-by-side comparison of different algorithms.
        
        Args:
            puzzle: Original puzzle
            algorithm_results: Dict of {algorithm_name: solution_steps}
            output_path: Path to save animation
            fps: Frames per second
        """
        n_algorithms = len(algorithm_results)
        fig, axes = plt.subplots(
            1, n_algorithms, 
            figsize=(5 * n_algorithms, 5)
        )
        
        if n_algorithms == 1:
            axes = [axes]
            
        # Set up each axis
        for ax, (alg_name, _) in zip(axes, algorithm_results.items()):
            ax.set_xlim(-0.5, puzzle.width - 0.5)
            ax.set_ylim(-0.5, puzzle.height - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(alg_name.upper(), fontsize=12)
            
            # Draw static elements
            self._draw_static_elements(ax, puzzle)
            
        # Track bridges for each algorithm
        all_bridge_lines = {alg: {} for alg in algorithm_results}
        
        # Find maximum steps
        max_steps = max(len(steps) for steps in algorithm_results.values())
        
        def animate(frame):
            for ax, (alg_name, steps) in zip(axes, algorithm_results.items()):
                # Clear existing bridges
                for line in all_bridge_lines[alg_name].values():
                    line.remove()
                all_bridge_lines[alg_name].clear()
                
                # Get current step
                if frame < len(steps):
                    step = steps[frame]
                    current_puzzle = step.get('puzzle')
                    
                    # Draw bridges
                    if current_puzzle:
                        for bridge in current_puzzle.bridges:
                            line = self._draw_bridge(ax, current_puzzle, bridge)
                            if line:
                                key = f"{bridge.island1_id}_{bridge.island2_id}"
                                all_bridge_lines[alg_name][key] = line
                                
                        # Update progress
                        progress = self._calculate_progress(current_puzzle, puzzle)
                        ax.set_xlabel(f"Progress: {progress:.1f}%", fontsize=10)
                        
            fig.suptitle(f"Step {frame + 1}/{max_steps}", fontsize=14)
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_steps,
            interval=1000/fps, blit=False
        )
        
        # Save
        if output_path.suffix == '.gif':
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            
        anim.save(str(output_path), writer=writer)
        plt.close(fig)
        
    def _draw_static_elements(self, ax, puzzle: Puzzle):
        """Draw static elements (islands and grid)"""
        # Draw grid
        for x in range(puzzle.width):
            ax.axvline(x, color='#E0E0E0', linewidth=0.5, alpha=0.5)
        for y in range(puzzle.height):
            ax.axhline(y, color='#E0E0E0', linewidth=0.5, alpha=0.5)
            
        # Draw islands
        for island in puzzle.islands:
            # Shadow
            shadow = Circle(
                (island.col + 0.02, island.row + 0.02),
                self.viz.island_radius,
                color='gray', alpha=0.3, zorder=1
            )
            ax.add_patch(shadow)
            
            # Island circle
            circle = Circle(
                (island.col, island.row),
                self.viz.island_radius,
                color=self.viz.island_color,
                zorder=2
            )
            ax.add_patch(circle)
            
            # Number
            ax.text(
                island.col, island.row,
                str(island.required_bridges),
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white', zorder=3
            )
            
    def _draw_bridge(self, ax, puzzle: Puzzle, bridge: Bridge):
        """Draw a single bridge"""
        island1 = puzzle._id_to_island[bridge.island1_id]
        island2 = puzzle._id_to_island[bridge.island2_id]
        
        # Calculate endpoints
        dx = island2.col - island1.col
        dy = island2.row - island1.row
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return None
            
        # Unit vector
        ux = dx / length
        uy = dy / length
        
        # Endpoints (offset by island radius)
        x1 = island1.col + ux * self.viz.island_radius
        y1 = island1.row + uy * self.viz.island_radius
        x2 = island2.col - ux * self.viz.island_radius
        y2 = island2.row - uy * self.viz.island_radius
        
        if bridge.count == 1:
            # Single bridge
            line = ax.plot(
                [x1, x2], [y1, y2],
                color=self.viz.bridge_color,
                linewidth=self.viz.bridge_width * 40,
                solid_capstyle='round',
                zorder=0
            )[0]
            return line
        else:
            # Double bridge - draw as thick line for simplicity
            line = ax.plot(
                [x1, x2], [y1, y2],
                color=self.viz.bridge_color,
                linewidth=self.viz.bridge_width * 60,
                solid_capstyle='round',
                zorder=0
            )[0]
            return line
            
    def _calculate_progress(self, current: Puzzle, target: Puzzle) -> float:
        """Calculate solving progress as percentage"""
        total_required = sum(island.required_bridges for island in target.islands)
        current_bridges = sum(current.get_island_bridges(island.id) 
                            for island in current.islands)
        
        if total_required == 0:
            return 100.0
            
        return (current_bridges / total_required) * 100
        
    def _format_stats(self, stats: Dict[str, Any], 
                     current: Puzzle, target: Puzzle) -> str:
        """Format statistics for display"""
        lines = ["SOLVING STATISTICS", "=" * 20, ""]
        
        # Progress
        progress = self._calculate_progress(current, target)
        lines.append(f"Progress: {progress:5.1f}%")
        
        # Bridge counts
        total_required = sum(island.required_bridges for island in target.islands)
        current_bridges = sum(current.get_island_bridges(island.id) 
                            for island in current.islands)
        lines.append(f"Bridges: {current_bridges}/{total_required}")
        
        # Algorithm-specific stats
        if 'iteration' in stats:
            lines.append(f"Iteration: {stats['iteration']}")
            
        if 'temperature' in stats:
            lines.append(f"Temperature: {stats['temperature']:.2f}")
            
        if 'current_cost' in stats:
            lines.append(f"Cost: {stats['current_cost']:.2f}")
            
        if 'time' in stats:
            lines.append(f"Time: {stats['time']:.2f}s")
            
        # Validation status
        lines.append("")
        lines.append("VALIDATION")
        lines.append("-" * 20)
        
        # Check completeness
        complete = current.is_complete()
        lines.append(f"Complete: {'Yes' if complete else 'No'}")
        
        # Check connectivity
        if complete:
            connected = current.is_connected()
            lines.append(f"Connected: {'Yes' if connected else 'No'}")
            
        # Check crossing
        has_crossing = current.has_crossing_bridges()
        lines.append(f"Crossings: {'Yes' if has_crossing else 'No'}")
        
        return '\n'.join(lines)


class SolvingRecorder:
    """Record solving process for later animation"""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        
    def record_step(self, puzzle: Puzzle, stats: Optional[Dict[str, Any]] = None):
        """Record a solving step"""
        step = {
            'puzzle': puzzle.copy(),
            'stats': stats or {},
            'timestamp': len(self.steps)
        }
        self.steps.append(step)
        
    def save(self, filepath: Path):
        """Save recorded steps to file"""
        data = []
        for step in self.steps:
            step_data = {
                'puzzle': step['puzzle'].to_dict(),
                'stats': step['stats'],
                'timestamp': step['timestamp']
            }
            data.append(step_data)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, filepath: Path) -> 'SolvingRecorder':
        """Load recorded steps from file"""
        recorder = cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for step_data in data:
            puzzle = Puzzle.from_dict(step_data['puzzle'])
            stats = step_data['stats']
            
            recorder.steps.append({
                'puzzle': puzzle,
                'stats': stats,
                'timestamp': step_data['timestamp']
            })
            
        return recorder
        
    def create_animation(self, original_puzzle: Puzzle, 
                        output_path: Path, **kwargs):
        """Create animation from recorded steps"""
        animator = AnimatedSolver()
        animator.animate_solution_steps(
            original_puzzle, self.steps, output_path, **kwargs
        )