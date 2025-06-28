"""
Utility functions for Hashiwokakero solver.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import time
from functools import wraps
import numpy as np
from ..core.puzzle import Puzzle, Island, Bridge, Difficulty


def setup_logger(name: str, log_file: Optional[Path] = None, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Try to get logger from first argument (usually self)
        if args and hasattr(args[0], 'logger'):
            args[0].logger.debug(f"{func.__name__} took {execution_time:.3f} seconds")
        else:
            print(f"{func.__name__} took {execution_time:.3f} seconds")
            
        return result
    return wrapper


def memory_usage():
    """Get current memory usage in MB"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class PuzzleConverter:
    """Convert puzzles between different formats"""
    
    @staticmethod
    def to_grid(puzzle: Puzzle) -> np.ndarray:
        """
        Convert puzzle to 2D grid representation.
        0: empty, 1-8: island with that many required bridges
        """
        grid = np.zeros((puzzle.height, puzzle.width), dtype=int)
        for island in puzzle.islands:
            grid[island.row, island.col] = island.required_bridges
        return grid
        
    @staticmethod
    def from_grid(grid: np.ndarray) -> Puzzle:
        """Create puzzle from 2D grid representation"""
        height, width = grid.shape
        puzzle = Puzzle(width, height)
        
        for row in range(height):
            for col in range(width):
                if grid[row, col] > 0:
                    puzzle.add_island(row, col, int(grid[row, col]))
                    
        return puzzle
        
    @staticmethod
    def to_string(puzzle: Puzzle, show_bridges: bool = False) -> str:
        """
        Convert puzzle to string representation.
        
        Args:
            puzzle: The puzzle to convert
            show_bridges: Whether to show bridges in the output
            
        Returns:
            String representation of the puzzle
        """
        # Create grid
        grid = [[' ' for _ in range(puzzle.width * 2 - 1)] 
                for _ in range(puzzle.height * 2 - 1)]
        
        # Place islands
        for island in puzzle.islands:
            grid[island.row * 2][island.col * 2] = str(island.required_bridges)
            
        # Place bridges if requested
        if show_bridges:
            for bridge in puzzle.bridges:
                i1 = puzzle._id_to_island[bridge.island1_id]
                i2 = puzzle._id_to_island[bridge.island2_id]
                
                if i1.row == i2.row:  # Horizontal bridge
                    row = i1.row * 2
                    start_col = min(i1.col, i2.col) * 2 + 1
                    end_col = max(i1.col, i2.col) * 2
                    
                    for col in range(start_col, end_col):
                        if bridge.count == 1:
                            grid[row][col] = '-'
                        else:
                            grid[row][col] = '='
                else:  # Vertical bridge
                    col = i1.col * 2
                    start_row = min(i1.row, i2.row) * 2 + 1
                    end_row = max(i1.row, i2.row) * 2
                    
                    for row in range(start_row, end_row):
                        if bridge.count == 1:
                            grid[row][col] = '|'
                        else:
                            grid[row][col] = '‖'
                            
        # Convert to string
        return '\n'.join([''.join(row) for row in grid])
        
    @staticmethod
    def from_string(s: str) -> Puzzle:
        """
        Create puzzle from string representation.
        Numbers 1-8 represent islands, other characters are ignored.
        """
        lines = s.strip().split('\n')
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0
        
        puzzle = Puzzle(width, height)
        
        for row, line in enumerate(lines):
            for col, char in enumerate(line):
                if char.isdigit() and '1' <= char <= '8':
                    puzzle.add_island(row, col, int(char))
                    
        return puzzle


class DifficultyEstimator:
    """Estimate puzzle difficulty based on various heuristics"""
    
    @staticmethod
    def estimate_difficulty(puzzle: Puzzle) -> Difficulty:
        """
        Estimate puzzle difficulty based on various factors.
        
        Args:
            puzzle: The puzzle to analyze
            
        Returns:
            Estimated difficulty level
        """
        score = 0
        
        # Factor 1: Number of islands (more islands = harder)
        num_islands = len(puzzle.islands)
        if num_islands < 10:
            score += 1
        elif num_islands < 20:
            score += 2
        elif num_islands < 30:
            score += 3
        else:
            score += 4
            
        # Factor 2: Grid size
        grid_size = puzzle.width * puzzle.height
        if grid_size < 50:
            score += 1
        elif grid_size < 150:
            score += 2
        elif grid_size < 300:
            score += 3
        else:
            score += 4
            
        # Factor 3: Average connections per island
        if puzzle.islands:
            avg_connections = sum(len(puzzle._valid_connections[i.id]) 
                                for i in puzzle.islands) / len(puzzle.islands)
            if avg_connections < 2:
                score += 1
            elif avg_connections < 3:
                score += 2
            elif avg_connections < 4:
                score += 3
            else:
                score += 4
                
        # Factor 4: Islands with high bridge requirements
        high_req_islands = sum(1 for i in puzzle.islands if i.required_bridges >= 6)
        if high_req_islands == 0:
            score += 1
        elif high_req_islands < 3:
            score += 2
        elif high_req_islands < 6:
            score += 3
        else:
            score += 4
            
        # Factor 5: Forced connections (islands with only one valid neighbor)
        forced = sum(1 for i in puzzle.islands 
                    if len(puzzle._valid_connections[i.id]) == 1)
        if forced > num_islands * 0.5:
            score -= 2  # Many forced connections make puzzle easier
        elif forced > num_islands * 0.3:
            score -= 1
            
        # Map score to difficulty
        if score <= 4:
            return Difficulty.EASY
        elif score <= 8:
            return Difficulty.MEDIUM
        elif score <= 12:
            return Difficulty.HARD
        else:
            return Difficulty.EXPERT


def save_puzzle_batch(puzzles: List[Puzzle], directory: Path, prefix: str = "puzzle"):
    """Save multiple puzzles to a directory"""
    directory.mkdir(parents=True, exist_ok=True)
    
    for i, puzzle in enumerate(puzzles):
        filename = directory / f"{prefix}_{i:04d}.json"
        puzzle.save(filename)
        

def load_puzzle_batch(directory: Path, pattern: str = "*.json") -> List[Puzzle]:
    """Load multiple puzzles from a directory"""
    puzzles = []
    
    for filepath in sorted(directory.glob(pattern)):
        try:
            puzzle = Puzzle.load(filepath)
            puzzles.append(puzzle)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    return puzzles


def calculate_solution_stats(puzzle: Puzzle) -> Dict[str, Any]:
    """Calculate statistics for a solved puzzle"""
    stats = {
        'total_bridges': sum(b.count for b in puzzle.bridges),
        'single_bridges': sum(1 for b in puzzle.bridges if b.count == 1),
        'double_bridges': sum(1 for b in puzzle.bridges if b.count == 2),
        'is_complete': puzzle.is_complete(),
        'is_connected': puzzle.is_connected(),
        'has_crossing': puzzle.has_crossing_bridges()
    }
    
    # Calculate average bridge length
    if puzzle.bridges:
        lengths = []
        for bridge in puzzle.bridges:
            i1 = puzzle._id_to_island[bridge.island1_id]
            i2 = puzzle._id_to_island[bridge.island2_id]
            length = abs(i1.row - i2.row) + abs(i1.col - i2.col)
            lengths.append(length)
        stats['avg_bridge_length'] = sum(lengths) / len(lengths)
        stats['max_bridge_length'] = max(lengths)
        stats['min_bridge_length'] = min(lengths)
    else:
        stats['avg_bridge_length'] = 0
        stats['max_bridge_length'] = 0
        stats['min_bridge_length'] = 0
        
    return stats