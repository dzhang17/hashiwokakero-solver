# src/core/__init__.py
"""
Core data structures and utilities for Hashiwokakero solver.
"""

from .puzzle import Puzzle, Island, Bridge, Difficulty
from .validator import PuzzleValidator, ValidationResult
from .utils import (
    setup_logger, timer, memory_usage,
    PuzzleConverter, DifficultyEstimator,
    save_puzzle_batch, load_puzzle_batch,
    calculate_solution_stats
)

__all__ = [
    # Data structures
    'Puzzle', 'Island', 'Bridge', 'Difficulty',
    
    # Validation
    'PuzzleValidator', 'ValidationResult',
    
    # Utilities
    'setup_logger', 'timer', 'memory_usage',
    'PuzzleConverter', 'DifficultyEstimator',
    'save_puzzle_batch', 'load_puzzle_batch',
    'calculate_solution_stats'
]