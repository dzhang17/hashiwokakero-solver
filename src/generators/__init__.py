"""
Puzzle generators for Hashiwokakero.
"""

from .puzzle_generator import PuzzleGenerator, PuzzleGeneratorConfig
from .pattern_generator import (
    PatternGenerator, PatternType, SymmetryType,
    BasePattern, SymmetricPattern, GridPattern,
    SpiralPattern, DiagonalPattern, StarPattern,
    FramePattern, ClusterPattern
)

__all__ = [
    # Main generators
    'PuzzleGenerator', 'PuzzleGeneratorConfig',
    'PatternGenerator',
    
    # Pattern types
    'PatternType', 'SymmetryType',
    
    # Pattern classes
    'BasePattern', 'SymmetricPattern', 'GridPattern',
    'SpiralPattern', 'DiagonalPattern', 'StarPattern',
    'FramePattern', 'ClusterPattern'
]