# utils/data_loader.py
from pathlib import Path
import re
from typing import Dict, List, Tuple
import sys
sys.path.append('..')

from src.core.puzzle import Puzzle

class PuzzleLoader:
    @staticmethod
    def load_from_has_file(file_path: Path) -> Puzzle:
        """Load a puzzle from .has format file"""
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse grid size from first line
        first_line = lines[0].strip()
        match = re.match(r'(\d+)\s+(\d+)', first_line)
        if not match:
            raise ValueError(f"Invalid format in {file_path}")
        
        width = int(match.group(1))
        height = int(match.group(2))
        
        # Create puzzle
        puzzle = Puzzle(width, height)
        
        # Parse grid
        for row in range(height):
            if row + 1 >= len(lines):
                break
                
            line = lines[row + 1].strip()
            for col, char in enumerate(line):
                if char.isdigit() and int(char) > 0:
                    puzzle.add_island(row, col, int(char))
        
        return puzzle
    
    @staticmethod
    def parse_filename(filename: str) -> Dict:
        """Parse instance filename to extract metadata"""
        
        # Expected format: Hs_GG_NNN_DD_OO_III.has
        match = re.match(
            r'Hs_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.has',
            filename
        )
        
        if not match:
            return {}
        
        return {
            'grid_size': int(match.group(1)),
            'num_islands': int(match.group(2)),
            'density': int(match.group(3)),
            'obstacles': int(match.group(4)),
            'instance_id': int(match.group(5))
        }