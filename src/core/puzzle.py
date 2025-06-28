"""
Core data structure for Hashiwokakero puzzles.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Union
import numpy as np
from enum import Enum
import json
from pathlib import Path
import re


class Difficulty(Enum):
    """Puzzle difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class Island:
    """Represents an island in the puzzle"""
    row: int
    col: int
    required_bridges: int
    id: int = field(default=-1)
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        if isinstance(other, Island):
            return self.row == other.row and self.col == other.col
        return False
    
    def __repr__(self):
        return f"Island({self.row}, {self.col}, bridges={self.required_bridges})"


@dataclass
class Bridge:
    """Represents a bridge between two islands"""
    island1_id: int
    island2_id: int
    count: int = 1  # 1 or 2 bridges
    
    def __hash__(self):
        return hash(tuple(sorted([self.island1_id, self.island2_id])))
    
    def __eq__(self, other):
        if isinstance(other, Bridge):
            ids1 = sorted([self.island1_id, self.island2_id])
            ids2 = sorted([other.island1_id, other.island2_id])
            return ids1 == ids2
        return False
    
    def __repr__(self):
        return f"Bridge({self.island1_id}<->{self.island2_id}, count={self.count})"


class Puzzle:
    """Main puzzle class for Hashiwokakero"""
    
    def __init__(self, width: int, height: int, islands: Optional[List[Island]] = None):
        """
        Initialize a Hashiwokakero puzzle.
        
        Args:
            width: Width of the puzzle grid
            height: Height of the puzzle grid
            islands: List of islands in the puzzle
        """
        self.width = width
        self.height = height
        self.islands: List[Island] = islands or []
        self.bridges: List[Bridge] = []
        self._island_map: Dict[Tuple[int, int], Island] = {}
        self._id_to_island: Dict[int, Island] = {}
        
        # Assign IDs to islands and build maps
        self._initialize_islands()
        
        # Precompute valid connections
        self._valid_connections: Dict[int, Set[int]] = {}
        self._compute_valid_connections()
        
    def _initialize_islands(self):
        """Initialize island IDs and build lookup maps"""
        for i, island in enumerate(self.islands):
            island.id = i
            self._island_map[(island.row, island.col)] = island
            self._id_to_island[island.id] = island
            
    def _compute_valid_connections(self):
        """Precompute which islands can be connected"""
        for island in self.islands:
            self._valid_connections[island.id] = set()
            
            # Check horizontal connections
            for other in self.islands:
                if island == other:
                    continue
                    
                if self._can_connect(island, other):
                    self._valid_connections[island.id].add(other.id)
    
    def _can_connect(self, island1: Island, island2: Island) -> bool:
        """Check if two islands can be connected with a bridge"""
        # Must be on same row or column
        if island1.row != island2.row and island1.col != island2.col:
            return False
            
        # Check if path is clear (no islands in between)
        if island1.row == island2.row:  # Horizontal
            min_col = min(island1.col, island2.col)
            max_col = max(island1.col, island2.col)
            for col in range(min_col + 1, max_col):
                if (island1.row, col) in self._island_map:
                    return False
        else:  # Vertical
            min_row = min(island1.row, island2.row)
            max_row = max(island1.row, island2.row)
            for row in range(min_row + 1, max_row):
                if (row, island1.col) in self._island_map:
                    return False
                    
        return True
    
    def add_island(self, row: int, col: int, required_bridges: int):
        """Add an island to the puzzle"""
        if (row, col) in self._island_map:
            raise ValueError(f"Island already exists at ({row}, {col})")
            
        island = Island(row, col, required_bridges)
        island.id = len(self.islands)
        self.islands.append(island)
        self._island_map[(row, col)] = island
        self._id_to_island[island.id] = island
        
        # Update valid connections
        self._compute_valid_connections()
        
    def add_bridge(self, island1_id: int, island2_id: int, count: int = 1):
        """Add a bridge between two islands"""
        if island1_id not in self._id_to_island or island2_id not in self._id_to_island:
            raise ValueError("Invalid island ID")
            
        if island2_id not in self._valid_connections.get(island1_id, set()):
            raise ValueError("Islands cannot be connected")
            
        # Check if bridge already exists
        for bridge in self.bridges:
            if bridge == Bridge(island1_id, island2_id, count):
                bridge.count = min(bridge.count + count, 2)
                return
                
        self.bridges.append(Bridge(island1_id, island2_id, count))
        
    def remove_bridge(self, island1_id: int, island2_id: int):
        """Remove a bridge between two islands"""
        self.bridges = [b for b in self.bridges 
                       if not (b == Bridge(island1_id, island2_id, 1))]
        
    def get_island_bridges(self, island_id: int) -> int:
        """Get the number of bridges connected to an island"""
        count = 0
        for bridge in self.bridges:
            if bridge.island1_id == island_id or bridge.island2_id == island_id:
                count += bridge.count
        return count
        
    def is_complete(self) -> bool:
        """Check if all islands have the required number of bridges"""
        for island in self.islands:
            if self.get_island_bridges(island.id) != island.required_bridges:
                return False
        return True
        
    def is_connected(self) -> bool:
        """Check if all islands are connected (single connected component)"""
        if not self.islands:
            return True
            
        # Build adjacency list
        adj = {island.id: set() for island in self.islands}
        for bridge in self.bridges:
            if bridge.count > 0:
                adj[bridge.island1_id].add(bridge.island2_id)
                adj[bridge.island2_id].add(bridge.island1_id)
                
        # DFS to check connectivity
        visited = set()
        stack = [self.islands[0].id]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adj[current] - visited)
            
        return len(visited) == len(self.islands)
        
    def bridges_cross(self, bridge1: Bridge, bridge2: Bridge) -> bool:
        """Check if two bridges cross each other"""
        # Get island positions
        i1_1 = self._id_to_island[bridge1.island1_id]
        i1_2 = self._id_to_island[bridge1.island2_id]
        i2_1 = self._id_to_island[bridge2.island1_id]
        i2_2 = self._id_to_island[bridge2.island2_id]
        
        # Determine if bridges are horizontal or vertical
        b1_horizontal = i1_1.row == i1_2.row
        b2_horizontal = i2_1.row == i2_2.row
        
        # Same orientation - cannot cross
        if b1_horizontal == b2_horizontal:
            return False
            
        if b1_horizontal:
            # Bridge 1 is horizontal, bridge 2 is vertical
            h_row = i1_1.row
            h_col_min = min(i1_1.col, i1_2.col)
            h_col_max = max(i1_1.col, i1_2.col)
            v_col = i2_1.col
            v_row_min = min(i2_1.row, i2_2.row)
            v_row_max = max(i2_1.row, i2_2.row)
            
            return (h_col_min < v_col < h_col_max and 
                   v_row_min < h_row < v_row_max)
        else:
            # Bridge 1 is vertical, bridge 2 is horizontal
            v_col = i1_1.col
            v_row_min = min(i1_1.row, i1_2.row)
            v_row_max = max(i1_1.row, i1_2.row)
            h_row = i2_1.row
            h_col_min = min(i2_1.col, i2_2.col)
            h_col_max = max(i2_1.col, i2_2.col)
            
            return (h_col_min < v_col < h_col_max and 
                   v_row_min < h_row < v_row_max)
            
    def has_crossing_bridges(self) -> bool:
        """Check if any bridges cross each other"""
        for i, bridge1 in enumerate(self.bridges):
            for bridge2 in self.bridges[i+1:]:
                if self.bridges_cross(bridge1, bridge2):
                    return True
        return False
        
    def copy(self) -> 'Puzzle':
        """Create a deep copy of the puzzle"""
        new_puzzle = Puzzle(self.width, self.height)
        new_puzzle.islands = [Island(i.row, i.col, i.required_bridges, i.id) 
                             for i in self.islands]
        new_puzzle.bridges = [Bridge(b.island1_id, b.island2_id, b.count) 
                             for b in self.bridges]
        new_puzzle._initialize_islands()
        new_puzzle._compute_valid_connections()
        return new_puzzle
        
    def to_dict(self) -> dict:
        """Convert puzzle to dictionary for serialization"""
        return {
            'width': self.width,
            'height': self.height,
            'islands': [
                {
                    'row': i.row,
                    'col': i.col,
                    'required_bridges': i.required_bridges
                }
                for i in self.islands
            ],
            'bridges': [
                {
                    'island1_id': b.island1_id,
                    'island2_id': b.island2_id,
                    'count': b.count
                }
                for b in self.bridges
            ]
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Puzzle':
        """Create puzzle from dictionary"""
        puzzle = cls(data['width'], data['height'])
        
        # Add islands
        for island_data in data['islands']:
            puzzle.add_island(
                island_data['row'],
                island_data['col'],
                island_data['required_bridges']
            )
            
        # Add bridges
        for bridge_data in data.get('bridges', []):
            puzzle.add_bridge(
                bridge_data['island1_id'],
                bridge_data['island2_id'],
                bridge_data['count']
            )
            
        return puzzle
        
    def save(self, filepath: Path):
        """Save puzzle to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: Path) -> 'Puzzle':
        """Load puzzle from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
        
    @classmethod
    def load_from_has(cls, filepath: Union[str, Path]) -> 'Puzzle':
        """
        Load a puzzle from a .has file format.
        
        Expected .has format:
        Line 1: width height num_islands
        Following lines: grid where digits (1-8) are islands, 0 are empty
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            raise ValueError(f"Empty file: {filepath}")
        
        # Parse dimensions from first line
        dimensions = lines[0].split()
        if len(dimensions) < 2:
            raise ValueError(f"Invalid format in {filepath}: expected at least 'width height' on first line")
        
        try:
            width = int(dimensions[0])
            height = int(dimensions[1])
            # Third number is num_islands if present (we don't need it for parsing)
            if len(dimensions) >= 3:
                num_islands = int(dimensions[2])
        except ValueError:
            raise ValueError(f"Invalid dimensions in {filepath}: {lines[0]}")
        
        # Create puzzle
        puzzle = cls(width, height)
        
        # Parse grid (starting from second line)
        for row_idx, line in enumerate(lines[1:]):
            if row_idx >= height:
                break
            
            # Split the line into individual numbers
            cells = line.split()
            
            for col_idx, cell in enumerate(cells):
                if col_idx >= width:
                    break
                
                try:
                    value = int(cell)
                    if 1 <= value <= 8:
                        # Add island with required bridges
                        puzzle.add_island(row_idx, col_idx, value)
                    # 0 means empty cell, so we skip it
                except ValueError:
                    # If it's not a number, treat it as empty or obstacle
                    if cell == '#':
                        # Obstacle - you might want to track these
                        pass
                    elif cell == '.':
                        # Empty cell
                        pass
                    else:
                        # Unknown character - log warning but continue
                        print(f"Warning: Cannot parse '{cell}' at ({row_idx}, {col_idx}) in {filepath}")
        
        return puzzle
    
    @classmethod
    def parse_has_filename(cls, filename: str) -> dict:
        """
        Parse metadata from .has filename.
        Format: Hs_GG_NNN_DD_OO_III.has
        where:
        - GG: grid size
        - NNN: number of islands
        - DD: density percentage
        - OO: obstacle percentage
        - III: instance ID
        """
        pattern = r'Hs_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.has'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'grid_size': int(match.group(1)),
                'num_islands': int(match.group(2)),
                'density': int(match.group(3)),
                'obstacles': int(match.group(4)),
                'instance_id': int(match.group(5))
            }
        return {}
    
    def save_to_has(self, filepath: Union[str, Path]):
        """
        Save puzzle to .has format.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            # Write dimensions
            f.write(f"{self.width} {self.height}\n")
            
            # Create grid representation
            grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
            
            # Place islands
            for island in self.islands:
                if 0 <= island.row < self.height and 0 <= island.col < self.width:
                    grid[island.row][island.col] = str(island.required_bridges)
            
            # Write grid
            for row in grid:
                f.write(''.join(row) + '\n')
    
    def __str__(self):
        """String representation of puzzle (useful for debugging)"""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Place islands
        for island in self.islands:
            if 0 <= island.row < self.height and 0 <= island.col < self.width:
                grid[island.row][island.col] = str(island.required_bridges)
        
        # Add bridges
        for bridge in self.bridges:
            island1 = self._id_to_island[bridge.island1_id]
            island2 = self._id_to_island[bridge.island2_id]
            
            if island1.row == island2.row:  # Horizontal bridge
                start_col = min(island1.col, island2.col) + 1
                end_col = max(island1.col, island2.col)
                for col in range(start_col, end_col):
                    if grid[island1.row][col] == '.':
                        grid[island1.row][col] = '─' if bridge.count == 1 else '═'
            elif island1.col == island2.col:  # Vertical bridge
                start_row = min(island1.row, island2.row) + 1
                end_row = max(island1.row, island2.row)
                for row in range(start_row, end_row):
                    if grid[row][island1.col] == '.':
                        grid[row][island1.col] = '│' if bridge.count == 1 else '║'
        
        return '\n'.join(''.join(row) for row in grid)
        
    def __repr__(self):
        return f"Puzzle({self.width}x{self.height}, {len(self.islands)} islands, {len(self.bridges)} bridges)"