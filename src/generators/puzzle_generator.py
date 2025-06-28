"""
Puzzle generator for Hashiwokakero.
"""

import random
from typing import List, Optional, Tuple, Set
import numpy as np
from pathlib import Path

from ..core.puzzle import Puzzle, Island, Difficulty
from ..core.validator import PuzzleValidator
from ..core.utils import setup_logger, DifficultyEstimator
from ..solvers import ILPSolver, ILPSolverConfig


class PuzzleGeneratorConfig:
    """Configuration for puzzle generator"""
    
    def __init__(self, **kwargs):
        self.min_islands: int = kwargs.get('min_islands', 5)
        self.max_islands: int = kwargs.get('max_islands', 30)
        self.min_bridges: int = kwargs.get('min_bridges', 1)
        self.max_bridges: int = kwargs.get('max_bridges', 8)
        self.ensure_unique: bool = kwargs.get('ensure_unique', True)
        self.max_attempts: int = kwargs.get('max_attempts', 100)
        self.solver_time_limit: float = kwargs.get('solver_time_limit', 10.0)
        self.random_seed: Optional[int] = kwargs.get('random_seed', None)
        
        # Difficulty-specific parameters
        self.difficulty_params = {
            Difficulty.EASY: {
                'density_range': (0.1, 0.2),
                'avg_bridges': (2.0, 3.0),
                'high_bridge_ratio': 0.1,  # Ratio of islands with 6+ bridges
                'grid_size_range': (7, 10)
            },
            Difficulty.MEDIUM: {
                'density_range': (0.15, 0.3),
                'avg_bridges': (2.5, 3.5),
                'high_bridge_ratio': 0.2,
                'grid_size_range': (10, 15)
            },
            Difficulty.HARD: {
                'density_range': (0.2, 0.35),
                'avg_bridges': (3.0, 4.0),
                'high_bridge_ratio': 0.3,
                'grid_size_range': (15, 25)
            },
            Difficulty.EXPERT: {
                'density_range': (0.25, 0.4),
                'avg_bridges': (3.5, 4.5),
                'high_bridge_ratio': 0.4,
                'grid_size_range': (20, 30)
            }
        }


class PuzzleGenerator:
    """Generate Hashiwokakero puzzles with various strategies"""
    
    def __init__(self, config: Optional[PuzzleGeneratorConfig] = None):
        self.config = config or PuzzleGeneratorConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            
        # Solver for uniqueness checking
        self.solver = ILPSolver(ILPSolverConfig(
            time_limit=self.config.solver_time_limit,
            verbose=False
        ))
        
    def generate(self, width: int, height: int, 
                difficulty: Difficulty = Difficulty.MEDIUM,
                strategy: str = 'random') -> Optional[Puzzle]:
        """
        Generate a puzzle with specified parameters.
        
        Args:
            width: Puzzle width
            height: Puzzle height
            difficulty: Target difficulty level
            strategy: Generation strategy ('random', 'solution_based', 'pattern')
            
        Returns:
            Generated puzzle or None if generation failed
        """
        self.logger.info(f"Generating {width}x{height} {difficulty.value} puzzle using {strategy} strategy")
        
        for attempt in range(self.config.max_attempts):
            if strategy == 'random':
                puzzle = self._generate_random(width, height, difficulty)
            elif strategy == 'solution_based':
                puzzle = self._generate_from_solution(width, height, difficulty)
            elif strategy == 'pattern':
                puzzle = self._generate_pattern_based(width, height, difficulty)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            if puzzle:
                # Validate puzzle
                validation = PuzzleValidator.validate_puzzle_structure(puzzle)
                if not validation:
                    self.logger.warning(f"Generated invalid puzzle: {validation.errors}")
                    continue
                    
                # Check if puzzle has solution
                if self._has_unique_solution(puzzle):
                    self.logger.info(f"Successfully generated puzzle on attempt {attempt + 1}")
                    return puzzle
                    
        self.logger.error(f"Failed to generate valid puzzle after {self.config.max_attempts} attempts")
        return None
        
    def _generate_random(self, width: int, height: int, difficulty: Difficulty) -> Optional[Puzzle]:
        """Generate puzzle by random island placement"""
        params = self.config.difficulty_params[difficulty]
        
        # Calculate number of islands based on density
        density = random.uniform(*params['density_range'])
        num_islands = int(width * height * density)
        num_islands = max(self.config.min_islands, 
                         min(self.config.max_islands, num_islands))
        
        # Create empty puzzle
        puzzle = Puzzle(width, height)
        
        # Place islands randomly
        positions = set()
        attempts = 0
        
        while len(positions) < num_islands and attempts < 1000:
            row = random.randint(0, height - 1)
            col = random.randint(0, width - 1)
            
            if (row, col) not in positions:
                # Check if position allows connections
                # Avoid placing islands too close to edges if they need many bridges
                edge_distance = min(row, col, height - row - 1, width - col - 1)
                max_possible_bridges = min(8, edge_distance * 2)
                
                if max_possible_bridges >= 2:
                    positions.add((row, col))
                    
            attempts += 1
            
        # Assign bridge requirements
        avg_bridges = random.uniform(*params['avg_bridges'])
        
        for row, col in positions:
            # Determine number of bridges for this island
            if random.random() < params['high_bridge_ratio']:
                # High bridge count island
                required = random.randint(6, 8)
            else:
                # Normal distribution around average
                required = int(np.random.normal(avg_bridges, 1.0))
                required = max(1, min(8, required))
                
            puzzle.add_island(row, col, required)
            
        # Ensure total bridges is even (handshaking lemma)
        total = sum(i.required_bridges for i in puzzle.islands)
        if total % 2 != 0:
            # Adjust a random island
            island = random.choice(puzzle.islands)
            if island.required_bridges < 8:
                island.required_bridges += 1
            else:
                island.required_bridges -= 1
                
        return puzzle
        
    def _generate_from_solution(self, width: int, height: int, 
                               difficulty: Difficulty) -> Optional[Puzzle]:
        """Generate puzzle by creating a solution first"""
        params = self.config.difficulty_params[difficulty]
        
        # Start with empty grid
        puzzle = Puzzle(width, height)
        
        # Generate connected graph of islands
        density = random.uniform(*params['density_range'])
        num_islands = int(width * height * density)
        num_islands = max(self.config.min_islands, 
                         min(self.config.max_islands, num_islands))
        
        # Use minimum spanning tree approach
        # First, place islands
        positions = self._generate_connected_positions(width, height, num_islands)
        
        for i, (row, col) in enumerate(positions):
            puzzle.add_island(row, col, 0)  # Temporary 0 bridges
            
        # Create a valid bridge configuration
        bridges = self._generate_valid_bridges(puzzle, params['avg_bridges'])
        
        # Set bridge requirements based on solution
        bridge_counts = {i: 0 for i in range(len(puzzle.islands))}
        
        for (i1, i2), count in bridges.items():
            bridge_counts[i1] += count
            bridge_counts[i2] += count
            
        # Update island requirements
        for island in puzzle.islands:
            island.required_bridges = bridge_counts[island.id]
            
        # Clear any bridges (we only want the puzzle, not the solution)
        puzzle.bridges = []
        
        return puzzle
        
    def _generate_pattern_based(self, width: int, height: int,
                               difficulty: Difficulty) -> Optional[Puzzle]:
        """Generate puzzle based on patterns (symmetry, shapes, etc.)"""
        params = self.config.difficulty_params[difficulty]
        
        puzzle = Puzzle(width, height)
        pattern_type = random.choice(['symmetric', 'grid', 'diagonal', 'spiral'])
        
        if pattern_type == 'symmetric':
            positions = self._generate_symmetric_pattern(width, height, params)
        elif pattern_type == 'grid':
            positions = self._generate_grid_pattern(width, height, params)
        elif pattern_type == 'diagonal':
            positions = self._generate_diagonal_pattern(width, height, params)
        else:  # spiral
            positions = self._generate_spiral_pattern(width, height, params)
            
        # Add islands with appropriate bridge counts
        avg_bridges = random.uniform(*params['avg_bridges'])
        
        for row, col in positions:
            required = int(np.random.normal(avg_bridges, 1.0))
            required = max(1, min(8, required))
            puzzle.add_island(row, col, required)
            
        # Ensure even total
        total = sum(i.required_bridges for i in puzzle.islands)
        if total % 2 != 0:
            island = random.choice(puzzle.islands)
            if island.required_bridges < 8:
                island.required_bridges += 1
            else:
                island.required_bridges -= 1
                
        return puzzle
        
    def _generate_connected_positions(self, width: int, height: int, 
                                    num_islands: int) -> List[Tuple[int, int]]:
        """Generate positions that form a connected graph"""
        positions = []
        available = set((r, c) for r in range(height) for c in range(width))
        
        # Start with random position
        start = random.choice(list(available))
        positions.append(start)
        available.remove(start)
        
        # Grow connected component
        while len(positions) < num_islands and available:
            # Pick random existing island
            base = random.choice(positions)
            
            # Find available positions that can connect to base
            candidates = []
            for r, c in available:
                if r == base[0] or c == base[1]:  # Same row or column
                    # Check if path is clear
                    if self._is_path_clear(base, (r, c), positions):
                        distance = abs(r - base[0]) + abs(c - base[1])
                        candidates.append(((r, c), distance))
                        
            if candidates:
                # Prefer closer positions
                candidates.sort(key=lambda x: x[1])
                # Take from top candidates
                idx = min(random.randint(0, 3), len(candidates) - 1)
                new_pos = candidates[idx][0]
                positions.append(new_pos)
                available.remove(new_pos)
                
        return positions[:num_islands]
        
    def _is_path_clear(self, pos1: Tuple[int, int], pos2: Tuple[int, int],
                      obstacles: List[Tuple[int, int]]) -> bool:
        """Check if path between two positions is clear"""
        if pos1[0] == pos2[0]:  # Same row
            row = pos1[0]
            start_col = min(pos1[1], pos2[1])
            end_col = max(pos1[1], pos2[1])
            for col in range(start_col + 1, end_col):
                if (row, col) in obstacles:
                    return False
        else:  # Same column
            col = pos1[1]
            start_row = min(pos1[0], pos2[0])
            end_row = max(pos1[0], pos2[0])
            for row in range(start_row + 1, end_row):
                if (row, col) in obstacles:
                    return False
        return True
        
    def _generate_valid_bridges(self, puzzle: Puzzle, 
                               avg_bridges: float) -> dict:
        """Generate a valid bridge configuration"""
        bridges = {}
        
        # Create minimum spanning tree first
        edges = []
        for i, island1 in enumerate(puzzle.islands):
            for j, island2 in enumerate(puzzle.islands[i+1:], i+1):
                if island2.id in puzzle._valid_connections[island1.id]:
                    distance = abs(island1.row - island2.row) + abs(island1.col - island2.col)
                    edges.append((distance, island1.id, island2.id))
                    
        edges.sort()
        
        # Kruskal's algorithm for MST
        parent = list(range(len(puzzle.islands)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
            
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
            
        # Build MST
        for dist, i1, i2 in edges:
            if union(i1, i2):
                bridges[(i1, i2)] = 1
                
        # Add extra bridges to reach target average
        current_avg = sum(bridges.values()) * 2 / len(puzzle.islands)
        
        while current_avg < avg_bridges:
            # Add random bridge
            i1 = random.randint(0, len(puzzle.islands) - 1)
            valid = list(puzzle._valid_connections[i1])
            if valid:
                i2 = random.choice(valid)
                key = (min(i1, i2), max(i1, i2))
                if key in bridges and bridges[key] < 2:
                    bridges[key] += 1
                elif key not in bridges:
                    bridges[key] = 1
                    
            current_avg = sum(bridges.values()) * 2 / len(puzzle.islands)
            
        return bridges
        
    def _generate_symmetric_pattern(self, width: int, height: int,
                                   params: dict) -> List[Tuple[int, int]]:
        """Generate symmetric island pattern"""
        positions = set()
        density = random.uniform(*params['density_range'])
        target_islands = int(width * height * density)
        
        # Generate on one half, mirror to other
        half_width = width // 2
        
        for _ in range(target_islands // 2):
            row = random.randint(0, height - 1)
            col = random.randint(0, half_width - 1)
            positions.add((row, col))
            # Mirror position
            positions.add((row, width - 1 - col))
            
        # Add center column if odd width
        if width % 2 == 1 and len(positions) < target_islands:
            row = random.randint(0, height - 1)
            positions.add((row, width // 2))
            
        return list(positions)
        
    def _generate_grid_pattern(self, width: int, height: int,
                              params: dict) -> List[Tuple[int, int]]:
        """Generate grid-based island pattern"""
        positions = []
        density = random.uniform(*params['density_range'])
        
        # Calculate grid spacing
        spacing = int(1 / (density ** 0.5))
        spacing = max(2, spacing)
        
        for row in range(0, height, spacing):
            for col in range(0, width, spacing):
                if random.random() < 0.8:  # Some randomness
                    # Add small offset
                    r = row + random.randint(-1, 1)
                    c = col + random.randint(-1, 1)
                    r = max(0, min(height - 1, r))
                    c = max(0, min(width - 1, c))
                    positions.append((r, c))
                    
        return positions
        
    def _generate_diagonal_pattern(self, width: int, height: int,
                                  params: dict) -> List[Tuple[int, int]]:
        """Generate diagonal island pattern"""
        positions = []
        density = random.uniform(*params['density_range'])
        num_diagonals = int((width + height) * density / 2)
        
        for _ in range(num_diagonals):
            # Random starting point on edge
            if random.random() < 0.5:
                # Start from top/bottom
                row = 0 if random.random() < 0.5 else height - 1
                col = random.randint(0, width - 1)
            else:
                # Start from left/right
                row = random.randint(0, height - 1)
                col = 0 if random.random() < 0.5 else width - 1
                
            # Generate diagonal line
            dr = random.choice([-1, 1])
            dc = random.choice([-1, 1])
            
            while 0 <= row < height and 0 <= col < width:
                if random.random() < 0.7:
                    positions.append((row, col))
                row += dr
                col += dc
                
        return list(set(positions))  # Remove duplicates
        
    def _generate_spiral_pattern(self, width: int, height: int,
                                params: dict) -> List[Tuple[int, int]]:
        """Generate spiral island pattern"""
        positions = []
        density = random.uniform(*params['density_range'])
        
        # Start from center
        row, col = height // 2, width // 2
        positions.append((row, col))
        
        # Spiral outward
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        dir_idx = 0
        steps = 1
        
        while len(positions) < width * height * density:
            for _ in range(2):  # Each distance appears twice in spiral
                dr, dc = directions[dir_idx]
                
                for _ in range(steps):
                    row += dr
                    col += dc
                    
                    if 0 <= row < height and 0 <= col < width:
                        if random.random() < 0.6:
                            positions.append((row, col))
                            
                dir_idx = (dir_idx + 1) % 4
                
            steps += 1
            
            if steps > max(width, height):
                break
                
        return positions
        
    def _has_unique_solution(self, puzzle: Puzzle) -> bool:
        """Check if puzzle has a unique solution"""
        if not self.config.ensure_unique:
            return True
            
        # First check if puzzle has any solution
        result = self.solver.solve(puzzle)
        
        if not result.success:
            return False
            
        # For true uniqueness checking, we would need to:
        # 1. Find one solution
        # 2. Add constraint to exclude this solution
        # 3. Try to find another solution
        # This is computationally expensive, so we use heuristics
        
        # Heuristic: puzzles with more forced moves tend to have unique solutions
        forced_moves = 0
        
        for island in puzzle.islands:
            valid_neighbors = list(puzzle._valid_connections[island.id])
            
            # Island with only one neighbor
            if len(valid_neighbors) == 1 and island.required_bridges > 0:
                forced_moves += 1
                
            # Island that must connect to all neighbors
            total_capacity = sum(min(2, puzzle._id_to_island[n].required_bridges) 
                               for n in valid_neighbors)
            if total_capacity <= island.required_bridges:
                forced_moves += 1
                
        # More forced moves generally means more likely to be unique
        uniqueness_score = forced_moves / len(puzzle.islands)
        
        return uniqueness_score > 0.2  # Threshold based on experience
        
    def generate_batch(self, count: int, width: int, height: int,
                      difficulty: Difficulty = Difficulty.MEDIUM,
                      save_dir: Optional[Path] = None) -> List[Puzzle]:
        """Generate multiple puzzles"""
        puzzles = []
        strategies = ['random', 'solution_based', 'pattern']
        
        for i in range(count):
            self.logger.info(f"Generating puzzle {i+1}/{count}")
            
            # Vary strategies
            strategy = strategies[i % len(strategies)]
            puzzle = self.generate(width, height, difficulty, strategy)
            
            if puzzle:
                puzzles.append(puzzle)
                
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    filename = save_dir / f"{difficulty.value}_{width}x{height}_{i:04d}.json"
                    puzzle.save(filename)
                    
        self.logger.info(f"Generated {len(puzzles)}/{count} valid puzzles")
        return puzzles