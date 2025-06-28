"""
Pattern-based puzzle generator for Hashiwokakero.
Generates puzzles with specific patterns and symmetries.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import random
from abc import ABC, abstractmethod

from ..core.puzzle import Puzzle, Island, Difficulty
from ..core.utils import setup_logger


class PatternType(Enum):
    """Types of patterns for puzzle generation"""
    SYMMETRIC = "symmetric"
    GRID = "grid"
    SPIRAL = "spiral"
    DIAGONAL = "diagonal"
    STAR = "star"
    FRAME = "frame"
    RANDOM_WALK = "random_walk"
    CLUSTERS = "clusters"


class SymmetryType(Enum):
    """Types of symmetry"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL_MAIN = "diagonal_main"
    DIAGONAL_ANTI = "diagonal_anti"
    ROTATIONAL_90 = "rotational_90"
    ROTATIONAL_180 = "rotational_180"
    FOUR_WAY = "four_way"


class BasePattern(ABC):
    """Abstract base class for pattern generators"""
    
    def __init__(self, width: int, height: int, density: float = 0.2):
        self.width = width
        self.height = height
        self.density = density
        self.logger = setup_logger(self.__class__.__name__)
        
    @abstractmethod
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate island positions based on pattern"""
        pass
        
    def adjust_density(self, positions: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Adjust number of positions to match target density"""
        target_count = int(self.width * self.height * self.density)
        current_count = len(positions)
        
        if current_count > target_count:
            # Remove random positions
            positions_list = list(positions)
            random.shuffle(positions_list)
            return set(positions_list[:target_count])
            
        elif current_count < target_count:
            # Add random positions
            all_positions = set(
                (r, c) for r in range(self.height) for c in range(self.width)
            )
            available = all_positions - positions
            
            if available:
                additional = random.sample(
                    list(available), 
                    min(len(available), target_count - current_count)
                )
                positions.update(additional)
                
        return positions


class SymmetricPattern(BasePattern):
    """Generate symmetric patterns"""
    
    def __init__(self, width: int, height: int, density: float = 0.2,
                 symmetry_type: SymmetryType = SymmetryType.HORIZONTAL):
        super().__init__(width, height, density)
        self.symmetry_type = symmetry_type
        
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate positions with specified symmetry"""
        positions = set()
        
        if self.symmetry_type == SymmetryType.HORIZONTAL:
            # Generate on top half, mirror to bottom
            for _ in range(int(self.width * self.height * self.density / 2)):
                row = random.randint(0, self.height // 2)
                col = random.randint(0, self.width - 1)
                positions.add((row, col))
                positions.add((self.height - 1 - row, col))
                
        elif self.symmetry_type == SymmetryType.VERTICAL:
            # Generate on left half, mirror to right
            for _ in range(int(self.width * self.height * self.density / 2)):
                row = random.randint(0, self.height - 1)
                col = random.randint(0, self.width // 2)
                positions.add((row, col))
                positions.add((row, self.width - 1 - col))
                
        elif self.symmetry_type == SymmetryType.DIAGONAL_MAIN:
            # Generate below main diagonal, mirror above
            for _ in range(int(min(self.width, self.height) * self.density)):
                size = min(self.width, self.height)
                i = random.randint(0, size - 1)
                j = random.randint(0, i)
                positions.add((i, j))
                positions.add((j, i))
                
        elif self.symmetry_type == SymmetryType.ROTATIONAL_180:
            # Generate on one half, rotate 180 degrees
            for _ in range(int(self.width * self.height * self.density / 2)):
                row = random.randint(0, self.height - 1)
                col = random.randint(0, self.width - 1)
                positions.add((row, col))
                positions.add((self.height - 1 - row, self.width - 1 - col))
                
        elif self.symmetry_type == SymmetryType.FOUR_WAY:
            # Four-way symmetry
            for _ in range(int(self.width * self.height * self.density / 4)):
                row = random.randint(0, self.height // 2)
                col = random.randint(0, self.width // 2)
                positions.add((row, col))
                positions.add((row, self.width - 1 - col))
                positions.add((self.height - 1 - row, col))
                positions.add((self.height - 1 - row, self.width - 1 - col))
                
        return self.adjust_density(positions)


class GridPattern(BasePattern):
    """Generate grid-based patterns"""
    
    def __init__(self, width: int, height: int, density: float = 0.2,
                 spacing: Optional[int] = None, jitter: float = 0.2):
        super().__init__(width, height, density)
        self.spacing = spacing or max(2, int(1 / (density ** 0.5)))
        self.jitter = jitter
        
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate grid positions with optional jitter"""
        positions = set()
        
        for row in range(0, self.height, self.spacing):
            for col in range(0, self.width, self.spacing):
                # Apply jitter
                if self.jitter > 0 and random.random() < 0.8:
                    jitter_row = int(random.uniform(-self.jitter, self.jitter) * self.spacing)
                    jitter_col = int(random.uniform(-self.jitter, self.jitter) * self.spacing)
                    
                    new_row = max(0, min(self.height - 1, row + jitter_row))
                    new_col = max(0, min(self.width - 1, col + jitter_col))
                    
                    positions.add((new_row, new_col))
                else:
                    if row < self.height and col < self.width:
                        positions.add((row, col))
                        
        return self.adjust_density(positions)


class SpiralPattern(BasePattern):
    """Generate spiral patterns"""
    
    def __init__(self, width: int, height: int, density: float = 0.2,
                 arms: int = 1, tightness: float = 0.5):
        super().__init__(width, height, density)
        self.arms = arms
        self.tightness = tightness
        
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate spiral positions"""
        positions = set()
        center_row = self.height // 2
        center_col = self.width // 2
        
        max_radius = min(self.width, self.height) // 2
        points_per_arm = int(self.width * self.height * self.density / self.arms)
        
        for arm in range(self.arms):
            angle_offset = arm * 2 * np.pi / self.arms
            
            for i in range(points_per_arm):
                # Spiral equation
                t = i / points_per_arm * 4 * np.pi
                radius = max_radius * t / (4 * np.pi) * self.tightness
                
                angle = t + angle_offset
                row = int(center_row + radius * np.sin(angle))
                col = int(center_col + radius * np.cos(angle))
                
                if 0 <= row < self.height and 0 <= col < self.width:
                    positions.add((row, col))
                    
        return self.adjust_density(positions)


class DiagonalPattern(BasePattern):
    """Generate diagonal line patterns"""
    
    def __init__(self, width: int, height: int, density: float = 0.2,
                 num_lines: Optional[int] = None, spacing: int = 3):
        super().__init__(width, height, density)
        self.num_lines = num_lines or max(3, int(density * 10))
        self.spacing = spacing
        
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate diagonal line positions"""
        positions = set()
        
        # Main diagonals
        for line in range(self.num_lines):
            # Starting positions
            if random.random() < 0.5:
                # Start from top/bottom
                start_row = 0 if random.random() < 0.5 else self.height - 1
                start_col = random.randint(0, self.width - 1)
                row_dir = 1 if start_row == 0 else -1
            else:
                # Start from left/right
                start_row = random.randint(0, self.height - 1)
                start_col = 0 if random.random() < 0.5 else self.width - 1
                col_dir = 1 if start_col == 0 else -1
                
            # Direction
            row_step = random.choice([-1, 0, 1])
            col_step = random.choice([-1, 0, 1])
            
            if row_step == 0 and col_step == 0:
                col_step = 1
                
            # Generate line
            row, col = start_row, start_col
            step_count = 0
            
            while 0 <= row < self.height and 0 <= col < self.width:
                if step_count % self.spacing == 0:
                    positions.add((row, col))
                    
                row += row_step
                col += col_step
                step_count += 1
                
        return self.adjust_density(positions)


class StarPattern(BasePattern):
    """Generate star/radial patterns"""
    
    def __init__(self, width: int, height: int, density: float = 0.2,
                 rays: int = 8, rings: int = 3):
        super().__init__(width, height, density)
        self.rays = rays
        self.rings = rings
        
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate star pattern positions"""
        positions = set()
        center_row = self.height // 2
        center_col = self.width // 2
        
        # Center point
        positions.add((center_row, center_col))
        
        max_radius = min(self.width, self.height) // 2
        
        # Rays
        for ray in range(self.rays):
            angle = ray * 2 * np.pi / self.rays
            
            for r in range(1, max_radius):
                row = int(center_row + r * np.sin(angle))
                col = int(center_col + r * np.cos(angle))
                
                if 0 <= row < self.height and 0 <= col < self.width:
                    if random.random() < 0.7:  # Some randomness
                        positions.add((row, col))
                        
        # Rings
        for ring in range(1, self.rings + 1):
            radius = max_radius * ring / self.rings
            points = int(2 * np.pi * radius * self.density)
            
            for i in range(points):
                angle = i * 2 * np.pi / points
                row = int(center_row + radius * np.sin(angle))
                col = int(center_col + radius * np.cos(angle))
                
                if 0 <= row < self.height and 0 <= col < self.width:
                    positions.add((row, col))
                    
        return self.adjust_density(positions)


class FramePattern(BasePattern):
    """Generate frame/border patterns"""
    
    def __init__(self, width: int, height: int, density: float = 0.2,
                 layers: int = 2, fill_corners: bool = True):
        super().__init__(width, height, density)
        self.layers = layers
        self.fill_corners = fill_corners
        
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate frame pattern positions"""
        positions = set()
        
        for layer in range(self.layers):
            # Top and bottom edges
            for col in range(layer, self.width - layer, 2):
                positions.add((layer, col))
                positions.add((self.height - 1 - layer, col))
                
            # Left and right edges
            for row in range(layer, self.height - layer, 2):
                positions.add((row, layer))
                positions.add((row, self.width - 1 - layer))
                
        # Fill corners if requested
        if self.fill_corners:
            corner_size = min(3, min(self.width, self.height) // 4)
            for r in range(corner_size):
                for c in range(corner_size):
                    # All four corners
                    positions.add((r, c))
                    positions.add((r, self.width - 1 - c))
                    positions.add((self.height - 1 - r, c))
                    positions.add((self.height - 1 - r, self.width - 1 - c))
                    
        return self.adjust_density(positions)


class ClusterPattern(BasePattern):
    """Generate clustered island patterns"""
    
    def __init__(self, width: int, height: int, density: float = 0.2,
                 num_clusters: int = 3, cluster_spread: float = 0.3):
        super().__init__(width, height, density)
        self.num_clusters = num_clusters
        self.cluster_spread = cluster_spread
        
    def generate_positions(self) -> Set[Tuple[int, int]]:
        """Generate clustered positions"""
        positions = set()
        
        # Generate cluster centers
        centers = []
        for _ in range(self.num_clusters):
            center_row = random.randint(
                int(self.height * 0.2), 
                int(self.height * 0.8)
            )
            center_col = random.randint(
                int(self.width * 0.2), 
                int(self.width * 0.8)
            )
            centers.append((center_row, center_col))
            
        # Generate points around each cluster
        points_per_cluster = int(self.width * self.height * self.density / self.num_clusters)
        
        for center_row, center_col in centers:
            spread = min(self.width, self.height) * self.cluster_spread
            
            for _ in range(points_per_cluster):
                # Gaussian distribution around center
                row = int(np.random.normal(center_row, spread / 3))
                col = int(np.random.normal(center_col, spread / 3))
                
                if 0 <= row < self.height and 0 <= col < self.width:
                    positions.add((row, col))
                    
        return self.adjust_density(positions)


class PatternGenerator:
    """Main pattern generator that combines different patterns"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        
    def generate(self, width: int, height: int, 
                pattern_type: PatternType,
                density: float = 0.2,
                **kwargs) -> List[Tuple[int, int]]:
        """
        Generate island positions based on pattern type.
        
        Args:
            width: Puzzle width
            height: Puzzle height
            pattern_type: Type of pattern to generate
            density: Target island density
            **kwargs: Additional pattern-specific parameters
            
        Returns:
            List of (row, col) positions
        """
        pattern_classes = {
            PatternType.SYMMETRIC: SymmetricPattern,
            PatternType.GRID: GridPattern,
            PatternType.SPIRAL: SpiralPattern,
            PatternType.DIAGONAL: DiagonalPattern,
            PatternType.STAR: StarPattern,
            PatternType.FRAME: FramePattern,
            PatternType.CLUSTERS: ClusterPattern
        }
        
        pattern_class = pattern_classes.get(pattern_type)
        if not pattern_class:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
        pattern = pattern_class(width, height, density, **kwargs)
        positions = pattern.generate_positions()
        
        self.logger.info(f"Generated {len(positions)} positions using {pattern_type.value} pattern")
        
        return list(positions)
        
    def create_patterned_puzzle(self, width: int, height: int,
                               pattern_type: PatternType,
                               difficulty: Difficulty,
                               **kwargs) -> Puzzle:
        """
        Create a complete puzzle with pattern-based island placement.
        
        Args:
            width: Puzzle width
            height: Puzzle height
            pattern_type: Type of pattern to use
            difficulty: Target difficulty level
            **kwargs: Pattern-specific parameters
            
        Returns:
            Generated puzzle
        """
        # Determine density based on difficulty
        density_map = {
            Difficulty.EASY: 0.15,
            Difficulty.MEDIUM: 0.25,
            Difficulty.HARD: 0.35,
            Difficulty.EXPERT: 0.45
        }
        
        density = kwargs.pop('density', density_map[difficulty])
        
        # Generate positions
        positions = self.generate(width, height, pattern_type, density, **kwargs)
        
        # Create puzzle
        puzzle = Puzzle(width, height)
        
        # Add islands with bridge requirements
        avg_bridges_map = {
            Difficulty.EASY: 2.5,
            Difficulty.MEDIUM: 3.0,
            Difficulty.HARD: 3.5,
            Difficulty.EXPERT: 4.0
        }
        
        avg_bridges = avg_bridges_map[difficulty]
        
        for row, col in positions:
            # Determine bridge count
            if random.random() < 0.2:  # 20% chance of high bridge count
                bridges = random.randint(5, 8)
            else:
                bridges = int(np.random.normal(avg_bridges, 1.0))
                bridges = max(1, min(8, bridges))
                
            puzzle.add_island(row, col, bridges)
            
        # Ensure even total (handshaking lemma)
        total = sum(i.required_bridges for i in puzzle.islands)
        if total % 2 != 0:
            # Adjust a random island
            island = random.choice(puzzle.islands)
            if island.required_bridges < 8:
                island.required_bridges += 1
            elif island.required_bridges > 1:
                island.required_bridges -= 1
                
        return puzzle
        
    def combine_patterns(self, width: int, height: int,
                        patterns: List[Tuple[PatternType, float, dict]],
                        merge_strategy: str = 'union') -> List[Tuple[int, int]]:
        """
        Combine multiple patterns.
        
        Args:
            width: Puzzle width
            height: Puzzle height  
            patterns: List of (pattern_type, weight, kwargs) tuples
            merge_strategy: How to merge patterns ('union', 'intersection', 'weighted')
            
        Returns:
            Combined positions
        """
        all_positions = []
        
        for pattern_type, weight, kwargs in patterns:
            positions = self.generate(width, height, pattern_type, **kwargs)
            
            if merge_strategy == 'weighted':
                # Sample positions based on weight
                sample_size = int(len(positions) * weight)
                positions = random.sample(positions, min(sample_size, len(positions)))
                
            all_positions.append(set(positions))
            
        # Merge based on strategy
        if merge_strategy == 'union':
            result = set()
            for pos_set in all_positions:
                result.update(pos_set)
                
        elif merge_strategy == 'intersection':
            result = all_positions[0]
            for pos_set in all_positions[1:]:
                result = result.intersection(pos_set)
                
        else:  # weighted
            result = set()
            for pos_set in all_positions:
                result.update(pos_set)
                
        return list(result)