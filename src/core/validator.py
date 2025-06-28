"""
Validator for Hashiwokakero puzzle constraints.
"""

from typing import List, Tuple, Optional, Set
from .puzzle import Puzzle, Island, Bridge
import networkx as nx


class ValidationResult:
    """Result of puzzle validation"""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)
        
    def __bool__(self):
        return self.is_valid
        
    def __repr__(self):
        status = "Valid" if self.is_valid else "Invalid"
        return f"ValidationResult({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


class PuzzleValidator:
    """Validates Hashiwokakero puzzle constraints"""
    
    @staticmethod
    def validate_puzzle_structure(puzzle: Puzzle) -> ValidationResult:
        """Validate basic puzzle structure"""
        result = ValidationResult()
        
        # Check dimensions
        if puzzle.width <= 0 or puzzle.height <= 0:
            result.add_error("Invalid puzzle dimensions")
            
        # Check if puzzle has islands
        if not puzzle.islands:
            result.add_error("Puzzle has no islands")
            
        # Check island positions
        for island in puzzle.islands:
            if not (0 <= island.row < puzzle.height):
                result.add_error(f"Island at ({island.row}, {island.col}) has invalid row")
            if not (0 <= island.col < puzzle.width):
                result.add_error(f"Island at ({island.row}, {island.col}) has invalid column")
            if not (1 <= island.required_bridges <= 8):
                result.add_error(f"Island at ({island.row}, {island.col}) has invalid bridge requirement: {island.required_bridges}")
                
        # Check for duplicate islands
        positions = set()
        for island in puzzle.islands:
            pos = (island.row, island.col)
            if pos in positions:
                result.add_error(f"Duplicate island at position {pos}")
            positions.add(pos)
            
        return result
        
    @staticmethod
    def validate_bridges(puzzle: Puzzle) -> ValidationResult:
        """Validate bridge constraints"""
        result = ValidationResult()
        
        # Check bridge counts
        for bridge in puzzle.bridges:
            if not (1 <= bridge.count <= 2):
                result.add_error(f"Bridge {bridge} has invalid count: {bridge.count}")
                
            # Check if islands exist
            if bridge.island1_id not in puzzle._id_to_island:
                result.add_error(f"Bridge references non-existent island ID: {bridge.island1_id}")
            if bridge.island2_id not in puzzle._id_to_island:
                result.add_error(f"Bridge references non-existent island ID: {bridge.island2_id}")
                
        # Check for crossing bridges
        if puzzle.has_crossing_bridges():
            result.add_error("Puzzle has crossing bridges")
            
        # Check bridge requirements for each island
        for island in puzzle.islands:
            bridge_count = puzzle.get_island_bridges(island.id)
            if bridge_count > island.required_bridges:
                result.add_error(f"Island {island.id} has too many bridges: {bridge_count} > {island.required_bridges}")
                
        return result
        
    @staticmethod
    def validate_solution(puzzle: Puzzle) -> ValidationResult:
        """Validate if puzzle is a complete solution"""
        result = ValidationResult()
        
        # First check basic structure and bridges
        structure_result = PuzzleValidator.validate_puzzle_structure(puzzle)
        if not structure_result:
            result.errors.extend(structure_result.errors)
            result.is_valid = False
            
        bridge_result = PuzzleValidator.validate_bridges(puzzle)
        if not bridge_result:
            result.errors.extend(bridge_result.errors)
            result.is_valid = False
            
        # Check if all islands have correct number of bridges
        for island in puzzle.islands:
            bridge_count = puzzle.get_island_bridges(island.id)
            if bridge_count != island.required_bridges:
                result.add_error(f"Island {island.id} has {bridge_count} bridges, requires {island.required_bridges}")
                
        # Check connectivity
        if not puzzle.is_connected():
            result.add_error("Not all islands are connected")
            
        # Check handshaking lemma (sum of degrees must be even)
        total_bridges = sum(island.required_bridges for island in puzzle.islands)
        if total_bridges % 2 != 0:
            result.add_error(f"Total bridge requirements ({total_bridges}) is odd - impossible to solve")
            
        return result
        
    @staticmethod
    def validate_partial_solution(puzzle: Puzzle) -> ValidationResult:
        """Validate a partial solution (for intermediate states)"""
        result = ValidationResult()
        
        # Basic structure validation
        structure_result = PuzzleValidator.validate_puzzle_structure(puzzle)
        if not structure_result:
            result.errors.extend(structure_result.errors)
            result.is_valid = False
            
        # Bridge validation (but allow incomplete bridges)
        for bridge in puzzle.bridges:
            if not (1 <= bridge.count <= 2):
                result.add_error(f"Bridge {bridge} has invalid count: {bridge.count}")
                
        # Check for crossing bridges
        if puzzle.has_crossing_bridges():
            result.add_error("Puzzle has crossing bridges")
            
        # Check that no island exceeds its requirement
        for island in puzzle.islands:
            bridge_count = puzzle.get_island_bridges(island.id)
            if bridge_count > island.required_bridges:
                result.add_error(f"Island {island.id} exceeds bridge requirement: {bridge_count} > {island.required_bridges}")
                
        # Add warnings for incomplete islands
        for island in puzzle.islands:
            bridge_count = puzzle.get_island_bridges(island.id)
            if bridge_count < island.required_bridges:
                result.add_warning(f"Island {island.id} is incomplete: {bridge_count} < {island.required_bridges}")
                
        return result
        
    @staticmethod
    def check_uniqueness(puzzle: Puzzle) -> bool:
        """
        Check if the puzzle has a unique solution.
        This is a simplified check - full uniqueness checking requires solving.
        """
        # Basic heuristic checks for uniqueness
        # A puzzle is more likely to have unique solution if:
        # 1. It has enough constraints (islands)
        # 2. Islands are well distributed
        # 3. No isolated subgroups
        
        if len(puzzle.islands) < 4:
            return False
            
        # Check for islands with only one possible neighbor
        forced_connections = 0
        for island in puzzle.islands:
            valid_neighbors = list(puzzle._valid_connections[island.id])
            if len(valid_neighbors) == 1 and island.required_bridges > 0:
                forced_connections += 1
                
        # More forced connections generally mean more constrained puzzle
        return forced_connections >= len(puzzle.islands) * 0.3
        
    @staticmethod
    def get_puzzle_statistics(puzzle: Puzzle) -> dict:
        """Get various statistics about the puzzle"""
        stats = {
            'width': puzzle.width,
            'height': puzzle.height,
            'num_islands': len(puzzle.islands),
            'num_bridges': len(puzzle.bridges),
            'total_bridge_requirements': sum(i.required_bridges for i in puzzle.islands),
            'avg_bridges_per_island': sum(i.required_bridges for i in puzzle.islands) / len(puzzle.islands) if puzzle.islands else 0,
            'density': len(puzzle.islands) / (puzzle.width * puzzle.height),
            'is_complete': puzzle.is_complete(),
            'is_connected': puzzle.is_connected(),
            'has_crossing': puzzle.has_crossing_bridges()
        }
        
        # Island degree distribution
        degree_dist = {}
        for island in puzzle.islands:
            deg = island.required_bridges
            degree_dist[deg] = degree_dist.get(deg, 0) + 1
        stats['degree_distribution'] = degree_dist
        
        # Connectivity statistics
        if puzzle.islands:
            # Average number of possible connections per island
            avg_connections = sum(len(puzzle._valid_connections[i.id]) 
                                for i in puzzle.islands) / len(puzzle.islands)
            stats['avg_possible_connections'] = avg_connections
            
        return stats