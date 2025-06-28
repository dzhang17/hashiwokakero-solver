"""
Enhanced Base solver class for Hashiwokakero puzzles.
Fixed brute force logic and added more solving rules.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, Tuple, Set
from dataclasses import dataclass, field
import time
import logging
from pathlib import Path
from datetime import datetime
import random
from collections import deque, defaultdict
import copy

from ..core.puzzle import Puzzle, Bridge, Island
from ..core.validator import PuzzleValidator, ValidationResult
from ..core.utils import setup_logger, memory_usage


@dataclass
class SolverConfig:
    """Configuration for puzzle solvers"""
    time_limit: float = 300.0  # seconds
    max_iterations: int = 100000
    max_depth: int = 5  # Maximum depth for brute force search
    verbose: bool = True
    log_file: Optional[Path] = None
    save_intermediate: bool = False
    intermediate_dir: Optional[Path] = None
    random_seed: Optional[int] = None
    check_multiple_solutions: bool = False
    use_advanced_rules: bool = True  # Enable advanced solving rules
    
    # Algorithm-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Result from puzzle solver"""
    success: bool
    solution: Optional[Puzzle] = None
    solve_time: float = 0.0
    iterations: int = 0
    memory_used: float = 0.0  # MB
    message: str = ""
    
    # Additional information
    has_multiple_solutions: bool = False
    steps: List[Puzzle] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        status = "Success" if self.success else "Failed"
        return f"SolverResult({status}, time={self.solve_time:.2f}s, iterations={self.iterations})"


class BaseSolver(ABC):
    """Abstract base class for Hashiwokakero solvers"""
    
    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize solver with configuration."""
        self.config = config or SolverConfig()
        self.logger = setup_logger(
            self.__class__.__name__,
            self.config.log_file,
            "DEBUG" if self.config.verbose else "INFO"
        )
        
        # Callbacks for monitoring progress
        self._progress_callbacks: List[Callable] = []
        
        # Statistics tracking
        self._start_time: Optional[float] = None
        self._iterations: int = 0
        
        # Set random seed if specified
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            
    def add_progress_callback(self, callback: Callable):
        """Add a callback function to monitor solving progress."""
        self._progress_callbacks.append(callback)
        
    def solve(self, puzzle: Puzzle) -> SolverResult:
        """Solve the puzzle."""
        self.logger.info(f"Starting {self.__class__.__name__} solver")
        self.logger.info(f"Puzzle: {puzzle}")
        
        # Validate input puzzle
        validation = PuzzleValidator.validate_puzzle_structure(puzzle)
        if not validation:
            return SolverResult(
                success=False,
                message=f"Invalid puzzle: {'; '.join(validation.errors)}"
            )
            
        # Initialize solving
        self._start_time = time.time()
        self._iterations = 0
        initial_memory = memory_usage()
        
        try:
            # Call the specific solver implementation
            result = self._solve(puzzle)
            
            # Validate solution if found
            if result.success and result.solution:
                validation = PuzzleValidator.validate_solution(result.solution)
                if not validation:
                    result.success = False
                    result.message = f"Invalid solution: {'; '.join(validation.errors)}"
                    
            # Add final statistics
            result.solve_time = time.time() - self._start_time
            result.memory_used = memory_usage() - initial_memory
            result.iterations = self._iterations
            
            # Log result
            if result.success:
                self.logger.info(f"Solved in {result.solve_time:.2f}s with {result.iterations} iterations")
            else:
                self.logger.warning(f"Failed to solve: {result.message}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error during solving: {str(e)}", exc_info=True)
            return SolverResult(
                success=False,
                message=f"Solver error: {str(e)}",
                solve_time=time.time() - self._start_time,
                iterations=self._iterations
            )
            
    @abstractmethod
    def _solve(self, puzzle: Puzzle) -> SolverResult:
        """Implement the specific solving algorithm."""
        pass
        
    def _check_time_limit(self) -> bool:
        """Check if time limit has been exceeded"""
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) > self.config.time_limit
        
    def _increment_iteration(self):
        """Increment iteration counter and check limits"""
        self._iterations += 1
        
        if self._iterations > self.config.max_iterations:
            raise RuntimeError(f"Maximum iterations ({self.config.max_iterations}) exceeded")
            
    def _call_progress_callbacks(self, current_solution: Optional[Puzzle] = None, 
                               stats: Optional[Dict[str, Any]] = None):
        """Call all registered progress callbacks"""
        for callback in self._progress_callbacks:
            try:
                callback(self._iterations, current_solution, stats or {})
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
                
    def _save_intermediate_solution(self, solution: Puzzle, suffix: str = ""):
        """Save intermediate solution if configured"""
        if not self.config.save_intermediate or not self.config.intermediate_dir:
            return
            
        self.config.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.__class__.__name__}_{timestamp}_{self._iterations}{suffix}.json"
        filepath = self.config.intermediate_dir / filename
        
        solution.save(filepath)
        self.logger.debug(f"Saved intermediate solution to {filepath}")


class GraphNode:
    """Represents an island in the graph with its connections"""
    
    def __init__(self, island: Island):
        self.island = island
        self.id = island.id
        self.value = island.required_bridges
        self.row = island.row
        self.col = island.col
        self.neighbors: List['GraphNode'] = []
        self.bridges: Dict[int, int] = {}  # neighbor_id -> bridge_count
        self.completed = False
        
    @property
    def current_bridges(self) -> int:
        """Total number of bridges connected to this node"""
        return sum(self.bridges.values())
        
    @property
    def remaining_bridges(self) -> int:
        """Number of bridges still needed"""
        return self.value - self.current_bridges
        
    def get_available_neighbors(self) -> List['GraphNode']:
        """Get neighbors that can still accept bridges"""
        available = []
        for neighbor in self.neighbors:
            if not neighbor.completed:
                current_bridges_to_neighbor = self.bridges.get(neighbor.id, 0)
                if current_bridges_to_neighbor < 2:  # Max 2 bridges between islands
                    available.append(neighbor)
        return available
        
    def __repr__(self):
        return f"Node({self.id}, val={self.value}, curr={self.current_bridges}, completed={self.completed})"


class IterativeSolver(BaseSolver):
    """
    Enhanced iterative solver with more solving rules and fixed brute force.
    """
    
    def _solve(self, puzzle: Puzzle) -> SolverResult:
        """Implement iterative solving strategy"""
        working_puzzle = puzzle.copy()
        steps = []
        
        # Statistics tracking
        rules_used = defaultdict(int)
        
        # Ensure at least one iteration is counted
        self._increment_iteration()
        
        old_state = ""
        new_state = self._puzzle_state_string(working_puzzle)
        no_progress_count = 0
        
        while old_state != new_state and not self._check_time_limit():
            old_state = new_state
            
            # Call progress callbacks
            self._call_progress_callbacks(working_puzzle, {
                'phase': 'solving',
                'rules_used': dict(rules_used)
            })
            
            # Perform one solving step
            step_result = self._solver_step(working_puzzle, rules_used)
            
            if step_result['new_bridges'] > 0:
                # Save step
                steps.append(working_puzzle.copy())
                new_state = self._puzzle_state_string(working_puzzle)
                no_progress_count = 0
                
                self.logger.debug(f"Added {step_result['new_bridges']} bridges in this step")
                
            else:
                no_progress_count += 1
                
            # Check if puzzle is solved
            if step_result['solved']:
                return SolverResult(
                    success=True,
                    solution=working_puzzle,
                    steps=steps,
                    message="Puzzle solved successfully",
                    stats={'rules_used': dict(rules_used)}
                )
                
            # If no progress for several iterations, try advanced techniques
            if no_progress_count > 3 and not step_result['solved']:
                self.logger.debug("No progress with normal rules, trying advanced techniques")
                
                # Try to make educated guesses
                guess_result = self._make_educated_guess(working_puzzle)
                
                if guess_result['success']:
                    steps.append(working_puzzle.copy())
                    new_state = self._puzzle_state_string(working_puzzle)
                    no_progress_count = 0
                else:
                    # No more moves possible
                    break
                    
            # Increment iteration
            if step_result['new_bridges'] > 0 or old_state != new_state:
                self._increment_iteration()
            
        # Final validation
        validation = PuzzleValidator.validate_solution(working_puzzle)
        
        return SolverResult(
            success=validation.is_valid,
            solution=working_puzzle,
            steps=steps,
            message="Solved" if validation.is_valid else "Could not find valid solution",
            stats={'rules_used': dict(rules_used), 'total_steps': len(steps)}
        )
        
    def _solver_step(self, puzzle: Puzzle, rules_used: Dict[str, int]) -> Dict[str, Any]:
        """Perform one step of solving with multiple rules"""
        graph = self._create_graph(puzzle)
        new_bridges = 0
        
        # Rule 1: Basic constraints (single neighbor, exact capacity)
        added = self._apply_basic_rules(graph, puzzle, rules_used)
        new_bridges += added
        
        if added > 0:
            self._update_graph(graph, puzzle)
        
        # Rule 2: Isolation prevention
        if self.config.use_advanced_rules:
            added = self._apply_isolation_rule(graph, puzzle, rules_used)
            new_bridges += added
            
            if added > 0:
                self._update_graph(graph, puzzle)
        
        # Rule 3: Connectivity maintenance
        if self.config.use_advanced_rules:
            added = self._apply_connectivity_rule(graph, puzzle, rules_used)
            new_bridges += added
            
            if added > 0:
                self._update_graph(graph, puzzle)
        
        # Rule 4: Required minimum bridges
        added = self._apply_minimum_bridges_rule(graph, puzzle, rules_used)
        new_bridges += added
        
        # Check if puzzle is completed
        solved = self._is_puzzle_completed(graph)
        
        return {
            'new_bridges': new_bridges,
            'solved': solved,
            'multiple_solutions': False
        }
        
    def _apply_basic_rules(self, graph: Dict[int, GraphNode], puzzle: Puzzle, 
                          rules_used: Dict[str, int]) -> int:
        """Apply basic constraint rules"""
        added_bridges = 0
        
        for node in graph.values():
            if node.completed:
                continue
                
            available_neighbors = []
            for neighbor in node.neighbors:
                if not neighbor.completed:
                    current_bridges = node.bridges.get(neighbor.id, 0)
                    max_possible = min(2 - current_bridges, neighbor.remaining_bridges)
                    if max_possible > 0:
                        available_neighbors.append({
                            'node': neighbor,
                            'max_bridges': max_possible,
                            'current_bridges': current_bridges
                        })
            
            if not available_neighbors:
                continue
                
            required = node.remaining_bridges
            
            # Rule 1: If only one neighbor, must use all required bridges
            if len(available_neighbors) == 1:
                neighbor_info = available_neighbors[0]
                neighbor = neighbor_info['node']
                bridges_to_add = min(required, neighbor_info['max_bridges'])
                
                if bridges_to_add > 0:
                    self._add_bridges(puzzle, node.id, neighbor.id, bridges_to_add)
                    added_bridges += bridges_to_add
                    rules_used['single_neighbor'] += 1
                    
            # Rule 2: If total capacity equals required, use all available
            else:
                total_capacity = sum(n['max_bridges'] for n in available_neighbors)
                
                if total_capacity == required:
                    for neighbor_info in available_neighbors:
                        neighbor = neighbor_info['node']
                        bridges_to_add = neighbor_info['max_bridges']
                        
                        if bridges_to_add > 0:
                            self._add_bridges(puzzle, node.id, neighbor.id, bridges_to_add)
                            added_bridges += bridges_to_add
                    rules_used['exact_capacity'] += 1
                    
        return added_bridges
        
    def _apply_isolation_rule(self, graph: Dict[int, GraphNode], puzzle: Puzzle,
                             rules_used: Dict[str, int]) -> int:
        """Prevent islands from becoming isolated"""
        added_bridges = 0
        
        for node in graph.values():
            if node.completed or node.remaining_bridges == 0:
                continue
                
            # Count neighbors that could still connect
            active_neighbors = []
            for neighbor in node.neighbors:
                current = node.bridges.get(neighbor.id, 0)
                if current < 2 and neighbor.remaining_bridges > 0:
                    active_neighbors.append(neighbor)
            
            # If this node would be isolated after neighbors complete
            critical_neighbors = []
            for neighbor in active_neighbors:
                if neighbor.remaining_bridges == 1:
                    # This neighbor will complete soon
                    other_options = sum(1 for n in active_neighbors 
                                      if n.id != neighbor.id and n.remaining_bridges > 1)
                    if other_options == 0 and node.remaining_bridges > 1:
                        # Must connect to this neighbor now
                        critical_neighbors.append(neighbor)
            
            for neighbor in critical_neighbors:
                current = node.bridges.get(neighbor.id, 0)
                if current < 2:
                    self._add_bridges(puzzle, node.id, neighbor.id, 1)
                    added_bridges += 1
                    rules_used['isolation_prevention'] += 1
                    
        return added_bridges
        
    def _apply_connectivity_rule(self, graph: Dict[int, GraphNode], puzzle: Puzzle,
                                rules_used: Dict[str, int]) -> int:
        """Ensure connectivity between components"""
        added_bridges = 0
        
        # Find weakly connected components
        components = self._find_components_weak(graph)
        
        if len(components) > 1:
            # Try to connect components
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components[i+1:], i+1):
                    # Find best connection between components
                    best_connection = None
                    best_score = float('inf')
                    
                    for node1 in comp1:
                        if node1.remaining_bridges == 0:
                            continue
                        for node2 in comp2:
                            if node2.remaining_bridges == 0:
                                continue
                            if node2 in node1.neighbors:
                                current = node1.bridges.get(node2.id, 0)
                                if current < 2:
                                    # Score based on how critical this connection is
                                    score = (node1.remaining_bridges + node2.remaining_bridges)
                                    if score < best_score:
                                        best_score = score
                                        best_connection = (node1, node2)
                    
                    if best_connection and best_score <= 4:  # Only connect if both nodes have few remaining
                        node1, node2 = best_connection
                        self._add_bridges(puzzle, node1.id, node2.id, 1)
                        added_bridges += 1
                        rules_used['connectivity'] += 1
                        
        return added_bridges
        
    def _apply_minimum_bridges_rule(self, graph: Dict[int, GraphNode], puzzle: Puzzle,
                                   rules_used: Dict[str, int]) -> int:
        """Apply minimum bridges required rule"""
        added_bridges = 0
        
        for node in graph.values():
            if node.completed or node.remaining_bridges == 0:
                continue
                
            available_neighbors = []
            for neighbor in node.neighbors:
                current = node.bridges.get(neighbor.id, 0)
                if current < 2 and neighbor.remaining_bridges > 0:
                    available_neighbors.append({
                        'node': neighbor,
                        'current': current,
                        'can_add': min(2 - current, neighbor.remaining_bridges)
                    })
            
            if not available_neighbors:
                continue
                
            # If we need many bridges and have few neighbors
            if node.remaining_bridges >= len(available_neighbors) and len(available_neighbors) > 1:
                # Must add at least one bridge to most neighbors
                min_per_neighbor = node.remaining_bridges // len(available_neighbors)
                
                for neighbor_info in available_neighbors:
                    if neighbor_info['current'] < min_per_neighbor:
                        to_add = min(min_per_neighbor - neighbor_info['current'], 
                                   neighbor_info['can_add'])
                        if to_add > 0:
                            self._add_bridges(puzzle, node.id, neighbor_info['node'].id, to_add)
                            added_bridges += to_add
                            rules_used['minimum_distribution'] += 1
                            
        return added_bridges
        
    def _make_educated_guess(self, puzzle: Puzzle) -> Dict[str, Any]:
        """Make an educated guess when stuck"""
        graph = self._create_graph(puzzle)
        
        # Find the best candidate for guessing
        best_guess = None
        best_score = float('inf')
        
        for node in graph.values():
            if node.completed or node.remaining_bridges == 0:
                continue
                
            # Prefer nodes with fewer options
            options = []
            for neighbor in node.neighbors:
                current = node.bridges.get(neighbor.id, 0)
                if current < 2 and neighbor.remaining_bridges > 0:
                    options.append((neighbor, current))
            
            if 1 < len(options) <= 3:  # Good candidates for guessing
                # Score based on impact
                score = node.remaining_bridges * len(options)
                if score < best_score:
                    best_score = score
                    best_guess = (node, options)
        
        if best_guess:
            node, options = best_guess
            # Try the most promising option
            neighbor, current = min(options, key=lambda x: x[1].remaining_bridges)
            self._add_bridges(puzzle, node.id, neighbor.id, 1)
            self.logger.debug(f"Made educated guess: bridge between {node.id} and {neighbor.id}")
            return {'success': True}
            
        return {'success': False}
        
    def _find_components_weak(self, graph: Dict[int, GraphNode]) -> List[List[GraphNode]]:
        """Find weakly connected components (considering potential connections)"""
        visited = set()
        components = []
        
        for node in graph.values():
            if node.id not in visited:
                component = []
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current.id in visited:
                        continue
                        
                    visited.add(current.id)
                    component.append(current)
                    
                    # Add all neighbors (not just connected ones)
                    for neighbor in current.neighbors:
                        if neighbor.id not in visited:
                            # Include if there's a bridge OR both have remaining capacity
                            has_bridge = current.bridges.get(neighbor.id, 0) > 0
                            can_connect = (current.remaining_bridges > 0 and 
                                         neighbor.remaining_bridges > 0 and
                                         current.bridges.get(neighbor.id, 0) < 2)
                            
                            if has_bridge or can_connect:
                                queue.append(neighbor)
                            
                if component:
                    components.append(component)
                    
        return components
        
    def _create_graph(self, puzzle: Puzzle) -> Dict[int, GraphNode]:
        """Create graph representation of puzzle"""
        graph = {}
        
        # Create nodes
        for island in puzzle.islands:
            graph[island.id] = GraphNode(island)
            
        # Add neighbors based on valid connections
        for island in puzzle.islands:
            node = graph[island.id]
            for neighbor_id in puzzle._valid_connections.get(island.id, []):
                if neighbor_id in graph:
                    neighbor = graph[neighbor_id]
                    if neighbor not in node.neighbors:
                        node.neighbors.append(neighbor)
                    
        # Add existing bridges
        for bridge in puzzle.bridges:
            if bridge.island1_id in graph and bridge.island2_id in graph:
                node1 = graph[bridge.island1_id]
                node2 = graph[bridge.island2_id]
                
                node1.bridges[bridge.island2_id] = bridge.count
                node2.bridges[bridge.island1_id] = bridge.count
            
        # Update completed status
        for node in graph.values():
            if node.current_bridges == node.value:
                node.completed = True
                
        return graph
        
    def _add_bridges(self, puzzle: Puzzle, island1_id: int, island2_id: int, count: int):
        """Add bridges between two islands"""
        # Check for existing bridge
        for bridge in puzzle.bridges:
            if ((bridge.island1_id == island1_id and bridge.island2_id == island2_id) or
                (bridge.island1_id == island2_id and bridge.island2_id == island1_id)):
                # Update existing bridge
                new_count = min(bridge.count + count, 2)
                bridge.count = new_count
                return
                
        # Add new bridge
        puzzle.add_bridge(island1_id, island2_id, min(count, 2))
        
    def _update_graph(self, graph: Dict[int, GraphNode], puzzle: Puzzle):
        """Update graph based on current puzzle state"""
        # Reset bridges in graph
        for node in graph.values():
            node.bridges.clear()
            node.completed = False
            
        # Re-add bridges from puzzle
        for bridge in puzzle.bridges:
            if bridge.island1_id in graph and bridge.island2_id in graph:
                node1 = graph[bridge.island1_id]
                node2 = graph[bridge.island2_id]
                
                node1.bridges[bridge.island2_id] = bridge.count
                node2.bridges[bridge.island1_id] = bridge.count
            
        # Update completed status
        for node in graph.values():
            if node.current_bridges == node.value:
                node.completed = True
                
    def _is_puzzle_completed(self, graph: Dict[int, GraphNode]) -> bool:
        """Check if all nodes are completed and connected"""
        # Check if all nodes are satisfied
        if not all(node.completed for node in graph.values()):
            return False
            
        # Check connectivity
        if not graph:
            return True
            
        # BFS to check if all nodes are reachable
        start = next(iter(graph.values()))
        visited = set()
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            if current.id in visited:
                continue
                
            visited.add(current.id)
            
            # Add connected neighbors
            for neighbor_id, bridge_count in current.bridges.items():
                if bridge_count > 0 and neighbor_id not in visited and neighbor_id in graph:
                    queue.append(graph[neighbor_id])
                    
        return len(visited) == len(graph)
        
    def _puzzle_state_string(self, puzzle: Puzzle) -> str:
        """Create a string representation of puzzle state for comparison"""
        bridges = sorted([(min(b.island1_id, b.island2_id), 
                          max(b.island1_id, b.island2_id), 
                          b.count) for b in puzzle.bridges])
        return str(bridges)
    
    def _find_components(self, graph: Dict[int, GraphNode]) -> List[List[GraphNode]]:
        """Find connected components in the graph (only via existing bridges)"""
        visited = set()
        components = []
        
        for node in graph.values():
            if node.id not in visited:
                component = []
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current.id in visited:
                        continue
                        
                    visited.add(current.id)
                    component.append(current)
                    
                    # Add connected neighbors (with bridges)
                    for neighbor_id, bridge_count in current.bridges.items():
                        if neighbor_id not in visited and bridge_count > 0 and neighbor_id in graph:
                            queue.append(graph[neighbor_id])
                            
                if component:
                    components.append(component)
                    
        return components


# For backward compatibility
GreedySolver = IterativeSolver