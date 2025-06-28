"""
Large Neighborhood Search (LNS) solver for Hashiwokakero - OPTIMIZED VERSION.
Includes performance optimizations, parallel repairs, and smart heuristics.
"""

from typing import Optional, Dict, Any, List, Set, Tuple
import random
import time
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq

from .base_solver import BaseSolver, SolverConfig, SolverResult
from .ilp_solver import ILPSolver, ILPSolverConfig
from ..core.puzzle import Puzzle, Bridge
from ..core.validator import PuzzleValidator


class DestroyMethod(Enum):
    """Different strategies for destroying partial solutions"""
    RANDOM = "random"
    WORST_CONNECTED = "worst_connected"
    GEOGRAPHICAL = "geographical"
    BRIDGE_REMOVAL = "bridge_removal"
    CONSTRAINT_BASED = "constraint_based"
    SMART_SELECTION = "smart_selection"  # NEW: Smart adaptive selection
    CRITICAL_PATH = "critical_path"      # NEW: Target critical connections


class LNSSolverConfig(SolverConfig):
    """Configuration for LNS solver"""
    
    def __init__(self, **kwargs):
        # Extract LNS-specific parameters before passing to parent
        self.initial_destroy_rate = kwargs.pop('initial_destroy_rate', 0.3)
        self.min_destroy_rate = kwargs.pop('min_destroy_rate', 0.1)
        self.max_destroy_rate = kwargs.pop('max_destroy_rate', 0.6)
        self.destroy_rate_increase = kwargs.pop('destroy_rate_increase', 1.1)
        self.destroy_rate_decrease = kwargs.pop('destroy_rate_decrease', 0.95)
        
        # Repair parameters
        self.repair_time_limit = kwargs.pop('repair_time_limit', 5.0)
        self.use_warm_start = kwargs.pop('use_warm_start', True)
        self.use_parallel_repair = kwargs.pop('use_parallel_repair', True)
        self.max_repair_threads = kwargs.pop('max_repair_threads', 3)
        
        # Acceptance criteria
        self.accept_worse_solutions = kwargs.pop('accept_worse_solutions', True)
        self.initial_temperature = kwargs.pop('initial_temperature', 10.0)
        self.cooling_rate = kwargs.pop('cooling_rate', 0.95)
        
        # Early stopping parameters
        self.max_iterations_without_improvement = kwargs.pop('max_iterations_without_improvement', 100)
        self.min_temperature = kwargs.pop('min_temperature', 0.01)
        
        # Performance optimizations
        self.cache_evaluations = kwargs.pop('cache_evaluations', True)
        self.adaptive_destroy = kwargs.pop('adaptive_destroy', True)
        self.use_smart_destroy = kwargs.pop('use_smart_destroy', True)
        
        # Method weights (adaptive)
        self.destroy_weights = kwargs.pop('destroy_weights', {
            DestroyMethod.RANDOM: 1.0,
            DestroyMethod.WORST_CONNECTED: 1.5,
            DestroyMethod.GEOGRAPHICAL: 1.0,
            DestroyMethod.BRIDGE_REMOVAL: 1.0,
            DestroyMethod.CONSTRAINT_BASED: 2.0,
            DestroyMethod.SMART_SELECTION: 2.5,
            DestroyMethod.CRITICAL_PATH: 2.0,
        })
        
        # Statistics tracking
        self.track_statistics = kwargs.pop('track_statistics', True)
        
        # Call parent constructor with remaining kwargs
        super().__init__(**kwargs)


@dataclass
class LNSStatistics:
    """Statistics for LNS performance analysis"""
    iterations: int = 0
    improvements: int = 0
    destroy_method_usage: Dict[DestroyMethod, int] = None
    destroy_method_success: Dict[DestroyMethod, int] = None
    destroy_rate_history: List[float] = None
    objective_history: List[float] = None
    time_history: List[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    repair_times: List[float] = None
    
    def __post_init__(self):
        if self.destroy_method_usage is None:
            self.destroy_method_usage = {method: 0 for method in DestroyMethod}
        if self.destroy_method_success is None:
            self.destroy_method_success = {method: 0 for method in DestroyMethod}
        if self.destroy_rate_history is None:
            self.destroy_rate_history = []
        if self.objective_history is None:
            self.objective_history = []
        if self.time_history is None:
            self.time_history = []
        if self.repair_times is None:
            self.repair_times = []


class SolutionCache:
    """Simple cache for solution evaluations"""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get_key(self, solution: Puzzle) -> str:
        """Generate unique key for solution"""
        bridges = []
        for b in solution.bridges:
            id1, id2 = min(b.island1_id, b.island2_id), max(b.island1_id, b.island2_id)
            bridges.append(f"{id1}-{id2}-{b.count}")
        return "|".join(sorted(bridges))
        
    def get(self, solution: Puzzle) -> Optional[float]:
        """Get cached evaluation"""
        key = self.get_key(solution)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
        
    def put(self, solution: Puzzle, value: float):
        """Store evaluation"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        key = self.get_key(solution)
        self.cache[key] = value


class LargeNeighborhoodSearchSolver(BaseSolver):
    """
    Large Neighborhood Search solver for Hashiwokakero with optimizations.
    
    This solver combines:
    1. Adaptive destroy operators
    2. ILP-based repair with parallel strategies
    3. Simulated annealing acceptance
    4. Statistical learning for operator selection
    5. Performance optimizations (caching, precomputation)
    """
    
    def __init__(self, config: Optional[LNSSolverConfig] = None):
        super().__init__(config or LNSSolverConfig())
        self.stats = LNSStatistics()
        self.current_destroy_rate = self.config.initial_destroy_rate
        self.temperature = self.config.initial_temperature
        
        # Caching
        self.solution_cache = SolutionCache() if self.config.cache_evaluations else None
        
        # Precomputed data
        self._neighbor_cache = {}
        self._distance_cache = {}
        self._critical_islands = set()
        self._last_solution_hash = None
        self._has_crossing = False
        
    def _solve(self, puzzle: Puzzle) -> SolverResult:
        """Main LNS algorithm with optimizations"""
        self.logger.info("Starting Large Neighborhood Search (Optimized)")
        
        # Precompute frequently used data
        self._precompute_data(puzzle)
        
        # Get initial solution
        current_solution = self._get_initial_solution(puzzle)
        if current_solution is None:
            return SolverResult(
                success=False,
                message="Failed to find initial solution"
            )
        
        best_solution = current_solution.copy()
        best_objective = self._evaluate_solution_fast(best_solution)
        
        # Check if initial solution is already optimal
        validation = PuzzleValidator.validate_solution(best_solution)
        if validation.is_valid and best_objective == 0:
            self.logger.info("Initial solution is already optimal!")
            return SolverResult(
                success=True,
                solution=best_solution,
                message="Found optimal solution immediately",
                stats={
                    'iterations': 0,
                    'improvements': 0,
                    'final_objective': 0
                }
            )
        
        # Main LNS loop
        no_improvement_counter = 0
        last_objective = best_objective
        destroy_success_rates = {method: 0.5 for method in DestroyMethod}
        
        while not self._check_time_limit():
            self._increment_iteration()
            self.stats.iterations += 1
            
            # Check early stopping conditions
            if no_improvement_counter >= self.config.max_iterations_without_improvement:
                self.logger.info(f"No improvement for {no_improvement_counter} iterations, stopping")
                break
                
            if self.temperature < self.config.min_temperature:
                self.logger.info(f"Temperature too low ({self.temperature:.6f}), stopping")
                break
            
            # Adaptive destroy rate
            if self.config.adaptive_destroy:
                self.current_destroy_rate = self._adaptive_destroy_rate(no_improvement_counter)
            
            # Select destroy method using adaptive weights
            destroy_method = self._select_destroy_method_adaptive(destroy_success_rates)
            self.stats.destroy_method_usage[destroy_method] += 1
            
            # Destroy part of the solution
            partial_solution, destroyed_components = self._destroy_optimized(
                current_solution, 
                destroy_method,
                self.current_destroy_rate
            )
            
            # Skip if nothing was destroyed
            if len(destroyed_components) == 0:
                no_improvement_counter += 1
                continue
            
            # Repair the solution
            repair_start = time.time()
            if self.config.use_parallel_repair and len(destroyed_components) > 10:
                new_solution = self._parallel_repair(
                    partial_solution, 
                    destroyed_components,
                    puzzle
                )
            else:
                new_solution = self._repair(
                    partial_solution, 
                    destroyed_components,
                    puzzle
                )
            repair_time = time.time() - repair_start
            self.stats.repair_times.append(repair_time)
            
            if new_solution is None:
                # Repair failed, try different destroy rate
                self._adapt_destroy_rate(success=False)
                no_improvement_counter += 1
                self._update_destroy_success(destroy_method, destroy_success_rates, False)
                continue
            
            # Evaluate new solution
            new_objective = self._evaluate_solution_fast(new_solution)
            
            # Check if solution is identical to current (avoid cycling)
            if self._solutions_are_identical(new_solution, current_solution):
                no_improvement_counter += 1
                # Force diversification
                if no_improvement_counter > 20:
                    self.current_destroy_rate = min(self.config.max_destroy_rate, 
                                                   self.current_destroy_rate * 1.5)
                continue
            
            # Acceptance decision
            current_objective = self._evaluate_solution_fast(current_solution)
            if self._accept_solution(new_objective, current_objective):
                current_solution = new_solution
                
                # Update best solution if improved
                if new_objective < best_objective:
                    best_solution = new_solution.copy()
                    best_objective = new_objective
                    self.stats.improvements += 1
                    self.stats.destroy_method_success[destroy_method] += 1
                    no_improvement_counter = 0
                    
                    self.logger.info(f"New best solution found: objective = {best_objective}")
                    
                    # Check if optimal
                    if best_objective == 0:
                        validation = PuzzleValidator.validate_solution(best_solution)
                        if validation.is_valid:
                            self.logger.info("Found optimal solution!")
                            break
                    
                    # Adapt parameters on success
                    self._adapt_destroy_rate(success=True)
                    self._update_destroy_success(destroy_method, destroy_success_rates, True)
                else:
                    no_improvement_counter += 1
            else:
                no_improvement_counter += 1
                self._update_destroy_success(destroy_method, destroy_success_rates, False)
            
            # Adaptive parameter updates
            if no_improvement_counter > 50:
                self._diversification_phase()
                no_improvement_counter = 0
            
            # Update temperature
            self.temperature *= self.config.cooling_rate
            
            # Statistics tracking
            if self.config.track_statistics:
                self.stats.destroy_rate_history.append(self.current_destroy_rate)
                self.stats.objective_history.append(new_objective)
                self.stats.time_history.append(time.time() - self._start_time)
            
            # Progress callback
            self._call_progress_callbacks(current_solution, {
                'iteration': self.stats.iterations,
                'best_objective': best_objective,
                'current_objective': current_objective,
                'destroy_rate': self.current_destroy_rate,
                'temperature': self.temperature,
                'no_improvement': no_improvement_counter
            })
        
        # Final validation
        validation = PuzzleValidator.validate_solution(best_solution)
        
        # Prepare statistics
        stats = {
            'iterations': self.stats.iterations,
            'improvements': self.stats.improvements,
            'final_objective': best_objective,
            'destroy_method_usage': dict(self.stats.destroy_method_usage),
            'destroy_method_success': dict(self.stats.destroy_method_success),
            'final_destroy_rate': self.current_destroy_rate,
            'avg_repair_time': np.mean(self.stats.repair_times) if self.stats.repair_times else 0
        }
        
        if self.solution_cache:
            stats['cache_hits'] = self.solution_cache.hits
            stats['cache_misses'] = self.solution_cache.misses
            stats['cache_hit_rate'] = self.solution_cache.hits / max(1, self.solution_cache.hits + self.solution_cache.misses)
        
        return SolverResult(
            success=validation.is_valid,
            solution=best_solution,
            message=f"LNS completed with {self.stats.improvements} improvements",
            stats=stats
        )
    
    def _precompute_data(self, puzzle: Puzzle):
        """Precompute frequently used data for efficiency"""
        # Cache valid neighbors
        for island in puzzle.islands:
            self._neighbor_cache[island.id] = set(puzzle._valid_connections.get(island.id, []))
        
        # Precompute distances between islands
        n = len(puzzle.islands)
        for i in range(n):
            for j in range(i+1, n):
                dist = abs(puzzle.islands[i].row - puzzle.islands[j].row) + \
                       abs(puzzle.islands[i].col - puzzle.islands[j].col)
                self._distance_cache[(i, j)] = dist
                self._distance_cache[(j, i)] = dist
        
        # Identify critical islands (high degree or bottlenecks)
        for island in puzzle.islands:
            neighbors = len(self._neighbor_cache.get(island.id, []))
            if island.required_bridges >= 6 or neighbors <= 2:
                self._critical_islands.add(island.id)
    
    def _evaluate_solution_fast(self, solution: Puzzle) -> float:
        """Fast evaluation with caching"""
        # Check cache first
        if self.solution_cache:
            cached = self.solution_cache.get(solution)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached
            self.stats.cache_misses += 1
        
        score = 0.0
        
        # Penalty for incorrect bridge counts (vectorized)
        violations = [abs(solution.get_island_bridges(island.id) - island.required_bridges) 
                      for island in solution.islands]
        score = sum(v ** 2 for v in violations)
        
        # Only check connectivity if needed
        if score > 0 and score < 1000:
            if not self._is_connected_fast(solution):
                score += 1000
        
        # Crossing check with caching
        solution_hash = hash(tuple(sorted((b.island1_id, b.island2_id, b.count) 
                                        for b in solution.bridges)))
        if solution_hash != self._last_solution_hash:
            self._has_crossing = solution.has_crossing_bridges()
            self._last_solution_hash = solution_hash
        
        if self._has_crossing:
            score += 500
        
        # Cache result
        if self.solution_cache:
            self.solution_cache.put(solution, score)
        
        return score
    
    def _is_connected_fast(self, solution: Puzzle) -> bool:
        """Fast connectivity check using BFS"""
        if not solution.bridges:
            return len(solution.islands) <= 1
        
        # Build adjacency list
        adj = defaultdict(set)
        for bridge in solution.bridges:
            adj[bridge.island1_id].add(bridge.island2_id)
            adj[bridge.island2_id].add(bridge.island1_id)
        
        # BFS from first island with bridges
        if not adj:
            return True
            
        start = next(iter(adj.keys()))
        visited = {start}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Check if all islands with required bridges are connected
        for island in solution.islands:
            if island.required_bridges > 0 and island.id not in visited:
                return False
        
        return True
    
    def _adaptive_destroy_rate(self, stagnation_counter: int) -> float:
        """Adaptive destroy rate based on search progress"""
        base_rate = self.current_destroy_rate
        
        if stagnation_counter > 30:
            # Increase destruction significantly when stuck
            factor = 1 + (stagnation_counter - 30) * 0.03
            return min(self.config.max_destroy_rate, base_rate * factor)
        elif stagnation_counter < 5:
            # Decrease when making good progress
            return max(self.config.min_destroy_rate, base_rate * 0.9)
        
        return base_rate
    
    def _select_destroy_method_adaptive(self, success_rates: Dict[DestroyMethod, float]) -> DestroyMethod:
        """Select destroy method based on historical performance"""
        methods = list(self.config.destroy_weights.keys())
        scores = []
        
        for method in methods:
            base_weight = self.config.destroy_weights[method]
            success_rate = success_rates.get(method, 0.5)
            # Combine base weight with success rate
            score = base_weight * (0.5 + success_rate)
            scores.append(score)
        
        # Normalize and select
        total = sum(scores)
        if total == 0:
            return random.choice(methods)
        
        probs = [s / total for s in scores]
        return np.random.choice(methods, p=probs)
    
    def _update_destroy_success(self, method: DestroyMethod, 
                               success_rates: Dict[DestroyMethod, float],
                               success: bool):
        """Update success rates with exponential smoothing"""
        alpha = 0.1  # Smoothing factor
        old_rate = success_rates.get(method, 0.5)
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_rate
        success_rates[method] = new_rate
    
    def _destroy_optimized(self, solution: Puzzle, method: DestroyMethod,
                          destroy_rate: float) -> Tuple[Puzzle, Set[int]]:
        """Optimized destroy with smart targeting"""
        partial = solution.copy()
        num_islands = len(solution.islands)
        num_to_destroy = max(2, int(num_islands * destroy_rate))
        
        # Ensure we don't destroy everything
        num_to_destroy = min(num_to_destroy, num_islands - 2)
        
        if method == DestroyMethod.SMART_SELECTION:
            destroyed = self._smart_destroy_selection(partial, num_to_destroy)
        elif method == DestroyMethod.CRITICAL_PATH:
            destroyed = self._critical_path_destroy(partial, num_to_destroy)
        elif method == DestroyMethod.RANDOM:
            destroyed = self._random_destroy(partial, num_to_destroy)
        elif method == DestroyMethod.WORST_CONNECTED:
            destroyed = self._worst_connected_destroy(partial, num_to_destroy)
        elif method == DestroyMethod.GEOGRAPHICAL:
            destroyed = self._geographical_destroy(partial, num_to_destroy)
        elif method == DestroyMethod.BRIDGE_REMOVAL:
            destroyed = self._bridge_removal_destroy(partial, num_to_destroy)
        elif method == DestroyMethod.CONSTRAINT_BASED:
            destroyed = self._constraint_based_destroy(partial, num_to_destroy)
        else:
            destroyed = self._random_destroy(partial, num_to_destroy)
        
        return partial, destroyed
    
    def _smart_destroy_selection(self, solution: Puzzle, num_to_destroy: int) -> Set[int]:
        """Smart selection of islands to destroy based on problem structure"""
        scores = []
        
        for island in solution.islands:
            score = 0
            
            # Violation score
            current = solution.get_island_bridges(island.id)
            required = island.required_bridges
            violation = abs(current - required)
            score += violation * 10
            
            # Connectivity score (prefer destroying hubs)
            num_neighbors = len(self._neighbor_cache.get(island.id, []))
            if num_neighbors > 4:
                score += 5
            
            # Critical island bonus
            if island.id in self._critical_islands:
                score += 3
            
            # Add randomness to avoid getting stuck
            score += random.random() * 3
            
            scores.append((island.id, score))
        
        # Select top scoring islands
        scores.sort(key=lambda x: x[1], reverse=True)
        destroyed = set(s[0] for s in scores[:num_to_destroy])
        
        # Remove bridges efficiently
        self._remove_bridges_batch(solution, destroyed)
        
        return destroyed
    
    def _critical_path_destroy(self, solution: Puzzle, num_to_destroy: int) -> Set[int]:
        """Destroy islands on critical paths"""
        # Find articulation points (cut vertices)
        articulation_points = self._find_articulation_points(solution)
        
        # Score islands
        scores = []
        for island in solution.islands:
            score = 0
            
            # Articulation points are critical
            if island.id in articulation_points:
                score += 15
            
            # Violation score
            current = solution.get_island_bridges(island.id)
            required = island.required_bridges
            score += abs(current - required) * 5
            
            # Degree
            score += len(self._neighbor_cache.get(island.id, [])) * 2
            
            scores.append((island.id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        destroyed = set(s[0] for s in scores[:num_to_destroy])
        
        self._remove_bridges_batch(solution, destroyed)
        return destroyed
    
    def _find_articulation_points(self, solution: Puzzle) -> Set[int]:
        """Find articulation points using DFS"""
        if not solution.bridges:
            return set()
        
        # Build adjacency list
        adj = defaultdict(set)
        for bridge in solution.bridges:
            adj[bridge.island1_id].add(bridge.island2_id)
            adj[bridge.island2_id].add(bridge.island1_id)
        
        if not adj:
            return set()
        
        visited = set()
        disc = {}
        low = {}
        parent = {}
        ap = set()
        time_counter = [0]
        
        def dfs(u):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time_counter[0]
            time_counter[0] += 1
            
            for v in adj[u]:
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        
        # Run DFS from all unvisited nodes
        for node in adj:
            if node not in visited:
                parent[node] = None
                dfs(node)
        
        return ap
    
    def _remove_bridges_batch(self, solution: Puzzle, islands: Set[int]):
        """Remove all bridges connected to given islands efficiently"""
        solution.bridges = [b for b in solution.bridges
                           if b.island1_id not in islands and b.island2_id not in islands]
    
    def _parallel_repair(self, partial_solution: Puzzle, destroyed_islands: Set[int], 
                        original_puzzle: Puzzle) -> Optional[Puzzle]:
        """Try multiple repair strategies in parallel"""
        strategies = [
            ('ilp_quick', 2.0),   # Quick ILP with short time limit
            ('ilp_full', self.config.repair_time_limit),    # Full ILP
            ('greedy', 0.1),      # Very fast greedy
        ]
        
        best_solution = None
        best_objective = float('inf')
        
        with ThreadPoolExecutor(max_workers=min(self.config.max_repair_threads, len(strategies))) as executor:
            futures = {}
            
            for strategy, time_limit in strategies:
                if strategy == 'greedy':
                    future = executor.submit(self._greedy_repair, partial_solution, 
                                           destroyed_islands, original_puzzle)
                else:
                    future = executor.submit(self._ilp_repair_with_timeout, partial_solution,
                                           destroyed_islands, original_puzzle, time_limit)
                futures[future] = strategy
            
            for future in as_completed(futures, timeout=self.config.repair_time_limit):
                try:
                    repaired = future.result()
                    if repaired:
                        obj = self._evaluate_solution_fast(repaired)
                        if obj < best_objective:
                            best_solution = repaired
                            best_objective = obj
                except:
                    continue
        
        return best_solution
    
    def _greedy_repair(self, partial: Puzzle, destroyed: Set[int], 
                       original: Puzzle) -> Optional[Puzzle]:
        """Fast greedy repair heuristic"""
        result = partial.copy()
        
        # Sort destroyed islands by required bridges (descending)
        destroyed_sorted = sorted(destroyed, 
                                key=lambda i: original.islands[i].required_bridges if i < len(original.islands) else 0, 
                                reverse=True)
        
        for island_id in destroyed_sorted:
            if island_id >= len(original.islands):
                continue
                
            island = original.islands[island_id]
            current = result.get_island_bridges(island_id)
            needed = island.required_bridges - current
            
            if needed <= 0:
                continue
            
            # Get neighbors sorted by available capacity
            neighbors = []
            for n_id in self._neighbor_cache.get(island_id, []):
                if n_id >= len(original.islands):
                    continue
                neighbor = original.islands[n_id]
                available = neighbor.required_bridges - result.get_island_bridges(n_id)
                if available > 0:
                    # Use cached distance
                    dist = self._distance_cache.get((island_id, n_id), float('inf'))
                    score = available / (1 + dist)
                    neighbors.append((n_id, available, score))
            
            # Sort by score
            neighbors.sort(key=lambda x: x[2], reverse=True)
            
            # Add bridges greedily
            for n_id, available, _ in neighbors:
                if needed <= 0:
                    break
                bridges_to_add = min(needed, available, 2)
                if bridges_to_add > 0:
                    result.add_bridge(island_id, n_id, bridges_to_add)
                    needed -= bridges_to_add
        
        return result
    
    def _ilp_repair_with_timeout(self, partial: Puzzle, destroyed: Set[int],
                                original: Puzzle, time_limit: float) -> Optional[Puzzle]:
        """ILP repair with specific time limit"""
        # Create subproblem
        subproblem_islands = set(destroyed)
        
        # Add only immediate neighbors
        for island_id in destroyed:
            for neighbor_id in self._neighbor_cache.get(island_id, []):
                subproblem_islands.add(neighbor_id)
        
        subpuzzle = self._create_subpuzzle(original, subproblem_islands, partial)
        
        config = ILPSolverConfig(
            time_limit=time_limit,
            solver_name='cbc',
            verbose=False,
            use_lazy_constraints=True,
            use_preprocessing=True,
        )
        
        solver = ILPSolver(config)
        result = solver.solve(subpuzzle)
        
        if not result.success:
            return None
        
        return self._merge_solutions(partial, result.solution, subproblem_islands)
    
    def _solutions_are_identical(self, sol1: Puzzle, sol2: Puzzle) -> bool:
        """Check if two solutions have identical bridges"""
        if len(sol1.bridges) != len(sol2.bridges):
            return False
            
        # Create sorted bridge representations
        bridges1 = set()
        for b in sol1.bridges:
            # Normalize bridge representation (smaller id first)
            id1, id2 = min(b.island1_id, b.island2_id), max(b.island1_id, b.island2_id)
            bridges1.add((id1, id2, b.count))
            
        bridges2 = set()
        for b in sol2.bridges:
            id1, id2 = min(b.island1_id, b.island2_id), max(b.island1_id, b.island2_id)
            bridges2.add((id1, id2, b.count))
            
        return bridges1 == bridges2
    
    def _get_initial_solution(self, puzzle: Puzzle) -> Optional[Puzzle]:
        """Generate initial solution using ILP with short time limit"""
        self.logger.info("Generating initial solution...")
        
        config = ILPSolverConfig(
            time_limit=min(30.0, self.config.time_limit * 0.1),
            solver_name='cbc',
            verbose=False,
            use_lazy_constraints=True,
            use_preprocessing=True
        )
        
        solver = ILPSolver(config)
        result = solver.solve(puzzle)
        
        if result.success:
            return result.solution
        else:
            # Try greedy if ILP fails
            self.logger.warning("ILP failed for initial solution, trying greedy")
            return self._greedy_repair(puzzle.copy(), set(range(len(puzzle.islands))), puzzle)
    
    def _random_destroy(self, solution: Puzzle, num_to_destroy: int) -> Set[int]:
        """Randomly select islands to destroy"""
        if num_to_destroy >= len(solution.islands):
            num_to_destroy = max(1, len(solution.islands) // 2)
            
        islands = random.sample(range(len(solution.islands)), num_to_destroy)
        destroyed = set(islands)
        
        self._remove_bridges_batch(solution, destroyed)
        return destroyed
    
    def _worst_connected_destroy(self, solution: Puzzle, num_to_destroy: int) -> Set[int]:
        """Destroy islands with worst connectivity"""
        # Calculate connectivity score for each island
        scores = []
        for island in solution.islands:
            current = solution.get_island_bridges(island.id)
            required = island.required_bridges
            violation = abs(current - required)
            scores.append((island.id, violation))
        
        # Sort by violation (worst first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select worst islands
        destroyed = set(score[0] for score in scores[:num_to_destroy])
        
        self._remove_bridges_batch(solution, destroyed)
        return destroyed
    
    def _geographical_destroy(self, solution: Puzzle, num_to_destroy: int) -> Set[int]:
        """Destroy islands in a geographical region"""
        if len(solution.islands) == 0:
            return set()
            
        # Select random center
        center = random.choice(solution.islands)
        
        # Calculate distances from center
        distances = []
        for island in solution.islands:
            dist = self._distance_cache.get((center.id, island.id), 
                                           abs(island.row - center.row) + abs(island.col - center.col))
            distances.append((island.id, dist))
        
        # Sort by distance and select closest
        distances.sort(key=lambda x: x[1])
        destroyed = set(d[0] for d in distances[:num_to_destroy])
        
        self._remove_bridges_batch(solution, destroyed)
        return destroyed
    
    def _bridge_removal_destroy(self, solution: Puzzle, num_to_destroy: int) -> Set[int]:
        """Remove bridges and affected islands"""
        if not solution.bridges:
            return self._random_destroy(solution, num_to_destroy)
        
        # Remove random bridges
        num_bridges_to_remove = max(1, min(num_to_destroy // 2, len(solution.bridges)))
        bridges_to_remove = random.sample(solution.bridges, num_bridges_to_remove)
        
        # Collect affected islands
        destroyed = set()
        for bridge in bridges_to_remove:
            destroyed.add(bridge.island1_id)
            destroyed.add(bridge.island2_id)
            solution.bridges.remove(bridge)
        
        # Limit destroyed islands
        if len(destroyed) > num_to_destroy:
            destroyed = set(list(destroyed)[:num_to_destroy])
        
        # Remove all bridges of affected islands
        self._remove_bridges_batch(solution, destroyed)
        return destroyed
    
    def _constraint_based_destroy(self, solution: Puzzle, num_to_destroy: int) -> Set[int]:
        """Destroy based on constraint violations"""
        # Identify constraint violations
        violations = []
        
        # Check bridge count violations
        for island in solution.islands:
            current = solution.get_island_bridges(island.id)
            required = island.required_bridges
            if current != required:
                violations.append((island.id, abs(current - required)))
        
        if not violations:
            return self._random_destroy(solution, num_to_destroy)
        
        # Sort by violation severity
        violations.sort(key=lambda x: x[1], reverse=True)
        
        # Select most violated islands
        destroyed = set(v[0] for v in violations[:num_to_destroy])
        
        # Also include neighbors of highly violated islands
        for island_id, violation in violations[:num_to_destroy//2]:
            if island_id in self._neighbor_cache:
                for neighbor_id in self._neighbor_cache[island_id]:
                    if len(destroyed) < num_to_destroy:
                        destroyed.add(neighbor_id)
        
        self._remove_bridges_batch(solution, destroyed)
        return destroyed
    
    def _repair(self, partial_solution: Puzzle, destroyed_islands: Set[int], 
                original_puzzle: Puzzle) -> Optional[Puzzle]:
        """Repair destroyed solution using ILP"""
        # Create subproblem for destroyed islands and their neighbors
        subproblem_islands = set(destroyed_islands)
        
        # Add neighbors of destroyed islands
        for island_id in destroyed_islands:
            if island_id in self._neighbor_cache:
                for neighbor_id in self._neighbor_cache[island_id]:
                    subproblem_islands.add(neighbor_id)
        
        # Create subproblem puzzle
        subpuzzle = self._create_subpuzzle(original_puzzle, subproblem_islands, partial_solution)
        
        # Solve subproblem with ILP
        config = ILPSolverConfig(
            time_limit=self.config.repair_time_limit,
            solver_name='cbc',
            verbose=False,
            use_lazy_constraints=True,
            use_preprocessing=True
        )
        
        if self.config.use_warm_start:
            # Extract partial solution for warm start
            warm_start = self._extract_warm_start(partial_solution, subproblem_islands)
            from .ilp_solver import WarmStartILPSolver
            solver = WarmStartILPSolver(config, warm_start)
        else:
            solver = ILPSolver(config)
        
        result = solver.solve(subpuzzle)
        
        if not result.success:
            return None
        
        # Merge subproblem solution back
        return self._merge_solutions(partial_solution, result.solution, subproblem_islands)
    
    def _create_subpuzzle(self, original: Puzzle, islands: Set[int], 
                          partial: Puzzle) -> Puzzle:
        """Create subpuzzle for given islands"""
        # Map from original to subpuzzle IDs
        id_map = {}
        subpuzzle = Puzzle(original.width, original.height)
        
        # Add islands
        new_id = 0
        for island_id in sorted(islands):
            island = original._id_to_island[island_id]
            
            # Calculate adjusted requirement
            current_external = 0
            for bridge in partial.bridges:
                if bridge.island1_id == island_id and bridge.island2_id not in islands:
                    current_external += bridge.count
                elif bridge.island2_id == island_id and bridge.island1_id not in islands:
                    current_external += bridge.count
            
            adjusted_requirement = island.required_bridges - current_external
            subpuzzle.add_island(island.row, island.col, max(0, adjusted_requirement))
            id_map[island_id] = new_id
            new_id += 1
        
        return subpuzzle
    
    def _extract_warm_start(self, partial: Puzzle, islands: Set[int]) -> Puzzle:
        """Extract bridges between subproblem islands for warm start"""
        warm_start = Puzzle(partial.width, partial.height)
        
        # Map islands to new IDs
        id_map = {}
        new_id = 0
        for island_id in sorted(islands):
            if island_id < len(partial.islands):
                island = partial.islands[island_id]
                warm_start.add_island(island.row, island.col, island.required_bridges)
                id_map[island_id] = new_id
                new_id += 1
        
        # Copy relevant bridges
        for bridge in partial.bridges:
            if bridge.island1_id in islands and bridge.island2_id in islands:
                if bridge.island1_id in id_map and bridge.island2_id in id_map:
                    warm_start.add_bridge(
                        id_map[bridge.island1_id], 
                        id_map[bridge.island2_id], 
                        bridge.count
                    )
        
        return warm_start
    
    def _merge_solutions(self, partial: Puzzle, subproblem_solution: Puzzle, 
                        subproblem_islands: Set[int]) -> Puzzle:
        """Merge subproblem solution back into partial solution"""
        result = partial.copy()
        
        # Remove all bridges between subproblem islands
        bridges_to_remove = []
        for bridge in result.bridges:
            if (bridge.island1_id in subproblem_islands and 
                bridge.island2_id in subproblem_islands):
                bridges_to_remove.append(bridge)
        
        for bridge in bridges_to_remove:
            result.bridges.remove(bridge)
        
        # Add bridges from subproblem solution
        # Need to map IDs correctly
        id_map = {}
        sorted_islands = sorted(subproblem_islands)
        for new_id, old_id in enumerate(sorted_islands):
            id_map[new_id] = old_id
        
        for bridge in subproblem_solution.bridges:
            if bridge.island1_id in id_map and bridge.island2_id in id_map:
                old_id1 = id_map[bridge.island1_id]
                old_id2 = id_map[bridge.island2_id]
                result.add_bridge(old_id1, old_id2, bridge.count)
        
        return result
    
    def _evaluate_solution(self, solution: Puzzle) -> float:
        """Evaluate solution quality (lower is better)"""
        # Use fast version
        return self._evaluate_solution_fast(solution)
    
    def _accept_solution(self, new_objective: float, current_objective: float) -> bool:
        """Simulated annealing acceptance criterion"""
        if new_objective <= current_objective:
            return True
        
        if not self.config.accept_worse_solutions:
            return False
        
        # Avoid division by zero
        if self.temperature <= 0:
            return False
        
        # Metropolis criterion
        delta = new_objective - current_objective
        probability = math.exp(-delta / self.temperature)
        return random.random() < probability
    
    def _adapt_destroy_rate(self, success: bool):
        """Adapt destroy rate based on success"""
        if success:
            # Decrease destroy rate on success (smaller neighborhoods)
            self.current_destroy_rate *= self.config.destroy_rate_decrease
            self.current_destroy_rate = max(self.config.min_destroy_rate, 
                                          self.current_destroy_rate)
        else:
            # Increase destroy rate on failure (larger neighborhoods)
            self.current_destroy_rate *= self.config.destroy_rate_increase
            self.current_destroy_rate = min(self.config.max_destroy_rate, 
                                          self.current_destroy_rate)
    
    def _update_method_weights(self, method: DestroyMethod, success: bool):
        """Update adaptive weights for destroy methods"""
        # Simple additive weight adjustment
        if success:
            self.config.destroy_weights[method] += 0.1
        else:
            self.config.destroy_weights[method] *= 0.95
            self.config.destroy_weights[method] = max(0.1, 
                                                     self.config.destroy_weights[method])
    
    def _diversification_phase(self):
        """Diversification when stuck"""
        self.logger.info("Entering diversification phase")
        
        # Reset temperature
        self.temperature = self.config.initial_temperature * 0.5
        
        # Increase destroy rate temporarily
        self.current_destroy_rate = min(self.config.max_destroy_rate, 
                                       self.current_destroy_rate * 2)
        
        # Reset method weights to equal
        for method in DestroyMethod:
            self.config.destroy_weights[method] = 1.0


# Convenience alias
LNSSolver = LargeNeighborhoodSearchSolver