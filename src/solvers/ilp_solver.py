"""
Integer Linear Programming solver for Hashiwokakero using AMPL.
Optimized version with lazy constraints and preprocessing.
"""

from typing import Optional, Dict, Any, Tuple, List, Set
import time
import traceback
from amplpy import AMPL, Environment

from .base_solver import BaseSolver, SolverConfig, SolverResult
from ..core.puzzle import Puzzle, Bridge


class ILPSolverConfig(SolverConfig):
    """Configuration specific to ILP solver"""
    
    def __init__(self, **kwargs):
        # Extract ILP-specific parameters before passing to parent
        self.solver_name = kwargs.pop('solver_name', 'cbc')
        self.solver_options = kwargs.pop('solver_options', {})
        self.use_symmetry_breaking = kwargs.pop('use_symmetry_breaking', False)
        self.use_valid_inequalities = kwargs.pop('use_valid_inequalities', False)
        self.use_lazy_constraints = kwargs.pop('use_lazy_constraints', True)  # New option
        self.use_preprocessing = kwargs.pop('use_preprocessing', True)  # New option
        self.presolve = kwargs.pop('presolve', True)
        self.debug_mode = kwargs.pop('debug_mode', False)
        
        # Call parent constructor with remaining kwargs
        super().__init__(**kwargs)
        
        # Set default solver options if not provided
        if not self.solver_options:
            if self.solver_name == 'cbc':
                # CBC-specific options
                self.solver_options = {
                    'timelimit': self.time_limit,
                    'ratioGap': 0.01,
                    # Note: CBC doesn't support 'cuts' and 'heuristics' in the same way as Gurobi
                }
            else:
                # Generic options for other solvers
                self.solver_options = {
                    'timelimit': self.time_limit,
                    'threads': 4,
                    'mipgap': 0.01,
                }


class ILPSolver(BaseSolver):
    """
    Solve Hashiwokakero using Integer Linear Programming.
    Optimized with lazy constraints and preprocessing.
    """
    
    def __init__(self, config: Optional[ILPSolverConfig] = None):
        super().__init__(config or ILPSolverConfig())
        self.ampl: Optional[AMPL] = None
        self.debug_mode = getattr(self.config, 'debug_mode', False)
        self.use_lazy = getattr(self.config, 'use_lazy_constraints', True)
        self.use_preprocessing = getattr(self.config, 'use_preprocessing', True)
        self.preprocessing_stats = {}
        
    def _debug_print(self, message: str, level: str = "INFO"):
        """Print debug message if debug mode is enabled"""
        if self.debug_mode:
            print(f"[ILP DEBUG {level}] {message}")
            
    def _solve(self, puzzle: Puzzle) -> SolverResult:
        """Implement ILP solving using AMPL with optimizations"""
        try:
            self._debug_print("Starting Optimized ILP solver")
            
            # Apply preprocessing if enabled
            if self.use_preprocessing:
                start_time = time.time()
                preprocessed_puzzle, fixed_bridges = self._preprocess(puzzle)
                preprocess_time = time.time() - start_time
                self._debug_print(f"Preprocessing took {preprocess_time:.3f}s, fixed {len(fixed_bridges)} bridges")
            else:
                preprocessed_puzzle = puzzle
                fixed_bridges = []
                preprocess_time = 0
            
            # Initialize AMPL
            self.ampl = AMPL()
            self._debug_print("AMPL initialized successfully")
            
            # Solve with appropriate strategy
            if self.use_lazy:
                result = self._solve_with_lazy_constraints(preprocessed_puzzle)
            else:
                result = self._solve_with_full_connectivity(preprocessed_puzzle)
            
            # Add back fixed bridges if preprocessing was used
            if result.success and fixed_bridges:
                for (i, j, count) in fixed_bridges:
                    result.solution.add_bridge(i, j, count)
                    
            # Update stats
            if result.stats:
                result.stats['preprocess_time'] = preprocess_time
                result.stats['fixed_bridges'] = len(fixed_bridges)
                result.stats['preprocessing'] = self.preprocessing_stats
                
            return result
                
        except Exception as e:
            error_msg = f"ILP solver error: {str(e)}"
            self._debug_print(f"ERROR: {error_msg}", "ERROR")
            self._debug_print(f"Traceback:\n{traceback.format_exc()}", "ERROR")
            self.logger.error(error_msg)
            
            return SolverResult(
                success=False,
                message=error_msg
            )
        finally:
            if self.ampl:
                self.ampl.close()
                self._debug_print("AMPL closed")
                
    def _preprocess(self, puzzle: Puzzle) -> Tuple[Puzzle, List[Tuple[int, int, int]]]:
        """
        Apply preprocessing techniques to reduce problem size
        Returns: (preprocessed_puzzle, fixed_bridges)
        """
        preprocessed = puzzle.copy()
        fixed_bridges = []
        
        # Track what we do
        self.preprocessing_stats = {
            'forced_bridges': 0,
            'impossible_bridges_removed': 0,
            'isolated_pairs': 0,
        }
        
        changed = True
        iterations = 0
        
        while changed and iterations < 10:
            changed = False
            iterations += 1
            
            # 1. Force bridges for islands with only one valid neighbor
            for island in preprocessed.islands:
                valid_neighbors = [n for n in preprocessed._valid_connections[island.id]]
                
                if len(valid_neighbors) == 1 and island.required_bridges > 0:
                    neighbor_id = valid_neighbors[0]
                    neighbor = preprocessed._id_to_island[neighbor_id]
                    
                    # Determine bridge count
                    bridge_count = min(island.required_bridges, neighbor.required_bridges, 2)
                    
                    if bridge_count > 0:
                        fixed_bridges.append((island.id, neighbor_id, bridge_count))
                        island.required_bridges -= bridge_count
                        neighbor.required_bridges -= bridge_count
                        
                        # Remove connection
                        preprocessed._valid_connections[island.id].remove(neighbor_id)
                        preprocessed._valid_connections[neighbor_id].remove(island.id)
                        
                        self.preprocessing_stats['forced_bridges'] += 1
                        changed = True
            
            # 2. Remove impossible connections
            for island in preprocessed.islands:
                neighbors_to_remove = []
                
                for neighbor_id in preprocessed._valid_connections[island.id]:
                    neighbor = preprocessed._id_to_island[neighbor_id]
                    
                    # If adding even one bridge would exceed requirements
                    if island.required_bridges == 0 or neighbor.required_bridges == 0:
                        neighbors_to_remove.append(neighbor_id)
                        
                for neighbor_id in neighbors_to_remove:
                    preprocessed._valid_connections[island.id].discard(neighbor_id)
                    preprocessed._valid_connections[neighbor_id].discard(island.id)
                    self.preprocessing_stats['impossible_bridges_removed'] += 1
                    changed = True
                    
            # 3. Handle isolated pairs
            for island in preprocessed.islands:
                valid_neighbors = list(preprocessed._valid_connections[island.id])
                
                if len(valid_neighbors) == 1:
                    neighbor_id = valid_neighbors[0]
                    neighbor = preprocessed._id_to_island[neighbor_id]
                    neighbor_valid = list(preprocessed._valid_connections[neighbor_id])
                    
                    if len(neighbor_valid) == 1 and neighbor_valid[0] == island.id:
                        bridge_count = min(island.required_bridges, neighbor.required_bridges, 2)
                        
                        if bridge_count > 0:
                            fixed_bridges.append((island.id, neighbor_id, bridge_count))
                            island.required_bridges = 0
                            neighbor.required_bridges = 0
                            
                            preprocessed._valid_connections[island.id].clear()
                            preprocessed._valid_connections[neighbor_id].clear()
                            
                            self.preprocessing_stats['isolated_pairs'] += 1
                            changed = True
                            
        return preprocessed, fixed_bridges
                
    def _solve_with_lazy_constraints(self, puzzle: Puzzle) -> SolverResult:
        """Solve using lazy connectivity constraints (more efficient)"""
        self._debug_print("Using lazy constraint approach")
        
        # Create basic model without connectivity constraints
        self._create_basic_model(puzzle)
        self._add_data(puzzle)
        
        # For CBC, use iterative approach
        if self.config.solver_name == 'cbc':
            return self._solve_iterative_lazy(puzzle)
        else:
            # For other solvers that support callbacks
            self._debug_print("Note: Full callback support not implemented for non-CBC solvers")
            return self._solve_iterative_lazy(puzzle)
            
    def _solve_iterative_lazy(self, puzzle: Puzzle) -> SolverResult:
        """Iterative approach for lazy constraints (CBC compatible)"""
        self._debug_print("Using iterative lazy constraint approach")
        
        max_iterations = 50
        iteration = 0
        total_solve_time = 0
        
        while iteration < max_iterations:
            iteration += 1
            self._debug_print(f"\nIteration {iteration}")
            
            # Solve current model
            solve_start = time.time()
            self._set_solver_options()
            self.ampl.solve()
            iter_solve_time = time.time() - solve_start
            total_solve_time += iter_solve_time
            
            solve_result = self.ampl.get_value("solve_result")
            
            if "optimal" not in solve_result.lower() and "solved" not in solve_result.lower():
                return SolverResult(
                    success=False,
                    message=f"ILP failed at iteration {iteration}: {solve_result}",
                    stats={'iterations': iteration, 'total_time': total_solve_time}
                )
            
            # Check connectivity
            solution = self._extract_solution(puzzle)
            components = self._find_connected_components(solution)
            
            if len(components) == 1:
                # Solution is connected!
                self._debug_print(f"Connected solution found after {iteration} iterations")
                return SolverResult(
                    success=True,
                    solution=solution,
                    message=f"ILP solved with lazy constraints in {iteration} iterations",
                    stats={
                        'solver_time': total_solve_time,
                        'iterations': iteration,
                        'lazy_constraints': True,
                        'objective_value': self.ampl.get_objective('TotalBridges').value()
                    }
                )
            
            # Add cut constraints for each component
            self._debug_print(f"Found {len(components)} components, adding cuts")
            cuts_added = 0
            
            for component in components:
                cut_edges = self._find_cut_edges(puzzle, component)
                if cut_edges:
                    # Add constraint that at least one cut edge must be used
                    constraint_name = f"Cut_{iteration}_{cuts_added}"
                    constraint = f"subject to {constraint_name}: "
                    constraint += " + ".join([f"x[{e},1]" for e in cut_edges])
                    constraint += " >= 1;"
                    
                    self.ampl.eval(constraint)
                    cuts_added += 1
            
            self._debug_print(f"Added {cuts_added} cut constraints")
            
        return SolverResult(
            success=False,
            message=f"Failed to find connected solution after {max_iterations} iterations",
            stats={'iterations': max_iterations, 'total_time': total_solve_time}
        )
        
    def _solve_with_full_connectivity(self, puzzle: Puzzle) -> SolverResult:
        """Solve with full connectivity constraints (original approach)"""
        self._debug_print("Using full connectivity constraints")
        
        # Create and solve the model
        self._create_model(puzzle)
        self._add_data(puzzle)
        
        # Get solver name and options
        solver_name = self.config.solver_name
        solver_options = self.config.solver_options
        
        # Solve
        self.logger.info(f"Solving with {solver_name}")
        self._debug_print(f"Using solver: {solver_name}")
        self._debug_print(f"Solver options: {solver_options}")
        
        solve_start = time.time()
        
        # Set solver and options
        self._set_solver_options()
        
        self._debug_print("Starting solve...")
        self.ampl.solve()
        
        solve_time = time.time() - solve_start
        self._debug_print(f"Solve completed in {solve_time:.2f}s")
        
        # Extract solution
        solve_result = self.ampl.get_value("solve_result")
        solve_message = self.ampl.get_value("solve_message")
        
        self._debug_print(f"Solve result: {solve_result}")
        self._debug_print(f"Solve message: {solve_message}")
        
        if "optimal" in solve_result.lower() or "solved" in solve_result.lower():
            solution = self._extract_solution(puzzle)
            return SolverResult(
                success=True,
                solution=solution,
                message=f"ILP solved: {solve_message}",
                stats={
                    'solver_time': solve_time,
                    'objective_value': self.ampl.get_objective('TotalBridges').value(),
                    'solver_result': solve_result
                }
            )
        else:
            return SolverResult(
                success=False,
                message=f"ILP failed: {solve_message}",
                stats={'solver_result': solve_result}
            )
        
    def _create_basic_model(self, puzzle: Puzzle):
        """Create basic AMPL model without connectivity constraints"""
        self._debug_print("Creating basic AMPL model...")
        
        num_islands = len(puzzle.islands)
        self._debug_print(f"Puzzle has {num_islands} islands")
        
        # Basic sets
        model_part1 = """
        # Sets
        set ISLANDS ordered;
        set EDGES ordered;  # Edge IDs
        set EDGE_MAP within {EDGES, ISLANDS, ISLANDS};  # Maps edge ID to its endpoints
        
        # Parameters
        param required{ISLANDS} >= 1, <= 8, integer;
        param num_edges >= 1, integer;
        """
        self.ampl.eval(model_part1)
        
        # Variables - simplified from original
        model_vars = """
        # Variables
        # x[e,l] = 1 if at least l bridges are built on edge e
        var x{e in EDGES, l in 1..2} binary;
        """
        self.ampl.eval(model_vars)
        
        # Objective - optional but helps guide solver
        model_obj = """
        # Objective - minimize total bridges
        minimize TotalBridges: sum{e in EDGES} (x[e,1] + x[e,2]);
        """
        self.ampl.eval(model_obj)
        
        # Basic constraints only
        # Bridge order constraint
        constraint_bridge_order = """
        subject to BridgeOrder{e in EDGES}:
            x[e,2] <= x[e,1];
        """
        self.ampl.eval(constraint_bridge_order)
        
        # Bridge requirements for each island
        constraint_bridge_req = """
        subject to BridgeRequirement{i in ISLANDS}:
            sum{(e,a,b) in EDGE_MAP: a = i or b = i} (x[e,1] + x[e,2]) = required[i];
        """
        self.ampl.eval(constraint_bridge_req)
        
        self._debug_print("✓ Basic model created")
        
    def _find_connected_components(self, puzzle: Puzzle) -> List[Set[int]]:
        """Find all connected components in the current solution"""
        # Build adjacency list from bridges
        adj = {island.id: [] for island in puzzle.islands}
        
        for bridge in puzzle.bridges:
            if bridge.count > 0:
                adj[bridge.island1_id].append(bridge.island2_id)
                adj[bridge.island2_id].append(bridge.island1_id)
        
        # BFS to find components
        visited = set()
        components = []
        
        for island in puzzle.islands:
            if island.id not in visited:
                # Start new component
                component = set()
                queue = [island.id]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                        
                    visited.add(current)
                    component.add(current)
                    
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
        
    def _find_cut_edges(self, puzzle: Puzzle, component: Set[int]) -> List[int]:
        """Find edges that connect component to rest of graph"""
        cut_edges = []
        
        # Get edge map from AMPL
        edge_map = list(self.ampl.set['EDGE_MAP'].members())
        
        for (e, i, j) in edge_map:
            # Check if this edge crosses the cut
            if (i in component and j not in component) or \
               (j in component and i not in component):
                cut_edges.append(e)
        
        return cut_edges
        
    def _set_solver_options(self):
        """Set solver options"""
        solver_name = self.config.solver_name
        solver_options = self.config.solver_options
        
        self.ampl.option['solver'] = solver_name
        
        # Format solver options
        solver_option_parts = []
        for key, value in solver_options.items():
            if solver_name == 'cbc':
                if key == 'timelimit':
                    solver_option_parts.append(f'lim:time={int(value)}')
                elif key == 'ratioGap':
                    solver_option_parts.append(f'ratioGap={value}')
                elif key == 'cuts':
                    # CBC doesn't support 'cuts' option directly
                    self._debug_print(f"  Note: CBC doesn't support 'cuts' option, skipping")
                    continue
                elif key == 'heuristics':
                    # CBC doesn't support 'heuristics' option directly
                    self._debug_print(f"  Note: CBC doesn't support 'heuristics' option, skipping")
                    continue
                else:
                    solver_option_parts.append(f'{key}={value}')
            else:
                solver_option_parts.append(f'{key}={value}')
        
        solver_option_string = ' '.join(solver_option_parts)
        self.ampl.option[f'{solver_name}_options'] = solver_option_string
                
    def _create_model(self, puzzle: Puzzle):
        """Create AMPL model for Hashiwokakero using Julia-inspired formulation"""
        self._debug_print("Creating AMPL model...")
        
        num_islands = len(puzzle.islands)
        self._debug_print(f"Puzzle has {num_islands} islands")
        
        try:
            # Basic sets
            model_part1 = """
            # Sets
            set ISLANDS ordered;
            set EDGES ordered;  # Edge IDs
            set EDGE_MAP within {EDGES, ISLANDS, ISLANDS};  # Maps edge ID to its endpoints
            
            # Parameters
            param required{ISLANDS} >= 1, <= 8, integer;
            param num_edges >= 1, integer;
            """
            self._debug_print("Defining sets and parameters...")
            self.ampl.eval(model_part1)
            self._debug_print("✓ Sets and parameters defined")
            
            # Variables - inspired by Julia implementation
            model_vars = """
            # Variables
            # x[e,l] = 1 if at least l bridges are built on edge e
            var x{e in EDGES, l in 1..2} binary;
            
            # y[e,l] = 1 if edge e can be reached from source edge in <= l-1 steps
            var y{e in EDGES, l in 1..num_edges} binary;
            """
            self._debug_print("Defining variables...")
            self.ampl.eval(model_vars)
            self._debug_print("✓ Variables defined")
            
            # Objective - minimize total bridges (optional, but helps solver)
            model_obj = """
            # Objective
            minimize TotalBridges: sum{e in EDGES} (x[e,1] + x[e,2]);
            """
            self._debug_print("Defining objective...")
            self.ampl.eval(model_obj)
            self._debug_print("✓ Objective defined")
            
            # Constraints
            self._debug_print("Adding constraints...")
            
            # By definition x[e,2] implies x[e,1]
            constraint_bridge_order = """
            subject to BridgeOrder{e in EDGES}:
                x[e,2] <= x[e,1];
            """
            self.ampl.eval(constraint_bridge_order)
            self._debug_print("  ✓ Bridge order constraints added")
            
            # Bridge requirements for each island
            constraint_bridge_req = """
            subject to BridgeRequirement{i in ISLANDS}:
                sum{(e,a,b) in EDGE_MAP: a = i or b = i} (x[e,1] + x[e,2]) = required[i];
            """
            self.ampl.eval(constraint_bridge_req)
            self._debug_print("  ✓ Bridge requirement constraints added")
            
            # No crossing bridges - will be handled in data section
            
            # Connectivity constraints - inspired by Julia implementation
            # Exactly one source edge
            constraint_source = """
            subject to OneSource:
                sum{e in EDGES} y[e,1] = 1;
            """
            self.ampl.eval(constraint_source)
            self._debug_print("  ✓ Source constraint added")
            
            # Monotonicity: if reachable in l-1 steps, then also in l steps
            constraint_monotone = """
            subject to Monotonicity{e in EDGES, l in 1..num_edges-1}:
                y[e,l] <= y[e,l+1];
            """
            self.ampl.eval(constraint_monotone)
            self._debug_print("  ✓ Monotonicity constraints added")
            
            # Every used edge must be reachable from source
            constraint_reachable = """
            subject to AllReachable{e in EDGES}:
                x[e,1] = y[e,num_edges];
            """
            self.ampl.eval(constraint_reachable)
            self._debug_print("  ✓ Reachability constraints added")
            
            # If edge is reachable in l steps, it or a neighbor must be reachable in l-1 steps
            constraint_neighbor = """
            subject to NeighborReachable{e in EDGES, l in 1..num_edges-1}:
                y[e,l+1] <= y[e,l] + 
                sum{(f,a,b) in EDGE_MAP, (g,c,d) in EDGE_MAP: 
                    f != e and g = e and (a = c or a = d or b = c or b = d)} y[f,l];
            """
            self.ampl.eval(constraint_neighbor)
            self._debug_print("  ✓ Neighbor reachability constraints added")
            
            self._debug_print("✓ Model creation completed successfully")
            
        except Exception as e:
            self._debug_print(f"ERROR in model creation: {str(e)}", "ERROR")
            self._debug_print(f"Model creation failed", "ERROR")
            raise
        
    def _add_data(self, puzzle: Puzzle):
        """Add puzzle data to AMPL model"""
        self._debug_print("Adding data to AMPL model...")
        
        try:
            # Create island set
            islands = list(range(len(puzzle.islands)))
            self._debug_print(f"Setting ISLANDS = {islands}")
            self.ampl.set['ISLANDS'] = islands
            self._debug_print("✓ ISLANDS set created")
            
            # Set island parameters
            self._debug_print("Setting island parameters...")
            required = self.ampl.param['required']
            for island in puzzle.islands:
                required[island.id] = island.required_bridges
                self._debug_print(f"  Island {island.id}: required={island.required_bridges}")
            self._debug_print("✓ Island parameters set")
            
            # Create edges and edge map (similar to Julia implementation)
            edges = []
            edge_map = []
            edge_id = 0
            edge_to_id = {}  # (i,j) -> edge_id mapping for crossing detection
            
            self._debug_print("Creating edges...")
            for i, island1 in enumerate(puzzle.islands):
                for j, island2 in enumerate(puzzle.islands):
                    if i < j and island2.id in puzzle._valid_connections[island1.id]:
                        edges.append(edge_id)
                        edge_map.append((edge_id, island1.id, island2.id))
                        edge_to_id[(island1.id, island2.id)] = edge_id
                        self._debug_print(f"  Edge {edge_id}: ({island1.id},{island2.id})")
                        edge_id += 1
                        
            self._debug_print(f"Setting EDGES with {len(edges)} edges")
            self.ampl.set['EDGES'] = edges
            self.ampl.set['EDGE_MAP'] = edge_map
            self.ampl.param['num_edges'] = len(edges)
            self._debug_print("✓ Edges and edge map created")
            
            # Add crossing constraints
            self._debug_print("Adding crossing constraints...")
            crossing_count = 0
            
            for idx1, (e1, i1, j1) in enumerate(edge_map):
                island1_i = puzzle._id_to_island[i1]
                island1_j = puzzle._id_to_island[j1]
                
                for idx2, (e2, i2, j2) in enumerate(edge_map):
                    if idx1 < idx2:  # Avoid duplicates
                        island2_i = puzzle._id_to_island[i2]
                        island2_j = puzzle._id_to_island[j2]
                        
                        # Check if bridges would cross
                        if self._bridges_would_cross(island1_i, island1_j, island2_i, island2_j):
                            # Add constraint: x[e1,1] + x[e2,1] <= 1
                            constraint = f"subject to NoCross_{e1}_{e2}: x[{e1},1] + x[{e2},1] <= 1;"
                            self.ampl.eval(constraint)
                            crossing_count += 1
                            self._debug_print(f"  Crossing constraint added for edges {e1} and {e2}")
            
            self._debug_print(f"✓ Added {crossing_count} crossing constraints")
            self._debug_print("✓ Data loading completed successfully")
            
        except Exception as e:
            self._debug_print(f"ERROR in data loading: {str(e)}", "ERROR")
            self._debug_print(f"Data loading failed", "ERROR")
            raise
        
    def _bridges_would_cross(self, island1_i, island1_j, island2_i, island2_j) -> bool:
        """Check if two bridges would cross"""
        # Bridge 1 coordinates
        r1_1, c1_1 = island1_i.row, island1_i.col
        r1_2, c1_2 = island1_j.row, island1_j.col
        
        # Bridge 2 coordinates
        r2_1, c2_1 = island2_i.row, island2_i.col
        r2_2, c2_2 = island2_j.row, island2_j.col
        
        # Check if one is horizontal and one is vertical
        bridge1_horizontal = (r1_1 == r1_2)
        bridge2_horizontal = (r2_1 == r2_2)
        
        if bridge1_horizontal == bridge2_horizontal:
            # Both horizontal or both vertical - they don't cross
            return False
        
        if bridge1_horizontal:
            # Bridge 1 is horizontal, bridge 2 is vertical
            h_row = r1_1
            h_col_min = min(c1_1, c1_2)
            h_col_max = max(c1_1, c1_2)
            
            v_col = c2_1
            v_row_min = min(r2_1, r2_2)
            v_row_max = max(r2_1, r2_2)
            
            # Check if they actually cross (not just at endpoints)
            crosses = (h_col_min < v_col < h_col_max) and (v_row_min < h_row < v_row_max)
            return crosses
        else:
            # Bridge 1 is vertical, bridge 2 is horizontal
            v_col = c1_1
            v_row_min = min(r1_1, r1_2)
            v_row_max = max(r1_1, r1_2)
            
            h_row = r2_1
            h_col_min = min(c2_1, c2_2)
            h_col_max = max(c2_1, c2_2)
            
            # Check if they actually cross (not just at endpoints)
            crosses = (h_col_min < v_col < h_col_max) and (v_row_min < h_row < v_row_max)
            return crosses
        
    def _extract_solution(self, puzzle: Puzzle) -> Puzzle:
        """Extract solution from AMPL model"""
        self._debug_print("Extracting solution from AMPL...")
        
        solution = puzzle.copy()
        solution.bridges = []
        
        try:
            x = self.ampl.var['x']
            edge_map = list(self.ampl.set['EDGE_MAP'].members())
            bridge_count = 0
            
            for (e, i, j) in edge_map:
                # Calculate number of bridges
                num_bridges = 0
                if x[e, 1].value() > 0.5:
                    num_bridges = 1
                    if x[e, 2].value() > 0.5:
                        num_bridges = 2
                
                if num_bridges > 0:
                    solution.add_bridge(i, j, num_bridges)
                    bridge_count += 1
                    self._debug_print(f"  Bridge ({i},{j}): {num_bridges}")
                    
            self._debug_print(f"✓ Extracted {bridge_count} bridges")
            
        except Exception as e:
            self._debug_print(f"ERROR extracting solution: {str(e)}", "ERROR")
            raise
            
        return solution
        
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about the ILP model"""
        if not self.ampl:
            return {}
            
        try:
            solver_name = 'cbc'
            if isinstance(self.config, ILPSolverConfig):
                solver_name = self.config.solver_name
                
            stats = {
                'num_variables': self.ampl.get_value('_nvars'),
                'num_constraints': self.ampl.get_value('_ncons'),
                'num_integers': self.ampl.get_value('_nivars'),
                'solver': solver_name
            }
            
            self._debug_print(f"Model statistics: {stats}")
            return stats
            
        except Exception as e:
            self._debug_print(f"ERROR getting model statistics: {str(e)}", "ERROR")
            return {'solver': solver_name, 'error': str(e)}
        
    def debug_ampl_state(self):
        """Print current AMPL state for debugging"""
        if not self.ampl:
            self._debug_print("AMPL not initialized", "WARNING")
            return
            
        try:
            print("\n" + "="*60)
            print("AMPL STATE DEBUG INFORMATION")
            print("="*60)
            
            # Sets
            print("\nSETS:")
            for s_name in ['ISLANDS', 'EDGES', 'EDGE_MAP']:
                try:
                    s = self.ampl.set[s_name]
                    print(f"  {s_name}: size = {s.size()}")
                    if s.size() < 20 and s_name != 'EDGE_MAP':
                        members = list(s.members())
                        print(f"    Members: {members}")
                except:
                    pass
                    
            # Parameters
            print("\nPARAMETERS:")
            for p_name in ['required', 'num_edges']:
                try:
                    p = self.ampl.param[p_name]
                    print(f"  {p_name}")
                except:
                    pass
                
            # Variables
            print("\nVARIABLES:")
            for v_name in ['x', 'y']:
                try:
                    v = self.ampl.var[v_name]
                    print(f"  {v_name}")
                except:
                    pass
                
            # Constraints
            print("\nCONSTRAINTS (types):")
            constraint_types = ['BridgeOrder', 'BridgeRequirement', 'OneSource', 
                              'Monotonicity', 'AllReachable', 'NeighborReachable']
            for c_type in constraint_types:
                try:
                    # Check if any constraint of this type exists
                    print(f"  {c_type}")
                except:
                    pass
                
            # Objectives
            print("\nOBJECTIVES:")
            try:
                o = self.ampl.objective['TotalBridges']
                print(f"  TotalBridges")
            except:
                pass
                
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"ERROR in debug_ampl_state: {str(e)}")


class WarmStartILPSolver(ILPSolver):
    """
    ILP solver with warm start capability.
    Can use an initial solution to speed up solving.
    """
    
    def __init__(self, config: Optional[ILPSolverConfig] = None,
                 initial_solution: Optional[Puzzle] = None):
        super().__init__(config)
        self.initial_solution = initial_solution
        
    def _add_warm_start(self, puzzle: Puzzle):
        """Add warm start solution to AMPL model"""
        if not self.initial_solution:
            return
            
        self._debug_print("Adding warm start solution...")
        
        x = self.ampl.var['x']
        edge_map = list(self.ampl.set['EDGE_MAP'].members())
        warm_count = 0
        
        # Create mapping from (i,j) to edge_id
        pair_to_edge = {}
        for (e, i, j) in edge_map:
            pair_to_edge[(i, j)] = e
            pair_to_edge[(j, i)] = e  # Both directions
        
        # Set initial values for bridge variables
        for bridge in self.initial_solution.bridges:
            i, j = bridge.island1_id, bridge.island2_id
            
            if (i, j) in pair_to_edge:
                e = pair_to_edge[(i, j)]
                
                try:
                    # Set x[e,1] = 1 if at least 1 bridge
                    if bridge.count >= 1:
                        x[e, 1].set_value(1)
                    # Set x[e,2] = 1 if 2 bridges
                    if bridge.count >= 2:
                        x[e, 2].set_value(1)
                    warm_count += 1
                    self._debug_print(f"  Set warm start: edge {e} ({i},{j}) = {bridge.count} bridges")
                except Exception as e:
                    self._debug_print(f"  WARNING: Could not set warm start for edge {e}: {e}", "WARNING")
                    
        self._debug_print(f"✓ Set {warm_count} warm start values")
            
    def _solve(self, puzzle: Puzzle) -> SolverResult:
        """Solve with warm start"""
        try:
            # Apply preprocessing if enabled
            if self.use_preprocessing:
                start_time = time.time()
                preprocessed_puzzle, fixed_bridges = self._preprocess(puzzle)
                preprocess_time = time.time() - start_time
                self._debug_print(f"Preprocessing took {preprocess_time:.3f}s")
            else:
                preprocessed_puzzle = puzzle
                fixed_bridges = []
                preprocess_time = 0
            
            # Initialize AMPL
            self.ampl = AMPL()
            self._debug_print("AMPL initialized for warm start")
            
            # Create model and add data
            if self.use_lazy:
                self._create_basic_model(preprocessed_puzzle)
            else:
                self._create_model(preprocessed_puzzle)
            self._add_data(preprocessed_puzzle)
            
            # Add warm start if available
            if self.initial_solution:
                self._add_warm_start(preprocessed_puzzle)
                
            # Solve
            if self.use_lazy:
                result = self._solve_iterative_lazy(preprocessed_puzzle)
            else:
                result = self._solve_with_full_connectivity(preprocessed_puzzle)
                
            # Add back fixed bridges
            if result.success and fixed_bridges:
                for (i, j, count) in fixed_bridges:
                    result.solution.add_bridge(i, j, count)
                    
            # Update stats
            if result.stats:
                result.stats['warm_start_used'] = self.initial_solution is not None
                result.stats['preprocess_time'] = preprocess_time
                result.stats['fixed_bridges'] = len(fixed_bridges)
                
            return result
                
        except Exception as e:
            error_msg = f"Warm start ILP solver error: {str(e)}"
            self._debug_print(f"ERROR: {error_msg}", "ERROR")
            self._debug_print(f"Traceback:\n{traceback.format_exc()}", "ERROR")
            self.logger.error(error_msg)
            
            return SolverResult(
                success=False,
                message=error_msg
            )
        finally:
            if self.ampl:
                self.ampl.close()
                self._debug_print("AMPL closed")