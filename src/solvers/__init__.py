"""
Solvers for Hashiwokakero puzzles.
"""

from .base_solver import BaseSolver, SolverConfig, SolverResult, GreedySolver
from .ilp_solver import ILPSolver, ILPSolverConfig, WarmStartILPSolver
from .lns_solver import LargeNeighborhoodSearchSolver, LNSSolver, LNSSolverConfig, DestroyMethod

__all__ = [
    # Base classes
    'BaseSolver',
    'SolverConfig',
    'SolverResult',
    
    # Simple solver
    'GreedySolver',
    
    # ILP solvers
    'ILPSolver',
    'ILPSolverConfig',
    'WarmStartILPSolver',
    
    # LNS solver
    'LargeNeighborhoodSearchSolver',
    'LNSSolver',
    'LNSSolverConfig',
    'DestroyMethod',
]


# Solver registry for easy access
SOLVER_REGISTRY = {
    'greedy': GreedySolver,
    'ilp': ILPSolver,
    'lns': LargeNeighborhoodSearchSolver,
}

# Config registry for each solver type
CONFIG_REGISTRY = {
    'greedy': SolverConfig,
    'ilp': ILPSolverConfig,
    'lns': LNSSolverConfig,
}


def get_solver(name: str, config: SolverConfig = None) -> BaseSolver:
    """
    Get a solver by name.
    
    Args:
        name: Solver name (greedy, ilp, lns)
        config: Optional solver configuration. If None, will create appropriate default config.
        
    Returns:
        Solver instance
        
    Raises:
        ValueError: If solver name is not recognized
    """
    solver_class = SOLVER_REGISTRY.get(name.lower())
    if not solver_class:
        raise ValueError(f"Unknown solver: {name}. Available: {list(SOLVER_REGISTRY.keys())}")
    
    # If no config provided, create appropriate default config
    if config is None:
        config_class = CONFIG_REGISTRY.get(name.lower(), SolverConfig)
        config = config_class()
    else:
        # If provided config is base SolverConfig but solver needs specific config,
        # create specific config with base config parameters
        expected_config_class = CONFIG_REGISTRY.get(name.lower(), SolverConfig)
        if expected_config_class != SolverConfig and type(config) == SolverConfig:
            # Transfer base config parameters to specific config
            new_config = expected_config_class(
                time_limit=config.time_limit,
                max_iterations=config.max_iterations,
                verbose=config.verbose,
                log_file=config.log_file,
                save_intermediate=config.save_intermediate,
                intermediate_dir=config.intermediate_dir,
                random_seed=config.random_seed,
                extra_params=config.extra_params
            )
            config = new_config
        
    return solver_class(config)