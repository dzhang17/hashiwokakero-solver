import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
PUZZLES_DIR = DATA_DIR / "puzzles"
SOLUTIONS_DIR = DATA_DIR / "solutions"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_SOLUTIONS_DIR = RESULTS_DIR / "solutions"
RESULTS_VIZ_DIR = RESULTS_DIR / "visualizations"
RESULTS_REPORTS_DIR = RESULTS_DIR / "reports"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"

# Create directories if they don't exist
for dir_path in [PUZZLES_DIR, SOLUTIONS_DIR, BENCHMARKS_DIR,
                  RESULTS_SOLUTIONS_DIR, RESULTS_VIZ_DIR, 
                  RESULTS_REPORTS_DIR, RESULTS_LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Algorithm parameters
ILP_TIME_LIMIT = 300  # seconds
SA_MAX_ITERATIONS = 10000
SA_INITIAL_TEMP = 100
SA_COOLING_RATE = 0.95

# Visualization settings
VIZ_DPI = 300
VIZ_FIGSIZE = (10, 10)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"