# Hashiwokakero Solver

# Hashiwokakero Solver

A comprehensive implementation of Integer Linear Programming (ILP) and Large Neighborhood Search (LNS) algorithms for solving Hashiwokakero (Bridges) puzzles. This project includes state-of-the-art optimization techniques, visualization tools, and extensive benchmarking capabilities.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)![License](https://img.shields.io/badge/license-MIT-green.svg)![Status](https://img.shields.io/badge/status-active-success.svg)

## 🎯 Overview

Hashiwokakero (橋をかけろ, "build bridges") is a logic puzzle where islands must be connected by bridges following specific rules:

* Each island shows the number of bridges that must connect to it
* Bridges run horizontally or vertically
* Bridges cannot cross each other
* At most two bridges can connect any pair of islands
* All islands must form a single connected network

This project provides two sophisticated solving approaches:

* **ILP Solver**: Exact method using Integer Linear Programming with lazy constraint generation
* **LNS Solver**: Metaheuristic approach with adaptive destruction operators and parallel repair strategies

## 🚀 Features

### Solvers

* **Integer Linear Programming (ILP)**
  * Lazy constraint generation for improved efficiency
  * Advanced preprocessing techniques
  * Warm-start capabilities
  * Compatible with CBC, Gurobi, and other AMPL-supported solvers
* **Large Neighborhood Search (LNS)**
  * 7 adaptive destruction operators
  * Parallel repair strategies
  * Simulated annealing acceptance criterion
  * Solution caching and performance optimizations

### Visualization & Analysis

* Interactive puzzle visualization with solution display
* Side-by-side solver comparison
* Performance statistics and validation
* Comprehensive benchmarking suite
* Automated figure generation for research papers

## 📋 Requirements

```bash
amplpy>=0.13.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.0
openpyxl>=3.1.0
tqdm>=4.65.0
click>=8.1.0
pytest>=7.3.0
pytest-cov>=4.0.0
psutil>=5.9.0
scikit-learn>=1.3.0
pyyaml>=6.0.0
statsmodels>=0.14.0
```

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hashiwokakero_solver.git
cd hashiwokakero_solver
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install and activate AMPL (required for ILP solver):

   * If you have an AMPL license:
     ```bash
     python -m amplpy.modules activate [your-license-key]
     ```
   * For evaluation, AMPL provides a size-limited demo version

   Note: The ILP solver requires AMPL to be properly activated. Without activation, you can still use the LNS solver.

## 🎮 Quick Start

### Visualizing and Solving Puzzles

The `visualize_solver.py` script provides an interactive way to solve and visualize puzzles:

```bash
# Basic usage - solves with ILP by default
python visualize_solver.py dataset/100/Hs_16_100_25_00_001.has

# Use LNS solver
python visualize_solver.py dataset/200/Hs_24_200_50_05_010.has --solver lns

# Compare both solvers
python visualize_solver.py dataset/400/Hs_34_400_75_15_020.has --solver both

# Set custom time limit and save images
python visualize_solver.py dataset/300/Hs_29_300_50_10_015.has --time-limit 600 --save
```

#### Command Line Options:

* `--solver {ilp|lns|both}`: Choose which solver to use (default: ilp)
* `--time-limit TIME`: Set time limit in seconds (default: 300)
* `--save`: Save visualizations as PNG files
* `--no-grid`: Hide grid lines in visualization

### Running Experiments

Generate comprehensive benchmark results and thesis figures:

```bash
python thesis_figures_generator.py
```

This will:

* Run both solvers on all 1,440 test instances
* Generate 8 publication-ready figures
* Create LaTeX tables for academic papers
* Save detailed results in CSV format

## 📊 Dataset Structure

The project includes 1,440 benchmark instances organized by size:

```
dataset/
├── 100/    # 360 puzzles with 100 islands
├── 200/    # 360 puzzles with 200 islands
├── 300/    # 360 puzzles with 300 islands
└── 400/    # 360 puzzles with 400 islands
```

Each file follows the naming convention: `Hs_[grid]_[islands]_[density]_[difficulty]_[id].has`

## 🔬 Research Results

Our extensive experiments on 1,440 instances show:

* **Small instances (100-200 islands)**: Both algorithms achieve >94% success rate with similar performance
* **Large instances (300-400 islands)**: LNS shows superior scalability with up to 37% faster solving times
* **Success vs Speed trade-off**: ILP maintains higher success rates (90.3% vs 86.1% on 400-island puzzles) while LNS offers better speed

### Performance Summary


| Size | ILP Success | LNS Success | ILP Avg Time | LNS Avg Time | Speedup |
| ---- | ----------- | ----------- | ------------ | ------------ | ------- |
| 100  | 94.4%       | 94.4%       | 1.37s        | 1.36s        | 1.01x   |
| 200  | 100.0%      | 100.0%      | 4.23s        | 4.25s        | 1.00x   |
| 300  | 96.1%       | 95.3%       | 15.06s       | 13.90s       | 1.08x   |
| 400  | 90.3%       | 86.1%       | 33.60s       | 21.10s       | 1.59x   |

## 📁 Project Structure

```
hashiwokakero_solver/
├── src/
│   ├── core/              # Core puzzle logic
│   │   ├── puzzle.py      # Puzzle data structure
│   │   ├── validator.py   # Solution validation
│   │   └── utils.py       # Utility functions
│   ├── solvers/           # ILP and LNS implementations
│   │   ├── base_solver.py # Base solver interface
│   │   ├── ilp_solver.py  # Integer Linear Programming
│   │   └── lns_solver.py  # Large Neighborhood Search
│   ├── visualization/     # Visualization utilities
│   │   ├── static_viz.py  # Static visualizations
│   │   └── performance_viz.py # Performance plots
│   └── analysis/          # Benchmarking tools
│       └── benchmark.py   # Benchmark runner
├── dataset/               # Test instances (1,440 puzzles)
├── experiment_framework/  # Experiment automation
├── thesis_figures/        # Generated figures
├── visualize_solver.py    # Interactive solver/visualizer
├── thesis_figures_generator.py  # Benchmark runner
└── requirements.txt       # Python dependencies
```

## 🧪 Example Usage

### Solving a Small Puzzle

```python
from src.core.puzzle import Puzzle
from src.solvers.ilp_solver import ILPSolver, ILPSolverConfig

# Create a puzzle
puzzle = Puzzle(10, 10)
puzzle.add_island(0, 0, 3)
puzzle.add_island(0, 5, 2)
puzzle.add_island(5, 0, 4)
puzzle.add_island(5, 5, 3)

# Configure and run solver
config = ILPSolverConfig(time_limit=60, use_lazy_constraints=True)
solver = ILPSolver(config)
result = solver.solve(puzzle)

if result.success:
    print("Solution found!")
    print(f"Bridges: {len(result.solution.bridges)}")
```

### Running Benchmarks

```python
from thesis_figures_generator import run_complete_benchmark, generate_all_figures

# Run benchmarks
results_df = run_complete_benchmark()

# Generate figures
generate_all_figures(results_df)
```

```## 🙏 Acknowledgments

* Puzzle instances adapted from various Hashiwokakero collections
* ILP formulation inspired by constraint programming literature
* LNS framework based on Ropke & Pisinger's adaptive large neighborhood search
* CBC solver from COIN-OR project
* AMPL modeling system for mathematical optimization

## 🐛 Known Issues

* ILP solver may require significant memory for very large instances (>500 islands)
* Visualization may be slow for puzzles with many bridges
* Some solver options are CBC-specific and may not work with other AMPL solvers

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact [your-email@example.com].

# Hashiwokakero Solver

A professional-grade solver for Hashiwokakero (Bridges) puzzles using multiple algorithms including Integer Linear Programming (ILP) and Large Neighborhood Search (LNS).

## Features

* **Multiple Solving Algorithms**
  * Integer Linear Programming (ILP) with CBC solver
  * Large Neighborhood Search (LNS) with adaptive strategies
  * Optimized with lazy constraints and preprocessing
* **Puzzle Generator** with configurable difficulty levels
* **Comprehensive Benchmarking System** for algorithm comparison
* **Professional Visualization** with color-coded solutions
* **Detailed Performance Reports** with statistical analysis
* **Command-line Interface** for easy puzzle solving

## Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd hashiwokakero_solver
   ```
2. Create virtual environment
   ```bash
   python -m venv menv
   ```
3. Activate environment
   * Windows: `menv\Scripts\activate`
   * Linux/Mac: `source menv/bin/activate`
4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
5. Install AMPL solvers
   ```bash
   python -m amplpy.modules install cbc highs
   ```

## Quick Start

### Solving and Visualizing Puzzles

The `visualize_solver.py` script provides an easy way to solve and visualize Hashiwokakero puzzles:

#### Basic Usage

```bash
# Solve with ILP (default)
python visualize_solver.py dataset/100/Hs_16_100_25_00_001.has

# Use LNS solver
python visualize_solver.py dataset/200/Hs_24_200_50_05_010.has --solver lns

# Compare both solvers
python visualize_solver.py dataset/400/Hs_34_400_75_15_020.has --solver both

# Set custom time limit and save images
python visualize_solver.py dataset/300/Hs_29_300_50_10_015.has --time-limit 600 --save
```
#### Command Line Options

* `puzzle_file`: Path to the puzzle file (.has format)
* `--solver {ilp|lns|both}`: Choose solver (default: ilp)
* `--time-limit TIME`: Set time limit in seconds (default: 300)
* `--save`: Save visualizations as PNG files
* `--no-grid`: Hide grid lines in visualization

#### Examples

```bash
# Quick solve of a small puzzle
python visualize_solver.py dataset/100/Hs_16_100_25_00_001.has

# Test LNS on a large puzzle with extended time limit
python visualize_solver.py dataset/400/Hs_34_400_75_15_030.has --solver lns --time-limit 600

# Compare both solvers and save all visualizations
python visualize_solver.py dataset/200/Hs_24_200_50_05_010.has --solver both --save

# Clean visualization without grid lines
python visualize_solver.py dataset/300/Hs_29_300_25_00_015.has --no-grid --save
```
### Running Benchmarks

To run performance benchmarks on the dataset:

```bash
# Run complete benchmark suite
python benchmark_script.py

# Run specific experiment
python -m experiment_framework.main --experiment algorithm_comparison
```
### Generating Puzzles

To generate new Hashiwokakero puzzles:

```bash
# Generate puzzles with default settings
python scripts/generate_puzzles.py

# Generate with specific parameters
python scripts/generate_puzzles.py --size 20x20 --islands 50 --difficulty hard
```
## Dataset Structure

The dataset is organized by puzzle size:

```
dataset/
├── 100/  # Puzzles with ~100 islands
├── 200/  # Puzzles with ~200 islands
├── 300/  # Puzzles with ~300 islands
└── 400/  # Puzzles with ~400 islands
```
Each puzzle file follows the naming convention: `Hs_[grid_size]_[islands]_[density]_[param]_[id].has`

## Visualization Features

The solver provides rich visualizations including:

* **Color-coded islands**: Green for satisfied, red for unsatisfied
* **Bridge visualization**: Single bridges in light blue, double bridges in dark blue
* **Statistics panel**: Solver performance metrics and solution quality
* **Validation status**: Automatic checking of solution correctness

## Algorithm Details

### ILP Solver

* Uses AMPL with CBC solver
* Implements lazy constraint generation for improved performance
* Includes preprocessing to reduce problem size
* Supports warm start capabilities

### LNS Solver

* Adaptive destroy and repair operators
* Multiple destroy strategies (random, worst-connected, geographical, etc.)
* Parallel repair with ILP sub-problems
* Simulated annealing acceptance criterion

## Performance

Based on benchmarks with 1440 test instances:


| Size | ILP Success Rate | LNS Success Rate | ILP Avg Time | LNS Avg Time |
| ---- | ---------------- | ---------------- | ------------ | ------------ |
| 100  | 94.4%            | 94.4%            | 1.37s        | 1.36s        |
| 200  | 100.0%           | 100.0%           | 4.23s        | 4.25s        |
| 300  | 96.1%            | 95.3%            | 15.06s       | 13.90s       |
| 400  | 90.3%            | 86.1%            | 33.60s       | 21.10s       |

## Project Structure

```
hashiwokakero_solver/
├── src/
│   ├── core/           # Core puzzle logic
│   ├── solvers/        # ILP and LNS implementations
│   ├── visualization/  # Visualization tools
│   └── analysis/       # Benchmarking and analysis
├── dataset/            # Puzzle instances
├── experiment_framework/  # Experiment runner
├── scripts/            # Utility scripts
├── visualize_solver.py # Main visualization script
└── requirements.txt    # Python dependencies
```
## Contributing

Contributions are welcome! Please ensure:

* Code follows PEP 8 style guidelines
* All tests pass
* New features include appropriate documentation
* Performance benchmarks are maintained

```## Acknowledgments

* Dataset from [PhoenixSmaug](https://github.com/PhoenixSmaug/hashi/commits?author=PhoenixSmaug)
* AMPL and CBC solver teams
* Prof. Mitchell offering me this chance

A professional-grade solver for Hashiwokakero (Bridges) puzzles using multiple algorithms.

## Features

- Multiple solving algorithms (ILP, Simulated Annealing, Hybrid)
- Puzzle generator with difficulty levels
- Comprehensive benchmarking system
- Professional visualization
- Detailed performance reports

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv menv`
3. Activate environment: `menv\Scripts\activate` (Windows) or `source menv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`
5. Install AMPL solvers: `python -m amplpy.modules install cbc highs`
6. activate using `python -m amplpy.modules activate [your ampl license key]`

## Usage

See documentation for detailed usage instructions.
