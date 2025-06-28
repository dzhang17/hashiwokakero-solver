#!/usr/bin/env python3
"""
Test script to verify the experiment framework setup
"""
import sys
from pathlib import Path

print("Testing Hashiwokakero Experiment Framework Setup")
print("=" * 50)

# Check Python version
print(f"Python version: {sys.version}")

# Check current directory
print(f"Current directory: {Path.cwd()}")

# Check if required directories exist
required_dirs = [
    'experiment_framework',
    'experiment_framework/config',
    'experiment_framework/experiments',
    'experiment_framework/analysis',
    'experiment_framework/utils',
    'experiment_framework/visualization',
    'src',
    'src/core',
    'src/solvers',
    'dataset',
    'dataset/100'
]

print("\nChecking directories:")
for dir_path in required_dirs:
    path = Path(dir_path)
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists} {dir_path}")

# Check if config files exist
print("\nChecking config files:")
config_files = [
    'experiment_framework/config/experiment_config.yaml',
    'experiment_framework/config/solver_configs.yaml'
]

for config_file in config_files:
    path = Path(config_file)
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists} {config_file}")

# Check if we can import required modules
print("\nChecking imports:")
try:
    sys.path.insert(0, str(Path.cwd()))
    from src.core.puzzle import Puzzle
    print("  ✓ Can import Puzzle")
    
    # Test load_from_has method
    if hasattr(Puzzle, 'load_from_has'):
        print("  ✓ Puzzle has load_from_has method")
    else:
        print("  ✗ Puzzle missing load_from_has method")
        
except ImportError as e:
    print(f"  ✗ Cannot import Puzzle: {e}")

# Check sample dataset file
print("\nChecking sample dataset:")
sample_file = Path("dataset/100/Hs_16_100_25_00_001.has")
if sample_file.exists():
    print(f"  ✓ Found sample file: {sample_file}")
    # Try to read first few lines
    with open(sample_file, 'r') as f:
        lines = f.readlines()[:5]
    print("  First few lines:")
    for i, line in enumerate(lines):
        print(f"    Line {i}: {line.strip()}")
else:
    print(f"  ✗ Sample file not found: {sample_file}")

print("\n" + "=" * 50)
print("Setup test complete!")

# Try to load a puzzle
if sample_file.exists():
    print("\nTrying to load a puzzle...")
    try:
        from src.core.puzzle import Puzzle
        puzzle = Puzzle.load_from_has(sample_file)
        print(f"  ✓ Successfully loaded puzzle: {puzzle}")
    except Exception as e:
        print(f"  ✗ Failed to load puzzle: {e}")
        import traceback
        traceback.print_exc()