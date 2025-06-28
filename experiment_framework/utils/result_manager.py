# utils/result_manager.py
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import pickle

class ResultManager:
    @staticmethod
    def save_results(results: List[Dict], output_dir: Path, prefix: str = ""):
        """Save results in multiple formats"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_dir / f"{prefix}results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as CSV
        df = pd.DataFrame(results)
        csv_path = output_dir / f"{prefix}results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as pickle for complex objects
        pickle_path = output_dir / f"{prefix}results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        return {
            'json': json_path,
            'csv': csv_path,
            'pickle': pickle_path
        }
    
    @staticmethod
    def load_results(file_path: Path) -> List[Dict]:
        """Load results from file"""
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def merge_results(result_files: List[Path]) -> List[Dict]:
        """Merge results from multiple files"""
        
        all_results = []
        
        for file_path in result_files:
            results = ResultManager.load_results(file_path)
            all_results.extend(results)
        
        return all_results