import os
import json
import time
from typing import Dict, Any

class ExperimentTracker:
    """Manages experiment parameter registries and run tracking logs for indexing runs."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.exp_dir = os.path.join(output_dir, "experiments")
        os.makedirs(self.exp_dir, exist_ok=True)
        self.registry_path = os.path.join(output_dir, "experiment_registry.json")

    def log_run(self, config_dict: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Logs an indexing experiment and appends details to the central registry.
        
        Returns:
            str: Unique experiment ID.
        """
        timestamp = int(time.time())
        exp_id = f"exp_{config_dict.get('document_version', 'v2')}_{config_dict.get('backend', 'st')}_{timestamp}"
        
        run_data = {
            "experiment_id": exp_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": config_dict,
            "metrics": metrics
        }
        
        # Save individual experiment log
        exp_file_path = os.path.join(self.exp_dir, f"{exp_id}.json")
        with open(exp_file_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)

        # Update central registry
        registry_data = []
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    registry_data = json.load(f)
                    if not isinstance(registry_data, list):
                        registry_data = []
            except (json.JSONDecodeError, IOError):
                pass
                
        registry_data.append({
            "experiment_id": exp_id,
            "timestamp": run_data["timestamp"],
            "model_name": config_dict.get("model_name"),
            "backend": config_dict.get("backend"),
            "document_version": config_dict.get("document_version"),
            "throughput_cps": metrics.get("throughput_docs_per_sec", 0.0),
            "elapsed_seconds": metrics.get("elapsed_time_seconds", 0.0)
        })
        
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)
            
        return exp_id
