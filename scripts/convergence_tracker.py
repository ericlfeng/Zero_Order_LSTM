#!/usr/bin/env python3
"""
Convergence Tracker for Cross-Run Early Stopping

This module provides functionality to track convergence across multiple runs
with the same (model_scale, num_perturbations) configuration. When one run
converges, it records the iteration count, allowing other runs to stop early
if they exceed that iteration count without converging.

The tracker uses a file-based approach with file locking for safe concurrent access.
"""

import json
import os
import time
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple


class ConvergenceTracker:
    """
    Manages convergence tracking across multiple runs.
    
    Uses a JSON file to store the minimum convergence iteration for each
    (model_scale, num_perturbations) pair.
    """
    
    def __init__(self, tracker_file: str):
        """
        Args:
            tracker_file: Path to the JSON file storing convergence data
        """
        self.tracker_file = Path(tracker_file)
        self._ensure_tracker_exists()
        
    def _ensure_tracker_exists(self):
        """Create the tracker file if it doesn't exist."""
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.tracker_file.exists():
            self._write_data({})
    
    def _get_key(self, solver: str, model_scale: int, num_perturbations: int) -> str:
        """Generate a unique key for the (solver, model_scale, num_perturbations) combination."""
        return f"{solver}_s{model_scale}_pert{num_perturbations}"
    
    def _read_data(self) -> Dict:
        """Read tracker data with file locking."""
        try:
            with open(self.tracker_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
        except FileNotFoundError:
            return {}
    
    def _write_data(self, data: Dict):
        """Write tracker data with file locking."""
        with open(self.tracker_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def record_convergence(self, solver: str, model_scale: int, num_perturbations: int, 
                          iteration: int, run_name: str, final_loss: float) -> bool:
        """
        Record a convergence event with atomic read-modify-write.
        
        Uses exclusive file locking to ensure that if a run converged in fewer
        iterations than the current record, it will correctly update the record
        even if runs finish out of order (not time-aligned).
        
        Args:
            solver: The solver type (1SPSA, 1.5-SPSA, etc.)
            model_scale: The model scale (1, 2, 4, 8, etc.)
            num_perturbations: Number of perturbations per batch (96, 512, etc.)
            iteration: The iteration at which convergence occurred
            run_name: Name of the run that converged
            final_loss: The final loss value
            
        Returns:
            True if this is a new record (fastest convergence), False otherwise
        """
        key = self._get_key(solver, model_scale, num_perturbations)
        is_new_record = False
        current_min = None
        
        # Ensure file exists before opening in r+ mode
        if not self.tracker_file.exists():
            self._ensure_tracker_exists()
        
        # Atomic read-modify-write with exclusive lock
        with open(self.tracker_file, 'r+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Read current data
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
                
                # Check if this is a new record (fewer iterations than current best)
                if key not in data or iteration < data[key]['min_iteration']:
                    is_new_record = True
                    data[key] = {
                        'min_iteration': iteration,
                        'run_name': run_name,
                        'final_loss': final_loss,
                        'timestamp': datetime.now().isoformat(),
                        'all_converged_runs': data.get(key, {}).get('all_converged_runs', [])
                    }
                
                current_min = data[key]['min_iteration']
                
                # Always append to the list of converged runs
                if 'all_converged_runs' not in data[key]:
                    data[key]['all_converged_runs'] = []
                
                data[key]['all_converged_runs'].append({
                    'run_name': run_name,
                    'iteration': iteration,
                    'final_loss': final_loss,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Write back to file
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
                
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        if is_new_record:
            print(f"[TRACKER] NEW RECORD: {key} converged at iteration {iteration} by {run_name}")
        else:
            print(f"[TRACKER] {key} converged at iteration {iteration} by {run_name} (record: {current_min})")
        
        return is_new_record
    
    def should_stop_early(self, solver: str, model_scale: int, num_perturbations: int, 
                         current_iteration: int) -> Tuple[bool, Optional[int]]:
        """
        Check if this run should stop early.
        
        Args:
            solver: The solver type
            model_scale: The model scale
            num_perturbations: Number of perturbations
            current_iteration: Current training iteration
            
        Returns:
            Tuple of (should_stop, min_iteration_recorded)
            If should_stop is True, min_iteration_recorded contains the iteration
            at which another run converged.
        """
        key = self._get_key(solver, model_scale, num_perturbations)
        data = self._read_data()
        
        if key in data:
            min_iter = data[key]['min_iteration']
            if current_iteration > min_iter:
                return True, min_iter
        
        return False, None
    
    def get_min_convergence_iteration(self, solver: str, model_scale: int, 
                                       num_perturbations: int) -> Optional[int]:
        """Get the minimum convergence iteration for a given configuration."""
        key = self._get_key(solver, model_scale, num_perturbations)
        data = self._read_data()
        
        if key in data:
            return data[key]['min_iteration']
        return None
    
    def get_summary(self) -> Dict:
        """Get a summary of all convergence data."""
        return self._read_data()
    
    def reset(self):
        """Reset all convergence data."""
        self._write_data({})
        print("[TRACKER] Convergence data reset")


def print_convergence_table(tracker_file: str):
    """Print a formatted table of convergence results."""
    tracker = ConvergenceTracker(tracker_file)
    data = tracker.get_summary()
    
    if not data:
        print("No convergence data recorded yet.")
        return
    
    print("\n" + "="*80)
    print("CONVERGENCE SUMMARY")
    print("="*80)
    print(f"{'Config':<25} {'Min Iter':>10} {'Run Name':<35} {'Loss':>8}")
    print("-"*80)
    
    for key in sorted(data.keys()):
        info = data[key]
        print(f"{key:<25} {info['min_iteration']:>10} {info['run_name']:<35} {info['final_loss']:>8.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convergence Tracker CLI")
    parser.add_argument("--tracker_file", type=str, default="./convergence_tracker.json",
                        help="Path to the tracker file")
    parser.add_argument("--action", type=str, choices=["summary", "reset", "watch"],
                        default="summary", help="Action to perform")
    parser.add_argument("--watch_interval", type=int, default=10,
                        help="Interval in seconds for watch mode")
    
    args = parser.parse_args()
    
    if args.action == "summary":
        print_convergence_table(args.tracker_file)
    elif args.action == "reset":
        tracker = ConvergenceTracker(args.tracker_file)
        tracker.reset()
        print("Tracker reset complete.")
    elif args.action == "watch":
        print("Watching convergence tracker (Ctrl+C to stop)...")
        try:
            while True:
                os.system('clear')
                print(f"Last updated: {datetime.now().isoformat()}")
                print_convergence_table(args.tracker_file)
                time.sleep(args.watch_interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")

