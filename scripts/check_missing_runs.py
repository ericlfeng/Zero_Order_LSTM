#!/usr/bin/env python3
"""Check which runs are missing or still in progress."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def check_missing(results_dir: str, scales: list, solvers: list, perts: list, lrs: list, alphas: list):
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"[ERROR] Directory not found: {results_dir}")
        return
    
    # Load all completed results
    completed = set()
    in_progress = []
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            args = data.get('args', {})
            hidden = args.get('hidden_size', 111)
            scale = hidden // 111
            solver = args.get('solver')
            pert = args.get('num_perturbations')
            lr = args.get('learning_rate')
            alpha = args.get('saturating_alpha')
            status = data.get('status', 'unknown')
            iters = data.get('iters', 0)
            
            key = (scale, solver, pert, lr, alpha)
            completed.add(key)
            
            if status not in ('converged', 'success', 'diverged'):
                in_progress.append({
                    'scale': scale, 'solver': solver, 'pert': pert,
                    'lr': lr, 'alpha': alpha, 'status': status, 'iters': iters,
                    'file': json_file.name
                })
        except Exception as e:
            print(f"[WARN] Error reading {json_file}: {e}")
    
    # Generate expected combinations
    expected = set()
    for scale in scales:
        for solver in solvers:
            # For 1SPSA, only alpha=1.0 (1SPSA ignores alpha anyway)
            solver_alphas = [1.0] if solver == '1SPSA' else alphas
            for pert in perts:
                for lr in lrs:
                    for alpha in solver_alphas:
                        expected.add((scale, solver, pert, lr, alpha))
    
    missing = expected - completed
    extra = completed - expected  # Configs in results but not expected
    
    print(f"Expected: {len(expected)} runs")
    print(f"Completed: {len(completed)} runs")
    print(f"Missing: {len(missing)} runs")
    print()
    
    if in_progress:
        print("=== STILL IN PROGRESS (status != converged/diverged) ===")
        for r in sorted(in_progress, key=lambda x: (x['scale'], x['solver'], x['pert'])):
            print(f"  scale={r['scale']}, solver={r['solver']}, pert={r['pert']}, "
                  f"lr={r['lr']}, alpha={r['alpha']}, status={r['status']}, iters={r['iters']}")
        print()
    
    if missing:
        print("=== MISSING (no JSON file found) ===")
        for m in sorted(missing):
            print(f"  scale={m[0]}, solver={m[1]}, pert={m[2]}, lr={m[3]}, alpha={m[4]}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_missing_runs.py <results_dir>")
        print("Example: python check_missing_runs.py ./results_lr_search/lstm3layer_overfit")
        sys.exit(1)
    
    # Configure expected combinations here:
    scales = [16, 32]  # Your current scales
    solvers = ['1SPSA', '1.5-SPSA']
    perts = [8, 96, 512]
    lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    alphas = [1.0, 0.5, 0.1]  # For 1.5-SPSA
    
    check_missing(sys.argv[1], scales, solvers, perts, lrs, alphas)
