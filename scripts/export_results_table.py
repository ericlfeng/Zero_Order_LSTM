#!/usr/bin/env python3
"""
Export training results into a formatted table (iterations to convergence).

Usage:
    python scripts/export_results_table.py --results_dir ./results_lstm3layer_overfit_best
    python scripts/export_results_table.py --results_dir ./results --format csv
"""

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import sys


def calculate_lstm_params(hidden_size: int, input_size: int, n_layers: int) -> int:
    """Calculate approximate parameter count for an LSTM."""
    layer1_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
    other_layer_params = 4 * (hidden_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
    total = layer1_params + (n_layers - 1) * other_layer_params
    total += hidden_size * input_size + input_size
    return total


def load_results(results_dir: str) -> List[Dict]:
    """Load all JSON result files from a directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"[ERROR] Results directory does not exist: {results_dir}", file=sys.stderr)
        return results
    
    json_files = list(results_path.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files", file=sys.stderr)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            args = data.get('args', {})
            if not args:
                continue
            
            hidden_size = args.get('hidden_size', 111)
            model_scale = hidden_size // 111 if hidden_size else 1
            
            result = {
                'model_scale': model_scale,
                'hidden_size': hidden_size,
                'solver': args.get('solver', 'unknown'),
                'num_perturbations': args.get('num_perturbations', 0),
                'saturating_alpha': args.get('saturating_alpha', 1.0),
                'learning_rate': args.get('learning_rate', 0.0),
                'n_layers': args.get('n_layers', 3),
                'input_size': args.get('input_size', 128),
                'final_loss': data.get('final_loss', float('inf')),
                'final_acc': data.get('final_acc', 0.0),
                'status': data.get('status', 'unknown'),
                'iters': data.get('iters', 0),
            }
            results.append(result)
            
        except Exception as e:
            print(f"[WARN] Failed to process {json_file}: {e}", file=sys.stderr)
    
    return results


def get_results_by_config(results: List[Dict]) -> Dict:
    """
    Group results by (scale, solver, perturbations).
    Returns dict mapping (scale, solver, pert) -> result
    """
    by_config = {}
    for r in results:
        key = (r['model_scale'], r['solver'], r['num_perturbations'])
        # If multiple results for same config, keep the one with more iterations
        if key not in by_config or r['iters'] > by_config[key]['iters']:
            by_config[key] = r
    return by_config


def format_table(results: List[Dict], output_format: str = 'tsv', 
                 all_scales: List[int] = None, all_perts: List[int] = None):
    """Format results into a table matching the user's format."""
    
    by_config = get_results_by_config(results)
    
    if not by_config:
        print("[ERROR] No valid results found", file=sys.stderr)
        return
    
    # Get scales from data, or use provided list
    if all_scales:
        scales = all_scales
    else:
        scales = sorted(set(k[0] for k in by_config.keys()))
    
    # Get perturbations from data, or use provided list
    if all_perts:
        perts = all_perts
    else:
        perts = sorted(set(k[2] for k in by_config.keys()))
    
    # Get solvers from data
    solvers = sorted(set(k[1] for k in by_config.keys()))
    
    # Get model info from first result
    sample_result = list(by_config.values())[0]
    n_layers = sample_result['n_layers']
    input_size = sample_result['input_size']
    base_hidden = 111
    
    sep = '\t' if output_format == 'tsv' else ','
    
    # Scale row
    print(f"Scale{sep}" + sep.join(str(s) for s in scales))
    
    # Params row
    params = []
    for s in scales:
        hidden = base_hidden * s
        p = calculate_lstm_params(hidden, input_size, n_layers)
        params.append(f"{p:,}")
    print(f"# params{sep}" + sep.join(params))
    
    # Hidden size row
    print(f"Model Size (hidden){sep}" + sep.join(str(base_hidden * s) for s in scales))
    
    # Duplicate Scale row (matching user's format)
    params_again = []
    for s in scales:
        hidden = base_hidden * s
        p = calculate_lstm_params(hidden, input_size, n_layers)
        params_again.append(f"{p:,}")
    print(f"Scale{sep}" + sep.join(params_again))
    
    # BPTT row (placeholder - include if you have BPTT results)
    if 'BPTT' in solvers:
        values = []
        for scale in scales:
            key = (scale, 'BPTT', 0)  # BPTT doesn't use perturbations
            if key in by_config:
                r = by_config[key]
                if r['status'] in ('converged', 'success'):
                    values.append(str(r['iters']))
                else:
                    values.append("")
            else:
                values.append("")
        print(f"BPTT{sep}" + sep.join(values))
    
    # Data rows for each solver and perturbation count
    for solver in ['1SPSA', '1.5-SPSA']:  # Fixed order
        if solver not in solvers and solver.replace('-', '') not in [s.replace('-', '') for s in solvers]:
            continue
        # Normalize solver name
        actual_solver = solver if solver in solvers else None
        for s in solvers:
            if s.replace('-', '').replace('.', '') == solver.replace('-', '').replace('.', ''):
                actual_solver = s
                break
        if not actual_solver:
            continue
            
        for pert in perts:
            row_name = f"{solver} @ {pert} perturbations/step"
            values = []
            for scale in scales:
                key = (scale, actual_solver, pert)
                if key in by_config:
                    r = by_config[key]
                    status = r['status']
                    iters = r['iters']
                    
                    # Only show iterations if converged
                    if status in ('converged', 'success'):
                        values.append(str(iters))
                    elif status == 'diverged':
                        values.append("")  # or "DIV" if you want to show it
                    else:
                        # Still training or didn't converge
                        values.append("")
                else:
                    values.append("")
            print(f"{row_name}{sep}" + sep.join(values))
        
        # Add empty row for 1024 perturbations (placeholder)
        if 1024 not in perts:
            print(f"{solver} @ 1024 perturbations/step{sep}" + sep.join([""] * len(scales)))


def main():
    parser = argparse.ArgumentParser(description='Export training results to table format')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing JSON result files')
    parser.add_argument('--format', type=str, default='tsv',
                        choices=['tsv', 'csv'],
                        help='Output format (default: tsv for easy pasting)')
    parser.add_argument('--scales', type=str, default=None,
                        help='Comma-separated list of scales to include (e.g., "1,2,4,8,16,32,64,128")')
    parser.add_argument('--perts', type=str, default=None,
                        help='Comma-separated list of perturbation counts (e.g., "8,96,512,1024")')
    
    args = parser.parse_args()
    
    # Parse scales and perts if provided
    all_scales = None
    all_perts = None
    if args.scales:
        all_scales = [int(x) for x in args.scales.split(',')]
    if args.perts:
        all_perts = [int(x) for x in args.perts.split(',')]
    
    results = load_results(args.results_dir)
    
    if not results:
        print("[ERROR] No results found", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(results)} results", file=sys.stderr)
    
    # Show status distribution
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['status']] += 1
    print(f"[INFO] Status distribution: {dict(status_counts)}", file=sys.stderr)
    print("", file=sys.stderr)
    
    format_table(results, args.format, all_scales, all_perts)


if __name__ == '__main__':
    main()
