#!/usr/bin/env python3
"""
Parse LR search results and find the best (learning_rate, saturating_alpha)
for each (model_scale, solver, num_perturbations) combination.

This script reads the JSON result files from an LR search run and outputs
a CSV table with the best hyperparameters for each configuration.

Usage:
    python scripts/parse_lr_search_results.py --results_dir ./results_lr_search/lstm3layer_overfit
    python scripts/parse_lr_search_results.py  # uses default directory
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import csv


def load_results_from_json_dir(results_dir: str) -> List[Dict]:
    """Load all JSON result files from a directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"[ERROR] Results directory does not exist: {results_dir}")
        return results
    
    json_files = list(results_path.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files in {results_dir}")
    
    skipped_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract from the saved args
            args = data.get('args', {})
            
            if not args:
                print(f"[WARN] No args found in {json_file}")
                skipped_count += 1
                continue
            
            # Calculate model_scale from hidden_size (base is 111)
            hidden_size = args.get('hidden_size', 111)
            if hidden_size is None:
                hidden_size = 111
            model_scale = hidden_size // 111 if hidden_size else 1
            
            # Handle potential None values with safe defaults
            learning_rate = args.get('learning_rate')
            if learning_rate is None:
                learning_rate = 0.0
            
            saturating_alpha = args.get('saturating_alpha')
            if saturating_alpha is None:
                saturating_alpha = 1.0
            
            num_perturbations = args.get('num_perturbations')
            if num_perturbations is None:
                num_perturbations = 0
            
            solver = args.get('solver')
            if solver is None:
                solver = 'unknown'
            
            n_layers = args.get('n_layers')
            if n_layers is None:
                n_layers = 3
            
            # Handle final_loss - could be None or invalid
            final_loss = data.get('final_loss')
            if final_loss is None:
                final_loss = float('inf')
            
            final_acc = data.get('final_acc')
            if final_acc is None:
                final_acc = 0.0
            
            result = {
                'n_layers': n_layers,
                'num_perturbations': num_perturbations,
                'model_scale': model_scale,
                'solver': solver,
                'learning_rate': float(learning_rate),
                'saturating_alpha': float(saturating_alpha),
                'final_loss': float(final_loss),
                'final_acc': float(final_acc),
                'status': data.get('status', 'unknown'),
                'iters': data.get('iters', 0),
                'source_file': str(json_file.name),
            }
            
            results.append(result)
            
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse JSON {json_file}: {e}")
            skipped_count += 1
        except Exception as e:
            print(f"[WARN] Failed to process {json_file}: {e}")
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} files due to errors")
    
    return results


def find_best_hyperparameters(results: List[Dict], metric: str = 'final_loss') -> Dict:
    """
    For each (model_scale, solver, num_perturbations) combination,
    find the best (learning_rate, saturating_alpha) based on the metric.
    
    Note: 1SPSA does NOT use saturating_alpha (only 1.5-SPSA uses it).
    For 1SPSA, the alpha in the output is just the default value tested.
    
    Returns a dictionary mapping (scale, solver, perturbations) to best config.
    """
    # Group results by (model_scale, solver, num_perturbations)
    grouped = defaultdict(list)
    
    for r in results:
        key = (r['model_scale'], r['solver'], r['num_perturbations'])
        grouped[key].append(r)
    
    print(f"[INFO] Found {len(grouped)} unique (scale, solver, perturbations) configurations")
    
    # Count solvers to show info about alpha relevance
    solver_counts = defaultdict(int)
    for key in grouped:
        solver_counts[key[1]] += 1
    if '1SPSA' in solver_counts:
        print(f"[INFO] Note: 1SPSA ({solver_counts['1SPSA']} configs) does NOT use saturating_alpha - alpha values shown are defaults")
    
    best_configs = {}
    
    for key, group in grouped.items():
        # Filter out runs with invalid loss values (NaN, inf, None)
        # Accept any status except explicit failures - "training", "converged", "diverged" are all ok
        # since we're picking based on loss value anyway
        valid_runs = []
        rejected_reasons = defaultdict(int)
        
        for r in group:
            loss = r['final_loss']
            status = r['status']
            
            # Check for valid loss (not NaN, not inf, not None)
            if loss is None:
                rejected_reasons['loss_is_none'] += 1
                continue
            if math.isnan(loss):
                rejected_reasons['loss_is_nan'] += 1
                continue
            if math.isinf(loss):
                rejected_reasons['loss_is_inf'] += 1
                continue
            
            # Only reject explicit failure statuses
            if status in ('failed', 'crashed', 'fail', 'error', 'oom'):
                rejected_reasons[f'status_{status}'] += 1
                continue
            
            valid_runs.append(r)
        
        if not valid_runs:
            print(f"[WARN] No valid runs for config: scale={key[0]}, solver={key[1]}, pert={key[2]}")
            print(f"       Rejection reasons: {dict(rejected_reasons)}")
            # Use the first run anyway as a fallback (with lowest loss if possible)
            if group:
                # Try to find the one with lowest finite loss
                finite_runs = [r for r in group if r['final_loss'] is not None 
                               and not math.isnan(r['final_loss']) 
                               and not math.isinf(r['final_loss'])]
                if finite_runs:
                    valid_runs = [min(finite_runs, key=lambda x: x['final_loss'])]
                else:
                    valid_runs = [group[0]]
            else:
                continue
        
        # Find the run with lowest loss (or highest accuracy)
        if metric == 'final_loss':
            best = min(valid_runs, key=lambda x: x['final_loss'])
        else:
            best = max(valid_runs, key=lambda x: x['final_acc'])
        
        best_configs[key] = {
            'model_scale': key[0],
            'solver': key[1],
            'num_perturbations': key[2],
            'best_learning_rate': best['learning_rate'],
            'best_saturating_alpha': best['saturating_alpha'],
            'best_final_loss': best['final_loss'],
            'best_final_acc': best['final_acc'],
            'best_status': best['status'],
            'num_runs_evaluated': len(group),
            'num_valid_runs': len(valid_runs),
            'source_file': best.get('source_file', '-'),
        }
    
    return best_configs


def save_to_csv(best_configs: Dict, output_file: str):
    """Save the best configurations to a CSV file."""
    if not best_configs:
        print("[ERROR] No configurations to save")
        return
    
    # Sort by (model_scale, solver, num_perturbations)
    sorted_keys = sorted(best_configs.keys())
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'model_scale', 'solver', 'num_perturbations',
            'best_learning_rate', 'best_saturating_alpha',
            'best_final_loss', 'best_final_acc', 'best_status',
            'num_runs_evaluated', 'num_valid_runs'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for key in sorted_keys:
            # Only write the fields we need (exclude source_file)
            row = {k: best_configs[key][k] for k in fieldnames}
            writer.writerow(row)
    
    print(f"[INFO] Saved best hyperparameters to {output_file}")


def print_summary_table(best_configs: Dict):
    """Print a formatted summary table."""
    if not best_configs:
        print("[ERROR] No configurations to display")
        return
    
    print("\n" + "=" * 130)
    print("BEST HYPERPARAMETERS FOR EACH (MODEL_SCALE, SOLVER, NUM_PERTURBATIONS) COMBINATION")
    print("=" * 130)
    
    header = f"{'Scale':>6} | {'Solver':>10} | {'Pert':>5} | {'Best LR':>12} | {'Best Alpha':>10} | {'Loss':>12} | {'Acc':>8} | {'Status':>10} | {'Runs':>5}"
    print(header)
    print("-" * 130)
    
    sorted_keys = sorted(best_configs.keys())
    for key in sorted_keys:
        cfg = best_configs[key]
        row = (
            f"{cfg['model_scale']:>6} | "
            f"{cfg['solver']:>10} | "
            f"{cfg['num_perturbations']:>5} | "
            f"{cfg['best_learning_rate']:>12.6f} | "
            f"{cfg['best_saturating_alpha']:>10.2f} | "
            f"{cfg['best_final_loss']:>12.6f} | "
            f"{cfg['best_final_acc']:>8.4f} | "
            f"{cfg['best_status']:>10} | "
            f"{cfg['num_runs_evaluated']:>5}"
        )
        print(row)
    
    print("-" * 130)
    print(f"Total configurations: {len(best_configs)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Parse LR search results and find best hyperparameters.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse results from default LR search directory (3-layer LSTM):
  python scripts/parse_lr_search_results.py

  # Parse results from a specific directory:
  python scripts/parse_lr_search_results.py --results_dir ./results_lr_search/lstm4layer_overfit

  # Specify output file:
  python scripts/parse_lr_search_results.py --output my_best_params.csv

  # Optimize for accuracy instead of loss:
  python scripts/parse_lr_search_results.py --metric final_acc
"""
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results_lr_search/lstm3layer_overfit',
        help='Directory containing JSON result files from LR search (default: ./results_lr_search/lstm3layer_overfit)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='best_lr_alpha.csv',
        help='Output CSV file path (default: best_lr_alpha.csv)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['final_loss', 'final_acc'],
        default='final_loss',
        help='Metric to optimize (default: final_loss, lower is better)'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=None,
        help='Number of layers (auto-detects results_dir if not specified)'
    )
    
    args = parser.parse_args()
    
    # If n_layers is specified, update results_dir accordingly
    if args.n_layers is not None and args.results_dir == './results_lr_search/lstm3layer_overfit':
        args.results_dir = f'./results_lr_search/lstm{args.n_layers}layer_overfit'
    
    print(f"[INFO] Looking for results in: {args.results_dir}")
    
    # Load results from local JSON files
    results = load_results_from_json_dir(args.results_dir)
    
    if not results:
        print("[ERROR] No results found. Make sure the LR search has completed and results are saved.")
        print(f"[INFO] Expected location: {args.results_dir}")
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(results)} total results")
    
    # Show distribution of results
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['status']] += 1
    print(f"[INFO] Status distribution: {dict(status_counts)}")
    
    # Find best hyperparameters
    best_configs = find_best_hyperparameters(results, metric=args.metric)
    
    if not best_configs:
        print("[ERROR] Could not determine best hyperparameters")
        sys.exit(1)
    
    # Print summary
    print_summary_table(best_configs)
    
    # Save to CSV
    save_to_csv(best_configs, args.output)
    
    print(f"[INFO] Done! Next step: run the longer experiments with:")
    print(f"       ./scripts/lstm_nlayer_overfit_best_lr.sh {args.output}")


if __name__ == '__main__':
    main()
