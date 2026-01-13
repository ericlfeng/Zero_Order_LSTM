#!/usr/bin/env python3
"""
Export best results from overfit sweep based on fastest convergence.

For each (model_scale, num_perturbations, solver) combination, finds the run 
that converged in the fewest iterations and exports it to CSV with the associated
learning rate.

Usage:
    python scripts/export_overfit_best_results.py --results_dir ./results_lstm3layer_overfit_fixed
    python scripts/export_overfit_best_results.py --results_dir ./results_lstm3layer_overfit_fixed --output best_overfit_results.csv
"""

import argparse
import json
import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


def load_results_from_json_dir(results_dir: str) -> List[Dict]:
    """Load all JSON result files from a directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"[ERROR] Results directory does not exist: {results_dir}", file=sys.stderr)
        return results
    
    json_files = list(results_path.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files in {results_dir}", file=sys.stderr)
    
    skipped_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract from the saved args
            args = data.get('args', {})
            
            if not args:
                print(f"[WARN] No args found in {json_file.name}", file=sys.stderr)
                skipped_count += 1
                continue
            
            # Calculate model_scale from hidden_size (base is 111)
            hidden_size = args.get('hidden_size', 111)
            if hidden_size is None:
                hidden_size = 111
            model_scale = hidden_size // 111 if hidden_size else 1
            
            # Handle potential None values with safe defaults
            learning_rate = args.get('learning_rate', 0.0)
            if learning_rate is None:
                learning_rate = 0.0
            
            saturating_alpha = args.get('saturating_alpha', 1.0)
            if saturating_alpha is None:
                saturating_alpha = 1.0
            
            num_perturbations = args.get('num_perturbations', 0)
            if num_perturbations is None:
                num_perturbations = 0
            
            solver = args.get('solver', 'unknown')
            if solver is None:
                solver = 'unknown'
            
            n_layers = args.get('n_layers', 3)
            if n_layers is None:
                n_layers = 3
            
            # Handle final_loss - could be None or invalid
            final_loss = data.get('final_loss', float('inf'))
            if final_loss is None:
                final_loss = float('inf')
            
            final_acc = data.get('final_acc', 0.0)
            if final_acc is None:
                final_acc = 0.0
            
            status = data.get('status', 'unknown')
            iters = data.get('iters', 0)
            if iters is None:
                iters = 0
            
            result = {
                'n_layers': n_layers,
                'hidden_size': hidden_size,
                'model_scale': model_scale,
                'num_perturbations': num_perturbations,
                'solver': solver,
                'learning_rate': float(learning_rate),
                'saturating_alpha': float(saturating_alpha),
                'final_loss': float(final_loss),
                'final_acc': float(final_acc),
                'status': status,
                'iters': int(iters),
                'source_file': str(json_file.name),
            }
            
            results.append(result)
            
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse JSON {json_file.name}: {e}", file=sys.stderr)
            skipped_count += 1
        except Exception as e:
            print(f"[WARN] Failed to process {json_file.name}: {e}", file=sys.stderr)
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} files due to errors", file=sys.stderr)
    
    return results


def find_best_converged_runs(results: List[Dict]) -> Dict:
    """
    For each (model_scale, num_perturbations, solver) combination,
    find the run that converged in the fewest iterations.
    
    Returns a dictionary mapping (scale, perturbations, solver) to best run.
    """
    # Group results by (model_scale, num_perturbations, solver)
    grouped = defaultdict(list)
    
    for r in results:
        # Only consider converged/successful runs
        if r['status'] not in ('converged', 'success'):
            continue
        
        key = (r['model_scale'], r['num_perturbations'], r['solver'])
        grouped[key].append(r)
    
    print(f"[INFO] Found {len(grouped)} unique (scale, perturbations, solver) configurations with converged runs", file=sys.stderr)
    
    best_runs = {}
    
    for key, group in grouped.items():
        # Find the run with minimum iterations
        best = min(group, key=lambda x: x['iters'])
        
        best_runs[key] = {
            'model_scale': best['model_scale'],
            'num_perturbations': best['num_perturbations'],
            'solver': best['solver'],
            'min_iterations': best['iters'],
            'learning_rate': best['learning_rate'],
            'saturating_alpha': best['saturating_alpha'],
            'final_loss': best['final_loss'],
            'final_acc': best['final_acc'],
            'hidden_size': best['hidden_size'],
            'n_layers': best['n_layers'],
            'num_converged_runs': len(group),
            'source_file': best['source_file'],
        }
    
    return best_runs


def save_to_csv(best_runs: Dict, output_file: str):
    """Save the best runs to a CSV file."""
    if not best_runs:
        print("[ERROR] No runs to save", file=sys.stderr)
        return
    
    # Sort by (model_scale, num_perturbations, solver)
    sorted_keys = sorted(best_runs.keys())
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'model_scale',
            'num_perturbations',
            'solver',
            'min_iterations',
            'learning_rate',
            'saturating_alpha',
            'final_loss',
            'final_acc',
            'hidden_size',
            'n_layers',
            'num_converged_runs',
            'source_file'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for key in sorted_keys:
            writer.writerow(best_runs[key])
    
    print(f"[INFO] Saved {len(best_runs)} best runs to {output_file}", file=sys.stderr)


def print_summary_table(best_runs: Dict):
    """Print a formatted summary table."""
    if not best_runs:
        print("[ERROR] No runs to display", file=sys.stderr)
        return
    
    print("\n" + "=" * 140, file=sys.stderr)
    print("BEST CONVERGED RUNS (MINIMUM ITERATIONS) FOR EACH (MODEL_SCALE, NUM_PERTURBATIONS, SOLVER)", file=sys.stderr)
    print("=" * 140, file=sys.stderr)
    
    header = (
        f"{'Scale':>6} | {'Perts':>6} | {'Solver':>10} | {'Min Iter':>10} | "
        f"{'LR':>12} | {'Alpha':>8} | {'Loss':>12} | {'Acc':>8} | {'#Conv':>6}"
    )
    print(header, file=sys.stderr)
    print("-" * 140, file=sys.stderr)
    
    sorted_keys = sorted(best_runs.keys())
    for key in sorted_keys:
        run = best_runs[key]
        row = (
            f"{run['model_scale']:>6} | "
            f"{run['num_perturbations']:>6} | "
            f"{run['solver']:>10} | "
            f"{run['min_iterations']:>10} | "
            f"{run['learning_rate']:>12.6f} | "
            f"{run['saturating_alpha']:>8.2f} | "
            f"{run['final_loss']:>12.6f} | "
            f"{run['final_acc']:>8.4f} | "
            f"{run['num_converged_runs']:>6}"
        )
        print(row, file=sys.stderr)
    
    print("-" * 140, file=sys.stderr)
    print(f"Total configurations: {len(best_runs)}", file=sys.stderr)
    print("", file=sys.stderr)


def print_status_distribution(results: List[Dict]):
    """Print distribution of run statuses."""
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['status']] += 1
    
    print(f"[INFO] Status distribution:", file=sys.stderr)
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}", file=sys.stderr)
    print("", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Export best converged results from overfit sweep.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export results from default directory:
  python scripts/export_overfit_best_results.py --results_dir ./results_lstm3layer_overfit_fixed

  # Specify output file:
  python scripts/export_overfit_best_results.py --results_dir ./results_lstm3layer_overfit_fixed --output my_best_results.csv
"""
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing JSON result files from overfit sweep'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='best_overfit_results.csv',
        help='Output CSV file path (default: best_overfit_results.csv)'
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] Looking for results in: {args.results_dir}", file=sys.stderr)
    
    # Load results from JSON files
    results = load_results_from_json_dir(args.results_dir)
    
    if not results:
        print("[ERROR] No results found.", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(results)} total results", file=sys.stderr)
    
    # Show distribution of results
    print_status_distribution(results)
    
    # Find best converged runs
    best_runs = find_best_converged_runs(results)
    
    if not best_runs:
        print("[ERROR] No converged runs found", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print_summary_table(best_runs)
    
    # Save to CSV
    save_to_csv(best_runs, args.output)
    
    print(f"[INFO] Done! Results saved to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()

