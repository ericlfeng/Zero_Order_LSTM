#!/usr/bin/env python3
"""
Format best results CSV into a table with solvers@perturbations as rows and model scales as columns.

Takes the output from export_overfit_best_results.py and creates a formatted table.

Usage:
    python scripts/format_results_table.py --input best_overfit_results.csv
    python scripts/format_results_table.py --input best_overfit_results.csv --format csv --output results_table.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def calculate_lstm_params(hidden_size: int, input_size: int, n_layers: int, output_size: int = None) -> int:
    """Calculate approximate parameter count for an LSTM."""
    if output_size is None:
        output_size = input_size
    
    # First layer: input_size -> hidden_size
    # 4 gates * (input weights + recurrent weights + biases)
    layer1_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    
    # Subsequent layers: hidden_size -> hidden_size
    other_layer_params = 4 * (hidden_size * hidden_size + hidden_size * hidden_size + hidden_size)
    
    total = layer1_params + (n_layers - 1) * other_layer_params
    
    # Output layer: hidden_size -> output_size
    total += hidden_size * output_size + output_size
    
    return total


def load_results_csv(input_file: str) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    
    if not Path(input_file).exists():
        print(f"[ERROR] Input file does not exist: {input_file}", file=sys.stderr)
        return results
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            result = {
                'model_scale': int(row['model_scale']),
                'num_perturbations': int(row['num_perturbations']),
                'solver': row['solver'],
                'min_iterations': int(row['min_iterations']),
                'learning_rate': float(row['learning_rate']),
                'saturating_alpha': float(row['saturating_alpha']),
                'final_loss': float(row['final_loss']),
                'final_acc': float(row['final_acc']),
                'hidden_size': int(row['hidden_size']),
                'n_layers': int(row['n_layers']),
            }
            results.append(result)
    
    print(f"[INFO] Loaded {len(results)} results from {input_file}", file=sys.stderr)
    return results


def format_table(results: List[Dict], output_format: str = 'tsv', 
                 show_params: bool = True, input_size: int = 128):
    """
    Format results into a table.
    
    Rows: solver @ perturbations
    Columns: model scale (or parameters)
    Values: min iterations
    """
    
    if not results:
        print("[ERROR] No results to format", file=sys.stderr)
        return
    
    # Get unique scales, perturbations, and solvers
    scales = sorted(set(r['model_scale'] for r in results))
    perturbations = sorted(set(r['num_perturbations'] for r in results))
    solvers = sorted(set(r['solver'] for r in results))
    
    # Get n_layers from first result (assuming all same)
    n_layers = results[0]['n_layers']
    base_hidden = 111  # Standard base hidden size
    
    # Create lookup dict: (scale, pert, solver) -> min_iterations
    lookup = {}
    for r in results:
        key = (r['model_scale'], r['num_perturbations'], r['solver'])
        lookup[key] = r['min_iterations']
    
    # Determine separator
    sep = '\t' if output_format == 'tsv' else ','
    
    # Header row: Scale
    print(f"Scale{sep}" + sep.join(str(s) for s in scales))
    
    # Parameter count row (if requested)
    if show_params:
        params = []
        for s in scales:
            hidden = base_hidden * s
            p = calculate_lstm_params(hidden, input_size, n_layers)
            params.append(f"{p:,}")
        print(f"# params{sep}" + sep.join(params))
    
    # Hidden size row
    print(f"Model Size (hidden){sep}" + sep.join(str(base_hidden * s) for s in scales))
    
    # Data rows: one row per (solver, perturbations) combination
    for solver in solvers:
        for pert in perturbations:
            row_label = f"{solver} @ {pert} perturbations/step"
            values = []
            
            for scale in scales:
                key = (scale, pert, solver)
                if key in lookup:
                    values.append(str(lookup[key]))
                else:
                    values.append("")  # Missing data
            
            print(f"{row_label}{sep}" + sep.join(values))


def main():
    parser = argparse.ArgumentParser(
        description='Format best results into a table.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format as TSV (tab-separated, easy to paste into spreadsheets):
  python scripts/format_results_table.py --input best_overfit_results.csv

  # Format as CSV and save to file:
  python scripts/format_results_table.py --input best_overfit_results.csv --format csv --output results_table.csv
  
  # Without parameter count row:
  python scripts/format_results_table.py --input best_overfit_results.csv --no_params
"""
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file (output from export_overfit_best_results.py)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='tsv',
        choices=['tsv', 'csv'],
        help='Output format (default: tsv for easy pasting into spreadsheets)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file (default: print to stdout)'
    )
    parser.add_argument(
        '--no_params',
        action='store_true',
        help='Do not show parameter count row'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=128,
        help='Input size for parameter calculation (default: 128)'
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_results_csv(args.input)
    
    if not results:
        print("[ERROR] No results found in input file", file=sys.stderr)
        sys.exit(1)
    
    # Redirect output to file if specified
    if args.output:
        original_stdout = sys.stdout
        sys.stdout = open(args.output, 'w')
    
    # Format and print table
    format_table(
        results, 
        output_format=args.format, 
        show_params=not args.no_params,
        input_size=args.input_size
    )
    
    # Restore stdout if we redirected it
    if args.output:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"[INFO] Table saved to {args.output}", file=sys.stderr)
    else:
        print("", file=sys.stderr)
        print("[INFO] Table printed to stdout (copy and paste into spreadsheet)", file=sys.stderr)


if __name__ == '__main__':
    main()

