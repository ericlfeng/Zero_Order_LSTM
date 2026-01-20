#!/usr/bin/env python3
"""
Parse experiment JSON checkpoint files and generate a results table.

Usage:
    python parse_results.py <directory>
    python parse_results.py <directory> --output results.csv
    python parse_results.py <directory> --output results.tsv
    python parse_results.py <directory> --format long
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def parse_json_file(filepath):
    """Extract relevant info from a checkpoint JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        args = data.get('args', {})
        best_test = data.get('best_test', {})
        
        # Extract solver name and normalize it
        solver = args.get('solver', 'unknown')
        
        # Extract other fields
        num_pert = args.get('num_perturbations', 0)
        model_scale = args.get('model_scale', 'unknown')
        learning_rate = args.get('learning_rate', None)
        
        # Best test results
        best_acc = best_test.get('accuracy', None)
        best_iter = best_test.get('iteration', None)
        
        # Also get status for context
        status = data.get('status', 'unknown')
        
        return {
            'solver': solver,
            'num_perturbations': num_pert,
            'model_scale': model_scale,
            'learning_rate': learning_rate,
            'best_accuracy': best_acc,
            'best_iteration': best_iter,
            'status': status,
            'filename': os.path.basename(filepath)
        }
    except Exception as e:
        print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
        return None


def select_best_lr(results):
    """
    For each combination of (solver, num_perturbations, model_scale),
    select the result with the best accuracy across different learning rates.
    """
    # Group by (solver, num_pert, scale)
    groups = defaultdict(list)
    for r in results:
        key = (r['solver'], r['num_perturbations'], r['model_scale'])
        groups[key].append(r)
    
    # Select best from each group
    best_results = []
    for key, group in groups.items():
        # Filter out results with no accuracy
        valid = [r for r in group if r['best_accuracy'] is not None]
        if valid:
            best = max(valid, key=lambda x: x['best_accuracy'])
        else:
            # If no valid results, just pick the first one
            best = group[0]
        best_results.append(best)
    
    return best_results


def format_solver_pert(solver, num_pert):
    """Format solver and perturbation count together."""
    return f"{solver} @ {num_pert}"


def format_cell(accuracy, iteration, learning_rate=None, include_lr=False):
    """Format accuracy and iteration for a table cell."""
    if accuracy is None:
        return "—"
    acc_pct = accuracy * 100
    parts = [f"{acc_pct:.2f}%"]
    
    if iteration is not None:
        # Use 'k' notation for thousands
        if iteration >= 1000:
            iter_str = f"{iteration/1000:.1f}k"
        else:
            iter_str = str(iteration)
        parts.append(f"@ {iter_str}")
    
    if include_lr and learning_rate is not None:
        parts.append(f"(lr={learning_rate})")
    
    return " ".join(parts)


def generate_wide_table(results, include_lr=False):
    """
    Generate a wide-format table with:
    - Rows: solver + perturbations
    - Columns: model scale
    - Cells: accuracy @ iteration (lr=X)
    """
    # First select best LR for each combination
    best_results = select_best_lr(results)
    
    # Organize data
    data = defaultdict(dict)
    raw_data = defaultdict(dict)  # Store raw values for CSV export
    all_scales = set()
    
    for r in best_results:
        key = format_solver_pert(r['solver'], r['num_perturbations'])
        scale = f"s{r['model_scale']}"
        all_scales.add(scale)
        
        cell = format_cell(r['best_accuracy'], r['best_iteration'], 
                          r['learning_rate'], include_lr)
        data[key][scale] = cell
        raw_data[key][scale] = r  # Store full result for CSV
    
    # Sort scales naturally (s1, s2, s4, s8, etc.)
    def scale_sort_key(s):
        try:
            return int(s[1:])
        except:
            return s
    
    scales = sorted(all_scales, key=scale_sort_key)
    
    # Sort solver+pert rows
    def solver_sort_key(s):
        # Extract solver name and pert count for sorting
        parts = s.split(' @ ')
        solver = parts[0]
        pert = int(parts[1]) if len(parts) > 1 else 0
        return (solver, pert)
    
    solvers = sorted(data.keys(), key=solver_sort_key)
    
    return solvers, scales, data, raw_data


def print_markdown_table(solvers, scales, data):
    """Print a markdown-formatted table."""
    # Header
    header = "| Solver | " + " | ".join(scales) + " |"
    separator = "|" + "---|" * (len(scales) + 1)
    
    print(header)
    print(separator)
    
    for solver in solvers:
        row = f"| {solver} |"
        for scale in scales:
            cell = data[solver].get(scale, "—")
            row += f" {cell} |"
        print(row)


def save_wide_csv(solvers, scales, raw_data, filepath, delimiter=','):
    """Save wide-format results to CSV/TSV with separate columns for acc, iter, lr."""
    with open(filepath, 'w') as f:
        # Header
        headers = ['Solver']
        for scale in scales:
            headers.extend([f'{scale}_accuracy', f'{scale}_iteration', f'{scale}_lr'])
        f.write(delimiter.join(headers) + '\n')
        
        # Data rows
        for solver in solvers:
            row = [solver]
            for scale in scales:
                r = raw_data[solver].get(scale)
                if r:
                    row.append(str(r['best_accuracy']) if r['best_accuracy'] else '')
                    row.append(str(r['best_iteration']) if r['best_iteration'] else '')
                    row.append(str(r['learning_rate']) if r['learning_rate'] else '')
                else:
                    row.extend(['', '', ''])
            f.write(delimiter.join(row) + '\n')
    
    print(f"Saved to: {filepath}")


def print_ascii_table(solvers, scales, data):
    """Print a nicely formatted ASCII table."""
    # Calculate column widths
    solver_width = max(len("Solver"), max(len(s) for s in solvers))
    col_widths = {}
    for scale in scales:
        max_cell = max(
            len(data[solver].get(scale, "—")) 
            for solver in solvers
        )
        col_widths[scale] = max(len(scale), max_cell)
    
    # Build format string
    total_width = solver_width + 3 + sum(w + 3 for w in col_widths.values())
    
    # Header
    print("=" * total_width)
    header = f" {'Solver':<{solver_width}} |"
    for scale in scales:
        header += f" {scale:^{col_widths[scale]}} |"
    print(header)
    print("-" * total_width)
    
    # Data rows
    for solver in solvers:
        row = f" {solver:<{solver_width}} |"
        for scale in scales:
            cell = data[solver].get(scale, "—")
            row += f" {cell:^{col_widths[scale]}} |"
        print(row)
    
    print("=" * total_width)


def print_long_table(results):
    """Print a long-format table with one row per experiment (best LR selected)."""
    best_results = select_best_lr(results)
    
    print(f"{'Solver':<12} {'Pert':>4} {'Scale':>5} {'Best Acc':>10} {'Best Iter':>10} {'LR':>8} {'Status':<10}")
    print("-" * 70)
    
    # Sort results
    sorted_results = sorted(best_results, key=lambda x: (
        x['solver'], 
        x['num_perturbations'], 
        x['model_scale']
    ))
    
    for r in sorted_results:
        acc = f"{r['best_accuracy']*100:.2f}%" if r['best_accuracy'] else "—"
        iter_val = str(r['best_iteration']) if r['best_iteration'] else "—"
        lr_val = str(r['learning_rate']) if r['learning_rate'] else "—"
        print(f"{r['solver']:<12} {r['num_perturbations']:>4} {r['model_scale']:>5} {acc:>10} {iter_val:>10} {lr_val:>8} {r['status']:<10}")


def save_long_csv(results, filepath, delimiter=','):
    """Save all results (before best-LR selection) to a CSV/TSV file."""
    with open(filepath, 'w') as f:
        headers = [
            'Solver', 'Num_Perturbations', 'Model_Scale', 
            'Learning_Rate', 'Best_Accuracy', 'Best_Iteration', 
            'Status', 'Filename'
        ]
        f.write(delimiter.join(headers) + '\n')
        
        for r in results:
            row = [
                r['solver'],
                str(r['num_perturbations']),
                str(r['model_scale']),
                str(r['learning_rate']) if r['learning_rate'] else '',
                str(r['best_accuracy']) if r['best_accuracy'] else '',
                str(r['best_iteration']) if r['best_iteration'] else '',
                r['status'],
                r['filename']
            ]
            f.write(delimiter.join(row) + '\n')
    
    print(f"Saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse experiment JSON files and generate results table'
    )
    parser.add_argument(
        'directory', 
        help='Directory containing JSON checkpoint files'
    )
    parser.add_argument(
        '--output', '-o',
        metavar='FILE',
        help='Save results to file (.csv or .tsv extension determines format)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['wide', 'long', 'markdown'],
        default='wide',
        help='Output format: wide (default), long, or markdown'
    )
    parser.add_argument(
        '--pattern',
        default='*.json',
        help='Glob pattern for JSON files (default: *.json)'
    )
    parser.add_argument(
        '--show-lr',
        action='store_true',
        help='Show learning rate in table cells'
    )
    parser.add_argument(
        '--all-lrs',
        action='store_true',
        help='For CSV output, include all LRs instead of just the best'
    )
    
    args = parser.parse_args()
    
    # Find all JSON files
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    json_files = list(directory.glob(args.pattern))
    
    if not json_files:
        print(f"No JSON files found in '{args.directory}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files\n")
    
    # Parse all files
    results = []
    for filepath in json_files:
        result = parse_json_file(filepath)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found", file=sys.stderr)
        sys.exit(1)
    
    # Determine delimiter for output file
    if args.output:
        ext = Path(args.output).suffix.lower()
        delimiter = '\t' if ext == '.tsv' else ','
        
        if args.all_lrs:
            # Save all results without best-LR selection
            save_long_csv(results, args.output, delimiter)
        else:
            # Save wide format with best LR selected
            solvers, scales, data, raw_data = generate_wide_table(results, args.show_lr)
            save_wide_csv(solvers, scales, raw_data, args.output, delimiter)
        print()
    
    # Print table to stdout
    if args.format == 'long':
        print_long_table(results)
    else:
        solvers, scales, data, raw_data = generate_wide_table(results, args.show_lr)
        if args.format == 'markdown':
            print_markdown_table(solvers, scales, data)
        else:
            print_ascii_table(solvers, scales, data)


if __name__ == '__main__':
    main()