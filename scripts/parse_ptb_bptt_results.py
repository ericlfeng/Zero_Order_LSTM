#!/usr/bin/env python3
"""
Parse BPTT experiment JSON checkpoint files and generate a results table.
Outputs best test loss for each model scale.

Usage:
    python parse_bptt_results.py <directory>
    python parse_bptt_results.py <directory> --output results.csv
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict


def parse_json_file(filepath):
    """Extract relevant info from a BPTT checkpoint JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        args = data.get('args', {})
        filename = os.path.basename(filepath)
        
        # Extract model_scale from filename - look for pattern like _s1_, _s4_, _s16_
        scale_match = re.search(r'_s(\d+)_', filename)
        if scale_match:
            model_scale = int(scale_match.group(1))
        else:
            model_scale = args.get('model_scale', 'unknown')
        
        # Best test loss - extract from test_metrics array
        test_metrics = data.get('test_metrics', {})
        if 'loss' in test_metrics and test_metrics['loss']:
            loss_array = test_metrics['loss']
            best_loss = min(loss_array)
        else:
            best_test = data.get('best_test', {})
            best_loss = best_test.get('loss', None)
        
        return {
            'model_scale': model_scale,
            'best_loss': best_loss,
            'filename': filename
        }
    except Exception as e:
        print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
        return None


def select_best_per_scale(results):
    """For each scale, select the result with the lowest loss."""
    by_scale = defaultdict(list)
    for r in results:
        by_scale[r['model_scale']].append(r)
    
    best_results = {}
    for scale, runs in by_scale.items():
        valid = [r for r in runs if r['best_loss'] is not None]
        if valid:
            best = min(valid, key=lambda x: x['best_loss'])
            best_results[scale] = best
    
    return best_results


def main():
    parser = argparse.ArgumentParser(
        description='Parse BPTT experiment JSON files and generate results table'
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
        '--pattern',
        default='*bptt*.json',
        help='Glob pattern for JSON files (default: *bptt*.json)'
    )
    
    args = parser.parse_args()
    
    # Find all JSON files
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    json_files = list(directory.glob(args.pattern))
    
    if not json_files:
        print(f"No JSON files found matching '{args.pattern}' in '{args.directory}'", file=sys.stderr)
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
    
    # Get best per scale
    best_by_scale = select_best_per_scale(results)
    
    # Sort scales naturally
    scales = sorted(best_by_scale.keys(), key=lambda x: x if isinstance(x, int) else 0)
    
    # Print table
    print("BPTT Best Test Loss by Scale")
    print("=" * 40)
    print(f"{'Scale':<10} {'Best Loss':<15}")
    print("-" * 40)
    for scale in scales:
        r = best_by_scale[scale]
        loss_str = f"{r['best_loss']:.4f}" if r['best_loss'] else "—"
        print(f"s{scale:<9} {loss_str:<15}")
    print("=" * 40)
    
    # Save to file if requested
    if args.output:
        ext = Path(args.output).suffix.lower()
        delimiter = '\t' if ext == '.tsv' else ','
        
        with open(args.output, 'w') as f:
            f.write(delimiter.join(['Scale', 'Best_Loss']) + '\n')
            for scale in scales:
                r = best_by_scale[scale]
                loss_str = str(r['best_loss']) if r['best_loss'] else ''
                f.write(delimiter.join([f's{scale}', loss_str]) + '\n')
        
        print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()