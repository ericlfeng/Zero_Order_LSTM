#!/usr/bin/env python3
"""
Analyze plateau status for LSTM PTB training runs.
Uses the same EMAPlateauDetector logic from rge_series_experiments.py

Usage:
    python analyze_plateaus.py --results_dir ./results_lstm3layer_ptb
    python analyze_plateaus.py --results_dir ./results_lstm3layer_ptb --patience 20 --threshold 0.005
"""

import argparse
import json
import glob
import os
from pathlib import Path


class EMAPlateauDetector:
    """
    Same implementation from rge_series_experiments.py
    Detects when training loss has plateaued using exponential moving average.
    """
    def __init__(self, alpha=0.1, patience=10, threshold=0.01):
        self.alpha = alpha
        self.patience = patience
        self.threshold = threshold
        self.ema = None
        self.best_ema = None
        self.iters_without_improvement = 0

    def update(self, loss):
        if self.ema is None:
            self.ema = loss
            self.best_ema = loss
            return False

        self.ema = self.alpha * loss + (1 - self.alpha) * self.ema
        relative_improvement = (self.best_ema - self.ema) / (self.best_ema + 1e-8)

        if relative_improvement > self.threshold:
            self.best_ema = self.ema
            self.iters_without_improvement = 0
        else:
            self.iters_without_improvement += 1

        return self.iters_without_improvement >= self.patience

    def reset(self):
        self.iters_without_improvement = 0


def analyze_run(json_path, patience=10, threshold=0.01, alpha=0.1):
    """Analyze a single run for plateau behavior."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return None

    # Extract losses - check different possible structures
    losses = None
    iterations = None
    
    if 'train_metrics' in data:
        losses = data['train_metrics'].get('loss', [])
        iterations = data['train_metrics'].get('iterations', [])
    elif 'loss' in data:
        losses = data['loss']
        iterations = data.get('iterations', list(range(len(losses))))
    
    if not losses or len(losses) < patience * 2:
        return None

    # Run plateau detector
    detector = EMAPlateauDetector(alpha=alpha, patience=patience, threshold=threshold)
    plateau_iter = None
    
    for i, loss in enumerate(losses):
        if loss is None:
            continue
        is_plateau = detector.update(loss)
        if is_plateau and plateau_iter is None:
            plateau_iter = iterations[i] if iterations else i

    # Extract run info from args if available
    args = data.get('args', {})
    run_name = args.get('wandb_run_name', Path(json_path).stem)
    
    return {
        'run_name': run_name,
        'file': str(json_path),
        'solver': args.get('solver', 'unknown'),
        'learning_rate': args.get('learning_rate', 'unknown'),
        'model_scale': args.get('model_scale', 'unknown'),
        'num_perturbations': args.get('num_perturbations', 'unknown'),
        'total_iterations': len(losses),
        'final_loss': losses[-1] if losses else None,
        'best_loss': min([l for l in losses if l is not None]) if losses else None,
        'plateau_detected': plateau_iter is not None,
        'plateau_iteration': plateau_iter,
        'current_ema': detector.ema,
        'best_ema': detector.best_ema,
        'iters_without_improvement': detector.iters_without_improvement,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze training runs for plateau behavior')
    parser.add_argument('--results_dir', type=str, default='./results_lstm3layer_ptb',
                        help='Directory containing result JSON files')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of log intervals without improvement')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum relative improvement (0.01 = 1%%)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='EMA smoothing factor')
    parser.add_argument('--sort_by', type=str, default='final_loss',
                        choices=['final_loss', 'best_loss', 'plateau_iteration', 'total_iterations'],
                        help='Sort results by this metric')
    args = parser.parse_args()

    # Find all JSON files
    json_files = glob.glob(os.path.join(args.results_dir, '**/*.json'), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {args.results_dir}")
        print("\nTo use this script, run it in the directory containing your experiment results.")
        return

    print(f"Found {len(json_files)} JSON files in {args.results_dir}")
    print(f"Plateau detection settings: patience={args.patience}, threshold={args.threshold}, alpha={args.alpha}")
    print("=" * 100)

    results = []
    for jf in json_files:
        result = analyze_run(jf, patience=args.patience, threshold=args.threshold, alpha=args.alpha)
        if result:
            results.append(result)

    if not results:
        print("No valid runs found with loss data.")
        return

    # Sort results
    results.sort(key=lambda x: x.get(args.sort_by) or float('inf'))

    # Summary statistics
    plateaued_runs = [r for r in results if r['plateau_detected']]
    active_runs = [r for r in results if not r['plateau_detected']]

    print(f"\n{'='*100}")
    print(f"SUMMARY: {len(plateaued_runs)} plateaued, {len(active_runs)} still improving")
    print(f"{'='*100}\n")

    # Print plateaued runs
    if plateaued_runs:
        print("PLATEAUED RUNS (consider stopping or adjusting LR):")
        print("-" * 100)
        for r in plateaued_runs:
            print(f"  {r['run_name'][:50]:50s} | solver={r['solver']:10s} | "
                  f"lr={r['learning_rate']:<8} | scale={r['model_scale']} | "
                  f"plateau@{r['plateau_iteration']:>6} | loss={r['final_loss']:.4f}")
        print()

    # Print still-improving runs
    if active_runs:
        print("STILL IMPROVING RUNS:")
        print("-" * 100)
        for r in active_runs:
            print(f"  {r['run_name'][:50]:50s} | solver={r['solver']:10s} | "
                  f"lr={r['learning_rate']:<8} | scale={r['model_scale']} | "
                  f"iter={r['total_iterations']:>6} | loss={r['final_loss']:.4f} | "
                  f"stale={r['iters_without_improvement']}/{args.patience}")
        print()

    # Best runs by final loss
    print("TOP 5 RUNS BY BEST LOSS:")
    print("-" * 100)
    by_best = sorted(results, key=lambda x: x.get('best_loss') or float('inf'))[:5]
    for r in by_best:
        status = "PLATEAU" if r['plateau_detected'] else "ACTIVE"
        print(f"  {r['run_name'][:50]:50s} | {status:8s} | best_loss={r['best_loss']:.4f} | "
              f"solver={r['solver']} | lr={r['learning_rate']}")


if __name__ == '__main__':
    main()