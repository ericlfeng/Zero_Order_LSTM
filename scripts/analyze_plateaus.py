#!/usr/bin/env python3
"""
Analyze plateau status for LIVE/RUNNING LSTM PTB training runs.
Reads from .pt checkpoint files that are saved during training.

Uses the same EMAPlateauDetector logic from rge_series_experiments.py

Usage:
    python analyze_plateaus.py --results_dir ./results_lstm3layer_ptb
    python analyze_plateaus.py --results_dir ./results_lstm3layer_ptb --patience 20 --threshold 0.005
"""

import argparse
import glob
import os
import torch
from pathlib import Path
from datetime import datetime


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


def analyze_pt_checkpoint(pt_path, patience=10, threshold=0.01, alpha=0.1):
    """Analyze a .pt checkpoint file for plateau behavior."""
    try:
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  Warning: Could not load {pt_path}: {e}")
        return None

    # Extract results from checkpoint
    results = checkpoint.get('results', {})
    args = checkpoint.get('args', {})
    current_iter = checkpoint.get('iteration', 0)
    
    # Get train metrics
    train_metrics = results.get('train_metrics', {})
    losses = train_metrics.get('loss', [])
    iterations = train_metrics.get('iterations', [])
    
    if not losses or len(losses) < 2:
        return None

    # Run plateau detector over the loss history
    detector = EMAPlateauDetector(alpha=alpha, patience=patience, threshold=threshold)
    plateau_iter = None
    
    for i, loss in enumerate(losses):
        if loss is None:
            continue
        is_plateau = detector.update(loss)
        if is_plateau and plateau_iter is None:
            plateau_iter = iterations[i] if i < len(iterations) else i

    # Get file modification time to see how recently it was updated
    mtime = os.path.getmtime(pt_path)
    last_updated = datetime.fromtimestamp(mtime)
    
    # Extract run info
    run_name = args.get('wandb_run_name', Path(pt_path).stem.replace('checkpoint_', ''))
    
    return {
        'run_name': run_name,
        'file': str(pt_path),
        'solver': args.get('solver', 'unknown'),
        'learning_rate': args.get('learning_rate', 'unknown'),
        'model_scale': args.get('model_scale', 'unknown'),
        'num_perturbations': args.get('num_perturbations', 'unknown'),
        'saturating_alpha': args.get('saturating_alpha', 'N/A'),
        'current_iteration': current_iter,
        'num_logged_points': len(losses),
        'final_loss': losses[-1] if losses else None,
        'best_loss': min([l for l in losses if l is not None]) if losses else None,
        'initial_loss': losses[0] if losses else None,
        'plateau_detected': plateau_iter is not None,
        'plateau_iteration': plateau_iter,
        'current_ema': detector.ema,
        'best_ema': detector.best_ema,
        'iters_without_improvement': detector.iters_without_improvement,
        'last_updated': last_updated,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze LIVE training runs for plateau behavior using .pt checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results_lstm3layer_ptb',
                        help='Directory containing checkpoint .pt files')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of log intervals without improvement before plateau')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum relative improvement required (0.01 = 1%%)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='EMA smoothing factor (higher = more weight on recent)')
    parser.add_argument('--sort_by', type=str, default='final_loss',
                        choices=['final_loss', 'best_loss', 'plateau_iteration', 'current_iteration', 'last_updated'],
                        help='Sort results by this metric')
    parser.add_argument('--show_all', action='store_true',
                        help='Show all runs, not just summary')
    args = parser.parse_args()

    # Find all .pt checkpoint files
    pt_files = glob.glob(os.path.join(args.results_dir, '**/checkpoint_*.pt'), recursive=True)
    
    if not pt_files:
        print(f"No .pt checkpoint files found in {args.results_dir}")
        print(f"\nLooking for files matching: {args.results_dir}/**/checkpoint_*.pt")
        print("\nMake sure your runs have saved at least one checkpoint (every 500 iterations by default).")
        return

    print(f"Found {len(pt_files)} checkpoint files in {args.results_dir}")
    print(f"Plateau detection: patience={args.patience} log intervals, threshold={args.threshold*100:.1f}% improvement, alpha={args.alpha}")
    print("=" * 120)

    results = []
    for pt_file in pt_files:
        result = analyze_pt_checkpoint(pt_file, patience=args.patience, threshold=args.threshold, alpha=args.alpha)
        if result:
            results.append(result)

    if not results:
        print("No valid checkpoints found with loss data.")
        return

    # Sort results
    if args.sort_by == 'last_updated':
        results.sort(key=lambda x: x.get('last_updated') or datetime.min, reverse=True)
    else:
        results.sort(key=lambda x: x.get(args.sort_by) or float('inf'))

    # Summary statistics
    plateaued_runs = [r for r in results if r['plateau_detected']]
    active_runs = [r for r in results if not r['plateau_detected']]

    print(f"\n{'='*120}")
    print(f"SUMMARY: {len(plateaued_runs)} PLATEAUED | {len(active_runs)} STILL IMPROVING | {len(results)} total")
    print(f"{'='*120}\n")

    # Print plateaued runs
    if plateaued_runs:
        print("PLATEAUED RUNS (consider stopping):")
        print("-" * 120)
        print(f"  {'Run Name':<55} | {'Solver':<10} | {'LR':<8} | {'Scale':<5} | {'Pert':<5} | {'Iter':<7} | {'Loss':<8} | {'Plateau@':<8}")
        print("-" * 120)
        for r in sorted(plateaued_runs, key=lambda x: x['final_loss'] or float('inf')):
            print(f"  {r['run_name'][:55]:<55} | {r['solver']:<10} | {str(r['learning_rate']):<8} | "
                  f"{str(r['model_scale']):<5} | {str(r['num_perturbations']):<5} | {r['current_iteration']:<7} | "
                  f"{r['final_loss']:.4f}  | {r['plateau_iteration']}")
        print()

    # Print still-improving runs
    if active_runs:
        print("STILL IMPROVING RUNS:")
        print("-" * 120)
        print(f"  {'Run Name':<55} | {'Solver':<10} | {'LR':<8} | {'Scale':<5} | {'Pert':<5} | {'Iter':<7} | {'Loss':<8} | {'Stale':<8}")
        print("-" * 120)
        for r in sorted(active_runs, key=lambda x: x['final_loss'] or float('inf')):
            stale_str = f"{r['iters_without_improvement']}/{args.patience}"
            print(f"  {r['run_name'][:55]:<55} | {r['solver']:<10} | {str(r['learning_rate']):<8} | "
                  f"{str(r['model_scale']):<5} | {str(r['num_perturbations']):<5} | {r['current_iteration']:<7} | "
                  f"{r['final_loss']:.4f}  | {stale_str}")
        print()

    # Best runs by current loss
    print("TOP 10 RUNS BY CURRENT LOSS:")
    print("-" * 120)
    by_loss = sorted(results, key=lambda x: x.get('final_loss') or float('inf'))[:10]
    for i, r in enumerate(by_loss, 1):
        status = "PLATEAU" if r['plateau_detected'] else "ACTIVE"
        loss_drop = (r['initial_loss'] - r['final_loss']) / r['initial_loss'] * 100 if r['initial_loss'] else 0
        print(f"  {i:2}. {status} | loss={r['final_loss']:.4f} | drop={loss_drop:5.1f}% | "
              f"{r['solver']:<10} | lr={r['learning_rate']} | scale={r['model_scale']} | pert={r['num_perturbations']} | "
              f"iter={r['current_iteration']}")
    
    print()
    print("=" * 120)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Re-run this script periodically to monitor progress.")


if __name__ == '__main__':
    main()