#!/usr/bin/env python3
"""
Analyze existing sweep results and optionally populate convergence tracker.

Usage:
    # Just view results summary
    python analyze_results.py --results_dir ./results_lstm3layer_overfit_fixed
    
    # View results and populate convergence tracker from successful runs
    python analyze_results.py --results_dir ./results_lstm3layer_overfit_fixed \
                              --populate_tracker ./convergence_tracker_lstm3L_overfit.json
    
    # Show what combinations are still missing/running
    python analyze_results.py --results_dir ./results_lstm3layer_overfit_fixed --show_missing
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def parse_result_file(filepath: Path) -> Optional[Dict]:
    """Parse a single result JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract key info
        args = data.get('args', {})
        
        # Try to extract model_scale from various sources
        model_scale = args.get('model_scale')
        if model_scale is None:
            # Try to infer from hidden_size (hidden_size = 111 * scale)
            hidden_size = args.get('hidden_size', 0)
            if hidden_size > 0:
                model_scale = hidden_size // 111
        
        return {
            'filepath': str(filepath),
            'filename': filepath.name,
            'status': data.get('status', 'unknown'),
            'model_scale': model_scale,
            'hidden_size': args.get('hidden_size'),
            'num_perturbations': args.get('num_perturbations'),
            'solver': args.get('solver'),
            'learning_rate': args.get('learning_rate'),
            'saturating_alpha': args.get('saturating_alpha'),
            'final_loss': data.get('final_loss'),
            'iters': data.get('iters'),
            'seed': data.get('seed'),
            'wandb_run_name': data.get('wandb_run_name'),
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return None


def load_all_results(results_dir: str) -> List[Dict]:
    """Load all JSON result files from directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return []
    
    results = []
    for json_file in results_path.glob("*.json"):
        parsed = parse_result_file(json_file)
        if parsed:
            results.append(parsed)
    
    return results


def group_results(results: List[Dict], include_solver: bool = False) -> Dict[Tuple, List[Dict]]:
    """Group results by (model_scale, num_perturbations) or (model_scale, num_perturbations, solver)."""
    grouped = defaultdict(list)
    for r in results:
        if include_solver:
            key = (r['model_scale'], r['num_perturbations'], r['solver'])
        else:
            key = (r['model_scale'], r['num_perturbations'])
        grouped[key].append(r)
    return grouped


def print_summary_table(results: List[Dict]):
    """Print a summary table of all results."""
    if not results:
        print("No results found.")
        return
    
    # Group by status
    by_status = defaultdict(list)
    for r in results:
        by_status[r['status']].append(r)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY BY STATUS")
    print("="*80)
    
    for status in ['success', 'diverged', 'early_stopped', 'training', 'unknown']:
        if status in by_status:
            print(f"\n{status.upper()}: {len(by_status[status])} runs")
    
    print(f"\nTOTAL: {len(results)} runs")


def print_convergence_summary(results: List[Dict]):
    """Print convergence summary grouped by (scale, perturbations, solver)."""
    grouped = group_results(results, include_solver=True)
    
    print("\n" + "="*90)
    print("CONVERGENCE BY (MODEL_SCALE, NUM_PERTURBATIONS, SOLVER)")
    print("="*90)
    print(f"{'Scale':<8} {'Perts':<8} {'Solver':<12} {'Success':<10} {'Diverged':<10} {'Best Iter':<12} {'Best LR'}")
    print("-"*90)
    
    for (scale, perts, solver), runs in sorted(grouped.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0, x[0][2] or '')):
        success = [r for r in runs if r['status'] == 'success']
        diverged = [r for r in runs if r['status'] == 'diverged']
        
        best_iter = min([r['iters'] for r in success]) if success else '-'
        best_run = min(success, key=lambda x: x['iters']) if success else None
        best_lr = f"{best_run['learning_rate']}" if best_run else '-'
        
        print(f"{scale:<8} {perts:<8} {solver or '-':<12} {len(success):<10} {len(diverged):<10} {str(best_iter):<12} {best_lr}")


def print_detailed_success(results: List[Dict]):
    """Print detailed info about successful runs."""
    success = [r for r in results if r['status'] == 'success']
    
    if not success:
        print("\nNo successful runs yet.")
        return
    
    print("\n" + "="*80)
    print("SUCCESSFUL RUNS (sorted by iterations)")
    print("="*80)
    print(f"{'Scale':<6} {'Perts':<6} {'Solver':<10} {'LR':<10} {'Alpha':<8} {'Iters':<8} {'Loss':<10}")
    print("-"*80)
    
    for r in sorted(success, key=lambda x: (x['model_scale'] or 0, x['iters'] or 0)):
        solver = r['solver'] or '-'
        lr = f"{r['learning_rate']:.4f}" if r['learning_rate'] else '-'
        alpha = f"{r['saturating_alpha']:.2f}" if r['saturating_alpha'] else '-'
        iters = r['iters'] or '-'
        loss = f"{r['final_loss']:.4f}" if r['final_loss'] else '-'
        
        print(f"{r['model_scale'] or '-':<6} {r['num_perturbations'] or '-':<6} {solver:<10} {lr:<10} {alpha:<8} {iters:<8} {loss:<10}")


def print_missing_combinations(results: List[Dict], expected_config: Dict):
    """Print which combinations haven't succeeded yet."""
    grouped = group_results(results)
    
    # Expected combinations from config
    scales = expected_config.get('scales', [1, 2, 4, 8])
    perturbations = expected_config.get('perturbations', [96, 512])
    solvers = expected_config.get('solvers', ['1SPSA', '1.5-SPSA'])
    learning_rates = expected_config.get('learning_rates', [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    
    print("\n" + "="*80)
    print("COMBINATIONS WITHOUT SUCCESS YET")
    print("="*80)
    
    missing = []
    for scale in scales:
        for perts in perturbations:
            key = (scale, perts)
            runs = grouped.get(key, [])
            success = [r for r in runs if r['status'] == 'success']
            
            if not success:
                diverged = [r for r in runs if r['status'] == 'diverged']
                other = [r for r in runs if r['status'] not in ['success', 'diverged']]
                missing.append({
                    'scale': scale,
                    'perts': perts,
                    'diverged': len(diverged),
                    'in_progress': len(other),
                })
    
    if missing:
        print(f"{'Scale':<8} {'Perts':<8} {'Diverged':<10} {'In Progress':<12}")
        print("-"*40)
        for m in missing:
            print(f"{m['scale']:<8} {m['perts']:<8} {m['diverged']:<10} {m['in_progress']:<12}")
    else:
        print("All combinations have at least one successful run!")


def populate_convergence_tracker(results: List[Dict], tracker_file: str):
    """Populate convergence tracker from successful results."""
    success = [r for r in results if r['status'] == 'success']
    
    if not success:
        print("No successful runs to populate tracker with.")
        return
    
    # Group by (scale, perts) and find best (minimum iterations)
    grouped = group_results(success)
    
    tracker_data = {}
    for (scale, perts), runs in grouped.items():
        if scale is None or perts is None:
            continue
            
        best = min(runs, key=lambda x: x['iters'] or float('inf'))
        key = f"s{scale}_pert{perts}"
        
        tracker_data[key] = {
            'min_iteration': best['iters'],
            'run_name': best['wandb_run_name'] or best['filename'],
            'final_loss': best['final_loss'],
            'timestamp': datetime.now().isoformat(),
            'all_converged_runs': [
                {
                    'run_name': r['wandb_run_name'] or r['filename'],
                    'iteration': r['iters'],
                    'final_loss': r['final_loss'],
                    'learning_rate': r['learning_rate'],
                }
                for r in sorted(runs, key=lambda x: x['iters'] or 0)
            ]
        }
    
    # Write tracker file
    tracker_path = Path(tracker_file)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tracker_path, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    print(f"\nPopulated convergence tracker: {tracker_file}")
    print(f"Wrote {len(tracker_data)} (scale, perturbations) combinations")
    
    # Print what was written
    print("\n" + "="*60)
    print("TRACKER CONTENTS")
    print("="*60)
    print(f"{'Config':<20} {'Min Iter':<10} {'Best LR':<12} {'Loss':<10}")
    print("-"*60)
    for key, data in sorted(tracker_data.items()):
        best_lr = data['all_converged_runs'][0]['learning_rate'] if data['all_converged_runs'] else '-'
        print(f"{key:<20} {data['min_iteration']:<10} {best_lr:<12} {data['final_loss']:.4f}")


def get_running_screens() -> Dict[str, str]:
    """Get all running screen sessions as {name: full_id}."""
    import subprocess
    result = subprocess.run(['screen', '-ls'], capture_output=True, text=True)
    
    screens = {}
    for line in result.stdout.split('\n'):
        # Parse lines like "	12345.session_name	(Detached)"
        match = re.search(r'\s+(\d+\.(\S+))\s+\(', line)
        if match:
            full_id = match.group(1)
            name = match.group(2)
            screens[name] = full_id
    
    return screens


def parse_screen_name(name: str) -> Optional[Dict]:
    """
    Parse screen name to extract hyperparameters.
    Expected format: lstm3L_overfit_123_pert96_s4_1SPSA_lr0.01_sa0.1
                 or: lstm3L_overfit_123_pert96_s4_1.5-SPSA_lr0.01_sa0.1
    """
    patterns = {
        'pert': r'pert(\d+)',
        'scale': r'_s(\d+)_',
        'solver': r'_(1SPSA|1\.5-SPSA|2SPSA|BanditSPSA|Sanger-SPSA)_',
        'lr': r'_lr([\d.]+)',
        'sa': r'_sa([\d.]+)',
    }
    
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            val = match.group(1)
            if key in ['pert', 'scale']:
                result[key] = int(val)
            elif key in ['lr', 'sa']:
                result[key] = float(val)
            else:
                result[key] = val
    
    return result if result else None


def get_screen_current_iteration(full_id: str, log_dir: str = None) -> Optional[int]:
    """
    Try to get the current iteration of a running screen.
    
    This attempts to capture the screen's scrollback buffer
    to find the current iteration.
    """
    import subprocess
    import tempfile
    import os
    
    # Create a unique temp file for this screen
    tmp_file = f'/tmp/screen_output_{os.getpid()}_{full_id.replace(".", "_")}.txt'
    
    try:
        # Method 1: Use hardcopy to dump screen contents
        result = subprocess.run(
            ['screen', '-S', full_id, '-X', 'hardcopy', '-h', tmp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Give it a moment to write
        import time
        time.sleep(0.1)
        
        if os.path.exists(tmp_file):
            try:
                with open(tmp_file, 'r') as f:
                    content = f.read()
                
                # Look for iteration pattern like "Iteration 123/500000"
                matches = re.findall(r'Iteration (\d+)/\d+', content)
                if matches:
                    # Return the last (most recent) iteration
                    return int(matches[-1])
            except Exception as e:
                pass
            finally:
                # Clean up
                try:
                    os.remove(tmp_file)
                except:
                    pass
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
    
    # Method 2: Try without -h flag (just visible screen, not scrollback)
    try:
        result = subprocess.run(
            ['screen', '-S', full_id, '-X', 'hardcopy', tmp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        import time
        time.sleep(0.1)
        
        if os.path.exists(tmp_file):
            try:
                with open(tmp_file, 'r') as f:
                    content = f.read()
                
                matches = re.findall(r'Iteration (\d+)/\d+', content)
                if matches:
                    return int(matches[-1])
            except:
                pass
            finally:
                try:
                    os.remove(tmp_file)
                except:
                    pass
    except:
        pass
    
    return None


def find_screens_to_prune(results: List[Dict], dry_run: bool = True) -> List[Tuple[str, str, int, int, str, int, Optional[int]]]:
    """
    Find running screens that should be pruned based on convergence.
    
    Now includes solver in the grouping key and checks current iteration.
    
    Returns list of (screen_name, full_id, scale, perts, solver, best_iter, current_iter) to kill.
    """
    # Get best iterations per (scale, perts, solver) from successful runs
    grouped = group_results(results, include_solver=True)
    best_iters = {}
    
    for (scale, perts, solver), runs in grouped.items():
        success = [r for r in runs if r['status'] == 'success']
        if success:
            best = min(r['iters'] for r in success if r['iters'])
            best_iters[(scale, perts, solver)] = best
    
    if not best_iters:
        print("No successful runs yet - nothing to prune against.")
        return []
    
    print("\n" + "="*70)
    print("CONVERGENCE BASELINES (from successful runs)")
    print("="*70)
    print(f"{'Scale':<8} {'Perts':<8} {'Solver':<12} {'Best Iter'}")
    print("-"*70)
    for (scale, perts, solver), iters in sorted(best_iters.items()):
        print(f"{scale:<8} {perts:<8} {solver:<12} {iters}")
    
    # Get running screens
    screens = get_running_screens()
    print(f"\nFound {len(screens)} running screen sessions")
    print("Checking current iterations (this may take a moment)...")
    
    # Find screens to prune
    to_prune = []
    to_keep = []
    no_baseline = []
    
    for name, full_id in screens.items():
        parsed = parse_screen_name(name)
        if not parsed:
            continue
        
        scale = parsed.get('scale')
        perts = parsed.get('pert')
        solver = parsed.get('solver')
        
        if scale is None or perts is None or solver is None:
            continue
        
        key = (scale, perts, solver)
        
        if key not in best_iters:
            # No successful run for this combo yet - keep it running
            no_baseline.append((name, full_id, scale, perts, solver))
            continue
        
        best_iter = best_iters[key]
        
        # Try to get current iteration
        current_iter = get_screen_current_iteration(full_id)
        
        if current_iter is not None and current_iter > best_iter:
            # Current iteration exceeds best - should prune
            to_prune.append((name, full_id, scale, perts, solver, best_iter, current_iter))
        elif current_iter is not None:
            # Still has a chance
            to_keep.append((name, full_id, scale, perts, solver, best_iter, current_iter))
        else:
            # Couldn't determine iteration - mark as unknown
            # Be conservative: don't prune if we can't verify
            to_keep.append((name, full_id, scale, perts, solver, best_iter, None))
    
    # Print summary of screens without baseline (KEEP - no success yet)
    if no_baseline:
        print("\n" + "="*70)
        print(f"SCREENS WITHOUT BASELINE (KEEP - no success for this combo yet): {len(no_baseline)}")
        print("="*70)
        print(f"{'Screen Name':<55} {'Scale':<6} {'Perts':<6} {'Solver'}")
        print("-"*70)
        for name, full_id, scale, perts, solver in no_baseline[:20]:
            print(f"{name:<55} {scale:<6} {perts:<6} {solver}")
        if len(no_baseline) > 20:
            print(f"... and {len(no_baseline) - 20} more")
    
    # Print screens we're keeping
    if to_keep:
        print("\n" + "="*70)
        print(f"SCREENS TO KEEP (iteration <= best): {len(to_keep)}")
        print("="*70)
        print(f"{'Screen Name':<50} {'Scale':<6} {'Perts':<6} {'Solver':<10} {'Best':<8} {'Curr'}")
        print("-"*90)
        for name, full_id, scale, perts, solver, best_iter, current_iter in to_keep[:20]:
            curr_str = str(current_iter) if current_iter is not None else '?'
            print(f"{name:<50} {scale:<6} {perts:<6} {solver:<10} {best_iter:<8} {curr_str}")
        if len(to_keep) > 20:
            print(f"... and {len(to_keep) - 20} more")
    
    return to_prune


def prune_screens(results: List[Dict], dry_run: bool = True):
    """Prune screens that should be stopped based on convergence."""
    import subprocess
    
    to_prune = find_screens_to_prune(results, dry_run)
    
    if not to_prune:
        print("\nNo screens to prune.")
        return
    
    print("\n" + "="*90)
    print(f"SCREENS TO PRUNE (iteration > best): {len(to_prune)}")
    print("="*90)
    print(f"{'Screen Name':<50} {'Scale':<6} {'Perts':<6} {'Solver':<10} {'Best':<8} {'Curr'}")
    print("-"*90)
    
    for name, full_id, scale, perts, solver, best_iter, current_iter in to_prune:
        curr_str = str(current_iter) if current_iter is not None else '?'
        print(f"{name:<50} {scale:<6} {perts:<6} {solver:<10} {best_iter:<8} {curr_str}")
    
    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN - No screens killed.")
        print("Run with --prune --no_dry_run to actually kill these screens.")
        print("="*60)
        
        # Print the commands that would be run
        print("\nCommands that would be executed:")
        for name, full_id, _, _, _, _, _ in to_prune[:5]:
            print(f"  screen -S \"{full_id}\" -X quit")
        if len(to_prune) > 5:
            print(f"  ... and {len(to_prune) - 5} more")
    else:
        print("\n" + "="*60)
        print("KILLING SCREENS...")
        print("="*60)
        
        killed = 0
        failed = 0
        for name, full_id, scale, perts, solver, _, _ in to_prune:
            try:
                result = subprocess.run(
                    ['screen', '-S', full_id, '-X', 'quit'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  ✓ Killed: {name}")
                    killed += 1
                else:
                    print(f"  ✗ Failed to kill: {name} - {result.stderr}")
                    failed += 1
            except Exception as e:
                print(f"  ✗ Error killing {name}: {e}")
                failed += 1
        
        print(f"\nKilled {killed} screens, {failed} failed.")


def list_running_screens_with_status(results: List[Dict]):
    """List all running screens with their convergence status."""
    # Get best iterations per (scale, perts, solver) from successful runs
    grouped = group_results(results, include_solver=True)
    best_iters = {}
    
    for (scale, perts, solver), runs in grouped.items():
        success = [r for r in runs if r['status'] == 'success']
        if success:
            best = min(r['iters'] for r in success if r['iters'])
            best_iters[(scale, perts, solver)] = best
    
    screens = get_running_screens()
    
    print("\n" + "="*100)
    print(f"RUNNING SCREENS: {len(screens)}")
    print("="*100)
    print(f"{'Screen Name':<50} {'Scale':<6} {'Perts':<6} {'Solver':<10} {'Status'}")
    print("-"*100)
    
    should_prune = 0
    still_needed = 0
    unknown = 0
    need_iteration_check = 0
    
    for name, full_id in sorted(screens.items()):
        parsed = parse_screen_name(name)
        if not parsed:
            print(f"{name:<50} {'?':<6} {'?':<6} {'?':<10} unknown format")
            unknown += 1
            continue
        
        scale = parsed.get('scale')
        perts = parsed.get('pert')
        solver = parsed.get('solver')
        
        if scale is None or perts is None or solver is None:
            print(f"{name:<50} {str(scale):<6} {str(perts):<6} {str(solver):<10} missing info")
            unknown += 1
            continue
        
        key = (scale, perts, solver)
        if key in best_iters:
            best = best_iters[key]
            status = f"HAS BASELINE (best={best}) - check iter"
            need_iteration_check += 1
        else:
            status = "KEEP (no success yet for this solver)"
            still_needed += 1
        
        print(f"{name:<50} {scale:<6} {perts:<6} {solver:<10} {status}")
    
    print("-"*100)
    print(f"Summary: {still_needed} still needed (no baseline), {need_iteration_check} need iteration check, {unknown} unknown")
    print("\nRun with --prune to check iterations and determine which to actually kill.")


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results and prune screens")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing result JSON files")
    parser.add_argument("--populate_tracker", type=str, default=None,
                        help="Path to convergence tracker file to populate")
    parser.add_argument("--show_missing", action="store_true",
                        help="Show combinations that haven't succeeded yet")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Expected model scales")
    parser.add_argument("--perturbations", type=int, nargs="+", default=[96, 512],
                        help="Expected perturbation counts")
    
    # Pruning options
    parser.add_argument("--list_screens", action="store_true",
                        help="List all running screens with their convergence status")
    parser.add_argument("--prune", action="store_true",
                        help="Prune screens that should be stopped (dry run by default)")
    parser.add_argument("--no_dry_run", action="store_true",
                        help="Actually kill screens (use with --prune)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_all_results(args.results_dir)
    print(f"Found {len(results)} result files")
    
    # Print summaries
    print_summary_table(results)
    print_convergence_summary(results)
    print_detailed_success(results)
    
    if args.show_missing:
        expected_config = {
            'scales': args.scales,
            'perturbations': args.perturbations,
        }
        print_missing_combinations(results, expected_config)
    
    # Optionally populate tracker
    if args.populate_tracker:
        populate_convergence_tracker(results, args.populate_tracker)
    
    # Screen management
    if args.list_screens:
        list_running_screens_with_status(results)
    
    if args.prune:
        dry_run = not args.no_dry_run
        prune_screens(results, dry_run=dry_run)


if __name__ == "__main__":
    main()