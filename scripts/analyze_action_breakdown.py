#!/usr/bin/env python3
"""Analyze action breakdown for model runs.

Usage:
    python analyze_action_breakdown.py              # Analyze all runs
    python analyze_action_breakdown.py 2026-02-*    # Filter by glob pattern
"""

import json
from pathlib import Path
from collections import defaultdict, Counter


def analyze_run_file(filepath: Path) -> Counter:
    """Analyze a single run file and return action counts."""
    action_counts = Counter()

    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'action' in data and data['action']:
                    # Handle action as dict with 'type' field or direct string
                    if isinstance(data['action'], dict):
                        action = data['action'].get('type', '').lower()
                    else:
                        action = data['action'].lower()

                    if action:
                        # Normalize action names
                        if 'poll_passive' in action or action == 'poll':
                            action = 'poll'
                        elif 'poll_active' in action or 'query' in action:
                            action = 'query'
                        elif 'lock' in action:
                            action = 'lock_in'
                        elif 'test_submit' in action:
                            action = 'test_submit'
                        elif 'abandon' in action:
                            action = 'abandon'

                        action_counts[action] += 1
            except json.JSONDecodeError:
                continue

    return action_counts


def main():
    import sys

    # Use relative path from repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    runs_dir = repo_root / 'runs'

    # Allow optional glob pattern from command line
    pattern = sys.argv[1] if len(sys.argv) > 1 else '*.jsonl'

    files_to_analyze = defaultdict(list)

    for filepath in sorted(runs_dir.glob(pattern)):
        # Extract model name from filename
        model = filepath.stem.split('_')[-1]
        files_to_analyze[model].append(str(filepath))

    # Analyze each model
    model_results = {}

    for model, filepaths in files_to_analyze.items():
        total_counts = Counter()
        file_details = []

        for filepath_str in filepaths:
            filepath = Path(filepath_str)
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping...")
                continue

            counts = analyze_run_file(filepath)
            total_counts.update(counts)

            file_details.append({
                'name': filepath.name,
                'counts': counts,
                'total': sum(counts.values())
            })

        model_results[model] = {
            'total_counts': total_counts,
            'file_details': file_details,
            'total_actions': sum(total_counts.values())
        }

    # Print results
    print("=" * 80)
    if len(sys.argv) > 1:
        print(f"ACTION BREAKDOWN ANALYSIS (pattern: {pattern})")
    else:
        print("ACTION BREAKDOWN ANALYSIS (all runs)")
    print("=" * 80)
    print()

    for model in ['gpt', 'claude', 'gemini']:
        if model not in model_results:
            continue

        results = model_results[model]
        total_counts = results['total_counts']
        total_actions = results['total_actions']

        print(f"Model: {model.upper()}")
        print(f"  Total files analyzed: {len(results['file_details'])}")
        print(f"  Total actions: {total_actions}")
        print()

        if total_actions > 0:
            print(f"  Overall Action Distribution:")
            for action in ['query', 'poll', 'lock_in', 'test_submit', 'abandon']:
                count = total_counts.get(action, 0)
                percentage = (count / total_actions) * 100
                if count > 0:  # Only show actions that occurred
                    print(f"    {action.upper():<12}: {count:>4} ({percentage:>5.1f}%)")

            # Check for any other unexpected actions
            tracked_actions = ['query', 'poll', 'lock_in', 'test_submit', 'abandon']
            other_actions = {k: v for k, v in total_counts.items()
                           if k not in tracked_actions}
            if other_actions:
                print(f"    OTHER       : {sum(other_actions.values()):>4} "
                      f"({sum(other_actions.values())/total_actions*100:>5.1f}%)")
                for action, count in other_actions.items():
                    print(f"      - {action}: {count}")

            # Verify totals
            counted = sum(total_counts.get(a, 0) for a in tracked_actions)
            if counted != total_actions:
                print(f"    ⚠️  WARNING: Counted actions ({counted}) != Total ({total_actions})")

        print()
        print(f"  Individual run files:")
        for file_info in results['file_details']:
            counts = file_info['counts']
            total = file_info['total']
            print(f"    {file_info['name']}")
            print(f"      Total actions: {total}")
            if total > 0:
                for action in ['query', 'poll', 'lock_in', 'test_submit', 'abandon']:
                    count = counts.get(action, 0)
                    percentage = (count / total) * 100
                    if count > 0:  # Only show actions that occurred
                        print(f"        {action.upper():<12}: {count:>3} ({percentage:>5.1f}%)")

        print()
        print("-" * 80)
        print()

    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Model':<10} {'Total':<8} {'Query':<15} {'Poll':<15} {'Lock In':<15} {'Test Submit':<15} {'Abandon':<15}")
    print("-" * 80)

    for model in ['gpt', 'claude', 'gemini']:
        if model not in model_results:
            continue

        results = model_results[model]
        total_counts = results['total_counts']
        total = results['total_actions']

        if total > 0:
            query_pct = (total_counts.get('query', 0) / total) * 100
            poll_pct = (total_counts.get('poll', 0) / total) * 100
            lock_in_pct = (total_counts.get('lock_in', 0) / total) * 100
            test_submit_pct = (total_counts.get('test_submit', 0) / total) * 100
            abandon_pct = (total_counts.get('abandon', 0) / total) * 100

            # Verify the sum
            counted = (total_counts.get('query', 0) + total_counts.get('poll', 0) +
                      total_counts.get('lock_in', 0) + total_counts.get('test_submit', 0) +
                      total_counts.get('abandon', 0))
            status = " ✓" if counted == total else f" ⚠️ ({counted}/{total})"

            print(f"{model.upper():<10} {total:<8} "
                  f"{total_counts.get('query', 0):>3} ({query_pct:>4.1f}%)   "
                  f"{total_counts.get('poll', 0):>3} ({poll_pct:>4.1f}%)   "
                  f"{total_counts.get('lock_in', 0):>3} ({lock_in_pct:>4.1f}%)   "
                  f"{total_counts.get('test_submit', 0):>3} ({test_submit_pct:>4.1f}%)   "
                  f"{total_counts.get('abandon', 0):>3} ({abandon_pct:>4.1f}%)"
                  f"{status}")

    print()


if __name__ == '__main__':
    main()
