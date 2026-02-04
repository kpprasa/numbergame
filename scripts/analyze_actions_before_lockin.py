#!/usr/bin/env python3
"""
Analyze average number of actions/turns before LOCK_IN for different models.

Usage:
    python analyze_actions_before_lockin.py              # Analyze all runs
    python analyze_actions_before_lockin.py 2026-02-01   # Filter to runs after date
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_timestamp_from_filename(filename: str) -> datetime:
    """Extract timestamp from filename like '20260201_144100_gemini.jsonl'"""
    parts = filename.split('_')
    if len(parts) >= 2:
        date_str = parts[0]  # 20260201
        time_str = parts[1]  # 144100
        # Parse: YYYYMMDD_HHMMSS
        dt_str = f"{date_str}_{time_str}"
        return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    return datetime.min


def analyze_run_file(filepath: Path) -> List[Dict]:
    """
    Analyze a single run file and extract game statistics.
    Returns list of game stats: [{game_idx, actions_before_lockin, outcome, ...}]
    """
    game_stats = {}  # game_idx -> stats

    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)

            # Skip metadata entries
            if entry.get('type') == 'metadata':
                continue

            # Process step entries
            if entry.get('type') == 'step':
                obs = entry.get('obs', {})
                game_idx = obs.get('game_index')

                if game_idx is None:
                    continue

                # Initialize game if we haven't seen it
                if game_idx not in game_stats:
                    game_stats[game_idx] = {
                        'game_idx': game_idx,
                        'actions': [],
                        'locked_in': False,
                        'outcome': None,
                        'actions_before_lockin': 0
                    }

                # Track the action taken in this step
                action = entry.get('action', {})
                action_type = action.get('type')

                if action_type:
                    game_stats[game_idx]['actions'].append(action_type)

                    # Check if this is a LOCK_IN
                    if action_type == 'LOCK_IN' and not game_stats[game_idx]['locked_in']:
                        game_stats[game_idx]['locked_in'] = True
                        # Count actions before this LOCK_IN (not including the LOCK_IN itself)
                        game_stats[game_idx]['actions_before_lockin'] = len(game_stats[game_idx]['actions']) - 1

                # Track outcome from result event
                result = entry.get('result', {})
                event = result.get('event', {})
                if event.get('type') == 'ROUND_END':
                    game_stats[game_idx]['outcome'] = event.get('outcome')

    return list(game_stats.values())


def main():
    import sys

    # Use relative path from repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    runs_dir = repo_root / 'runs'

    # Allow optional cutoff date from command line (format: YYYY-MM-DD)
    # Otherwise, process all runs
    cutoff_date = None
    if len(sys.argv) > 1:
        try:
            cutoff_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
            print(f"Filtering to runs after {sys.argv[1]}")
        except ValueError:
            print(f"Invalid date format. Use YYYY-MM-DD. Processing all runs.")
            cutoff_date = None

    # Group by model type
    model_stats = defaultdict(lambda: {
        'games': [],
        'total_actions': 0,
        'total_games': 0,
        'games_with_lockin': 0
    })

    # Process all JSONL files in runs directory
    for filepath in sorted(runs_dir.glob('*.jsonl')):
        timestamp = parse_timestamp_from_filename(filepath.name)

        # Filter to games after cutoff date if specified
        if cutoff_date and timestamp < cutoff_date:
            continue

        # Determine model type from filename
        model = None
        if 'gpt' in filepath.name.lower():
            model = 'GPT'
        elif 'claude' in filepath.name.lower():
            model = 'Claude'
        elif 'gemini' in filepath.name.lower():
            model = 'Gemini'
        else:
            continue

        print(f"Processing {filepath.name} ({model})...")

        # Analyze this run
        games = analyze_run_file(filepath)

        for game in games:
            if game['locked_in']:
                model_stats[model]['games'].append(game)
                model_stats[model]['total_actions'] += game['actions_before_lockin']
                model_stats[model]['total_games'] += 1
                model_stats[model]['games_with_lockin'] += 1

    # Print results
    print("\n" + "=" * 80)
    if cutoff_date:
        print(f"ANALYSIS: Average Actions Before LOCK_IN (Games after {cutoff_date.strftime('%Y-%m-%d')})")
    else:
        print("ANALYSIS: Average Actions Before LOCK_IN (All Games)")
    print("=" * 80)

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]

        if stats['games_with_lockin'] > 0:
            avg_actions = stats['total_actions'] / stats['games_with_lockin']

            print(f"\n{model}:")
            print(f"  Total games with LOCK_IN: {stats['games_with_lockin']}")
            print(f"  Average actions before LOCK_IN: {avg_actions:.2f}")

            # Show distribution
            actions_list = [g['actions_before_lockin'] for g in stats['games']]
            actions_list.sort()

            if actions_list:
                print(f"  Min: {min(actions_list)}")
                print(f"  Max: {max(actions_list)}")
                print(f"  Median: {actions_list[len(actions_list)//2]}")

                # Show histogram
                print(f"\n  Distribution of actions before LOCK_IN:")
                bins = defaultdict(int)
                for count in actions_list:
                    bin_key = (count // 2) * 2  # Group into bins of 2
                    bins[bin_key] += 1

                for bin_start in sorted(bins.keys()):
                    bin_end = bin_start + 1
                    count = bins[bin_start]
                    bar = '█' * count
                    print(f"    {bin_start:2d}-{bin_end:2d}: {bar} ({count})")
        else:
            print(f"\n{model}:")
            print(f"  No games with LOCK_IN found")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
