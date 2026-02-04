#!/usr/bin/env python3
"""
Analyze performance: time spent per successful game.
Lower is better (more efficient).
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
        dt_str = f"{date_str}_{time_str}"
        return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    return datetime.min


def analyze_run_file(filepath: Path) -> Dict:
    """
    Analyze a single run file.
    Returns overall stats and per-game breakdown.
    """
    game_data = {}  # game_idx -> {first_step_time, outcome, end_time}

    with open(filepath, 'r') as f:
        lines = list(f)

    prev_game_end_time = 0  # Track when previous game ended

    for i, line in enumerate(lines):
        entry = json.loads(line)

        if entry.get('type') != 'step':
            continue

        obs = entry.get('obs', {})
        game_idx = obs.get('game_index')

        if game_idx is None:
            continue

        cumulative = entry.get('cumulative', {})
        time_spent = cumulative.get('time_spent', 0)

        # Initialize game data if first time seeing this game
        if game_idx not in game_data:
            game_data[game_idx] = {
                'first_step_time': time_spent,
                'outcome': None,
                'end_time': None
            }

        # Check for game end event
        result = entry.get('result', {})
        event = result.get('event', {})

        if event.get('type') == 'ROUND_END':
            game_data[game_idx]['outcome'] = event.get('outcome')
            # The time spent is AFTER the action, so this is the cumulative time when game ended
            game_data[game_idx]['end_time'] = time_spent

    # Calculate time per game
    # For each game, time = (cumulative time after its ROUND_END) - (cumulative time after previous game's ROUND_END)
    successful_games = []
    failed_games = []
    total_time = 0

    prev_end_time = 0
    for game_idx in sorted(game_data.keys()):
        data = game_data[game_idx]
        outcome = data['outcome']
        end_time = data['end_time']

        if end_time is None:
            continue  # Game didn't finish

        time_for_game = end_time - prev_end_time

        game_info = {
            'game_idx': game_idx,
            'outcome': outcome,
            'time': time_for_game,
            'cumulative_time': end_time
        }

        # A "successful" game is one where we locked in (right or wrong) or submitted a test
        if outcome and outcome not in ['TIMEOUT', 'ABANDON_RIGHTFUL', 'ABANDON_WRONGFUL']:
            successful_games.append(game_info)
            total_time += time_for_game
        else:
            failed_games.append(game_info)

        prev_end_time = end_time

    return {
        'successful_games': successful_games,
        'failed_games': failed_games,
        'num_successful': len(successful_games),
        'num_failed': len(failed_games),
        'total_time_on_successful': total_time,
        'avg_time_per_successful': total_time / len(successful_games) if successful_games else 0,
    }


def main():
    import sys

    # Use relative path from repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    runs_dir = repo_root / 'runs'

    # Allow optional cutoff date from command line (format: YYYY-MM-DD)
    cutoff_date = None
    if len(sys.argv) > 1:
        try:
            cutoff_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
            print(f"Filtering to runs after {sys.argv[1]}")
        except ValueError:
            print(f"Invalid date format. Use YYYY-MM-DD. Processing all runs.")
            cutoff_date = None

    # Group by model type
    model_results = defaultdict(lambda: {
        'runs': [],
        'total_successful_games': 0,
        'total_time_on_successful': 0,
        'test_correct': 0,
        'test_wrong': 0,
        'lockin_correct': 0,
        'lockin_wrong': 0
    })

    # Process all JSONL files
    for filepath in sorted(runs_dir.glob('*.jsonl')):
        timestamp = parse_timestamp_from_filename(filepath.name)

        # Filter to games after cutoff date if specified
        if cutoff_date and timestamp < cutoff_date:
            continue

        # Determine model type
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
        stats = analyze_run_file(filepath)

        model_results[model]['runs'].append({
            'filename': filepath.name,
            'stats': stats
        })

        model_results[model]['total_successful_games'] += stats['num_successful']
        model_results[model]['total_time_on_successful'] += stats['total_time_on_successful']

        # Count outcome types
        for game in stats['successful_games']:
            outcome = game['outcome']
            if 'TEST_SUBMIT_CORRECT' in outcome:
                model_results[model]['test_correct'] += 1
            elif 'TEST_SUBMIT_WRONG' in outcome:
                model_results[model]['test_wrong'] += 1
            elif 'LOCK_IN_CORRECT' in outcome:
                model_results[model]['lockin_correct'] += 1
            elif 'LOCK_IN_WRONG' in outcome:
                model_results[model]['lockin_wrong'] += 1

    # Print results
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS: Time Per Successful Game (Lower is Better)")
    if cutoff_date:
        print(f"After {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        print("All Games")
    print("=" * 80)

    results_summary = []

    for model in sorted(model_results.keys()):
        data = model_results[model]

        if data['total_successful_games'] > 0:
            avg_time = data['total_time_on_successful'] / data['total_successful_games']

            results_summary.append({
                'model': model,
                'avg_time': avg_time,
                'num_runs': len(data['runs']),
                'total_games': data['total_successful_games'],
                'test_correct': data['test_correct'],
                'test_wrong': data['test_wrong'],
                'lockin_correct': data['lockin_correct'],
                'lockin_wrong': data['lockin_wrong']
            })

    # Sort by avg_time (lower is better)
    results_summary.sort(key=lambda x: x['avg_time'])

    for i, result in enumerate(results_summary, 1):
        print(f"\n{i}. {result['model']}:")
        print(f"   Average time per successful game: {result['avg_time']:.2f} time units")
        print(f"   Total successful games: {result['total_games']} (across {result['num_runs']} runs)")
        print(f"   Outcomes:")
        print(f"     - TEST_SUBMIT_CORRECT: {result['test_correct']}")
        print(f"     - TEST_SUBMIT_WRONG: {result['test_wrong']}")
        print(f"     - LOCK_IN_CORRECT: {result['lockin_correct']}")
        print(f"     - LOCK_IN_WRONG: {result['lockin_wrong']}")

        total_outcomes = result['test_correct'] + result['test_wrong'] + result['lockin_correct'] + result['lockin_wrong']
        if total_outcomes > 0:
            test_success_rate = result['test_correct'] / (result['test_correct'] + result['test_wrong']) * 100 if (result['test_correct'] + result['test_wrong']) > 0 else 0
            print(f"     - Test success rate: {test_success_rate:.1f}%")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  - Lower avg time = more efficient (completes games faster)")
    print("  - 'Successful game' = any game ending in LOCK_IN or TEST_SUBMIT")
    print("  - Excludes TIMEOUT and ABANDON from this metric")
    print("=" * 80)


if __name__ == '__main__':
    main()
