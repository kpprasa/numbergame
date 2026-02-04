#!/usr/bin/env python3
"""
Extract raw reasoning from Claude runs without formatting.
Just the pure reasoning text for each game.
"""

import json
from pathlib import Path


def extract_raw_reasoning(filepath: Path, output_file: Path):
    """Extract raw reasoning from a run file."""

    with open(filepath, 'r') as f_in, open(output_file, 'w') as f_out:
        f_out.write(f"# RAW REASONING: {filepath.name}\n")
        f_out.write("=" * 80 + "\n\n")

        current_game = None
        step_count = 0

        for line in f_in:
            entry = json.loads(line)

            # Get metadata
            if entry.get('type') == 'metadata':
                metadata = entry.get('metadata', {})
                seed = metadata.get('env_config', {}).get('seed')
                f_out.write(f"Seed: {seed}\n")
                f_out.write(f"Model: {metadata.get('llm_config', {}).get('model')}\n")
                f_out.write(f"Reasoning Effort: {metadata.get('llm_config', {}).get('reasoning_effort')}\n\n")
                continue

            if entry.get('type') != 'step':
                continue

            obs = entry.get('obs', {})
            game_idx = obs.get('game_index')

            if game_idx is None:
                continue

            # New game
            if current_game is None or current_game != game_idx:
                if current_game is not None:
                    f_out.write("\n" + "=" * 80 + "\n\n")

                current_game = game_idx
                step_count = 0
                f_out.write(f"GAME {game_idx}\n")
                f_out.write("-" * 80 + "\n\n")

            # Get action and reasoning
            action = entry.get('action', {})
            action_type = action.get('type')
            reasoning = entry.get('reasoning', '')

            if not reasoning:
                continue

            # Track outcome
            result = entry.get('result', {})
            event = result.get('event', {})
            outcome = event.get('outcome') if event.get('type') == 'ROUND_END' else None

            # Write step
            f_out.write(f"Step {step_count}: {action_type}\n")
            if outcome:
                f_out.write(f"OUTCOME: {outcome}\n")
            f_out.write("\n")
            f_out.write(reasoning)
            f_out.write("\n\n")
            f_out.write("-" * 40 + "\n\n")

            step_count += 1


def main():
    import sys

    # Use relative paths from repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    runs_dir = repo_root / 'runs'
    output_dir = repo_root / 'analysis'
    output_dir.mkdir(exist_ok=True)

    # Get run files from command line args or process all runs
    if len(sys.argv) > 1:
        # Process specific files provided as arguments
        run_files = sys.argv[1:]
    else:
        # Process all JSONL files in runs directory
        run_files = [f.name for f in sorted(runs_dir.glob('*.jsonl'))]

    if not run_files:
        print("No run files found. Provide files as arguments or ensure runs/ directory has .jsonl files.")
        return

    for input_file in run_files:
        # Handle both full paths and just filenames
        if '/' in input_file:
            input_path = Path(input_file)
        else:
            input_path = runs_dir / input_file

        if not input_path.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue

        # Generate output filename from input filename
        output_file = input_path.stem + '_raw_reasoning.md'
        output_path = output_dir / output_file

        print(f"Extracting reasoning from {input_path.name}...")
        extract_raw_reasoning(input_path, output_path)
        print(f"  → Saved to {output_file}")
        print(f"  → Size: {output_path.stat().st_size} bytes")

    print("\n✓ Extraction complete")


if __name__ == '__main__':
    main()
