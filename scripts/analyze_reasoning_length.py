#!/usr/bin/env python3
"""Analyze average reasoning length per action across different model runs."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict


def count_words(text: str) -> int:
    """Count words in a string."""
    if not text:
        return 0
    return len(text.split())


def analyze_run_file(filepath: Path) -> Dict:
    """Analyze a single run file and return statistics."""
    word_counts = []

    with open(filepath, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "reasoning" in data and data["reasoning"]:
                    word_count = count_words(data["reasoning"])
                    word_counts.append(word_count)
            except json.JSONDecodeError:
                continue

    if not word_counts:
        return {
            "count": 0,
            "total_words": 0,
            "avg_words": 0,
            "min_words": 0,
            "max_words": 0,
        }

    return {
        "count": len(word_counts),
        "total_words": sum(word_counts),
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    runs_dir = repo_root / "runs"

    # Group by model
    model_stats = defaultdict(
        lambda: {
            "files": [],
            "total_actions": 0,
            "total_words": 0,
            "all_word_counts": [],
        }
    )
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*.jsonl"

    for filepath in sorted(runs_dir.glob(pattern)):
        # Extract model name from filename (last part before .jsonl)
        model = filepath.stem.split("_")[-1]

        stats = analyze_run_file(filepath)

        if stats["count"] > 0:
            model_stats[model]["files"].append({"name": filepath.name, "stats": stats})
            model_stats[model]["total_actions"] += stats["count"]
            model_stats[model]["total_words"] += stats["total_words"]

    # Print results
    print("=" * 80)
    print("REASONING LENGTH ANALYSIS BY MODEL")
    print("=" * 80)
    print()

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        print(f"Model: {model.upper()}")
        print(f"  Total run files: {len(stats['files'])}")
        print(f"  Total actions with reasoning: {stats['total_actions']}")
        print(f"  Total words: {stats['total_words']:,}")

        if stats["total_actions"] > 0:
            avg_words = stats["total_words"] / stats["total_actions"]
            print(f"  Average words per action: {avg_words:.2f}")
        else:
            print("  Average words per action: N/A")

        print()

        # Show individual file stats
        print("  Individual run files:")
        for file_info in stats["files"]:
            file_stats = file_info["stats"]
            print(f"    {file_info['name']}")
            print(
                f"      Actions: {file_stats['count']}, "
                f"Avg: {file_stats['avg_words']:.2f} words, "
                f"Range: {file_stats['min_words']}-{file_stats['max_words']} words"
            )
        print()
        print("-" * 80)
        print()

    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Model':<15} {'Files':<10} {'Actions':<15} {'Avg Words/Action':<20}")
    print("-" * 80)

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        if stats["total_actions"] > 0:
            avg_words = stats["total_words"] / stats["total_actions"]
            print(
                f"{model.upper():<15} {len(stats['files']):<10} "
                f"{stats['total_actions']:<15} {avg_words:<20.2f}"
            )

    print()


if __name__ == "__main__":
    main()
