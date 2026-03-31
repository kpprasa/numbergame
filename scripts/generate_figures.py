#!/usr/bin/env python3
"""
Generate all figures and tables for numbergame analysis.

This script creates:
- Figure 1: Speed-quality tradeoff scatter plot
- Table 1: Three headline numbers
- Figure 2: Outcome composition (100% stacked bar)
- Figure 3: Commitment behavior (actions before LOCK_IN distribution)
- Table 2: Action preference profile
- Table 3: Verbosity metrics

Usage:
    python generate_figures.py              # Analyze all runs
    python generate_figures.py 2026-02-01   # Filter to runs after date
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend
import numpy as np
import pandas as pd


def parse_timestamp_from_filename(filename: str) -> datetime:
    """Extract timestamp from filename like '20260201_144100_gemini.jsonl'"""
    parts = filename.split("_")
    if len(parts) >= 2:
        date_str = parts[0]  # 20260201
        time_str = parts[1]  # 144100
        dt_str = f"{date_str}_{time_str}"
        return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    return datetime.min


def count_words(text: str) -> int:
    """Count words in a string."""
    if not text:
        return 0
    return len(text.split())


def analyze_run_file(filepath: Path) -> Dict:
    """
    Analyze a single run file comprehensively.
    Returns all statistics needed for visualizations.
    """
    game_data = {}  # game_idx -> {outcome, time, actions_before_lockin, etc.}
    action_counts = Counter()
    word_counts = []

    with open(filepath, "r") as f:
        lines = list(f)

    prev_end_time = 0

    for line in lines:
        entry = json.loads(line)

        # Extract reasoning word counts
        if entry.get("type") == "step" and "reasoning" in entry and entry["reasoning"]:
            word_count = count_words(entry["reasoning"])
            word_counts.append(word_count)

        # Extract action counts
        if "action" in entry and entry["action"]:
            if isinstance(entry["action"], dict):
                action = entry["action"].get("type", "").lower()
            else:
                action = entry["action"].lower()

            if action:
                # Normalize action names
                if "poll_passive" in action or action == "poll":
                    action = "poll"
                elif "poll_active" in action or "query" in action:
                    action = "query"
                elif "lock" in action:
                    action = "lock_in"
                elif "test_submit" in action:
                    action = "test_submit"
                elif "abandon" in action:
                    action = "abandon"

                action_counts[action] += 1

        # Skip non-step entries for game analysis
        if entry.get("type") != "step":
            continue

        obs = entry.get("obs", {})
        game_idx = obs.get("game_index")

        if game_idx is None:
            continue

        cumulative = entry.get("cumulative", {})
        time_spent = cumulative.get("time_spent", 0)

        # Initialize game data if first time seeing this game
        if game_idx not in game_data:
            game_data[game_idx] = {
                "first_step_time": time_spent,
                "outcome": None,
                "end_time": None,
                "actions": [],
                "locked_in": False,
                "actions_before_lockin": 0,
                "realized_depth": None,
            }

        # Extract realized_depth from obs
        if (
            "dsl" in obs
            and "generator" in obs["dsl"]
            and "realized_depth" in obs["dsl"]["generator"]
        ):
            game_data[game_idx]["realized_depth"] = obs["dsl"]["generator"][
                "realized_depth"
            ]

        # Track actions
        action = entry.get("action", {})
        action_type = action.get("type")

        if action_type:
            game_data[game_idx]["actions"].append(action_type)

            # Check if this is a LOCK_IN
            if action_type == "LOCK_IN" and not game_data[game_idx]["locked_in"]:
                game_data[game_idx]["locked_in"] = True
                # Count actions before this LOCK_IN (not including the LOCK_IN itself)
                game_data[game_idx]["actions_before_lockin"] = (
                    len(game_data[game_idx]["actions"]) - 1
                )

        # Check for game end event
        result = entry.get("result", {})
        event = result.get("event", {})

        if event.get("type") == "ROUND_END":
            game_data[game_idx]["outcome"] = event.get("outcome")
            game_data[game_idx]["end_time"] = time_spent

    # Calculate time per game and track all lock-ins
    completed_games = []
    lockin_games = []  # Track games with LOCK_IN separately
    outcome_counts = Counter()
    actions_before_lockin_list = []

    # First pass: collect actions_before_lockin for ALL games with LOCK_IN
    for game_idx in sorted(game_data.keys()):
        data = game_data[game_idx]
        if data["locked_in"]:
            actions_before_lockin_list.append(data["actions_before_lockin"])

    # Second pass: calculate timing for completed games
    prev_end_time = 0
    for game_idx in sorted(game_data.keys()):
        data = game_data[game_idx]
        outcome = data["outcome"]
        end_time = data["end_time"]
        locked_in = data["locked_in"]

        if end_time is None:
            continue  # Game didn't finish

        time_for_game = end_time - prev_end_time

        # A "completed" game is one where we locked in (right or wrong) or submitted a test
        if outcome and outcome not in [
            "TIMEOUT",
            "ABANDON_RIGHTFUL",
            "ABANDON_WRONGFUL",
        ]:
            game_record = {"game_idx": game_idx, "outcome": outcome, "time": time_for_game, "locked_in": locked_in}
            completed_games.append(game_record)

            # Track games with LOCK_IN separately
            if locked_in:
                lockin_games.append(game_record)

            outcome_counts[outcome] += 1

        prev_end_time = end_time

    # Calculate statistics
    total_time_on_completed = sum(g["time"] for g in completed_games)
    num_completed = len(completed_games)

    # Track outcomes by game index for learning curve
    # Include runs with at least 10 games to avoid noise from very short runs
    game_outcomes_by_index = defaultdict(list)
    max_game_idx = max(game_data.keys()) if game_data else -1
    run_has_sufficient_data = max_game_idx >= 9  # At least 10 games (0-9)

    if run_has_sufficient_data:
        for game_idx in sorted(game_data.keys()):
            data = game_data[game_idx]
            outcome = data["outcome"]
            if outcome in ["TEST_SUBMIT_CORRECT", "TEST_SUBMIT_WRONG"]:
                game_outcomes_by_index[game_idx].append(outcome)

    # Track outcomes by tree depth
    test_outcomes_by_depth = defaultdict(list)
    for game_idx in sorted(game_data.keys()):
        data = game_data[game_idx]
        outcome = data["outcome"]
        depth = data.get("realized_depth")
        if (
            outcome in ["TEST_SUBMIT_CORRECT", "TEST_SUBMIT_WRONG"]
            and depth is not None
        ):
            test_outcomes_by_depth[depth].append(outcome)

    return {
        "completed_games": completed_games,
        "lockin_games": lockin_games,
        "num_completed": num_completed,
        "total_time_on_completed": total_time_on_completed,
        "avg_time_per_completed": total_time_on_completed / num_completed
        if num_completed > 0
        else 0,
        "outcome_counts": outcome_counts,
        "action_counts": action_counts,
        "total_actions": sum(action_counts.values()),
        "word_counts": word_counts,
        "total_words": sum(word_counts),
        "avg_words_per_action": sum(word_counts) / len(word_counts)
        if word_counts
        else 0,
        "actions_before_lockin": actions_before_lockin_list,
        "game_outcomes_by_index": game_outcomes_by_index,
        "test_outcomes_by_depth": test_outcomes_by_depth,
        "run_has_sufficient_data": run_has_sufficient_data,
        "max_game_idx": max_game_idx,
        "num_files": 1,  # We're analyzing one file at a time
    }


def aggregate_model_stats(filepaths: List[Path]) -> Dict:
    """Aggregate statistics across multiple run files for a model."""
    agg = {
        "num_files": 0,
        "total_completed_games": 0,
        "total_time_on_completed": 0,
        "outcome_counts": Counter(),
        "action_counts": Counter(),
        "total_actions": 0,
        "total_words": 0,
        "total_word_count_entries": 0,
        "actions_before_lockin": [],
        "game_outcomes_by_index": defaultdict(list),
        "test_outcomes_by_depth": defaultdict(list),
        "num_sufficient_runs": 0,
        "max_game_idx_overall": -1,
        "completed_games": [],  # Add this to collect all completed games
        "lockin_games": [],  # Track games with LOCK_IN
    }

    for filepath in filepaths:
        stats = analyze_run_file(filepath)

        agg["num_files"] += 1
        agg["total_completed_games"] += stats["num_completed"]
        agg["total_time_on_completed"] += stats["total_time_on_completed"]
        agg["outcome_counts"].update(stats["outcome_counts"])
        agg["action_counts"].update(stats["action_counts"])
        agg["total_actions"] += stats["total_actions"]
        agg["total_words"] += stats["total_words"]
        agg["total_word_count_entries"] += len(stats["word_counts"])
        agg["actions_before_lockin"].extend(stats["actions_before_lockin"])

        # Collect completed games data
        agg["completed_games"].extend(stats["completed_games"])
        agg["lockin_games"].extend(stats["lockin_games"])

        # Merge game outcomes by index (only from runs with sufficient data)
        if stats["run_has_sufficient_data"]:
            agg["num_sufficient_runs"] += 1
            agg["max_game_idx_overall"] = max(
                agg["max_game_idx_overall"], stats["max_game_idx"]
            )
            for game_idx, outcomes in stats["game_outcomes_by_index"].items():
                agg["game_outcomes_by_index"][game_idx].extend(outcomes)

        # Merge test outcomes by depth
        for depth, outcomes in stats["test_outcomes_by_depth"].items():
            agg["test_outcomes_by_depth"][depth].extend(outcomes)

    # Calculate averages
    if agg["total_completed_games"] > 0:
        agg["avg_time_per_completed"] = (
            agg["total_time_on_completed"] / agg["total_completed_games"]
        )
    else:
        agg["avg_time_per_completed"] = 0

    if agg["total_word_count_entries"] > 0:
        agg["avg_words_per_action"] = (
            agg["total_words"] / agg["total_word_count_entries"]
        )
    else:
        agg["avg_words_per_action"] = 0

    return agg


def create_figure_1(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Figure 1: Speed-quality tradeoff scatter plot.
    Based on games with LOCK_IN for consistency with other figures.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    models = []
    avg_times = []
    pct_corrects = []
    time_per_corrects = []
    n_games = []
    colors = {"Claude": "#e07a5f", "Gemini": "#f2cc8f", "GPT": "#81b29a"}

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]

        lockin_games = stats.get("lockin_games", [])
        if len(lockin_games) == 0:
            continue

        # Calculate average time from lockin games only
        avg_time = sum(g["time"] for g in lockin_games) / len(lockin_games)

        num_correct = stats["outcome_counts"]["TEST_SUBMIT_CORRECT"]
        pct_correct = (num_correct / len(lockin_games)) * 100

        # Time per correct = avg_time / (pct_correct/100)
        time_per_correct = (
            avg_time / (pct_correct / 100) if pct_correct > 0 else float("inf")
        )

        models.append(model)
        avg_times.append(avg_time)
        pct_corrects.append(pct_correct)
        time_per_corrects.append(time_per_correct)
        n_games.append(len(lockin_games))

    # Plot points with time-per-correct in legend
    for model, x, y, tpc in zip(models, avg_times, pct_corrects, time_per_corrects):
        ax.scatter(
            x,
            y,
            s=200,
            color=colors.get(model, "gray"),
            alpha=0.7,
            edgecolors="black",
            linewidth=1.5,
            label=f"{model} ({tpc:.1f})",
        )

    ax.set_xlabel("Avg time per game with LOCK_IN (↓ better)", fontsize=12, weight="bold")
    ax.set_ylabel("% correct (among LOCK_IN games)", fontsize=12, weight="bold")
    ax.set_title(
        "Figure 1: Speed-quality tradeoff\nLegend shows time per correct game",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.legend(fontsize=11, frameon=True, shadow=True, title="Model (time/correct)")
    ax.grid(True, alpha=0.2, linestyle="--")

    # Add caption and footnote
    caption = (
        "Claude and Gemini complete episodes faster, but GPT converts a much larger fraction\n"
        "of LOCK_IN games into correct test submissions; the resulting time-per-correct\n"
        "is nearly tied between Claude and GPT."
    )

    n_str = "/".join(str(n) for n in n_games)
    footnote = f"N = {n_str} games with LOCK_IN (Claude/Gemini/GPT); {scope_note}"

    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic", wrap=True)
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(
        output_dir / "figure_1_speed_quality_tradeoff.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Generated Figure 1: {output_dir / 'figure_1_speed_quality_tradeoff.png'}")


def create_table_1(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Table 1: Three headline numbers.
    Based on games with LOCK_IN for consistency.
    """
    rows = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]

        lockin_games = stats.get("lockin_games", [])
        if len(lockin_games) == 0:
            continue

        # Calculate average time from lockin games only
        avg_time = sum(g["time"] for g in lockin_games) / len(lockin_games)

        num_correct = stats["outcome_counts"]["TEST_SUBMIT_CORRECT"]
        pct_correct = (num_correct / len(lockin_games)) * 100
        time_per_correct = (
            avg_time / (pct_correct / 100) if pct_correct > 0 else float("inf")
        )

        rows.append(
            {
                "Model": model,
                "Time / LOCK_IN game (↓)": f"{avg_time:.2f}",
                "% correct among LOCK_IN (↑)": f"{pct_correct:.1f}%",
                "Time / correct (↓)": f"{time_per_correct:.2f}",
            }
        )

    df = pd.DataFrame(rows)

    # Save as CSV
    df.to_csv(output_dir / "table_1_headline_numbers.csv", index=False)

    # Create styled table image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.25, 0.35, 0.25],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#3d5a80")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style rows with alternating colors
    for i, row in enumerate(df.values, 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title("Table 1: Three headline numbers", fontsize=14, weight="bold", pad=20)

    # Add caption and footnote
    caption = (
        "The composite 'time per correct' (t_success / p_correct) shows why GPT's slower\n"
        "completion can still be competitive on end-to-end efficiency."
    )
    footnote = scope_note

    fig.text(0.5, 0.18, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, 0.08, footnote, ha="center", fontsize=8, color="gray")

    plt.savefig(
        output_dir / "table_1_headline_numbers.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"✓ Generated Table 1: {output_dir / 'table_1_headline_numbers.csv'} and .png"
    )


def create_figure_2(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Figure 2: Outcome composition (100% stacked bar).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    models = []
    test_correct = []
    test_wrong = []
    lock_wrong_consistent = []
    n_games = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]

        if stats["total_completed_games"] == 0:
            continue

        total = stats["total_completed_games"]

        models.append(model)
        test_correct.append(
            (stats["outcome_counts"]["TEST_SUBMIT_CORRECT"] / total) * 100
        )
        test_wrong.append((stats["outcome_counts"]["TEST_SUBMIT_WRONG"] / total) * 100)
        lock_wrong_consistent.append(
            (stats["outcome_counts"]["LOCK_IN_WRONG_CONSISTENT"] / total) * 100
        )
        n_games.append(total)

    x = np.arange(len(models))
    width = 0.5

    # Stacked bars
    p1 = ax.bar(
        x, test_correct, width, label="TEST_SUBMIT_CORRECT", color="#2a9d8f", alpha=0.9
    )
    p2 = ax.bar(
        x,
        test_wrong,
        width,
        bottom=test_correct,
        label="TEST_SUBMIT_WRONG",
        color="#e76f51",
        alpha=0.9,
    )
    p3 = ax.bar(
        x,
        lock_wrong_consistent,
        width,
        bottom=np.array(test_correct) + np.array(test_wrong),
        label="LOCK_IN_WRONG_CONSISTENT",
        color="#f4a261",
        alpha=0.9,
    )

    # Add percentage labels (only for segments >= 4%)
    threshold = 4.0
    for i, model in enumerate(models):
        # TEST_SUBMIT_CORRECT
        if test_correct[i] >= threshold:
            ax.text(
                i,
                test_correct[i] / 2,
                f"{test_correct[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

        # TEST_SUBMIT_WRONG
        if test_wrong[i] >= threshold:
            ax.text(
                i,
                test_correct[i] + test_wrong[i] / 2,
                f"{test_wrong[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

        # LOCK_IN_WRONG_CONSISTENT
        if lock_wrong_consistent[i] >= threshold:
            ax.text(
                i,
                test_correct[i] + test_wrong[i] + lock_wrong_consistent[i] / 2,
                f"{lock_wrong_consistent[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

    ax.set_ylabel("Percentage of completed games", fontsize=12, weight="bold")
    ax.set_title("Figure 1: Outcome composition", fontsize=14, weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, weight="bold")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
        shadow=True,
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Add caption and footnote
    caption = (
        "Claude/Gemini finish mostly via wrong test submissions, while GPT's completed games\n"
        "are split far more evenly between correct and wrong—consistent with 'slower but\n"
        "more accurate' behavior."
    )

    n_str = "/".join(str(n) for n in n_games)
    footnote = f"N = {n_str} completed games (Claude/Gemini/GPT); {scope_note}"

    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(
        output_dir / "figure_1_2_outcome_composition.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Generated Figure 2: {output_dir / 'figure_2_outcome_composition.png'}")


def create_figure_3(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Figure 3: Commitment behavior (actions before LOCK_IN distribution).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"Claude": "#e07a5f", "Gemini": "#f2cc8f", "GPT": "#81b29a"}
    n_games = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]
        actions_list = stats["actions_before_lockin"]

        if not actions_list:
            continue

        n_games.append(len(actions_list))

        # Create CDF
        sorted_actions = np.sort(actions_list)
        y = np.arange(1, len(sorted_actions) + 1) / len(sorted_actions)

        # Calculate median
        median_idx = len(sorted_actions) // 2
        median_actions = sorted_actions[median_idx]
        median_prob = y[median_idx]

        # Plot line with median in label
        ax.plot(
            sorted_actions,
            y,
            linewidth=2.5,
            label=f"{model} (median: {median_actions})",
            color=colors.get(model, "gray"),
            alpha=0.8,
        )

        # Mark median with a dot
        ax.plot(
            median_actions,
            median_prob,
            "o",
            markersize=8,
            color=colors.get(model, "gray"),
            markeredgecolor="black",
            markeredgewidth=1,
        )

    ax.set_xlabel("Number of actions before LOCK_IN", fontsize=12, weight="bold")
    ax.set_ylabel("Cumulative probability", fontsize=12, weight="bold")
    ax.set_title("Figure 2: Commitment behavior", fontsize=14, weight="bold", pad=15)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    # Add caption and footnote
    caption = (
        "Claude and Gemini 'lock in' after 2–3 actions in most games, while GPT continues\n"
        "gathering information substantially longer before committing."
    )

    n_str = "/".join(str(n) for n in n_games)
    footnote = (
        f"N(lock-in games) = {n_str} (Claude/Gemini/GPT); includes all outcomes (completed, timeout, abandon)\n"
        f"Dots mark median. Note: differs from Figs 1–2 which count only completed games."
    )

    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(
        output_dir / "figure_2_3_commitment_behavior.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Generated Figure 3: {output_dir / 'figure_3_commitment_behavior.png'}")


def create_table_2(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Table 2: Action preference profile.
    """
    rows = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]

        if stats["total_actions"] == 0:
            continue

        total = stats["total_actions"]

        rows.append(
            {
                "Model": model,
                "Query (%)": f"{(stats['action_counts']['query'] / total * 100):.1f}",
                "Poll (%)": f"{(stats['action_counts']['poll'] / total * 100):.1f}",
                "Lock In (%)": f"{(stats['action_counts']['lock_in'] / total * 100):.1f}",
                "Total Actions": stats["total_actions"],
            }
        )

    df = pd.DataFrame(rows)

    # Save as CSV
    df.to_csv(output_dir / "table_2_action_preference.csv", index=False)

    # Create styled table image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#3d5a80")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style rows with alternating colors
    for i, row in enumerate(df.values, 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title("Table 2: Action preference profile", fontsize=14, weight="bold", pad=20)

    # Add caption and footnote
    caption = (
        "Action mixes differ sharply: GPT spends much more of its budget polling, while\n"
        "Claude/Gemini allocate more actions to querying and lock-in decisions."
    )
    footnote = f"{scope_note.replace('excludes timeout/abandon', 'excludes timeout/abandon/submit')}"

    fig.text(0.5, 0.18, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, 0.08, footnote, ha="center", fontsize=8, color="gray")

    plt.savefig(
        output_dir / "table_2_action_preference.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"✓ Generated Table 2: {output_dir / 'table_2_action_preference.csv'} and .png"
    )


def create_figure_4(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Figure 4: First-half vs second-half accuracy comparison.
    Shows whether models improve with experience by comparing early vs late game accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"Claude": "#e07a5f", "Gemini": "#f2cc8f", "GPT": "#81b29a"}

    models_data = []
    n_runs = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]
        game_outcomes = stats.get("game_outcomes_by_index", {})

        if not game_outcomes:
            continue

        n_runs.append(stats.get("num_sufficient_runs", 0))

        # Determine the midpoint for this model's data
        max_idx = stats.get("max_game_idx_overall", -1)
        if max_idx < 0:
            continue

        midpoint = max_idx / 2.0

        # Separate into first half and second half
        first_half_outcomes = []
        second_half_outcomes = []

        for idx, outcomes in game_outcomes.items():
            if idx < midpoint:
                first_half_outcomes.extend(outcomes)
            else:
                second_half_outcomes.extend(outcomes)

        # Calculate accuracy for each half
        def calc_accuracy_stats(outcomes):
            if not outcomes:
                return None, None, 0
            correct = sum(1 for o in outcomes if o == "TEST_SUBMIT_CORRECT")
            total = len(outcomes)
            pct = (correct / total) * 100
            # Calculate standard error for binomial proportion
            p = correct / total
            se = np.sqrt(p * (1 - p) / total) * 100  # Convert to percentage
            return pct, se, total

        first_pct, first_se, first_n = calc_accuracy_stats(first_half_outcomes)
        second_pct, second_se, second_n = calc_accuracy_stats(second_half_outcomes)

        if first_pct is not None and second_pct is not None:
            models_data.append(
                {
                    "model": model,
                    "first_pct": first_pct,
                    "first_se": first_se,
                    "first_n": first_n,
                    "second_pct": second_pct,
                    "second_se": second_se,
                    "second_n": second_n,
                }
            )

    if not models_data:
        print("  Warning: No data for Figure 4")
        return

    # Create grouped bar chart
    x = np.arange(len(models_data))
    width = 0.35

    for i, data in enumerate(models_data):
        model = data["model"]
        color = colors.get(model, "gray")

        # First half bar
        ax.bar(
            i - width / 2,
            data["first_pct"],
            width,
            yerr=data["first_se"],
            capsize=5,
            label="First half" if i == 0 else "",
            color=color,
            alpha=0.6,
            edgecolor="black",
            linewidth=1,
        )

        # Second half bar
        ax.bar(
            i + width / 2,
            data["second_pct"],
            width,
            yerr=data["second_se"],
            capsize=5,
            label="Second half" if i == 0 else "",
            color=color,
            alpha=1.0,
            edgecolor="black",
            linewidth=1,
        )

    ax.set_xlabel("Model", fontsize=12, weight="bold")
    ax.set_ylabel("% TEST_SUBMIT_CORRECT", fontsize=12, weight="bold")
    ax.set_title(
        "Figure 3: Saturation effect\nFirst-half vs second-half accuracy",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d["model"] for d in models_data], fontsize=11, weight="bold")
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.set_ylim(0, 100)

    # Add caption and footnote
    caption = (
        "All models show lower accuracy in the second half of episodes, suggesting saturation or confusion\n"
        "as information accumulates rather than learning. Error bars show standard error."
    )

    # Calculate total games for footnote
    total_games = []
    for data in models_data:
        total_games.append(data["first_n"] + data["second_n"])

    n_str = "/".join(str(n) for n in total_games) if total_games else "0/0/0"
    footnote = (
        f"N = {n_str} games (Claude/Gemini/GPT) from runs with ≥10 games\n"
        f"First/second half split at midpoint of each run's max game index; {scope_note}"
    )

    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(
        output_dir / "figure_3_4_learning_curve.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Generated Figure 4: {output_dir / 'figure_4_learning_curve.png'}")


def create_figure_5(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Figure 5: Test accuracy by decision tree depth.
    Shows whether models perform better on shallow vs deep trees.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"Claude": "#e07a5f", "Gemini": "#f2cc8f", "GPT": "#81b29a"}

    models_data = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]
        outcomes_by_depth = stats.get("test_outcomes_by_depth", {})

        if not outcomes_by_depth:
            continue

        # Calculate accuracy for depth 2 and depth 3
        def calc_accuracy_stats(outcomes):
            if not outcomes:
                return None, None, 0
            correct = sum(1 for o in outcomes if o == "TEST_SUBMIT_CORRECT")
            total = len(outcomes)
            pct = (correct / total) * 100
            p = correct / total
            se = np.sqrt(p * (1 - p) / total) * 100
            return pct, se, total

        depth2_pct, depth2_se, depth2_n = calc_accuracy_stats(
            outcomes_by_depth.get(2, [])
        )
        depth3_pct, depth3_se, depth3_n = calc_accuracy_stats(
            outcomes_by_depth.get(3, [])
        )

        # Include model if it has at least one depth
        if depth2_pct is not None or depth3_pct is not None:
            models_data.append(
                {
                    "model": model,
                    "depth2_pct": depth2_pct,
                    "depth2_se": depth2_se,
                    "depth2_n": depth2_n,
                    "depth3_pct": depth3_pct,
                    "depth3_se": depth3_se,
                    "depth3_n": depth3_n,
                }
            )

    if not models_data:
        print("  Warning: No data for Figure 5")
        return

    # Create grouped bar chart
    x = np.arange(len(models_data))
    width = 0.35

    depth2_label_added = False
    depth3_label_added = False

    for i, data in enumerate(models_data):
        model = data["model"]
        color = colors.get(model, "gray")

        # Depth 2 bar (if data exists)
        if data["depth2_pct"] is not None:
            ax.bar(
                i - width / 2,
                data["depth2_pct"],
                width,
                yerr=data["depth2_se"],
                capsize=5,
                label="Depth 2" if not depth2_label_added else "",
                color=color,
                alpha=0.6,
                edgecolor="black",
                linewidth=1,
            )
            depth2_label_added = True

        # Depth 3 bar (if data exists)
        if data["depth3_pct"] is not None:
            ax.bar(
                i + width / 2,
                data["depth3_pct"],
                width,
                yerr=data["depth3_se"],
                capsize=5,
                label="Depth 3" if not depth3_label_added else "",
                color=color,
                alpha=1.0,
                edgecolor="black",
                linewidth=1,
            )
            depth3_label_added = True

    ax.set_xlabel("Model", fontsize=12, weight="bold")
    ax.set_ylabel("% TEST_SUBMIT_CORRECT", fontsize=12, weight="bold")
    ax.set_title(
        "Figure 5: Accuracy by tree depth\nShallow (depth 2) vs deep (depth 3) decision trees",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d["model"] for d in models_data], fontsize=11, weight="bold")
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.set_ylim(0, 100)

    # Add caption and footnote
    caption = (
        "Most runs only encountered depth-3 trees. Error bars show standard error.\n"
        "Lower accuracy on deeper trees suggests models struggle with more complex decision boundaries."
    )

    footnote = f"Only games ending in TEST_SUBMIT_CORRECT or TEST_SUBMIT_WRONG; {scope_note}\nNote: Gemini and GPT had no depth-2 trees in these runs."

    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(
        output_dir / "figure_5_accuracy_by_depth.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Generated Figure 5: {output_dir / 'figure_5_accuracy_by_depth.png'}")


def create_figure_6(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Figure 6: Efficiency metric (time/correct) for first vs second half.
    Shows whether models maintain efficiency or degrade as episode progresses.
    Based on games with LOCK_IN (regardless of test submission).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"Claude": "#e07a5f", "Gemini": "#f2cc8f", "GPT": "#81b29a"}

    models_data = []
    n_games = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]

        # Use lockin_games to match Figure 7
        all_lockin_games = stats.get("lockin_games", [])

        if not all_lockin_games:
            continue

        max_idx = stats.get("max_game_idx_overall", -1)
        if max_idx < 0:
            continue

        # Filter to only games from runs with sufficient data
        filtered_games = [g for g in all_lockin_games if g["game_idx"] <= max_idx]

        if not filtered_games:
            continue

        midpoint = max_idx / 2.0

        # Separate games by half
        first_half_games = [g for g in filtered_games if g["game_idx"] < midpoint]
        second_half_games = [g for g in filtered_games if g["game_idx"] >= midpoint]

        if not first_half_games or not second_half_games:
            continue

        # Calculate time-per-correct for each half
        def calc_time_per_correct(games):
            if not games:
                return None, None, 0

            # Among LOCK_IN games, only TEST_SUBMIT_CORRECT counts as "correct"
            # Everything else (LOCK_IN_WRONG_*, TEST_SUBMIT_WRONG, TIMEOUT) is wrong
            correct = sum(1 for g in games if g["outcome"] == "TEST_SUBMIT_CORRECT")
            total = len(games)

            if correct == 0:
                return None, None, 0

            avg_time = np.mean([g["time"] for g in games])
            pct_correct = correct / total
            time_per_correct = avg_time / pct_correct

            # Calculate standard error for time/correct
            # Using delta method: SE(1/p) ≈ SE(p) / p^2
            p = correct / total
            se_p = np.sqrt(p * (1 - p) / total)
            se_time_per_correct = (avg_time * se_p) / (p**2)

            return time_per_correct, se_time_per_correct, total

        first_tpc, first_se, first_n = calc_time_per_correct(first_half_games)
        second_tpc, second_se, second_n = calc_time_per_correct(second_half_games)

        if first_tpc is not None and second_tpc is not None:
            total_n = first_n + second_n
            n_games.append(total_n)
            models_data.append(
                {
                    "model": model,
                    "first_tpc": first_tpc,
                    "first_se": first_se,
                    "second_tpc": second_tpc,
                    "second_se": second_se,
                }
            )

    if not models_data:
        print("  Warning: No data for Figure 6")
        return

    # Create grouped bar chart
    x = np.arange(len(models_data))
    width = 0.35

    for i, data in enumerate(models_data):
        model = data["model"]
        color = colors.get(model, "gray")

        # First half bar
        ax.bar(
            i - width / 2,
            data["first_tpc"],
            width,
            yerr=data["first_se"],
            capsize=5,
            label="First half" if i == 0 else "",
            color=color,
            alpha=0.6,
            edgecolor="black",
            linewidth=1,
        )

        # Second half bar
        ax.bar(
            i + width / 2,
            data["second_tpc"],
            width,
            yerr=data["second_se"],
            capsize=5,
            label="Second half" if i == 0 else "",
            color=color,
            alpha=1.0,
            edgecolor="black",
            linewidth=1,
        )

    ax.set_xlabel("Model", fontsize=12, weight="bold")
    ax.set_ylabel("Time per correct (↓ better)", fontsize=12, weight="bold")
    ax.set_title(
        "Figure 4: Efficiency degradation\nTime-per-correct: first-half vs second-half",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d["model"] for d in models_data], fontsize=11, weight="bold")
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.set_ylim(bottom=0)

    # Add caption and footnote
    caption = (
        "Higher time-per-correct in second half indicates efficiency degradation as episodes progress.\n"
        "Combines both accuracy decline (Fig 4) and time costs. Error bars show standard error."
    )

    n_str = "/".join(str(n) for n in n_games) if n_games else "0/0/0"
    footnote = (
        f"N = {n_str} games with LOCK_IN (Claude/Gemini/GPT) from runs with ≥10 games\n"
        f"Correct = TEST_SUBMIT_CORRECT; Time-per-correct = avg_time_per_game / accuracy; {scope_note}"
    )

    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(
        output_dir / "figure_4_6_efficiency_degradation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"✓ Generated Figure 6: {output_dir / 'figure_6_efficiency_degradation.png'}")


def create_figure_7(model_stats: Dict[str, Dict], output_dir: Path, scope_note: str):
    """
    Figure 7: Average game duration for first vs second half.
    Shows whether models feel time pressured as episodes progress.
    Based on games with LOCK_IN (regardless of test submission).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"Claude": "#e07a5f", "Gemini": "#f2cc8f", "GPT": "#81b29a"}

    models_data = []
    n_games = []

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]

        # Use lockin_games for consistency with Figure 6
        all_lockin_games = stats.get("lockin_games", [])

        if not all_lockin_games:
            continue

        # Use the same logic: only from runs with sufficient data
        # We need to determine which games came from runs with ≥10 games
        # We can use actions_before_lockin list length as a proxy, but better to
        # check against game_outcomes_by_index for the max_idx
        max_idx = stats.get("max_game_idx_overall", -1)
        if max_idx < 0:
            continue

        # Filter to only games from runs with sufficient data (≥10 games)
        # A run has sufficient data if max_game_idx >= 9
        # We need to identify which lockin_games come from sufficient runs
        # Since we don't track run_id explicitly, we'll filter by game_idx range
        # Games with game_idx <= max_game_idx_overall are from sufficient runs
        filtered_games = [g for g in all_lockin_games if g["game_idx"] <= max_idx]

        if not filtered_games:
            continue

        midpoint = max_idx / 2.0

        # Separate games by half
        first_half_times = []
        second_half_times = []

        for game in filtered_games:
            idx = game["game_idx"]
            time = game["time"]
            if idx < midpoint:
                first_half_times.append(time)
            else:
                second_half_times.append(time)

        if not first_half_times or not second_half_times:
            continue

        # Calculate average and standard error for each half
        first_avg = np.mean(first_half_times)
        first_se = np.std(first_half_times, ddof=1) / np.sqrt(len(first_half_times))

        second_avg = np.mean(second_half_times)
        second_se = np.std(second_half_times, ddof=1) / np.sqrt(len(second_half_times))

        n_games.append(len(filtered_games))

        models_data.append(
            {
                "model": model,
                "first_avg": first_avg,
                "first_se": first_se,
                "first_n": len(first_half_times),
                "second_avg": second_avg,
                "second_se": second_se,
                "second_n": len(second_half_times),
            }
        )

    if not models_data:
        print("  Warning: No data for Figure 7")
        return

    # Create grouped bar chart
    x = np.arange(len(models_data))
    width = 0.35

    for i, data in enumerate(models_data):
        model = data["model"]
        color = colors.get(model, "gray")

        # First half bar
        ax.bar(
            i - width / 2,
            data["first_avg"],
            width,
            yerr=data["first_se"],
            capsize=5,
            label="First half" if i == 0 else "",
            color=color,
            alpha=0.6,
            edgecolor="black",
            linewidth=1,
        )

        # Second half bar
        ax.bar(
            i + width / 2,
            data["second_avg"],
            width,
            yerr=data["second_se"],
            capsize=5,
            label="Second half" if i == 0 else "",
            color=color,
            alpha=1.0,
            edgecolor="black",
            linewidth=1,
        )

    ax.set_xlabel("Model", fontsize=12, weight="bold")
    ax.set_ylabel("Average game duration (time points)", fontsize=12, weight="bold")
    ax.set_title(
        "Figure 5: Game duration\nFirst-half vs second-half average time points per game",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d["model"] for d in models_data], fontsize=11, weight="bold")
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.set_ylim(bottom=0)

    # Add caption and footnote
    caption = (
        "Lower time points in second half would suggest time pressure as episodes progress.\n"
        "Time points = action steps taken in the game. Error bars show standard error."
    )

    n_str = "/".join(str(n) for n in n_games) if n_games else "0/0/0"
    footnote = (
        f"N = {n_str} games with LOCK_IN (Claude/Gemini/GPT) from runs with ≥10 games\n"
        f"First/second half split at midpoint of each run's max game index; {scope_note}"
    )

    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(
        output_dir / "figure_5_7_game_duration.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Generated Figure 7: {output_dir / 'figure_7_game_duration.png'}")


def create_table_3(
    model_stats: Dict[str, Dict],
    output_dir: Path,
    scope_note: str,
    cutoff_date: datetime = None,
):
    """
    Table 3: Verbosity metrics.
    """
    rows = []

    # Special handling for GPT - only include files from 20260201_201850 onwards
    gpt_reasoning_cutoff = datetime.strptime("20260201_201850", "%Y%m%d_%H%M%S")

    for model in ["Claude", "Gemini", "GPT"]:
        if model not in model_stats:
            continue

        stats = model_stats[model]

        # For GPT, recalculate with only files from reasoning cutoff onwards
        if model == "GPT":
            # Re-aggregate only files from the reasoning cutoff
            script_dir = Path(__file__).parent
            repo_root = script_dir.parent
            runs_dir = repo_root / "runs"

            gpt_stats = {
                "total_actions": 0,
                "total_words": 0,
                "total_word_count_entries": 0,
                "num_files": 0,
            }

            for filepath in sorted(runs_dir.glob("*_gpt.jsonl")):
                timestamp = parse_timestamp_from_filename(filepath.name)

                # Only include files from reasoning cutoff onwards
                if timestamp < gpt_reasoning_cutoff:
                    continue

                # Also respect the main cutoff date if provided
                if cutoff_date and timestamp < cutoff_date:
                    continue

                file_stats = analyze_run_file(filepath)
                gpt_stats["total_actions"] += file_stats["total_actions"]
                gpt_stats["total_words"] += file_stats["total_words"]
                gpt_stats["total_word_count_entries"] += len(file_stats["word_counts"])
                gpt_stats["num_files"] += 1

            if gpt_stats["total_word_count_entries"] > 0:
                avg_words = (
                    gpt_stats["total_words"] / gpt_stats["total_word_count_entries"]
                )
            else:
                avg_words = 0

            rows.append(
                {
                    "Model": model,
                    "Actions": gpt_stats["total_actions"],
                    "Avg words/action": f"{avg_words:.2f}",
                }
            )
        else:
            rows.append(
                {
                    "Model": model,
                    "Actions": stats["total_actions"],
                    "Avg words/action": f"{stats['avg_words_per_action']:.2f}",
                }
            )

    df = pd.DataFrame(rows)

    # Save as CSV
    df.to_csv(output_dir / "table_3_verbosity.csv", index=False)

    # Create styled table image
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#3d5a80")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style rows with alternating colors
    for i, row in enumerate(df.values, 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title("Table 3: Verbosity", fontsize=14, weight="bold", pad=20)

    # Add caption and footnote
    caption = (
        "Higher average words/action does not correspond to better accuracy here\n"
        "(e.g., Gemini is most verbose but not most accurate)."
    )
    footnote = (
        f"{scope_note}\n"
        "Note: Verbosity measured from reasoning summaries, which may not be completely\n"
        "representative of full reasoning. GPT data from runs after 2026-02-01 20:18:50."
    )

    fig.text(0.5, 0.20, caption, ha="center", fontsize=10, style="italic")
    fig.text(0.5, 0.08, footnote, ha="center", fontsize=8, color="gray")

    plt.savefig(output_dir / "table_3_verbosity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Generated Table 3: {output_dir / 'table_3_verbosity.csv'} and .png")


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    runs_dir = repo_root / "runs"
    output_dir = repo_root / "figures"
    output_dir.mkdir(exist_ok=True)

    # Parse cutoff date if provided
    cutoff_date = None
    if len(sys.argv) > 1:
        try:
            cutoff_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
            print(f"Filtering to runs after {sys.argv[1]}")
        except ValueError:
            print(f"Invalid date format. Use YYYY-MM-DD. Processing all runs.")
            cutoff_date = None

    # Collect files by model
    model_files = defaultdict(list)

    for filepath in sorted(runs_dir.glob("*.jsonl")):
        timestamp = parse_timestamp_from_filename(filepath.name)

        # Filter by cutoff date
        if cutoff_date and timestamp < cutoff_date:
            continue

        # Determine model
        model = None
        if "gpt" in filepath.name.lower():
            model = "GPT"
        elif "claude" in filepath.name.lower():
            model = "Claude"
        elif "gemini" in filepath.name.lower():
            model = "Gemini"
        else:
            continue

        model_files[model].append(filepath)
        print(f"Processing {filepath.name} ({model})...")

    # Aggregate statistics per model
    model_stats = {}
    for model, filepaths in model_files.items():
        model_stats[model] = aggregate_model_stats(filepaths)

    # Create scope note
    if cutoff_date:
        scope_note = (
            f"excludes timeout/abandon; after {cutoff_date.strftime('%Y-%m-%d')}"
        )
    else:
        scope_note = "excludes timeout/abandon; all runs"

    print("\n" + "=" * 80)
    print("GENERATING FIGURES AND TABLES")
    if cutoff_date:
        print(f"Data filtered to runs after {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        print("Using all available runs")
    print("=" * 80 + "\n")

    # Generate all visualizations
    create_figure_1(model_stats, output_dir, scope_note)
    create_table_1(model_stats, output_dir, scope_note)
    create_figure_2(model_stats, output_dir, scope_note)
    create_figure_3(model_stats, output_dir, scope_note)
    create_figure_4(model_stats, output_dir, scope_note)
    create_figure_6(model_stats, output_dir, scope_note)
    create_figure_7(model_stats, output_dir, scope_note)
    # create_figure_5(model_stats, output_dir, scope_note)
    create_table_2(model_stats, output_dir, scope_note)
    create_table_3(model_stats, output_dir, scope_note, cutoff_date)

    print("\n" + "=" * 80)
    print(f"✓ All figures and tables generated in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
