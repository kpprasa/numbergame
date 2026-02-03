"""LLM evaluation harness for numbergame."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from numbergame import (
    Abandon,
    AnyAction,
    EnvConfig,
    LockIn,
    LUTEnv,
    PollActive,
    PollPassive,
    TestSubmit,
    format_event,
    format_instructions,
    format_observation,
)

# ============================================================================
# Settings
# ============================================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class GameAction(BaseModel):
    """LLM action output with structured format."""

    type: Literal["POLL_PASSIVE", "POLL_ACTIVE", "LOCK_IN", "TEST_SUBMIT", "ABANDON"]
    reasoning: str = Field(
        description="Scratchpad for what were the steps to reach this action"
    )

    # Optional fields for different action types
    a: Optional[int] = Field(
        None, description="First input for POLL_ACTIVE (required if type=POLL_ACTIVE)"
    )
    b: Optional[int] = Field(
        None, description="Second input for POLL_ACTIVE (required if type=POLL_ACTIVE)"
    )
    answer: Optional[int] = Field(
        None, description="Predicted answer for LOCK_IN (required if type=LOCK_IN)"
    )
    predictions: Optional[List[int]] = Field(
        None,
        description="List of predictions for TEST_SUBMIT (required if type=TEST_SUBMIT)",
    )


# ============================================================================
# Functional Core - Pure Functions
# ============================================================================


def action_to_env_action(action: GameAction) -> AnyAction:
    """Convert Pydantic action model to numbergame action object."""
    if action.type == "POLL_PASSIVE":
        return PollPassive()
    elif action.type == "POLL_ACTIVE":
        return PollActive(a=action.a, b=action.b)
    elif action.type == "LOCK_IN":
        return LockIn(answer=action.answer)
    elif action.type == "TEST_SUBMIT":
        return TestSubmit(predictions=tuple(action.predictions))
    elif action.type == "ABANDON":
        return Abandon()
    else:
        raise ValueError(f"Unknown action type: {action.type}")


def create_jsonl_line(
    run_id: str,
    line_type: str,
    step_idx: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    obs: Optional[Dict[str, Any]] = None,
    action: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    cumulative: Optional[Dict[str, Any]] = None,
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a JSONL line dict (pure function)."""
    line = {"run_id": run_id, "type": line_type}

    if line_type == "metadata":
        line["timestamp"] = datetime.now().isoformat()
        line["metadata"] = metadata
    elif line_type == "step":
        line["step_idx"] = step_idx
        line["obs"] = obs
        line["action"] = action
        if reasoning:
            line["reasoning"] = reasoning
        line["result"] = result
        line["cumulative"] = cumulative

    return line


def serialize_event(event: Any) -> Dict[str, Any]:
    """Serialize event object to dict, handling non-JSON-serializable fields."""
    if not hasattr(event, "type"):
        return {"type": str(type(event).__name__)}

    # Manually construct dict based on event type
    result = {"type": event.type}

    # Add fields based on what's in the event
    if hasattr(event, "action_type"):
        result["action_type"] = event.action_type
    if hasattr(event, "reason"):
        result["reason"] = event.reason
    if hasattr(event, "heldout_k"):
        result["heldout_k"] = event.heldout_k
    if hasattr(event, "outcome"):
        result["outcome"] = event.outcome
    if hasattr(event, "reward_delta"):
        result["reward_delta"] = event.reward_delta
    if hasattr(event, "meta_delta"):
        result["meta_delta"] = event.meta_delta
    if hasattr(event, "generator") and event.generator is not None:
        # Don't serialize the full generator, just note its type
        result["generator_type"] = type(event.generator).__name__

    return result


def format_outcome_frequency_chart(outcome_counts: Dict[str, int]) -> str:
    """Format outcome frequency chart as a string."""
    if not outcome_counts:
        return "No outcomes recorded."

    total = sum(outcome_counts.values())
    lines = []
    lines.append("\nOutcome Frequency Chart:")
    lines.append("-" * 60)

    # Sort by count (descending) then by name
    sorted_outcomes = sorted(outcome_counts.items(), key=lambda x: (-x[1], x[0]))

    for outcome, count in sorted_outcomes:
        pct = (count / total) * 100 if total > 0 else 0
        bar_length = int(pct / 2)  # Scale to max 50 chars
        bar = "█" * bar_length
        lines.append(f"{outcome:30} {count:3} ({pct:5.1f}%) {bar}")

    lines.append("-" * 60)
    lines.append(f"{'Total':30} {total:3}")
    return "\n".join(lines)


# ============================================================================
# Imperative Shell - I/O and Side Effects
# ============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def call_llm(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 8,
) -> GameAction:
    """Call OpenAI API with structured output and retry on rate limits.

    Uses aggressive exponential backoff to handle rate limits properly:
    - Retry 1: ~30-45s
    - Retry 2: ~60-90s
    - Retry 3: ~120-180s
    - Retry 4: ~240-360s (capped at 300s max)
    """
    kwargs = {
        "model": model,
        "input": messages,
        "text_format": GameAction,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}

    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            response = client.responses.parse(**kwargs)
            action = response.output_parsed

            if action is None:
                raise ValueError("API returned None for output_parsed")

            # Extract reasoning summaries from response.output
            if reasoning_effort and hasattr(response, "output"):
                reasoning_summaries = []
                for item in response.output:
                    # Check if this is a reasoning block
                    item_dict = item if isinstance(item, dict) else item.model_dump()
                    if item_dict.get("type") == "reasoning" and "summary" in item_dict:
                        for summary_item in item_dict["summary"]:
                            if summary_item.get("type") == "summary_text":
                                reasoning_summaries.append(summary_item.get("text", ""))

                if reasoning_summaries:
                    thinking_summary = " ".join(reasoning_summaries)
                    action.reasoning = (
                        f"{action.reasoning} |Thinking ->| {thinking_summary}"
                    )

            return action

        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            # Aggressive exponential backoff for rate limits
            # Base wait: 30s, 60s, 120s, 240s, 300s (capped)
            base_wait = min(300, 30 * (2**attempt))
            # Add jitter: multiply by 0.5-1.5x
            jitter = random.uniform(0.5, 1.5)
            wait_time = base_wait * jitter
            print(
                f"Rate limit hit. Waiting {wait_time:.1f}s before retry "
                f"{attempt + 1}/{max_retries}..."
            )
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error during LLM call (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            # Shorter backoff for non-rate-limit errors
            base_wait = min(60, 5 * (2**attempt))
            jitter = random.uniform(0.5, 1.5)
            wait_time = base_wait * jitter
            print(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)


def write_jsonl_line(file_handle, data: Dict[str, Any]) -> None:
    """Write a single JSONL line to file."""
    file_handle.write(json.dumps(data) + "\n")
    file_handle.flush()


def run_episode(
    env: LUTEnv,
    env_config: EnvConfig,
    client: OpenAI,
    llm_config: Dict[str, Any],
    interactive: bool,
    jsonl_file,
    run_id: str,
) -> Dict[str, Any]:
    """Run a single episode (imperative shell)."""
    # Create system prompt with game instructions
    system_prompt = format_instructions(env_config)

    messages = [{"role": "system", "content": system_prompt}]

    obs_dict = env.reset().to_dict()
    done = False
    step_idx = 0

    cumulative = {
        "time_spent": 0,
        "time_remaining": obs_dict.get("lifetime_time_budget", 0),
        "reward": 0,
        "meta": 0,
        "games_completed": 0,
    }

    if interactive:
        print(system_prompt)
        print("\n" + "=" * 60)
        print("Starting interactive evaluation...")
        print("=" * 60)

    while not done:
        # Format observation for LLM
        obs_text = format_observation(obs_dict)

        if interactive:
            print(f"\n{'=' * 60}")
            print(f"STEP {step_idx}")
            print("=" * 60)
            print(obs_text)
            print("\nPress ENTER for next LLM action (or 'q' to quit)...")
            user_input = input("> ")
            if user_input.strip().lower() == "q":
                print("Exiting...")
                break

        # Add observation to messages
        messages.append({"role": "user", "content": obs_text})

        # Call LLM
        try:
            llm_action = call_llm(
                client,
                messages,
                llm_config["model"],
                llm_config.get("reasoning_effort"),
            )
        except Exception as e:
            print(f"LLM error: {e}")
            break

        # Convert to env action
        env_action = action_to_env_action(llm_action)
        action_dict = asdict(env_action)

        if interactive:
            print(f"\nLLM reasoning: {llm_action.reasoning}")
            print(f"Action: {action_dict}")

        # Step environment
        result = env.step(env_action)
        obs_dict = result.observation.to_dict()
        done = result.done

        # Format event feedback (what the human would see)
        event_feedback = format_event(result.event)
        if event_feedback:
            messages.append({"role": "user", "content": event_feedback})
            if interactive:
                print(event_feedback)

        # Update cumulative metrics
        cumulative["time_spent"] = obs_dict.get("time_spent", 0)
        cumulative["time_remaining"] = obs_dict.get("time_remaining", 0)
        cumulative["reward"] = obs_dict.get("accuracy", 0)
        cumulative["meta"] = obs_dict.get("meta", 0)
        cumulative["games_completed"] = env.games_completed

        # Serialize result for JSONL
        result_dict = {
            "done": result.done,
            "info": result.info,
            "event": serialize_event(result.event),
        }

        # Write JSONL line
        jsonl_line = create_jsonl_line(
            run_id=run_id,
            line_type="step",
            step_idx=step_idx,
            obs=obs_dict,
            action=action_dict,
            result=result_dict,
            cumulative=cumulative,
            reasoning=llm_action.reasoning,
        )
        write_jsonl_line(jsonl_file, jsonl_line)

        if interactive:
            print(f"\nResult: {result.info.get('msg', 'OK')}")
            print(
                f"Event: {result.event.type if hasattr(result.event, 'type') else result.event}"
            )
            print(
                f"Cumulative - Time: {cumulative['time_spent']}/{obs_dict.get('lifetime_time_budget', '?')}, "
                f"Reward: {cumulative['reward']}, Meta: {cumulative['meta']}"
            )

        step_idx += 1

    return cumulative


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation on numbergame")
    parser.add_argument(
        "--config",
        type=str,
        default="eval_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive step-through mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: runs/{timestamp}_gpt.jsonl)",
    )
    args = parser.parse_args()

    # Load settings from environment
    settings = Settings()

    # Load config
    config = load_config(args.config)
    env_config_dict = config["env"]
    llm_config = config["llm"]

    # Validate GPT model
    model = llm_config["model"]
    if not (model == "gpt-5.2" or model.startswith("gpt-5.2-")):
        raise ValueError(
            f"Unsupported GPT model: {model}. "
            f"Supported models: gpt-5.2, gpt-5.2-YYYY-MM-DD (dated versions)"
        )

    # Create EnvConfig
    env_config = EnvConfig(
        n=env_config_dict["n"],
        seed_examples=env_config_dict["seed_examples"],
        lifetime_time_budget=env_config_dict["lifetime"],
        c_poll=env_config_dict["c_poll"],
        c_active=env_config_dict["c_active"],
        c_lock=env_config_dict["c_lock"],
        c_quit=env_config_dict["c_quit"],
        active_includes_passive=env_config_dict["active_includes_passive"],
        q_illposed=env_config_dict["q_illposed"],
        difficulty=env_config_dict["difficulty"],
        max_keys=env_config_dict["max_keys"],
        generator_archetype=env_config_dict["generator"],
        min_tree_depth=env_config_dict.get("min_tree_depth", 2),
        heldout_k=env_config_dict["heldout_k"],
        reward_info=env_config_dict["reward_info"],
    )

    # Create environment
    env = LUTEnv(env_config, rng=random.Random(env_config_dict.get("seed")))

    # Setup output file
    if args.output:
        output_path = Path(args.output)
    else:
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = runs_dir / f"{timestamp}_gpt.jsonl"

    # Create OpenAI client
    client = OpenAI(api_key=settings.openai_api_key)

    # Run episode
    run_id = str(env_config_dict.get("seed", "default"))

    with open(output_path, "w") as f:
        # Write metadata line
        metadata_line = create_jsonl_line(
            run_id=run_id,
            line_type="metadata",
            metadata={"env_config": env_config_dict, "llm_config": llm_config},
        )
        write_jsonl_line(f, metadata_line)

        # Run episode
        final_metrics = run_episode(
            env, env_config, client, llm_config, args.interactive, f, run_id
        )

    # Print summary
    games_played = final_metrics.get("games_completed", 0)
    # If the last game wasn't completed (timeout), games_played is accurate
    # If the last game was completed, games_completed reflects number of finished games
    max_possible_points = games_played * env_config.base_reward

    print("\n" + "=" * 60)
    print("EPISODE COMPLETE")
    print("=" * 60)
    print(f"Games played: {games_played}")
    print(f"Time spent: {final_metrics['time_spent']}/{env_config_dict['lifetime']}")
    print(f"Reward: {final_metrics['reward']}")
    print(f"Max possible points: {max_possible_points}")
    print(f"Meta: {final_metrics['meta']}")

    # Print outcome frequency chart
    print(format_outcome_frequency_chart(env.outcome_counts))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
