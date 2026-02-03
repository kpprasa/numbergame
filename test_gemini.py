"""LLM evaluation harness for numbergame using Google Gemini."""

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
from google import genai
from google.genai import types
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

    google_api_key: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class GameAction(BaseModel):
    """LLM action output with structured format."""

    type: Literal["POLL_PASSIVE", "POLL_ACTIVE", "LOCK_IN", "TEST_SUBMIT", "ABANDON"]
    reasoning: str = Field(description="Brief explanation of why you chose this action")

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

    result = {"type": event.type}

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

    sorted_outcomes = sorted(outcome_counts.items(), key=lambda x: (-x[1], x[0]))

    for outcome, count in sorted_outcomes:
        pct = (count / total) * 100 if total > 0 else 0
        bar_length = int(pct / 2)
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


def build_prompt(messages: List[Dict[str, str]], system_prompt: str) -> str:
    """Flatten chat history into a single prompt for Gemini."""
    parts = [f"SYSTEM:\n{system_prompt}\n"]
    for m in messages:
        if m["role"] == "system":
            continue
        role = m["role"].upper()
        parts.append(f"{role}:\n{m['content']}\n")
    parts.append("MODEL:")  # Gemini will complete after this marker.
    return "\n".join(parts)


def call_llm(
    client: genai.Client,
    messages: List[Dict[str, str]],
    model: str,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 8,
) -> GameAction:
    """Call Google Gemini with structured output and retry on rate limits.

    Uses aggressive exponential backoff to handle rate limits properly:
    - Retry 1: ~30-45s
    - Retry 2: ~60-90s
    - Retry 3: ~120-180s
    - Retry 4: ~240-360s (capped at 300s max)
    """
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    prompt = build_prompt(messages, system_prompt)

    config_dict = {
        "response_mime_type": "application/json",
        "response_json_schema": GameAction.model_json_schema(),
    }

    if reasoning_effort:
        config_dict["thinking_config"] = types.ThinkingConfig(
            thinking_level=reasoning_effort, include_thoughts=True
        )

    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config_dict,
            )

            # Collect thoughts and action text separately
            thoughts = []
            action_text = None

            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thoughts.append(part.text)
                else:
                    action_text = part.text

            if action_text:
                action = GameAction.model_validate_json(action_text)
            else:
                action = GameAction.model_validate_json(response.text)

            if action is None:
                raise ValueError("API returned None for action")

            if thoughts:
                thinking_summary = " ".join(thoughts)
                action.reasoning = (
                    f"{action.reasoning} |Thinking ->| {thinking_summary}"
                )

            return action

        except Exception as e:
            # Check if it's a rate limit error
            error_str = str(e).lower()
            is_rate_limit = (
                "rate" in error_str
                or "429" in error_str
                or "quota" in error_str
                or "resource_exhausted" in error_str
            )

            if not is_rate_limit:
                # Not a rate limit error
                print(
                    f"Error during LLM call (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    raise
                # Shorter backoff for non-rate-limit errors
                base_wait = min(60, 5 * (2**attempt))
                jitter = random.uniform(0.5, 1.5)
                wait_time = base_wait * jitter
                print(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # Rate limit error
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


def write_jsonl_line(file_handle, data: Dict[str, Any]) -> None:
    """Write a single JSONL line to file."""
    file_handle.write(json.dumps(data) + "\n")
    file_handle.flush()


def run_episode(
    env: LUTEnv,
    env_config: EnvConfig,
    client: genai.Client,
    llm_config: Dict[str, Any],
    interactive: bool,
    jsonl_file,
    run_id: str,
) -> Dict[str, Any]:
    """Run a single episode (imperative shell)."""
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

        messages.append({"role": "user", "content": obs_text})

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

        env_action = action_to_env_action(llm_action)
        action_dict = asdict(env_action)

        if interactive:
            print(f"\nLLM reasoning: {llm_action.reasoning}")
            print(f"Action: {action_dict}")

        result = env.step(env_action)
        obs_dict = result.observation.to_dict()
        done = result.done

        event_feedback = format_event(result.event)
        if event_feedback:
            messages.append({"role": "user", "content": event_feedback})
            if interactive:
                print(event_feedback)

        cumulative["time_spent"] = obs_dict.get("time_spent", 0)
        cumulative["time_remaining"] = obs_dict.get("time_remaining", 0)
        cumulative["reward"] = obs_dict.get("accuracy", 0)
        cumulative["meta"] = obs_dict.get("meta", 0)
        cumulative["games_completed"] = env.games_completed

        result_dict = {
            "done": result.done,
            "info": result.info,
            "event": serialize_event(result.event),
        }

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
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation on numbergame with Gemini"
    )
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
        help="Output JSONL file path (default: runs/{timestamp}_gemini.jsonl)",
    )
    args = parser.parse_args()

    settings = Settings()

    config = load_config(args.config)
    env_config_dict = config["env"]
    llm_config = config["llm"]

    # Validate Gemini model
    supported_models = ["gemini-3-pro-preview", "gemini-3-flash-preview"]
    if llm_config["model"] not in supported_models:
        raise ValueError(
            f"Unsupported Gemini model: {llm_config['model']}. "
            f"Supported models: {', '.join(supported_models)}"
        )

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

    env = LUTEnv(env_config, rng=random.Random(env_config_dict.get("seed")))

    if args.output:
        output_path = Path(args.output)
    else:
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = runs_dir / f"{timestamp}_gemini.jsonl"

    client = genai.Client(api_key=settings.google_api_key)

    run_id = str(env_config_dict.get("seed", "default"))

    with open(output_path, "w") as f:
        metadata_line = create_jsonl_line(
            run_id=run_id,
            line_type="metadata",
            metadata={"env_config": env_config_dict, "llm_config": llm_config},
        )
        write_jsonl_line(f, metadata_line)

        final_metrics = run_episode(
            env, env_config, client, llm_config, args.interactive, f, run_id
        )

    games_played = final_metrics.get("games_completed", 0)
    max_possible_points = games_played * env_config.base_reward

    print("\n" + "=" * 60)
    print("EPISODE COMPLETE")
    print("=" * 60)
    print(f"Games played: {games_played}")
    print(f"Time spent: {final_metrics['time_spent']}/{env_config_dict['lifetime']}")
    print(f"Reward: {final_metrics['reward']}")
    print(f"Max possible points: {max_possible_points}")
    print(f"Meta: {final_metrics['meta']}")

    print(format_outcome_frequency_chart(env.outcome_counts))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
