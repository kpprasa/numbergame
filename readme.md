# Numbergame

## Motivation
This game aims to be the simplest game that has some of the interesting properties of life: 
- handling imperfect information (confidence in internal models), 
- enough structure that there is information gain from "similar" examples, 
- varying costs according to the difficulty of the action, 
- and the possibility that the game you are playing isn't the right one.

To be useful as an evaluation environment, I wanted to make the game easy to verify but extremely difficult to solve (to avoid it feeling like primarily a math/computation problem).


**Numbergame** is a rule-inference / active-learning game.  
Each round hides a deterministic operator `#` that maps integer pairs `(a, b)` to an output via a structured generator (e.g. affine rules or decision trees over latent features). The player gathers limited evidence through passive samples or targeted queries, then predicts a designated target; correct predictions may unlock a held-out generalization test to reward true inference over memorization. Some rounds are intentionally ill-posed, making abandonment a valid scored strategy.

## Game Mechanics

Objective: Learn a hidden binary operator # from examples
 and predict outputs for target queries within a limited
time budget.

Setup:
- Hidden rule maps integer pairs (a, b) to outputs using
a key-based lookup system
- Players start each game with 3 seed examples and a
target query to solve
- Shared time budget of 50 units across all games in a
run

Available Actions:
- POLL_PASSIVE (1 time): Receive a random example
- POLL_ACTIVE (2 time): Query a specific pair of your
choice
- LOCK_IN (1 time): Submit your answer for the target
query
- TEST_SUBMIT (0 time): After correct lock-in, predict K
held-out queries
- ABANDON (1 time): Quit if the game appears ill-posed
(contradictory)

Scoring:
- +100 points: Correct lock-in AND all test predictions
correct
- 0 points: Wrong lock-in (but consistent) OR failed test
 phase
- -100 points: Wrong lock-in inconsistent with observed
data
- ±1 meta: Rightful/wrongful abandons

Difficulty Levels (1-5):
- Level 1: Full formula revealed, keys shown with
examples
- Level 2: Feature names given but order unknown
- Level 3: Must identify which features from a pool are
used
- Level 4: Only key structure revealed (black-box
features)
- Level 5: Pure black-box, no structural hints

Strategic Challenge: Players must balance exploration
(gathering examples) vs. exploitation (completing games)
under time pressure, while detecting unsolvable games and
 generalizing learned rules to novel queries.
 
## Running the game

With UV (recommended):

```bash
uv run numbergame.py --play
```

Or with additional options:

```bash
uv run numbergame.py --play --generator tree --heldout_k 2 --difficulty 1
```

Alternatively, run directly with Python:

```bash
python numbergame.py --play
```

## LLM Evaluation

Test LLM agents on numbergame.

**Setup**: Create a `.env` file with your OpenAI API key:
```bash
cp .env.template .env
# Edit .env and add your key
```

**Batch mode** (runs to completion):
```bash
uv run test_gpt.py
```

**Interactive mode** (step through with debugger):
```bash
uv run test_gpt.py --interactive
```

**Custom config**:
```bash
uv run test_gpt.py --config my_config.yaml
```

Results are saved to `runs/{timestamp}_gpt.jsonl` with full episode replay data (observations, actions, reasoning, events). Configure via `eval_config.yaml`.
