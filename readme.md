# Numbergame

## Motivation
This game aims to be the simplest game that has some of the interesting properties of life: some structure, imperfect information, definitely not computationally tractable with certainty, and a possibility that the game you're playing isn't the right one. In order to be useful, it needed to be extremely difficult to solve with certainty, but easy to verify.

**Numbergame** is a rule-inference / active-learning game.  
Each round hides a deterministic operator `#` that maps integer pairs `(a, b)` to an output via a structured generator (e.g. affine rules or decision trees over latent features). The player gathers limited evidence through passive samples or targeted queries, then predicts a designated target; correct predictions may unlock a held-out generalization test to reward true inference over memorization. Some rounds are intentionally ill-posed, making abandonment a valid scored strategy.

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
