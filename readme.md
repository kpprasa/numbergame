# Numbergame

**Numbergame** is a rule-inference / active-learning game.  
Each round hides a deterministic operator `#` that maps integer pairs `(a, b)` to an output via a structured generator (e.g. affine rules or decision trees over latent features). The player gathers limited evidence through passive samples or targeted queries, then predicts a designated target; correct predictions may unlock a held-out generalization test to reward true inference over memorization. Some rounds are intentionally ill-posed, making abandonment a valid scored strategy.

## Running the game

Basic:

```bash
python numbergame.py --play
```

** Recommended: **

```bash
python3 numbergame.py --play --generator tree --heldout_k 2 --difficulty 1
```
