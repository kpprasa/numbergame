# Analysis Scripts

Reusable tools for analyzing game runs.

## Scripts

### `analyze_actions_before_lockin.py`
Analyzes how many actions models take before locking in their answers.

**Usage:**
```bash
python analyze_actions_before_lockin.py              # All runs
python analyze_actions_before_lockin.py 2026-02-01   # Runs after date
```

**Outputs:**
- Average, min, max, median actions before LOCK_IN
- Distribution histogram

---

### `analyze_action_breakdown.py`
Analyzes the types of actions taken (POLL_PASSIVE, POLL_ACTIVE, LOCK_IN, etc.).

**Usage:**
```bash
python analyze_action_breakdown.py              # All runs
python analyze_action_breakdown.py "2026-02-*"  # Filtered by pattern
```

**Outputs:**
- Percentage breakdown of action types per model
- Per-file statistics

---

### `analyze_performance.py`
Analyzes time efficiency and test success rates.

**Usage:**
```bash
python analyze_performance.py              # All runs
python analyze_performance.py 2026-02-01   # Runs after date
```

**Outputs:**
- Average time per successful game
- Test success rates
- Outcomes breakdown

---

### `analyze_reasoning_length.py`
Analyzes word count in model reasoning.

**Usage:**
```bash
python analyze_reasoning_length.py
```

**Outputs:**
- Average reasoning length per model
- Statistics on verbosity

---

### `extract_raw_reasoning.py`
Extracts raw reasoning text from run files.

**Usage:**
```bash
# Extract specific files
python extract_raw_reasoning.py run1.jsonl run2.jsonl

# Extract all runs
python extract_raw_reasoning.py
```

**Outputs:**
- Markdown files in `analysis/` with raw reasoning per game

---

## Requirements

All scripts use relative paths and work from any directory. Run from repository root or scripts directory.
