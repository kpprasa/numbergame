"""
Rule-inference / active learning game: each round hides a deterministic operator '#' that maps
integer pairs (a,b) to an output via a structured generator (e.g., affine/additive or decision-tree
over latent features of (a,b)). The player gathers limited evidence through passive samples or
targeted queries, then predicts a designated target query; correct target predictions can trigger
a held-out generalization test to reward true model inference over memorization. Some rounds may
be intentionally ill-posed (inconsistent with the hypothesis class), making abandonment a scored
option.

Run:
  python numbergame.py --play

  or what I'd recommend to start:
  `python3 numbergame.py --play --generator tree --heldout_k 2 --difficulty 1`
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

ActionType = Literal["POLL_PASSIVE", "POLL_ACTIVE", "LOCK_IN", "ABANDON", "TEST_SUBMIT"]
RewardInfoRegime = Literal["perfect", "hinted", "hidden"]
GeneratorArchetype = Literal["tree", "random"]  # Add additive in future


# ------------------- Event System -------------------


class EventType(str, Enum):
    """Event types emitted by environment actions."""

    NONE = "NONE"
    ENTER_TEST = "ENTER_TEST"
    REJECTED = "REJECTED"
    ROUND_END = "ROUND_END"


Outcome = Literal[
    "LOCK_IN_CORRECT",
    "LOCK_IN_CORRECT_NO_TEST",
    "LOCK_IN_WRONG_CONSISTENT",
    "LOCK_IN_WRONG_INCONSISTENT",
    "TEST_SUBMIT_CORRECT",
    "TEST_SUBMIT_WRONG",
    "TEST_SUBMIT_INVALID",
    "ABANDON_RIGHTFUL",
    "ABANDON_WRONGFUL",
    "TIMEOUT",
]


@dataclass(frozen=True)
class EventNone:
    """No-op event emitted when no state transition occurs."""

    type: Literal[EventType.NONE] = EventType.NONE


@dataclass(frozen=True)
class RejectedEvent:
    """Event emitted when an action is rejected as invalid."""

    action_type: ActionType
    reason: str
    type: Literal[EventType.REJECTED] = EventType.REJECTED


@dataclass(frozen=True)
class EnterTestEvent:
    """Event emitted when entering held-out test phase."""

    type: Literal[EventType.ENTER_TEST] = EventType.ENTER_TEST
    heldout_k: int = 0


@dataclass(frozen=True)
class RoundEndEvent:
    """Event emitted when a round ends and advances to next game."""

    type: Literal[EventType.ROUND_END] = EventType.ROUND_END
    outcome: Outcome = "TIMEOUT"
    reward_delta: int = 0
    meta_delta: int = 0
    generator: Optional[LUTGenerator] = None


Event = Union[EventNone, RejectedEvent, EnterTestEvent, RoundEndEvent]


# ------------------- Action System -------------------


@dataclass(frozen=True)
class PollPassive:
    """Passive poll: add a random example."""

    type: Literal["POLL_PASSIVE"] = "POLL_PASSIVE"


@dataclass(frozen=True)
class PollActive:
    """Active poll: query a specific (a, b) pair."""

    a: int
    b: int
    type: Literal["POLL_ACTIVE"] = "POLL_ACTIVE"


@dataclass(frozen=True)
class LockIn:
    """Lock in an answer for the target query."""

    answer: int
    type: Literal["LOCK_IN"] = "LOCK_IN"


@dataclass(frozen=True)
class TestSubmit:
    """Submit predictions for held-out test queries."""

    predictions: Tuple[int, ...]
    type: Literal["TEST_SUBMIT"] = "TEST_SUBMIT"


@dataclass(frozen=True)
class Abandon:
    """Abandon the current game as ill-posed."""

    type: Literal["ABANDON"] = "ABANDON"


AnyAction = Union[PollPassive, PollActive, LockIn, TestSubmit, Abandon]


# ------------------- DSL Info System -------------------


@dataclass(frozen=True)
class GeneratorInfo:
    """Metadata about the LUT generator structure."""

    archetype: str
    max_depth: int
    allowed_split_types: Tuple[str, ...]
    realized_depth: Optional[int] = None
    num_leaves: Optional[int] = None
    num_internal_nodes: Optional[int] = None


@dataclass(frozen=True)
class DSLInfoL1:
    """DSL information for difficulty level 1 (full key formula revealed)."""

    key_definition: str
    feature_names: Tuple[str, ...]
    key_output_sizes: Tuple[int, ...]
    examples_with_keys: Tuple[Tuple[int, int, int, Tuple[int, ...]], ...]
    target_key: Tuple[int, ...]
    observed_key_map: Dict[str, int]
    conflicts: Tuple[Tuple[str, int, int], ...]
    generator: Optional[GeneratorInfo] = None


@dataclass(frozen=True)
class DSLInfoL2:
    """DSL information for difficulty level 2 (features shown, order unknown)."""

    selected_features: Tuple[Tuple[str, int], ...]  # (name, output_size)
    note: str
    examples_with_keys: Tuple[Tuple[int, int, int, Tuple[int, ...]], ...]
    generator: Optional[GeneratorInfo] = None


@dataclass(frozen=True)
class DSLInfoL3:
    """DSL information for difficulty level 3 (feature pool shown)."""

    feature_pool: Tuple[Tuple[str, int], ...]  # (name, output_size)
    note: str
    generator: Optional[GeneratorInfo] = None


DSLInfo = Union[DSLInfoL1, DSLInfoL2, DSLInfoL3]


# ------------------- Observation System -------------------


@dataclass(frozen=True)
class Observation:
    """Immutable observation of the game state visible to the player."""

    status: Literal["no_game", "running", "run_over"]
    time_spent: int
    time_remaining: int
    accuracy: int
    meta: int
    allowed_commands: Tuple[ActionType, ...]

    # Game-specific fields (None when no active game)
    n: Optional[int] = None
    target_query: Optional[Tuple[int, int]] = None
    examples: Tuple[Tuple[int, int, int], ...] = ()
    phase: Literal["PLAY", "TEST"] = "PLAY"
    reward_hint: Optional[int] = None
    pair_conflicts_detected: bool = False
    key_conflicts_detected: Optional[bool] = None

    # Test phase
    heldout_queries: Tuple[Tuple[int, int], ...] = ()

    dsl: Optional[DSLInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility with UI."""
        result: Dict[str, Any] = {
            "status": self.status,
            "time_spent": self.time_spent,
            "time_remaining": self.time_remaining,
            "accuracy": self.accuracy,
            "meta": self.meta,
            "allowed_commands": list(self.allowed_commands),
        }

        if self.n is not None:
            result["n"] = self.n
        if self.target_query is not None:
            result["target_query"] = self.target_query
        result["examples"] = list(self.examples)
        result["test_phase"] = self.phase == "TEST"
        if self.reward_hint is not None:
            result["reward_hint"] = self.reward_hint
        result["pair_conflicts_detected"] = self.pair_conflicts_detected
        if self.key_conflicts_detected is not None:
            result["key_conflicts_detected"] = self.key_conflicts_detected
        if self.heldout_queries:
            result["heldout_queries"] = list(self.heldout_queries)

        if self.dsl is not None:
            if isinstance(self.dsl, DSLInfoL1):
                result["dsl"] = {
                    "key_definition": self.dsl.key_definition,
                    "feature_names": self.dsl.feature_names,
                    "key_output_sizes": self.dsl.key_output_sizes,
                }
                if self.dsl.generator is not None:
                    result["dsl"]["generator"] = {
                        "archetype": self.dsl.generator.archetype,
                        "max_depth": self.dsl.generator.max_depth,
                        "realized_depth": self.dsl.generator.realized_depth,
                        "num_leaves": self.dsl.generator.num_leaves,
                        "num_internal_nodes": self.dsl.generator.num_internal_nodes,
                        "allowed_split_types": self.dsl.generator.allowed_split_types,
                    }
                result["examples_with_keys"] = list(self.dsl.examples_with_keys)
                result["target_key"] = self.dsl.target_key
                result["observed_key_map"] = self.dsl.observed_key_map
                result["conflicts"] = list(self.dsl.conflicts)

            elif isinstance(self.dsl, DSLInfoL2):
                result["dsl"] = {
                    "selected_features": [
                        {"name": name, "output_size": size}
                        for name, size in self.dsl.selected_features
                    ],
                    "note": self.dsl.note,
                }
                if self.dsl.generator is not None:
                    result["dsl"]["generator"] = {
                        "archetype": self.dsl.generator.archetype,
                        "max_depth": self.dsl.generator.max_depth,
                        "realized_depth": self.dsl.generator.realized_depth,
                        "num_leaves": self.dsl.generator.num_leaves,
                        "num_internal_nodes": self.dsl.generator.num_internal_nodes,
                        "allowed_split_types": self.dsl.generator.allowed_split_types,
                    }
                result["examples_with_keys"] = list(self.dsl.examples_with_keys)

            elif isinstance(self.dsl, DSLInfoL3):
                result["dsl"] = {
                    "feature_pool": [
                        {"name": name, "output_size": size}
                        for name, size in self.dsl.feature_pool
                    ],
                    "note": self.dsl.note,
                }
                if self.dsl.generator is not None:
                    result["dsl"]["generator"] = {
                        "archetype": self.dsl.generator.archetype,
                        "max_depth": self.dsl.generator.max_depth,
                        "allowed_split_types": self.dsl.generator.allowed_split_types,
                    }

        return result


# ------------------- LUT Generator Architecture -------------------


class LUTGenerator(ABC):
    """Abstract base class for LUT generation strategies."""

    @abstractmethod
    def eval(self, key: Tuple[int, ...]) -> int:
        """Evaluate the generator for a given key."""
        pass

    @abstractmethod
    def materialize(
        self, key_space: List[Tuple[int, ...]]
    ) -> Dict[Tuple[int, ...], int]:
        """Materialize the full LUT over the key space."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the generator structure."""
        pass


@dataclass
class DecisionTreeNode:
    """Node in a decision tree for LUT generation."""

    is_leaf: bool
    output: Optional[int] = None  # For leaves

    # For internal nodes
    split_dim: Optional[int] = None
    split_type: Optional[str] = None  # "threshold", "equality", "set"
    split_value: Optional[Any] = None  # threshold value, equality value, or set
    left: Optional[DecisionTreeNode] = None
    right: Optional[DecisionTreeNode] = None

    def eval(self, key: Tuple[int, ...]) -> int:
        """Evaluate this node/tree for the given key."""
        if self.is_leaf:
            return self.output

        # Evaluate split condition
        key_value = key[self.split_dim]
        if self.split_type == "threshold":
            goes_left = key_value <= self.split_value
        elif self.split_type == "equality":
            goes_left = key_value == self.split_value
        elif self.split_type == "set":
            goes_left = key_value in self.split_value
        else:
            raise ValueError(f"Unknown split type: {self.split_type}")

        return self.left.eval(key) if goes_left else self.right.eval(key)


class DecisionTreeLUT(LUTGenerator):
    """Decision tree-based LUT generator."""

    def __init__(self, root: DecisionTreeNode, metadata: Dict[str, Any]):
        self.root = root
        self.metadata = metadata

    def eval(self, key: Tuple[int, ...]) -> int:
        """Evaluate the tree for a given key."""
        return self.root.eval(key)

    def materialize(
        self, key_space: List[Tuple[int, ...]]
    ) -> Dict[Tuple[int, ...], int]:
        """Materialize the full LUT by evaluating all keys."""
        return {k: self.eval(k) for k in key_space}

    def get_metadata(self) -> Dict[str, Any]:
        """Return tree metadata (depth, num_leaves, etc.)."""
        return self.metadata.copy()


@dataclass
class EnvConfig:
    # Domain size for inputs/outputs: values are in [0, n-1]
    n: int = 5

    seed_examples: int = 3
    lifetime_time_budget: int = 50

    # Time costs
    c_poll: int = 1
    c_active: int = 2
    c_lock: int = 1
    c_quit: int = 1

    # This represents whether choosing active limits attention to passive (environmental) observations.
    active_includes_passive: bool = False

    difficulty: int = 1

    max_keys: int = 128  # product of feature output sizes

    num_features: Tuple[int, int] = (3, 5)

    generator_archetype: GeneratorArchetype = "tree"

    heldout_k: int = 3  # test post lock in

    auto_advance_on_lock: bool = True
    auto_advance_on_abandon: bool = True

    base_reward: int = 100
    penalize_inconsistent_wrong: bool = True

    reward_info: RewardInfoRegime = "perfect"
    reward_hint_noise: float = 0.25

    # Sampling control: Default is false because the intuition is sometimes the world repeats itself.
    no_repeat_pairs_within_game: bool = False

    q_illposed: float = 0.0

    def __post_init__(self):
        if self.difficulty >= 3:
            self.penalize_inconsistent_wrong = False


@dataclass(frozen=True)
class Example:
    a: int
    b: int
    c: int
    key: Tuple[int, ...]


@dataclass
class GameState:
    x: int
    y: int
    true_answer: int
    true_key: Tuple[int, ...]

    lut: Dict[Tuple[int, ...], int]
    exceptions: Dict[Tuple[int, int], int] = field(default_factory=dict)

    examples: List[Example] = field(default_factory=list)
    seen_pairs: Set[Tuple[int, int]] = field(default_factory=set)

    reward: int = 100

    is_over: bool = False
    outcome: Optional[Outcome] = None
    instructions_shown: bool = False

    feature_pool: List[Tuple[str, int, Callable]] = field(
        default_factory=list
    )  # (name, output_size, function)
    selected_features: List[Tuple[str, int, Callable]] = field(default_factory=list)
    feature_names: Tuple[str, ...] = ()
    key_output_sizes: Tuple[int, ...] = ()
    key_definition: str = ""

    generator: Optional[LUTGenerator] = None

    test_phase: bool = False
    heldout_pairs: List[Tuple[int, int]] = field(default_factory=list)
    heldout_answers: List[int] = field(default_factory=list)


@dataclass
class StepResult:
    observation: Observation
    done: bool
    info: Dict
    event: Event = field(default_factory=EventNone)


class LUTEnv:
    """
    Hidden rule class (DSL):
      key(a,b) = (a%2, b%2, 1[a<b])
      output = LUT[key] in [0..n-1]
    """

    def __init__(self, cfg: EnvConfig, rng: Optional[random.Random] = None):
        self.cfg = cfg
        self.rng = rng or random.Random()

        self.time_spent = 0
        self.accuracy = 0
        self.meta = 0

        self.game: Optional[GameState] = None
        self.run_over = False

    # ------------------- public API -------------------

    def reset(self) -> Dict:
        self.time_spent = 0
        self.accuracy = 0
        self.meta = 0
        self.run_over = False
        self.game = self._generate_new_game()
        return self._make_observation()

    def step(self, action: AnyAction) -> StepResult:
        if self.run_over:
            return StepResult(
                self._make_observation(),
                True,
                self._info("run_over"),
                event=RejectedEvent(action_type=action.type, reason="run_over"),
            )

        if self.game is None:
            self.game = self._generate_new_game()
            return StepResult(
                self._make_observation(),
                False,
                self._info("spawned_game"),
                event=EventNone(),
            )

        cost = self._time_cost_for_action(action.type)
        remaining = self.cfg.lifetime_time_budget - self.time_spent
        if cost > remaining:
            if self.game and not self.game.is_over:
                self.game.is_over = True
                self.game.outcome = "TIMEOUT"
            self.run_over = True
            return StepResult(
                self._make_observation(),
                True,
                self._info("insufficient_time"),
                event=RejectedEvent(
                    action_type=action.type, reason="insufficient_time"
                ),
            )

        self.game, event = self._apply_action(self.game, action)

        self.time_spent += cost
        if self.time_spent == self.cfg.lifetime_time_budget:
            self.run_over = True

        return StepResult(
            self._make_observation(), self.run_over, self._info("ok"), event=event
        )

    # ------------------- DSL / oracle -------------------

    def _key(self, game: GameState, a: int, b: int) -> Tuple[int, ...]:
        """Compute key using game's selected features."""
        return tuple(fn(a, b) for _, _, fn in game.selected_features)

    def _oracle(self, game: GameState, a: int, b: int) -> Tuple[int, Tuple[int, ...]]:
        a %= self.cfg.n
        b %= self.cfg.n
        k = self._key(game, a, b)
        if (a, b) in game.exceptions:
            return game.exceptions[(a, b)], k
        return game.lut[k], k

    # ------------------- feature pool -------------------

    def _build_feature_pool(self) -> List[Tuple[str, int, Any]]:
        """Build pool of features with (name, output_size, function)."""
        pool = []

        # Modulo operations (output_size 2-4)
        pool.append(("a%2", 2, lambda a, b: a % 2))
        pool.append(("a%3", 3, lambda a, b: a % 3))
        pool.append(("a%4", 4, lambda a, b: a % 4))
        pool.append(("b%2", 2, lambda a, b: b % 2))
        pool.append(("b%3", 3, lambda a, b: b % 3))
        pool.append(("b%4", 4, lambda a, b: b % 4))

        # Comparisons (output_size 2)
        pool.append(("a<b", 2, lambda a, b: 1 if a < b else 0))
        pool.append(("a>b", 2, lambda a, b: 1 if a > b else 0))
        pool.append(("a==b", 2, lambda a, b: 1 if a == b else 0))

        # Arithmetic (output_size 4)
        pool.append(("(a+b)%4", 4, lambda a, b: (a + b) % 4))
        pool.append(("(a-b)%4", 4, lambda a, b: (a - b) % 4))
        pool.append(("(a*b)%4", 4, lambda a, b: (a * b) % 4))

        # Bitwise (output_size 2-4)
        pool.append(("a&1", 2, lambda a, b: a & 1))
        pool.append(("a&3", 4, lambda a, b: a & 3))
        pool.append(("b&1", 2, lambda a, b: b & 1))
        pool.append(("b&3", 4, lambda a, b: b & 3))
        pool.append(("(a^b)%4", 4, lambda a, b: (a ^ b) % 4))

        return pool

    def _select_features(self) -> Tuple[List[Tuple[str, int, Any]], List[int]]:
        """
        Select features for this game, respecting max_keys constraint.
        Returns: (selected_features, output_sizes)
        """
        pool = self._build_feature_pool()
        self.rng.shuffle(pool)

        min_features, max_features = self.cfg.num_features
        selected = []
        output_sizes = []

        for feature in pool:
            name, output_size, fn = feature
            test_output_sizes = output_sizes + [output_size]
            key_space = 1
            for size in test_output_sizes:
                key_space *= size
                if key_space > self.cfg.max_keys:
                    break

            if key_space <= self.cfg.max_keys:
                selected.append(feature)
                output_sizes.append(output_size)

                if len(selected) >= max_features:
                    break

        if len(selected) < min_features:
            raise ValueError(
                f"Could not select {min_features} features with max_keys={self.cfg.max_keys}"
            )

        return selected, output_sizes

    # ------------------- decision tree generation -------------------

    def _generate_decision_tree(
        self, output_sizes: List[int], n: int, difficulty: int
    ) -> DecisionTreeLUT:
        """
        Generate a decision tree for the LUT.

        Args:
            output_sizes: List of output sizes for each key dimension
            n: Domain size for outputs
            difficulty: Difficulty level (1-4)

        Returns:
            DecisionTreeLUT instance
        """

        max_depths = {1: 3, 2: 3, 3: 4, 4: 4}
        max_depth = max_depths[difficulty]

        stop_probs = {1: 0.35, 2: 0.25, 3: 0.15, 4: 0.10}
        stop_prob = stop_probs[difficulty]

        allow_set_splits = difficulty >= 2

        root = self._build_tree_node(
            output_sizes, n, max_depth, 0, stop_prob, allow_set_splits
        )

        metadata = self._compute_tree_metadata(root, max_depth, allow_set_splits)

        return DecisionTreeLUT(root, metadata)

    def _build_tree_node(
        self,
        output_sizes: List[int],
        n: int,
        max_depth: int,
        current_depth: int,
        stop_prob: float,
        allow_set_splits: bool,
    ) -> DecisionTreeNode:
        """Recursively build a decision tree node."""
        if current_depth >= max_depth or self.rng.random() < stop_prob:
            return DecisionTreeNode(is_leaf=True, output=self.rng.randrange(n))

        weights = [size for size in output_sizes]
        total_weight = sum(weights)
        r = self.rng.random() * total_weight
        cumsum = 0
        split_dim = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r < cumsum:
                split_dim = i
                break

        dim_size = output_sizes[split_dim]

        possible_splits = []

        # Equality split (always available)
        possible_splits.append("equality")

        # Threshold split (for output_size >= 3)
        if dim_size >= 3:
            possible_splits.append("threshold")

        # Set-membership split (for difficulty >= 2 and output_size >= 2)
        if allow_set_splits and dim_size >= 2:
            possible_splits.append("set")

        split_type = self.rng.choice(possible_splits)

        if split_type == "threshold":
            # k_i <= t, where t in [0..dim_size-2]
            split_value = self.rng.randrange(dim_size - 1)
        elif split_type == "equality":
            # k_i == v, where v in [0..dim_size-1]
            split_value = self.rng.randrange(dim_size)
        elif split_type == "set":
            # k_i in S, where |S| in {1, 2} depending on dim_size
            max_set_size = min(2, dim_size - 1)
            set_size = self.rng.randint(1, max_set_size)
            split_value = frozenset(self.rng.sample(range(dim_size), set_size))

        left = self._build_tree_node(
            output_sizes, n, max_depth, current_depth + 1, stop_prob, allow_set_splits
        )
        right = self._build_tree_node(
            output_sizes, n, max_depth, current_depth + 1, stop_prob, allow_set_splits
        )

        return DecisionTreeNode(
            is_leaf=False,
            split_dim=split_dim,
            split_type=split_type,
            split_value=split_value,
            left=left,
            right=right,
        )

    def _compute_tree_metadata(
        self, root: DecisionTreeNode, max_depth: int, allow_set_splits: bool
    ) -> Dict[str, Any]:
        """Compute metadata about the tree structure."""

        def traverse(node: DecisionTreeNode) -> Tuple[int, int, int]:
            """Returns (depth, num_leaves, num_internal_nodes)."""
            if node.is_leaf:
                return (1, 1, 0)

            left_depth, left_leaves, left_internal = traverse(node.left)
            right_depth, right_leaves, right_internal = traverse(node.right)

            depth = 1 + max(left_depth, right_depth)
            num_leaves = left_leaves + right_leaves
            num_internal = 1 + left_internal + right_internal

            return (depth, num_leaves, num_internal)

        depth, num_leaves, num_internal = traverse(root)

        allowed_split_types = ["threshold", "equality"]
        if allow_set_splits:
            allowed_split_types.append("set")

        return {
            "archetype": "decision_tree",
            "max_depth": max_depth,
            "realized_depth": depth,
            "num_leaves": num_leaves,
            "num_internal_nodes": num_internal,
            "allowed_split_types": allowed_split_types,
        }

    # ------------------- game generation -------------------

    def _create_generator(
        self, archetype: GeneratorArchetype, output_sizes: List[int], n: int
    ) -> Optional[LUTGenerator]:
        """Factory method to create LUT generator based on archetype."""

        generators = {
            "tree": lambda: self._generate_decision_tree(
                output_sizes, n, self.cfg.difficulty
            ),
            "random": lambda: None,
        }

        if archetype not in generators:
            raise ValueError(f"Unknown generator archetype: {archetype}")

        return generators[archetype]()

    def _generate_new_game(self) -> GameState:
        n = self.cfg.n

        selected_features, output_sizes = self._select_features()
        feature_pool = self._build_feature_pool()

        key_space_ranges = [range(output_size) for output_size in output_sizes]
        key_space = list(itertools.product(*key_space_ranges))

        generator = self._create_generator(
            self.cfg.generator_archetype, output_sizes, n
        )
        if generator is not None:
            lut = generator.materialize(key_space)
        else:
            lut = {k: self.rng.randrange(0, n) for k in key_space}

        feature_names = tuple(name for name, _, _ in selected_features)
        key_definition = f"key(a,b) = ({', '.join(feature_names)})"

        x = self.rng.randrange(0, n)
        y = self.rng.randrange(0, n)

        temp = GameState(
            x=x,
            y=y,
            true_answer=0,
            true_key=(),
            lut=lut,
            exceptions={},
            reward=self.cfg.base_reward,
            feature_pool=feature_pool,
            selected_features=selected_features,
            feature_names=feature_names,
            key_output_sizes=tuple(output_sizes),
            key_definition=key_definition,
            generator=generator,
        )

        true_answer, true_key = self._oracle(temp, x, y)
        temp.true_answer = true_answer
        temp.true_key = true_key

        if self.rng.random() < self.cfg.q_illposed:
            self._inject_ill_posed_contradictions(temp)

        for _ in range(self.cfg.seed_examples):
            self._append_random_example(temp)

        return temp

    def _inject_ill_posed_contradictions(self, game: GameState) -> None:
        """Inject contradictions based on difficulty level."""
        n = self.cfg.n

        if self.cfg.difficulty <= 2:
            # Difficulty 1-2: inject key contradictions (same key, different outputs)
            # Find a key in the LUT and create 2 pairs that map to it with different outputs
            keys = list(game.lut.keys())
            if keys:
                target_key = self.rng.choice(keys)
                base_output = game.lut[target_key]

                # Find 2 different (a,b) pairs that produce this key
                pairs_with_key = []
                for _ in range(1000):  # Try to find pairs
                    a = self.rng.randrange(0, n)
                    b = self.rng.randrange(0, n)
                    if (a, b) != (game.x, game.y):  # Don't use target
                        k = self._key(game, a, b)
                        if k == target_key:
                            pairs_with_key.append((a, b))
                            if len(pairs_with_key) >= 2:
                                break

                # Inject contradictions
                if len(pairs_with_key) >= 2:
                    # First pair gets LUT output, second gets different output
                    a2, b2 = pairs_with_key[1]
                    alt_output = self.rng.randrange(0, n - 1)
                    if alt_output >= base_output:
                        alt_output += 1
                    game.exceptions[(a2, b2)] = alt_output

                    a1, b1 = pairs_with_key[0]
                    c1, k1 = self._oracle(game, a1, b1)
                    game.examples.append(Example(a1, b1, c1, k1))
                    game.seen_pairs.add((a1, b1))

                    c2, k2 = self._oracle(game, a2, b2)
                    game.examples.append(Example(a2, b2, c2, k2))
                    game.seen_pairs.add((a2, b2))
        else:
            # Difficulty 3-4: inject pair contradictions (same (a,b), different outputs)
            # This makes the operator non-deterministic (ABANDON-worthy)
            a = self.rng.randrange(0, n)
            b = self.rng.randrange(0, n)
            if (a, b) != (game.x, game.y):
                c1, k1 = self._oracle(game, a, b)
                c2 = self.rng.randrange(0, n - 1)
                if c2 >= c1:
                    c2 += 1

                # Force both contradictory examples into seed
                game.examples.append(Example(a, b, c1, k1))
                game.examples.append(Example(a, b, c2, k1))
                game.seen_pairs.add((a, b))

    def _append_random_example(self, game: GameState) -> None:
        n = self.cfg.n
        for _ in range(2000):
            a = self.rng.randrange(0, n)
            b = self.rng.randrange(0, n)
            if (not self.cfg.no_repeat_pairs_within_game) or (
                (a, b) not in game.seen_pairs
            ):
                c, k = self._oracle(game, a, b)
                game.examples.append(Example(a, b, c, k))
                game.seen_pairs.add((a, b))
                return

        a = self.rng.randrange(0, n)
        b = self.rng.randrange(0, n)
        c, k = self._oracle(game, a, b)
        game.examples.append(Example(a, b, c, k))
        game.seen_pairs.add((a, b))

    # ------------------- actions -------------------

    def _do_passive_poll(self) -> Event:
        assert self.game is not None
        if self.game.is_over or self.game.test_phase:
            return RejectedEvent(action_type="POLL_PASSIVE", reason="poll_not_allowed")
        self._append_random_example(self.game)
        return EventNone()

    def _do_active_poll(self, a: int, b: int) -> Event:
        assert self.game is not None
        if self.game.is_over or self.game.test_phase:
            return RejectedEvent(action_type="POLL_ACTIVE", reason="poll_not_allowed")

        a %= self.cfg.n
        b %= self.cfg.n

        # can't give away the answer!
        if (a, b) == (self.game.x, self.game.y):
            return RejectedEvent(
                action_type="POLL_ACTIVE", reason="target_pair_forbidden"
            )

        if (not self.cfg.no_repeat_pairs_within_game) or (
            (a, b) not in self.game.seen_pairs
        ):
            c, k = self._oracle(self.game, a, b)
            self.game.examples.append(Example(a, b, c, k))
            self.game.seen_pairs.add((a, b))

        if self.cfg.active_includes_passive:
            self._append_random_example(self.game)

        return EventNone()

    def _do_lock_in(self, answer: int) -> Event:
        assert self.game is not None
        if self.game.is_over or self.game.test_phase:
            return RejectedEvent(action_type="LOCK_IN", reason="lock_not_allowed")

        answer %= self.cfg.n
        reward_delta = 0

        if answer == self.game.true_answer:
            self.game.outcome = "LOCK_IN_CORRECT"

            n = self.cfg.n
            all_pairs = [(a, b) for a in range(n) for b in range(n)]
            excluded = self.game.seen_pairs | {(self.game.x, self.game.y)}
            available = [p for p in all_pairs if p not in excluded]

            num_heldout = min(self.cfg.heldout_k, len(available))
            if num_heldout > 0:
                heldout_pairs = self.rng.sample(available, num_heldout)
                heldout_answers = []
                for a, b in heldout_pairs:
                    c, _ = self._oracle(self.game, a, b)
                    heldout_answers.append(c)

                self.game.heldout_pairs = heldout_pairs
                self.game.heldout_answers = heldout_answers
                self.game.test_phase = True
                return EnterTestEvent(heldout_k=len(heldout_pairs))
            else:
                reward_delta = self.cfg.base_reward
                self.accuracy += reward_delta
                self.game.outcome = "LOCK_IN_CORRECT_NO_TEST"
                self.game.is_over = True

                ended_game = self.game
                if self.cfg.auto_advance_on_lock:
                    self.game = self._generate_new_game()

                return RoundEndEvent(
                    outcome=ended_game.outcome,
                    reward_delta=reward_delta,
                    meta_delta=0,
                    generator=ended_game.generator,
                )
        else:
            consistent = self._exists_lut_solution_for_answer(
                self.game.examples, self.game.true_key, answer
            )
            if consistent:
                self.game.outcome = "LOCK_IN_WRONG_CONSISTENT"
            else:
                if self.cfg.penalize_inconsistent_wrong:
                    self.accuracy -= self.game.reward
                    reward_delta = -self.game.reward
                self.game.outcome = "LOCK_IN_WRONG_INCONSISTENT"

            self.game.is_over = True

            ended_game = self.game
            if self.cfg.auto_advance_on_lock:
                self.game = self._generate_new_game()

            return RoundEndEvent(
                outcome=ended_game.outcome,
                reward_delta=reward_delta,
                meta_delta=0,
                generator=ended_game.generator,
            )

    def _do_test_submit(self, predictions: List[int]) -> Event:
        """Handle test submission after correct lock-in."""
        assert self.game is not None
        if not self.game.test_phase or self.game.is_over:
            return RejectedEvent(
                action_type="TEST_SUBMIT", reason="test_submit_not_allowed"
            )

        if len(predictions) != len(self.game.heldout_answers):
            self.game.outcome = "TEST_SUBMIT_INVALID"
            self.game.test_phase = False
            self.game.is_over = True

            ended_game = self.game
            self.game = self._generate_new_game()

            return RoundEndEvent(
                outcome=ended_game.outcome,
                reward_delta=0,
                meta_delta=0,
                generator=ended_game.generator,
            )

        predictions = [p % self.cfg.n for p in predictions]

        all_correct = all(
            pred == truth for pred, truth in zip(predictions, self.game.heldout_answers)
        )

        reward_delta = 0
        if all_correct:
            self.accuracy += self.cfg.base_reward
            reward_delta = self.cfg.base_reward
            self.game.outcome = "TEST_SUBMIT_CORRECT"
        else:
            self.game.outcome = "TEST_SUBMIT_WRONG"

        self.game.test_phase = False
        self.game.is_over = True

        ended_game = self.game
        self.game = self._generate_new_game()

        return RoundEndEvent(
            outcome=ended_game.outcome,
            reward_delta=reward_delta,
            meta_delta=0,
            generator=ended_game.generator,
        )

    def _do_abandon(self) -> Event:
        assert self.game is not None
        if self.game.is_over or self.game.test_phase:
            return RejectedEvent(action_type="ABANDON", reason="abandon_not_allowed")

        rightful = not self._exists_any_lut_solution(self.game.examples)
        meta_delta = 1 if rightful else -1
        self.meta += meta_delta
        self.game.outcome = "ABANDON_RIGHTFUL" if rightful else "ABANDON_WRONGFUL"
        self.game.is_over = True

        ended_game = self.game
        if self.cfg.auto_advance_on_abandon:
            self.game = self._generate_new_game()

        return RoundEndEvent(
            outcome=ended_game.outcome,
            reward_delta=0,
            meta_delta=meta_delta,
            generator=ended_game.generator,
        )

    # ------------------- consistency checks -------------------

    def _detect_pair_conflicts(self, examples: List[Example]) -> bool:
        """Detect if same (a,b) pair has different outputs (always safe to expose)."""
        seen_pairs: Dict[Tuple[int, int], int] = {}
        for e in examples:
            pair = (e.a, e.b)
            if pair in seen_pairs and seen_pairs[pair] != e.c:
                return True
            seen_pairs[pair] = e.c
        return False

    def _detect_key_conflicts(self, examples: List[Example]) -> bool:
        """Detect if same key has different outputs (only expose when DSL revealed)."""
        seen_keys: Dict[Tuple[int, ...], int] = {}
        for e in examples:
            if e.key in seen_keys and seen_keys[e.key] != e.c:
                return True
            seen_keys[e.key] = e.c
        return False

    def _exists_any_lut_solution(self, examples: List[Example]) -> bool:
        seen: Dict[Tuple[int, ...], int] = {}
        for e in examples:
            if e.key in seen and seen[e.key] != e.c:
                return False
            seen[e.key] = e.c
        return True

    def _exists_lut_solution_for_answer(
        self, examples: List[Example], target_key: Tuple[int, ...], proposed_answer: int
    ) -> bool:
        seen: Dict[Tuple[int, ...], int] = {}
        for e in examples:
            if e.key in seen and seen[e.key] != e.c:
                return False
            seen[e.key] = e.c
        if target_key in seen:
            return seen[target_key] == proposed_answer
        return 0 <= proposed_answer < self.cfg.n

    # ------------------- bookkeeping / observations -------------------

    def _time_cost_for_action(self, act: ActionType) -> int:
        if act == "POLL_PASSIVE":
            return self.cfg.c_poll
        if act == "POLL_ACTIVE":
            return self.cfg.c_active
        if act == "LOCK_IN":
            return self.cfg.c_lock
        if act == "ABANDON":
            return self.cfg.c_quit
        if act == "TEST_SUBMIT":
            return 0  # Test submission is free (no time cost)
        raise ValueError(act)

    def _apply_action(
        self, game: GameState, action: AnyAction
    ) -> Tuple[GameState, Event]:
        """Apply action to game state and return (game, event).

        This makes state transitions explicit. Even though it may mutate
        game internally for performance, returning the game makes data flow clear.
        """
        event: Event
        if isinstance(action, PollPassive):
            event = self._do_passive_poll()
        elif isinstance(action, PollActive):
            event = self._do_active_poll(action.a, action.b)
        elif isinstance(action, LockIn):
            event = self._do_lock_in(action.answer)
        elif isinstance(action, Abandon):
            event = self._do_abandon()
        elif isinstance(action, TestSubmit):
            event = self._do_test_submit(list(action.predictions))
        else:
            # Shouldn't happen with type checking, but be defensive
            event = RejectedEvent(action_type=action.type, reason="unknown_action_type")

        return game, event

    def _instructions_block(self) -> Dict:
        difficulty_hints = {
            1: "Full key formula + observed key map revealed",
            2: "Selected feature names + output_sizes shown (order unknown)",
            3: "Full feature pool shown (selected features unknown)",
            4: "No DSL information (black box)",
        }

        return {
            "you_are_in": "A rule-inference game with a hidden operator '#'.",
            "goal": "Use examples to predict the output for the target query, then LOCK_IN, or ABANDON if you believe the game is ill-posed under the allowed rule class.",
            "actions": {
                "POLL_PASSIVE": f"Receive a random example (cost {self.cfg.c_poll} time).",
                "POLL_ACTIVE": f"Choose (a,b) and receive its output (cost {self.cfg.c_active} time).",
                "LOCK_IN": f"Submit your answer for the target query (cost {self.cfg.c_lock} time).",
                "ABANDON": f"Quit the game; you forgo accuracy for this game and get meta +/-1 depending on whether quitting was rightful (cost {self.cfg.c_quit} time).",
            },
            "scoring": {
                "time": "Lifetime-cumulative; run ends when time_spent >= lifetime_time_budget.",
                "accuracy": f"+{self.cfg.base_reward} for correct LOCK_IN and Test performance; 0 for wrong-but-consistent; "
                f"{'-' + str(self.cfg.base_reward) if self.cfg.penalize_inconsistent_wrong else '0'} for wrong-and-inconsistent.",
                "meta": "+1 for rightful ABANDON (no in-class rule fits observed examples), -1 otherwise.",
                "timeout": "If time runs out before you LOCK_IN or ABANDON, the current game's accuracy is 0 and the run ends.",
            },
            "what_is_known": {
                "domain": f"Inputs and outputs are integers in [0, {self.cfg.n - 1}].",
                "difficulty": self.cfg.difficulty,
                "difficulty_hint": difficulty_hints.get(self.cfg.difficulty, "Unknown"),
                "inconsistent_penalties": "enabled"
                if self.cfg.penalize_inconsistent_wrong
                else "disabled",
                "ill_posed_games": f"{self.cfg.q_illposed * 100:.0f}% of games may be ill-posed"
                if self.cfg.q_illposed > 0
                else "All games are well-posed",
            },
        }

    def _make_observation(self) -> Observation:
        if self.game is None:
            return Observation(
                status="no_game",
                time_spent=self.time_spent,
                time_remaining=max(0, self.cfg.lifetime_time_budget - self.time_spent),
                accuracy=self.accuracy,
                meta=self.meta,
                allowed_commands=(),
            )

        reward_hint = None
        if self.cfg.reward_info == "perfect":
            reward_hint = self.game.reward
        elif self.cfg.reward_info == "hinted":
            eps = (2 * self.rng.random() - 1) * self.cfg.reward_hint_noise
            reward_hint = max(0, int(round(self.game.reward * (1.0 + eps))))
        else:
            reward_hint = None

        pair_conflicts = self._detect_pair_conflicts(self.game.examples)
        key_conflicts = self._detect_key_conflicts(self.game.examples)

        key_to_c: Dict[Tuple[int, ...], int] = {}
        conflicts: List[Tuple[Tuple[int, ...], int, int]] = []
        for e in self.game.examples:
            if e.key in key_to_c and key_to_c[e.key] != e.c:
                conflicts.append((e.key, key_to_c[e.key], e.c))
            else:
                key_to_c[e.key] = e.c

        allowed_commands: Tuple[ActionType, ...]
        if not self.run_over and not self.game.is_over:
            if self.game.test_phase:
                allowed_commands = ("TEST_SUBMIT",)
            else:
                allowed_commands = ("POLL_PASSIVE", "POLL_ACTIVE", "LOCK_IN", "ABANDON")
        else:
            allowed_commands = ()

        dsl_info: Optional[DSLInfo] = None

        if self.cfg.difficulty == 1:
            # Difficulty 1: Show full key formula + observed key map
            generator_info = None
            if self.game.generator is not None:
                metadata = self.game.generator.get_metadata()
                generator_info = GeneratorInfo(
                    archetype=metadata["archetype"],
                    max_depth=metadata["max_depth"],
                    allowed_split_types=tuple(metadata["allowed_split_types"]),
                    realized_depth=metadata["realized_depth"],
                    num_leaves=metadata["num_leaves"],
                    num_internal_nodes=metadata["num_internal_nodes"],
                )

            dsl_info = DSLInfoL1(
                key_definition=self.game.key_definition,
                feature_names=self.game.feature_names,
                key_output_sizes=self.game.key_output_sizes,
                examples_with_keys=tuple(
                    (e.a, e.b, e.c, e.key) for e in self.game.examples
                ),
                target_key=self.game.true_key,
                observed_key_map={str(k): v for k, v in key_to_c.items()},
                conflicts=tuple((str(k), c1, c2) for (k, c1, c2) in conflicts),
                generator=generator_info,
            )

        elif self.cfg.difficulty == 2:
            # Difficulty 2: Show selected feature names + output_sizes, not order
            generator_info = None
            if self.game.generator is not None:
                metadata = self.game.generator.get_metadata()
                generator_info = GeneratorInfo(
                    archetype=metadata["archetype"],
                    max_depth=metadata["max_depth"],
                    allowed_split_types=tuple(metadata["allowed_split_types"]),
                    realized_depth=metadata["realized_depth"],
                    num_leaves=metadata["num_leaves"],
                    num_internal_nodes=metadata["num_internal_nodes"],
                )

            dsl_info = DSLInfoL2(
                selected_features=tuple(
                    (name, output_size)
                    for name, output_size, _ in self.game.selected_features
                ),
                note="Features shown (unordered). Find the correct order.",
                examples_with_keys=tuple(
                    (e.a, e.b, e.c, e.key) for e in self.game.examples
                ),
                generator=generator_info,
            )

        elif self.cfg.difficulty == 3:
            # Difficulty 3: Show full feature pool, not which selected
            generator_info = None
            if self.game.generator is not None:
                metadata = self.game.generator.get_metadata()
                generator_info = GeneratorInfo(
                    archetype=metadata["archetype"],
                    max_depth=metadata["max_depth"],
                    allowed_split_types=tuple(metadata["allowed_split_types"]),
                )

            dsl_info = DSLInfoL3(
                feature_pool=tuple(
                    (name, output_size)
                    for name, output_size, _ in self.game.feature_pool
                ),
                note="Full feature pool shown. Subset selected for this game (unknown).",
                generator=generator_info,
            )

        # Difficulty 4: Show nothing (dsl_info remains None)

        return Observation(
            status="run_over" if self.run_over else "running",
            time_spent=self.time_spent,
            time_remaining=max(0, self.cfg.lifetime_time_budget - self.time_spent),
            accuracy=self.accuracy,
            meta=self.meta,
            allowed_commands=allowed_commands,
            n=self.cfg.n,
            target_query=(self.game.x, self.game.y),
            examples=tuple((e.a, e.b, e.c) for e in self.game.examples),
            phase="TEST" if self.game.test_phase else "PLAY",
            reward_hint=reward_hint,
            pair_conflicts_detected=pair_conflicts,
            key_conflicts_detected=key_conflicts if self.cfg.difficulty <= 2 else None,
            heldout_queries=(
                tuple(self.game.heldout_pairs) if self.game.test_phase else ()
            ),
            dsl=dsl_info,
        )

    def _info(self, msg: str) -> Dict:
        return {
            "msg": msg,
            "time_spent": self.time_spent,
            "time_remaining": max(0, self.cfg.lifetime_time_budget - self.time_spent),
            "accuracy": self.accuracy,
            "meta": self.meta,
            "run_over": self.run_over,
        }


# ----------------------------
# Terminal UI
# ----------------------------


def _print_decision_tree(
    node: DecisionTreeNode, prefix: str = "", is_last: bool = True
) -> None:
    """Pretty-print a decision tree.

    Branch navigation:
    - First branch (├──): condition is TRUE
    - Second branch (└──): condition is FALSE
    """
    connector = "└── " if is_last else "├── "

    if node.is_leaf:
        print(f"{prefix}{connector}Leaf: output={node.output}")
    else:
        if node.split_type == "threshold":
            condition = f"k[{node.split_dim}] <= {node.split_value}"
        elif node.split_type == "equality":
            condition = f"k[{node.split_dim}] == {node.split_value}"
        elif node.split_type == "set":
            condition = f"k[{node.split_dim}] in {{{', '.join(map(str, sorted(node.split_value)))}}}"
        else:
            condition = f"k[{node.split_dim}] ? {node.split_value}"

        print(f"{prefix}{connector}Split: {condition}")

        extension = "    " if is_last else "│   "
        _print_decision_tree(node.left, prefix + extension, False)
        _print_decision_tree(node.right, prefix + extension, True)


def _print_instructions(cfg: EnvConfig) -> None:
    print("\n" + "=" * 78)
    print("NEW GAME - INSTRUCTIONS")
    print("=" * 78)
    print("Goal: Use examples to predict the output for the target query.")
    print("      Then LOCK_IN your answer, or ABANDON if the game is ill-posed.")
    print("\nActions (with time costs):")
    print(f"  poll           - Get random example (cost: {cfg.c_poll})")
    print(f"  query <a> <b>  - Get example for specific (a,b) (cost: {cfg.c_active})")
    print(f"  lock <answer>  - Submit target prediction (cost: {cfg.c_lock})")
    print("                   If wrong: round ends immediately and auto-advances.")
    print("                   If correct: enters held-out test phase (no points yet).")
    print(
        f"  submit <y1>... - Submit answers for {cfg.heldout_k} held-out queries (cost: 0)"
    )
    print(
        f"                   Perfect submission earns +{cfg.base_reward}. Ends round."
    )
    print(
        f"  abandon        - Quit this game and auto-advance to next (cost: {cfg.c_quit})"
    )
    print("\nDifficulty Level:", cfg.difficulty)
    if cfg.difficulty == 1:
        print("  - Full key formula + observed key map revealed")
    elif cfg.difficulty == 2:
        print("  - Selected feature names + output_sizes shown (order unknown)")
    elif cfg.difficulty == 3:
        print("  - Full feature pool shown (selected features unknown)")
    elif cfg.difficulty == 4:
        print("  - No DSL information (black box)")
    print(
        f"\nInconsistent penalties: {'ENABLED' if cfg.penalize_inconsistent_wrong else 'DISABLED'}"
    )
    if cfg.q_illposed > 0:
        print(f"Ill-posed games: {cfg.q_illposed * 100:.0f}% of games may be ill-posed")
    print("\nScoring:")
    print("  Points are awarded only from the held-out test after correct lock-in:")
    print(
        f"    - Perfect test (all {cfg.heldout_k} queries correct): +{cfg.base_reward}"
    )
    print("    - Any wrong: +0")
    print(
        f"  Wrong lock-in: 0 for consistent, -{cfg.base_reward} for inconsistent"
        if cfg.penalize_inconsistent_wrong
        else "  Wrong lock-in: +0"
    )
    print("  Meta: +1 for rightful abandon, -1 for wrongful abandon")
    print(f"  Time budget: {cfg.lifetime_time_budget} (run ends when exhausted)")
    print("=" * 78 + "\n")


def _print_help(cfg: EnvConfig) -> None:
    print("""
Commands:
  poll
      Passive example.
  query <a> <b>
      Active example for chosen (a,b).
  lock <answer>
      Submit target prediction.
      If correct: transitions to test phase (no points yet).
      If wrong: ends round and auto-advances.
  submit <y1> <y2> ... <yK>""")
    print(f"      Submit answers for {cfg.heldout_k} held-out test queries.")
    print(f"      Perfect submission earns +{cfg.base_reward}, otherwise +0.")
    print("""      Submitting ends the round and advances.
  abandon
      Quit this game (meta +/-1). Auto-advances to next game.
  status
      Reprint state.
  help
      Show this help.
  quit
      Exit.
""")


def _print_obs(obs: Dict) -> None:
    print("\n" + "=" * 78)
    print(f"Status: {obs['status']}")
    print(f"Time: {obs['time_spent']} spent | {obs['time_remaining']} remaining")
    print(f"Scores: accuracy={obs['accuracy']} | meta={obs['meta']}")
    print(f"Domain: 0..{obs.get('n', '?') - 1 if 'n' in obs else '?'}")

    if obs.get("reward_hint") is not None:
        print(f"Reward hint: {obs['reward_hint']}")

    if obs.get("pair_conflicts_detected"):
        print(
            "!! Pair conflict: same (a,b) with different outputs (non-deterministic) !!"
        )
    if obs.get("key_conflicts_detected"):
        print("!! Key conflict: same key with different outputs (contradiction) !!")

    if "dsl" in obs:
        dsl = obs["dsl"]

        if "key_definition" in dsl:
            # Difficulty 1: Full revelation
            print(f"DSL: {dsl['key_definition']}")
            if "generator" in dsl:
                gen = dsl["generator"]
                print(f"Generator: {gen['archetype']}")
                print(
                    f"  Depth: max={gen['max_depth']}, realized={gen['realized_depth']}"
                )
                print(
                    f"  Nodes: {gen['num_leaves']} leaves, "
                    f"{gen['num_internal_nodes']} internal"
                )
                print(f"  Split types: {', '.join(gen['allowed_split_types'])}")
            print("Examples (a # b = c) [key shown]:")
            for a, b, c, k in obs.get("examples_with_keys", []):
                print(f"  {a} # {b} = {c}   key={k}")
            x, y = obs["target_query"]
            print(f"Target query: {x} # {y} = ?   (target key={obs.get('target_key')})")
            if obs.get("observed_key_map"):
                print("Observed key→value map:", obs["observed_key_map"])

        elif "selected_features" in dsl:
            # Difficulty 2: Feature names + output_sizes (unordered)
            print("Selected features (unordered):")
            for feat in dsl["selected_features"]:
                print(f"  {feat['name']} (output_size {feat['output_size']})")
            if "generator" in dsl:
                gen = dsl["generator"]
                print(f"Generator: {gen['archetype']}")
                print(
                    f"  Depth: max={gen['max_depth']}, realized={gen['realized_depth']}"
                )
                print(
                    f"  Nodes: {gen['num_leaves']} leaves, "
                    f"{gen['num_internal_nodes']} internal"
                )
                print(f"  Split types: {', '.join(gen['allowed_split_types'])}")
            print("Examples (a # b = c) [key shown]:")
            for a, b, c, k in obs.get("examples_with_keys", []):
                print(f"  {a} # {b} = {c}   key={k}")
            x, y = obs["target_query"]
            print(f"Target query: {x} # {y} = ?")

        elif "feature_pool" in dsl:
            # Difficulty 3: Full pool (selected unknown)
            print("Feature pool (subset selected for this game):")
            for feat in dsl["feature_pool"]:
                print(f"  {feat['name']} (output_size {feat['output_size']})")
            if "generator" in dsl:
                gen = dsl["generator"]
                print(f"Generator: {gen['archetype']}")
                print(f"  Max depth: {gen['max_depth']}")
                print(f"  Split types: {', '.join(gen['allowed_split_types'])}")
            print("Examples (a # b = c):")
            for a, b, c in obs.get("examples", []):
                print(f"  {a} # {b} = {c}")
            x, y = obs["target_query"]
            print(f"Target query: {x} # {y} = ?")
    else:
        # Difficulty 4: No DSL info
        print("Examples (a # b = c):")
        for a, b, c in obs.get("examples", []):
            print(f"  {a} # {b} = {c}")
        x, y = obs.get("target_query", ("?", "?"))
        print(f"Target query: {x} # {y} = ?")

    if obs.get("test_phase"):
        print("\n" + "=" * 78)
        print("TEST PHASE: Submit answers for held-out queries")
        print("=" * 78)
        print(
            "You correctly answered the target query! Now predict outputs for these held-out queries:"
        )
        for i, (a, b) in enumerate(obs["heldout_queries"], 1):
            print(f"  Query {i}: {a} # {b} = ?")
        print(
            f"\nUse 'submit <y1> <y2> ... <y{len(obs['heldout_queries'])}>' to submit your predictions."
        )
        print("All predictions must be correct to earn +100 points.")
        print("=" * 78)

    print("=" * 78)


def play_terminal(cfg: EnvConfig, seed: Optional[int] = None) -> None:
    rng = random.Random(seed) if seed is not None else random.Random()
    env = LUTEnv(cfg, rng=rng)
    obs = env.reset().to_dict()

    if env.game and not env.game.instructions_shown:
        _print_instructions(cfg)
        env.game.instructions_shown = True

    _print_obs(obs)
    _print_help(cfg)

    while True:
        if obs["status"] == "run_over":
            print("\n*** Lifetime time budget exhausted. Run over. ***")
            break

        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "help":
            _print_help(cfg)
            continue
        if cmd == "status":
            _print_obs(obs)
            continue

        cmd_to_action: Dict[str, ActionType] = {
            "poll": "POLL_PASSIVE",
            "query": "POLL_ACTIVE",
            "lock": "LOCK_IN",
            "submit": "TEST_SUBMIT",
            "abandon": "ABANDON",
        }

        allowed = obs.get("allowed_commands", [])
        if cmd in cmd_to_action:
            action_type = cmd_to_action[cmd]
            if action_type not in allowed:
                if "TEST_SUBMIT" in allowed:
                    print(
                        "Test phase active. Use 'submit <y1> <y2> ...' to submit test predictions."
                    )
                else:
                    print(f"Command '{cmd}' is not allowed in the current game state.")
                continue

        if cmd == "poll":
            res = env.step(PollPassive())
        elif cmd == "query":
            if len(parts) != 3:
                print("Usage: query <a> <b>")
                continue
            try:
                a = int(parts[1])
                b = int(parts[2])
            except ValueError:
                print("a and b must be integers.")
                continue
            res = env.step(PollActive(a=a, b=b))
        elif cmd == "lock":
            if len(parts) != 2:
                print("Usage: lock <answer>")
                continue
            try:
                ans = int(parts[1])
            except ValueError:
                print("Answer must be an integer.")
                continue
            res = env.step(LockIn(answer=ans))
        elif cmd == "submit":
            if len(parts) < 2:
                print("Usage: submit <y1> <y2> ... <yK>")
                continue
            try:
                predictions = [int(p) for p in parts[1:]]
            except ValueError:
                print("All predictions must be integers.")
                continue
            res = env.step(TestSubmit(predictions=tuple(predictions)))
        elif cmd == "abandon":
            res = env.step(Abandon())
        else:
            print("Unknown command. Type 'help' for options.")
            continue

        obs = res.observation.to_dict()

        if isinstance(res.event, EventNone):
            pass
        elif isinstance(res.event, RejectedEvent):
            print(f"Action rejected ({res.event.action_type}): {res.event.reason}")
        elif isinstance(res.event, EnterTestEvent):
            print("\n✓ Correct target!")
            print(
                f"Proceeding to test phase with {res.event.heldout_k} held-out queries..."
            )
            print("(Answer all test queries correctly to earn +100 points)")
        elif isinstance(res.event, RoundEndEvent):
            outcome = res.event.outcome
            reward_delta = res.event.reward_delta
            meta_delta = res.event.meta_delta
            generator = res.event.generator

            if outcome == "LOCK_IN_CORRECT_NO_TEST":
                print(
                    f"\n✓ Correct answer! (No held-out available, +{reward_delta} points)"
                )
            elif outcome == "LOCK_IN_WRONG_CONSISTENT":
                print("\n❌ Wrong answer (but consistent with observations). +0 points")
            elif outcome == "LOCK_IN_WRONG_INCONSISTENT":
                print(
                    f"\n❌ Wrong answer (inconsistent with observations). {reward_delta} points"
                )
            elif outcome == "TEST_SUBMIT_CORRECT":
                print(f"\n🎉🎉 Perfect test score! +{reward_delta} points!")
            elif outcome == "TEST_SUBMIT_WRONG":
                print("\n❌ Test failed. Some predictions were incorrect. +0 points")
            elif outcome == "TEST_SUBMIT_INVALID":
                print(
                    "\n❌ Invalid test submission (wrong number of predictions). +0 points"
                )
            elif outcome == "ABANDON_RIGHTFUL":
                print(f"\n✓ Rightful abandon! Meta +{meta_delta}")
            elif outcome == "ABANDON_WRONGFUL":
                print(f"\n✗ Wrongful abandon. Meta {meta_delta}")

            if generator is not None and isinstance(generator, DecisionTreeLUT):
                print("\n" + "=" * 78)
                print("COMPLETED GAME - DECISION TREE STRUCTURE")
                print("(First branch ├── = TRUE, Second branch └── = FALSE)")
                print("=" * 78)
                _print_decision_tree(generator.root)
                print("=" * 78)

        if env.game and not env.game.instructions_shown:
            _print_instructions(cfg)
            env.game.instructions_shown = True

        _print_obs(obs)

        if res.done:
            print("\n*** Run complete. Final totals ***")
            print(f"Time spent: {obs['time_spent']}")
            print(f"Accuracy:   {obs['accuracy']}")
            print(f"Meta:       {obs['meta']}")
            break


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true", help="Play in the terminal")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")

    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--lifetime", type=int, default=50)
    ap.add_argument("--seed_examples", type=int, default=3)

    ap.add_argument("--c_poll", type=int, default=1)
    ap.add_argument("--c_active", type=int, default=2)
    ap.add_argument("--c_lock", type=int, default=1)
    ap.add_argument("--c_quit", type=int, default=1)

    ap.add_argument("--active_includes_passive", action="store_true")
    ap.add_argument("--q_illposed", type=float, default=0.0)

    ap.add_argument(
        "--difficulty",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Difficulty level (1=easiest, 4=hardest)",
    )
    ap.add_argument("--max_keys", type=int, default=128, help="Maximum key space size")

    ap.add_argument(
        "--generator",
        type=str,
        default="tree",
        choices=["tree", "random"],
        help="LUT generator archetype (tree=decision tree, random=i.i.d.)",
    )
    ap.add_argument(
        "--heldout_k",
        type=int,
        default=8,
        help="Number of held-out test queries after correct lock-in",
    )

    ap.add_argument(
        "--reward_info", choices=["perfect", "hinted", "hidden"], default="perfect"
    )
    args = ap.parse_args(argv)

    cfg = EnvConfig(
        n=args.n,
        seed_examples=args.seed_examples,
        lifetime_time_budget=args.lifetime,
        c_poll=args.c_poll,
        c_active=args.c_active,
        c_lock=args.c_lock,
        c_quit=args.c_quit,
        active_includes_passive=args.active_includes_passive,
        q_illposed=max(0.0, min(1.0, args.q_illposed)),
        difficulty=args.difficulty,
        max_keys=args.max_keys,
        generator_archetype=args.generator,
        heldout_k=args.heldout_k,
        reward_info=args.reward_info,
    )

    if args.play:
        play_terminal(cfg, seed=args.seed)
        return 0

    env = LUTEnv(cfg, rng=random.Random(args.seed))
    print(env.reset())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
