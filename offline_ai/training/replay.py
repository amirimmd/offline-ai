"""Experience replay buffer strategies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal


Strategy = Literal["random", "stratified", "importance"]


@dataclass
class ReplayExample:
    example_id: str
    text: str
    label: str | None = None
    importance: float = 1.0
    stratum: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self) -> None:
        self._items: list[ReplayExample] = []
        self._used: list[str] = []

    def add(self, example: ReplayExample) -> None:
        self._items.append(example)

    def extend(self, examples: list[ReplayExample]) -> None:
        self._items.extend(examples)

    def sample(
        self,
        n: int,
        *,
        strategy: Strategy = "random",
        new_examples: list[ReplayExample] | None = None,
        replay_ratio: float = 0.3,
    ) -> list[ReplayExample]:
        """Mix new examples with replayed old examples."""
        new_examples = new_examples or []
        n_replay = int(n * replay_ratio) if new_examples else n
        n_new = n - n_replay
        chosen_new = new_examples[:n_new]
        pool = self._items
        if strategy == "random":
            chosen_old = random.sample(pool, min(n_replay, len(pool))) if pool else []
        elif strategy == "stratified":
            by = {}
            for ex in pool:
                by.setdefault(ex.stratum, []).append(ex)
            chosen_old = []
            strata = list(by.keys()) or ["default"]
            i = 0
            while len(chosen_old) < n_replay and strata:
                s = strata[i % len(strata)]
                if by.get(s):
                    chosen_old.append(by[s].pop())
                i += 1
                if i > n_replay * 5:
                    break
        else:  # importance
            ranked = sorted(pool, key=lambda x: x.importance, reverse=True)
            chosen_old = ranked[:n_replay]
        batch = chosen_new + chosen_old
        self._used.extend(ex.example_id for ex in chosen_old)
        return batch

    @property
    def used_ids(self) -> list[str]:
        return list(self._used)
