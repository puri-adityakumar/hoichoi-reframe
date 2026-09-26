"""Structured AI decision log: every AI choice in a run is captured here and
passed between modules, then written to the manifest / decisions table."""
from __future__ import annotations

from typing import Any, Optional


class DecisionLog:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add(self, stage: str, choice: str, reason: str,
            confidence: Optional[float] = None) -> None:
        self._items.append({
            "stage": stage,
            "choice": choice,
            "reason": reason,
            "confidence": confidence,
        })

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)
