"""The typed questions we ask about each clip, in Jev / SystemOne form."""

from __future__ import annotations

import itertools

ECR_INSTRUCTIONS = (
    "This is a short user-generated video from Snapchat. What share of viewers keep "
    "watching past the first 5 seconds instead of swiping away?"
)


def ecr_criteria(edges: list[float]) -> list[str]:
    """One description per level. edges are the inner bin edges, ascending."""
    bounds = [0.0, *edges, 1.0]
    return [
        f"{lo:.0%} to {hi:.0%} of viewers keep watching"
        for lo, hi in itertools.pairwise(bounds)
    ]


def state_for(rec: dict, duration: float | None, text: bool) -> dict:
    state: dict = {}
    if duration is not None:
        state["duration_seconds"] = round(duration, 1)
    if text:
        state["title"] = rec.get("title", "")
        state["description"] = rec.get("description", "")
    return state


def record_for(rec: dict, edges: list[float], duration: float | None, text: bool) -> dict:
    return {
        "state": state_for(rec, duration, text),
        "questions": {
            "ecr": {
                "type": "score",
                "instructions": ECR_INSTRUCTIONS,
                "criteria": ecr_criteria(edges),
            }
        },
    }
