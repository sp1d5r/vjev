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


# VideoFeedback (TIGER-Lab): five questions, each scored 1 to 4 by human raters.
# Keys map to the dataset's column names. Criteria are in order, score 1 first.
VF_QUESTIONS = {
    "visual_quality": (
        "visual quality",
        ("This is a short AI-generated or real video. How good is its visual quality: "
         "clarity, resolution, brightness and colour, and freedom from visual artifacts?"),
        ["bad: blurry, heavily distorted or hard to recognise",
         "average: clearly visible artifacts or low resolution",
         "good: clear, with only small flaws",
         "perfect: as clean as real camera footage"]),
    "temporal_consistency": (
        "temporal consistency",
        ("Do the subjects and background stay consistent from frame to frame, without "
         "flicker, morphing, or things changing shape or identity?"),
        ["bad: objects change shape or identity, or the video flickers heavily",
         "average: noticeable flicker, morphing or jumps",
         "good: mostly stable, with small glitches",
         "perfect: fully consistent, like real footage"]),
    "dynamic_degree": (
        "dynamic degree",
        "How much do the subjects and the camera move?",
        ["almost static: nearly nothing moves",
         "small motion: a little movement",
         "clear motion: subjects or camera move noticeably",
         "rich motion: large, natural movement, like real footage"]),
    "text_alignment": (
        "text-to-video alignment",
        "How well does the video match the text prompt it was generated from?",
        ["does not match the prompt",
         "matches only part of the prompt",
         "matches most of the prompt",
         "matches the whole prompt"]),
    "factual_consistency": (
        "factual consistency",
        ("Is the video consistent with physics, anatomy and common sense? Look for impossible "
         "motion, malformed bodies or hands, and objects passing through each other."),
        ["bad: clear violations of physics, anatomy or common sense",
         "average: several implausible details",
         "good: small implausible details",
         "perfect: fully plausible, like real footage"]),
}


def vf_record(rec: dict, duration: float | None) -> dict:
    state: dict = {"prompt": rec.get("prompt", "")}
    if duration is not None:
        state["duration_seconds"] = round(duration, 1)
    return {
        "state": state,
        "questions": {k: {"type": "score", "instructions": ins, "criteria": crit}
                      for k, (_col, ins, crit) in VF_QUESTIONS.items()},
    }
