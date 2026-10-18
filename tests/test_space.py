"""The demo's preset questions must match what the model was trained on."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "space"))

import presets

from vjev import tasks, vggsound, wordings
from vjev.questions import VF_QUESTIONS


def test_presets_match_training():
    assert presets.AI_QUESTION == tasks.AI_QUESTION
    assert presets.SOUND_CLASSES == vggsound.CLASSES
    assert {k: (ins, crit) for k, (_c, ins, crit) in VF_QUESTIONS.items()} == \
        presets.QUALITY_QUESTIONS
    assert presets.QUERY == wordings.WORDINGS[("qv", "moment")][0]
    assert presets.FUNNY in wordings.FQ_HELDOUT["H1"]
    assert presets.MAGIC in wordings.FQ_HELDOUT["M1"]


def test_profiles_match_training_decode():
    for profile, task in [("moment", "fq"), ("query", "qv"), ("sound", "vs"), ("quality", "vf"),
                          ("ai", "vfai")]:
        fps, audio = tasks.TASKS[task]
        assert presets.PROFILES[profile].fps == fps
        assert presets.PROFILES[profile].audio == audio
    assert presets.MATCHED_FRAMES == tasks.MATCHED_FRAMES
    assert presets.MATCHED_SIDE == tasks.MATCHED_SIDE


def test_build_groups_by_profile():
    g = presets.build(presets.CHOICES, query="a dog barks", prompt="")
    assert set(g) == {"moment", "query", "sound", "ai", "quality"}
    assert "text_alignment" not in g["quality"]["questions"]  # no prompt given
    assert '"a dog barks"' in g["query"]["questions"]["described_moment"]["instructions"]
    assert presets.build([], "", "") == {}
