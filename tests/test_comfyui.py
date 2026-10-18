"""The ComfyUI package vendors shared code; it must match, and its resampling must follow the
training decode settings."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NODE = ROOT / "comfyui" / "ComfyUI-VideoJev"


def _engine():
    if "videojev.engine" not in sys.modules:
        sys.path.insert(0, str(NODE))
        spec = importlib.util.spec_from_file_location("videojev", NODE / "presets.py",
                                                      submodule_search_locations=[str(NODE)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["videojev"] = mod  # package shell without nodes.py (needs no ComfyUI)
    import videojev.engine

    return videojev.engine


def test_vendored_code_matches():
    assert (NODE / "presets.py").read_bytes() == (ROOT / "space" / "presets.py").read_bytes()
    for name in ("__init__.py", "clef_schema.py", "model.py"):
        assert (NODE / "vjev" / name).read_bytes() == (ROOT / "release" / "vjev" / name).read_bytes(), \
            f"run scripts/sync_comfyui.sh ({name})"


def test_prepare_resamples_to_profile():
    engine = _engine()
    frames = np.zeros((300, 720, 1280, 3), np.uint8)  # 10 s at 30 fps
    audio = np.zeros(44100 * 10, np.float32)
    f, a, dur = engine.prepare(frames, 30.0, audio, 44100, engine.PROFILES["moment"])
    assert f.shape == (20, 190, 336, 3) and dur == 10.0 and len(a) == 160000
    f, a, _ = engine.prepare(frames, 30.0, audio, 44100, engine.PROFILES["ai"])
    assert f.shape == (12, 256, 256, 3) and a is None
    f, _, dur = engine.prepare(frames, 30.0, None, 44100, engine.PROFILES["quality"])
    assert f.shape[0] == 48 and dur == 6
    # Sizes ffmpeg's scale=336:-2 gives for these inputs (measured).
    for (h, w), out in {(700, 1000): 236, (1081, 1920): 190, (481, 640): 252}.items():
        f, _, _ = engine.prepare(np.zeros((4, h, w, 3), np.uint8), 2.0, None, 1,
                                 engine.PROFILES["sound"])
        assert f.shape[1:3] == (out, 336)


def test_span_grows_over_half_peak():
    engine = _engine()
    bins = [[0, 1, .05], [1, 2, .2], [2, 3, .4], [3, 4, .25], [4, 5, .1]]
    assert engine.span(bins) == {"start_s": 1, "end_s": 4, "peak_s": 2, "p_peak": .4,
                                 "p_span": .85}
