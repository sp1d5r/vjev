from vjev.model import _moment_to_score, moment_bins, video_group_spans


def test_bins_follow_frame_pairs():
    assert moment_bins(32, 2.0) == [(i * 1.0, i * 1.0 + 1.0) for i in range(16)]
    assert moment_bins(32, 2.0, bin_s=2.0)[-1] == (14.0, 16.0)
    assert moment_bins(10, 2.0, bin_s=2.0)[-1] == (4.0, 5.0)  # last bin is short
    assert len(moment_bins(24, 8.0, bin_s=1.0)) == 3


def test_group_spans_with_and_without_audio():
    v, a, o = 9, 5, 1
    assert video_group_spans([o, v, v, a, v, v, a, o], v, 2) == [(1, 3), (4, 6)]
    assert video_group_spans([o, v, v, v, v, o], v, 2) == [(1, 3), (3, 5)]


def test_moment_becomes_score_with_one_option_per_bin():
    rec = {"state": {}, "questions": {
        "when": {"type": "moment", "instructions": "When is the goal?"},
        "ai": {"type": "noul", "instructions": "Is this AI?"}}}
    out, bins = _moment_to_score(rec, 8, 2.0)
    assert out["questions"]["when"]["type"] == "score"
    assert out["questions"]["when"]["criteria"] == ["0 to 1 s", "1 to 2 s", "2 to 3 s", "3 to 4 s"]
    assert out["questions"]["ai"] == rec["questions"]["ai"]
    assert list(bins) == ["when"]


def test_matched_frames_hide_length_and_shape():
    import numpy as np

    from vjev.tasks import prepare_frames

    long_wide = prepare_frames("gv", np.zeros((48, 360, 640, 3), np.uint8))
    short_tall = prepare_frames("gv", np.zeros((9, 576, 320, 3), np.uint8))
    assert long_wide.shape == short_tall.shape == (12, 256, 256, 3)
    assert prepare_frames("vf", np.zeros((9, 4, 6, 3), np.uint8)).shape == (9, 4, 6, 3)
