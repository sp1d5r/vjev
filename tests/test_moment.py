from vjev.model import _moment_to_score, moment_bins, video_group_spans


def test_bins_follow_frame_pairs():
    assert moment_bins(32, 2.0) == [(i * 1.0, i * 1.0 + 1.0) for i in range(16)]
    assert moment_bins(32, 2.0, bin_s=2.0)[-1] == (14.0, 16.0)
    assert moment_bins(10, 2.0, bin_s=2.0)[-1] == (4.0, 5.0)  # last bin is short
    assert len(moment_bins(24, 8.0, bin_s=1.0)) == 3


def test_group_spans_skip_audio():
    v, a, o = 9, 5, 1
    ids = [o, v, v, a, v, v, a, o]
    assert video_group_spans(ids, v) == [(1, 3), (4, 6)]


def test_moment_becomes_score_with_one_option_per_bin():
    rec = {"state": {}, "questions": {
        "when": {"type": "moment", "instructions": "When is the goal?"},
        "ai": {"type": "noul", "instructions": "Is this AI?"}}}
    out, bins = _moment_to_score(rec, 8, 2.0)
    assert out["questions"]["when"]["type"] == "score"
    assert out["questions"]["when"]["criteria"] == ["0 to 1 s", "1 to 2 s", "2 to 3 s", "3 to 4 s"]
    assert out["questions"]["ai"] == rec["questions"]["ai"]
    assert list(bins) == ["when"]
