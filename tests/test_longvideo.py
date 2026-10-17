import itertools

import numpy as np

from vjev.longvideo import WINDOW_S, combine, window_starts


def test_windows_cover_the_whole_video():
    assert window_starts(40) == [0]
    starts = window_starts(185)
    assert starts[0] == 0 and starts[-1] + WINDOW_S == 185
    assert all(b - a <= 30 for a, b in itertools.pairwise(starts))


def test_combine_keeps_the_strongest_window_per_second():
    a = np.zeros(60)
    b = np.zeros(60)
    b[10] = 5.0  # second 40 of the video
    p_logit, p_prob = combine([(0, a), (30, b)], 90)
    assert p_logit.argmax() == 40 and p_prob.argmax() == 40
    assert np.isclose(p_logit.sum(), 1) and np.isclose(p_prob.sum(), 1)
