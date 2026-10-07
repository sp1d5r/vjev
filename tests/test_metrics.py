import numpy as np

from vjev import metrics


def test_perfect_ranking():
    x = np.array([0.1, 0.4, 0.7, 0.9])
    assert metrics.srocc(x, x) == 1.0
    assert abs(metrics.plcc(x, x) - 1.0) < 1e-9


def test_ecr_level_bins():
    edges = np.array([0.25, 0.5, 0.75])
    assert metrics.ecr_level(np.array([0.0, 0.25, 0.6, 0.99]), edges).tolist() == [0, 1, 2, 3]


def test_brier_and_ece_perfect():
    probs = np.eye(3)
    levels = np.array([0, 1, 2])
    assert metrics.brier(probs, levels) == 0.0
    assert metrics.ece(probs, levels) == 0.0


def test_brier_uniform():
    probs = np.full((2, 4), 0.25)
    # (0.75^2 + 3 * 0.25^2) = 0.75
    assert abs(metrics.brier(probs, np.array([0, 3])) - 0.75) < 1e-9


def test_ece_overconfident():
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    levels = np.array([0, 1])  # 50% right at 90% confidence
    assert abs(metrics.ece(probs, levels) - 0.4) < 1e-9


def test_ecr_criteria_levels():
    from vjev.questions import ecr_criteria

    crit = ecr_criteria([0.25, 0.5, 0.75])
    assert len(crit) == 4
    assert crit[0] == "0% to 25% of viewers keep watching"
    assert crit[-1] == "75% to 100% of viewers keep watching"
