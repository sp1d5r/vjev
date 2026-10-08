import numpy as np

from vjev.simple_baselines import _scale, _temperature


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 5000)
    logits = rng.normal(size=(5000, 10))
    logits[np.arange(5000), y] += 1.0
    p = np.exp(3 * logits)
    p /= p.sum(1, keepdims=True)
    assert abs(_temperature(p, y) - 3.0) < 0.2


def test_scale_keeps_distributions():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert np.allclose(_scale(p, 1.0), p)
    assert np.allclose(_scale(p, 2.0).sum(1), 1.0)
    assert _scale(p, 2.0)[0, 0] < 0.7
