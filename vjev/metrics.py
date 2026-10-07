"""Metrics for engagement prediction and calibration.

SROCC and PLCC follow the VQualA 2025 EVQA-SnapUGC challenge. The challenge final score is
reported as the mean of the two.
Brier and ECE score the typed answer distribution against the true ECR level.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def srocc(pred: np.ndarray, target: np.ndarray) -> float:
    return float(stats.spearmanr(pred, target).statistic)


def plcc(pred: np.ndarray, target: np.ndarray) -> float:
    return float(stats.pearsonr(pred, target).statistic)


def ecr_level(ecr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map ECR values to level indices given the inner bin edges."""
    return np.searchsorted(edges, ecr, side="right")


def brier(probs: np.ndarray, levels: np.ndarray) -> float:
    """Multi-class Brier score. probs is (n, k), levels is (n,) of true indices."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(levels)), levels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(probs: np.ndarray, levels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == levels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def report(expected: np.ndarray, probs: np.ndarray, ecr: np.ndarray, edges: np.ndarray) -> dict:
    """All headline numbers for one evaluation run."""
    levels = ecr_level(ecr, edges)
    s, p = srocc(expected, ecr), plcc(expected, ecr)
    return {
        "srocc": s,
        "plcc": p,
        "final": (s + p) / 2,
        "brier": brier(probs, levels),
        "ece": ece(probs, levels),
        "level_acc": float((probs.argmax(axis=1) == levels).mean()),
        "n": len(ecr),
    }
