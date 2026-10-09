import numpy as np
import torch

from vjev.questions import VF_QUESTIONS
from vjev.train import evaluate


class FakeNet:
    """Returns one logit vector per question; the top level follows the target."""

    def __init__(self, k: int):
        self.k = k

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, enc):
        return [torch.nn.functional.one_hot(torch.tensor(t), self.k).float() * 3 for t in enc]


def test_single_question_keeps_snapugc_rows():
    edges = [0.25, 0.5, 0.75]
    loader = [([lv], [lv], [ecr], f"id{i}")
              for i, (lv, ecr) in enumerate([(0, 0.1), (1, 0.3), (2, 0.6), (3, 0.9)])]
    res, rows = evaluate(FakeNet(4), loader, edges)
    assert res["srocc"] > 0.99 and res["n"] == 4
    assert set(rows[0]) == {"id", "ecr", "probs", "expected"}


def test_vf_reports_every_question_and_means():
    rng = np.random.default_rng(0)
    loader = []
    for i in range(20):
        lv = rng.integers(0, 4, len(VF_QUESTIONS)).tolist()
        loader.append((lv, lv, [float(v) for v in lv], f"vf{i}"))
    res, rows = evaluate(FakeNet(4), loader, [], task="vf")
    for q in VF_QUESTIONS:
        assert res[f"{q}/srocc"] > 0.99
    assert res["srocc"] > 0.99 and res["n"] == 20
    assert set(rows[0]["scores"]) == set(VF_QUESTIONS)
