import numpy as np
import torch

from vjev.questions import VF_QUESTIONS
from vjev.tasks import Item, moment_report, noul_report
from vjev.train import evaluate


class FakeNet:
    """Returns one logit vector per question, peaked at the given index."""

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, enc):
        return [torch.nn.functional.one_hot(torch.tensor(t), k).float() * 3 for t, k in enc]


def _item(task, qs, kinds, targets, extra):
    return Item({"state": {}, "questions": {q: {} for q in qs}}, 1.0, False,
                dict(zip(qs, targets, strict=True)), dict(zip(qs, kinds, strict=True)),
                dict(zip(qs, extra, strict=True)), task)


def test_single_question_keeps_snapugc_rows():
    edges = [0.25, 0.5, 0.75]
    loader = []
    for i, (lv, ecr) in enumerate([(0, 0.1), (1, 0.3), (2, 0.6), (3, 0.9)]):
        it = _item("snapugc", ["ecr"], ["score"], [lv], [{"raw": ecr, "edges": edges}])
        loader.append(([(lv, 4)], it, f"id{i}"))
    res, rows = evaluate(FakeNet(), loader, edges)
    assert res["srocc"] > 0.99 and res["n"] == 4
    assert set(rows[0]) == {"id", "ecr", "probs", "expected"}


def test_vf_reports_every_question_and_means():
    rng = np.random.default_rng(0)
    qs = list(VF_QUESTIONS)
    loader = []
    for i in range(20):
        lv = rng.integers(0, 4, len(qs)).tolist()
        it = _item("vf", qs, ["score"] * len(qs), lv,
                   [{"raw": v, "edges": [0.5, 1.5, 2.5]} for v in lv])
        loader.append(([(v, 4) for v in lv], it, f"vf{i}"))
    res, rows = evaluate(FakeNet(), loader, task="vf")
    for q in qs:
        assert res[f"{q}/srocc"] > 0.99
    assert res["srocc"] > 0.99 and res["n"] == 20
    assert set(rows[0]["scores"]) == set(qs)


def test_mixed_tasks_prefix_task_and_average_final():
    loader = []
    for i in range(10):
        y = i % 2
        loader.append(([(y, 2)], _item("gv", ["ai_generated"], ["noul"], [y], [{}]), f"gv{i}"))
        lv = i % 4
        it = _item("vf", ["visual_quality"], ["score"], [lv], [{"raw": lv, "edges": [0.5, 1.5, 2.5]}])
        loader.append(([(lv, 4)], it, f"vf{i}"))
    res, _ = evaluate(FakeNet(), loader, task="vf+gv")
    assert res["gv/ai_generated/auc"] == 1.0
    assert "vf/visual_quality/final" in res
    assert abs(res["final"] - np.mean([res["gv/ai_generated/final"],
                                       res["vf/visual_quality/final"]])) < 1e-9


def test_noul_report_auc_and_calibration_inputs():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    r = noul_report(probs, np.array([0, 1, 0, 1]))
    assert r["auc"] == 1.0 and r["acc"] == 1.0


def test_moment_hit_and_r1():
    sal = [[0, 0, 0], [4, 4, 3], [4, 3, 3], [0, 0, 0]]
    ex = {"saliency": sal, "windows": [[2, 6]], "bin_s": 2.0}
    p = np.array([0.05, 0.5, 0.4, 0.05])
    r = moment_report([p], [ex])
    assert abs(r["hit1"] - 2 / 3) < 1e-9
    assert r["r1_05"] == 1.0  # predicted span 2 to 6 s matches exactly
    assert abs(r["mass_in_moment"] - 0.9) < 1e-9


def test_window_target_and_windowed_hit():
    from vjev.tasks import window_target

    bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert window_target([[0.5, 1.5]], bins) == [0.5, 0.5, 0.0]
    assert window_target([[1.2, 1.3]], bins)[1] == 1.0
    ex = {"windows": [[1.0, 2.0]], "bin_s": 1.0}
    r = moment_report([np.array([0.1, 0.8, 0.1])], [ex])
    assert r["hit1"] == 1.0 and r["miou"] == 1.0


def test_choice_target_follows_sorted_option_order():
    from vjev.clef_schema import question_options
    from vjev.tasks import make_item

    it = make_item("vs", {"id": "vsx", "label": "people crowd"}, 20, 10.0, [])
    q = it.record["questions"]["sound"]
    keys = [k for k, _ in question_options(q)]
    assert keys[it.targets["sound"]] == "people crowd"


def test_moment_map_perfect_and_wrong():
    ex = {"windows": [[2.0, 6.0]], "bin_s": 2.0, "saliency": [[0, 0, 0], [4, 4, 4], [4, 4, 4], [0, 0, 0]]}
    good = moment_report([np.array([0.0, 0.5, 0.5, 0.0])], [ex])
    bad = moment_report([np.array([0.9, 0.0, 0.0, 0.1])], [ex])
    assert good["map_05"] == 1.0 and good["map_avg"] == 1.0
    assert bad["map_05"] == 0.0
