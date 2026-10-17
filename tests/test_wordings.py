import random

from vjev.questions import VF_QUESTIONS
from vjev.tasks import AI_QUESTION, keep, make_item
from vjev.wordings import N_TRAIN, WORDINGS, wording


def test_first_wording_is_the_canonical_one():
    for k, (_col, ins, _crit) in VF_QUESTIONS.items():
        assert WORDINGS[("vf", k)][0] == ins
    assert WORDINGS[("gv", "ai_generated")][0] == AI_QUESTION["instructions"]


def test_train_and_heldout_wordings_never_overlap():
    rng = random.Random(0)
    rec = {"id": "x", "query": "a dog runs"}
    for (task, q), options in WORDINGS.items():
        seen = {wording(task, q, rec | {"id": str(i)}, "train", rng) for i in range(200)}
        held = {wording(task, q, rec | {"id": str(i)}, "heldout") for i in range(200)}
        fmt = {o.format(query=rec["query"]) for o in options}
        assert seen <= {o.format(query=rec["query"]) for o in options[:N_TRAIN]}
        assert held <= {o.format(query=rec["query"]) for o in options[N_TRAIN:]}
        assert not seen & held and fmt


def test_heldout_wording_is_fixed_per_clip_and_shared_record_untouched():
    rec = {"id": "gv1", "real": False}
    a = make_item("gv", rec, 12, 1.5, [], wording_mode="heldout")
    b = make_item("gv", rec, 12, 1.5, [], wording_mode="heldout")
    q = a.record["questions"]["ai_generated"]["instructions"]
    assert q == b.record["questions"]["ai_generated"]["instructions"]
    assert q != AI_QUESTION["instructions"]
    assert AI_QUESTION["instructions"] == WORDINGS[("gv", "ai_generated")][0]


def test_qv_wording_keeps_the_query():
    rec = {"id": "qv1", "query": "a man opens a fridge", "windows": [[2, 6]],
           "clip_ids": [1, 2], "saliency_scores": [[4, 4, 4], [2, 2, 2]]}
    it = make_item("qv", rec, 20, 20.0, [], wording_mode="heldout")
    assert "a man opens a fridge" in it.record["questions"]["moment"]["instructions"]


def test_held_out_family_and_laugh_question():
    c1, h1 = {"kind": "C1"}, {"kind": "H1"}
    assert not keep("fq", c1, True) and keep("fq", c1, False) and keep("fq", h1, True)
    assert keep("fqc", c1, False) and not keep("fqc", h1, False)
    laugh = make_item("vsl", {"id": "vs1", "label": "baby laughter"}, 20, 10.0, [])
    talk = make_item("vsl", {"id": "vs2", "label": "male speech, man speaking"}, 20, 10.0, [])
    assert laugh.targets["laughing"] == 0 and talk.targets["laughing"] == 1
