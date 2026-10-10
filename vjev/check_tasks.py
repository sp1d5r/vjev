"""Encode a few real clips per task through the full data path and check the result.

    modal run vjev/check_tasks.py::check_tasks --task vf+gv+qv

Checks: one target per question, a moment's option spans lie on video tokens and match its
number of time bins, and the soft target sums to 1.
"""

from __future__ import annotations

import json

from vjev.common import DATA, HF, app, data_vol, hf_vol, train_image


@app.function(image=train_image, volumes={DATA: data_vol, HF: hf_vol}, cpu=8, memory=32768,
              timeout=1800)
def check(task: str, n: int, audio: bool = True) -> list[dict]:
    import pathlib

    from huggingface_hub import snapshot_download

    from vjev import model
    from vjev.train import ClipDataset, _records

    path = snapshot_download(model.OMNI_ID)
    edges = json.loads((pathlib.Path(DATA) / "index" / "meta.json").read_text())["level_edges"]
    recs = []
    for t in task.split("+"):
        recs += _records("val", n, 0, t)
    ds = ClipDataset(recs, edges, path, audio, task)
    vid = None
    out = []
    for i in range(len(ds)):
        enc, item, rid = ds[i]
        tok = ds.processor.tokenizer
        vid = vid or tok.convert_tokens_to_ids("<|video_pad|>")
        ids = enc.record.input_ids
        row = {"id": rid, "task": item.task, "tokens": len(ids), "audio": enc.audio}
        assert [q.question_id for q in enc.record.questions] == list(item.record["questions"])
        for q in enc.record.questions:
            t = item.targets[q.question_id]
            if item.kinds[q.question_id] == "moment":
                assert len(t) == len(q.option_spans), (len(t), len(q.option_spans))
                assert abs(sum(t) - 1) < 1e-6
                for a, b in q.option_spans:
                    assert ids[a] == vid and ids[b - 1] == vid, (a, b)
                row["bins"] = len(t)
                row["peak_bin"] = int(max(range(len(t)), key=t.__getitem__))
            else:
                assert 0 <= t < len(q.option_spans)
        out.append(row)
    return out


@app.local_entrypoint()
def check_tasks(task: str = "vf+gv", n: int = 3, audio: bool = True):
    for r in check.remote(task, n, audio):
        print(r)
