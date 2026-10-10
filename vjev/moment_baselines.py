"""Content-free baselines for moment questions: what do the metrics give without looking?

    modal run vjev/moment_baselines.py::moment_priors --task fq --split val --limit 400

- uniform: the same probability on every time bin.
- position prior: the mean target over train, by relative position in the clip (10 slots),
  stretched to each clip's length. It knows where moments usually are, not what happens.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from vjev.common import DATA, app, data_vol, gpu_image


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=8, memory=16384, timeout=3600)
def priors(task: str, split: str, limit: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from vjev.preprocess import pre_path
    from vjev.tasks import TASKS, make_item, moment_report
    from vjev.train import _records

    root = pathlib.Path(DATA)
    fps = TASKS[task][0]

    def item(rec: dict):
        z = np.load(pre_path(root, rec["id"]))
        n_frames = len(z["offsets"]) - 1
        return make_item(task, rec, n_frames, float(z["duration"]), [])

    with ThreadPoolExecutor(16) as ex:
        train = list(ex.map(item, _records("train", 0, 0, task)))
        test = list(ex.map(item, _records(split, limit, 0, task)))  # same clips as the run
    slots = np.zeros(10)
    for it in train:
        t = np.array(it.targets["moment"])
        pos = (np.arange(len(t)) + 0.5) / len(t)
        np.add.at(slots, np.minimum((pos * 10).astype(int), 9), t / max(1, len(t)) * 10)
    slots /= slots.sum()

    def prior(n: int) -> np.ndarray:
        p = slots[np.minimum(((np.arange(n) + 0.5) / n * 10).astype(int), 9)]
        return p / p.sum()

    extras = [it.extra["moment"] for it in test]
    lens = [len(it.targets["moment"]) for it in test]
    out = {"n": len(test), "fps": fps, "mean_bins": float(np.mean(lens)),
           "position_slots": slots.round(3).tolist(),
           "uniform": moment_report([np.ones(n) / n for n in lens], extras),
           "position_prior": moment_report([prior(n) for n in lens], extras)}
    return out


@app.local_entrypoint()
def moment_priors(task: str = "fq", split: str = "val", limit: int = 400):
    print(json.dumps(priors.remote(task, split, limit), indent=1))
