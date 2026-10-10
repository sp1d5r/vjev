"""CLIP + small classifier baseline for any score or yes/no task (VideoFeedback, GenVideo).

    modal run vjev/clip_baseline.py::embed --task gv          # CLIP features, once per task
    modal run vjev/clip_baseline.py::fit --task gv            # fit on train, score on val
    modal run vjev/clip_baseline.py::fit --task gv --test     # also score test, once

Features: CLIP ViT-L/14 on 8 evenly spaced frames, first frame plus frame mean (as in E003).
Each question gets an MLP classifier (one hidden layer of 512) fit on 90% of train, and one
temperature fit on the other 10%. Nothing is tuned on val. Metrics come from `vjev.tasks`,
so they match the Omni runs exactly.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from vjev.common import DATA, HF, app, baseline_image, data_vol, hf_vol
from vjev.simple_baselines import CLIP_MODEL, FRAMES, _scale, _temperature


@app.function(image=baseline_image, gpu="L40S", cpu=8, memory=32768, timeout=3 * 3600,
              volumes={DATA: data_vol, HF: hf_vol})
def embed_task(task: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from transformers import CLIPModel, CLIPProcessor

    from vjev.preprocess import load, pre_path
    from vjev.train import _records

    root = pathlib.Path(DATA)
    clip = CLIPModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    counts = {}
    for split in ("train", "val", "test"):
        out = root / "feats" / f"{task}_{split}.npz"
        if out.exists():
            continue
        recs = _records(split, 0, 0, task)

        def frames(rec: dict) -> np.ndarray:
            f, _, _ = load(pre_path(root, rec["id"]))
            return f[np.linspace(0, len(f) - 1, FRAMES).round().astype(int)]

        feats = []
        with ThreadPoolExecutor(8) as ex, torch.inference_mode():
            for f in ex.map(frames, recs):
                px = proc(images=list(f), return_tensors="pt")["pixel_values"].cuda().half()
                emb = clip.get_image_features(pixel_values=px)
                emb = (emb if isinstance(emb, torch.Tensor) else emb.pooler_output).float()
                emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
                feats.append(np.concatenate([emb[0], emb.mean(0)]))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, ids=np.array([r["id"] for r in recs]), x=np.stack(feats),
                 recs=np.array([json.dumps(r) for r in recs]))
        data_vol.commit()
        counts[split] = len(recs)
    return counts


@app.function(image=baseline_image, cpu=16, memory=32768, timeout=3600,
              volumes={DATA: data_vol})
def fit_task(task: str, test: bool = False) -> dict:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from vjev.tasks import make_item, question_report

    root = pathlib.Path(DATA)
    edges = json.loads((root / "index" / "meta.json").read_text())["level_edges"]
    splits = ["train", "val", *(["test"] if test else [])]
    data = {}
    for s in splits:
        z = np.load(root / "feats" / f"{task}_{s}.npz")
        recs = [json.loads(r) for r in z["recs"]]
        # n_frames and duration only matter for moment questions, which CLIP cannot answer.
        items = [make_item(task, r, 2, 1.0, edges) for r in recs]
        data[s] = (z["x"], items, recs)
    x_tr, it_tr, _ = data["train"]
    n_fit = int(len(x_tr) * 0.9)
    results, preds = {}, {}
    for q, kind in it_tr[0].kinds.items():
        y = np.array([it.targets[q] for it in it_tr])
        k = int(max(y.max() + 1, 2 if kind == "noul" else 4))
        clf = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(512,), early_stopping=True, max_iter=200, random_state=0))
        clf.fit(x_tr[:n_fit], y[:n_fit])

        def proba(x, clf=clf, k=k):
            p = np.zeros((len(x), k))
            p[:, clf.classes_] = clf.predict_proba(x)
            return p

        t = _temperature(proba(x_tr[n_fit:]), y[n_fit:])
        for s in splits[1:]:
            x, items, recs = data[s]
            p = _scale(proba(x), t)
            r = question_report(kind, list(p), [it.targets[q] for it in items],
                                [it.extra.get(q, {}) for it in items])
            results[f"{s}/{q}"] = r | {"temperature": t}
            preds.setdefault(s, {})[q] = p
    for s in splits[1:]:
        qs = list(it_tr[0].kinds)
        results[f"{s}/final"] = float(np.mean([results[f"{s}/{q}"]["final"] for q in qs]))
    if test:
        out = root / "preds" / f"baseline-clip-{task}-test.jsonl"
        if out.exists():
            raise RuntimeError(f"{out} exists; test is scored once per final model")
        _, items, recs = data["test"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps({"id": r["id"], "scores": {
            q: {"target": it.targets[q], "probs": preds["test"][q][i].tolist()}
            for q in preds["test"]}}) + "\n" for i, (r, it) in enumerate(zip(recs, items, strict=True))))
        data_vol.commit()
    return results


@app.local_entrypoint()
def embed(task: str = "gv"):
    print(embed_task.remote(task))


@app.local_entrypoint()
def fit(task: str = "gv", test: bool = False):
    print(json.dumps(fit_task.remote(task, test), indent=1, default=float))
