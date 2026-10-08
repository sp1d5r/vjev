"""Cheap baselines retrained on our split: text embeddings and CLIP frame features.

    modal run vjev/simple_baselines.py::embed_all   # features for every clip, once
    modal run vjev/simple_baselines.py::fit_all     # fit on train, score on val
    modal run vjev/simple_baselines.py::fit_all --calibrate   # + temperature scaling

Each model predicts the 10 ECR levels as a class distribution, so it gets the same typed
metrics (Brier, ECE) as the Omni model. A regression variant is also fit for ranking only.
Hyperparameters are fixed up front. Nothing is tuned on val, and test is not touched.
"""

from __future__ import annotations

import json
import pathlib

import modal
import numpy as np

from vjev.common import DATA, HF, app, baseline_image, data_vol, hf_vol

TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CLIP_MODEL = "openai/clip-vit-large-patch14"
FRAMES = 8  # sampled evenly from the first 16 s
SHARD = 2000
FEATS = "feats"


def _text(rec: dict) -> str:
    return f"{rec.get('title') or ''}\n{rec.get('description') or ''}".strip()


@app.function(image=baseline_image, gpu="L40S", cpu=8, memory=32768, timeout=3600,
              volumes={DATA: data_vol, HF: hf_vol}, max_containers=10)
def embed_shard(i: int, recs: list[dict]) -> int:
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    from vjev.preprocess import load, pre_path

    root = pathlib.Path(DATA)
    out = root / FEATS / f"shard_{i:03d}.npz"
    if out.exists():
        return 0
    recs = [r for r in recs if pre_path(root, r["id"]).exists()]
    text_model = SentenceTransformer(TEXT_MODEL, device="cuda")
    text = text_model.encode([_text(r) for r in recs], batch_size=128, normalize_embeddings=True)

    clip = CLIPModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def frames(rec: dict) -> np.ndarray:
        f, _, _ = load(pre_path(root, rec["id"]))
        return f[np.linspace(0, len(f) - 1, FRAMES).round().astype(int)]

    first, mean = [], []
    with ThreadPoolExecutor(8) as ex, torch.inference_mode():
        for f in ex.map(frames, recs):
            px = proc(images=list(f), return_tensors="pt")["pixel_values"].cuda().half()
            emb = clip.get_image_features(pixel_values=px)
            # transformers 5 returns an output object; pooler_output is the projected embedding.
            emb = (emb if isinstance(emb, torch.Tensor) else emb.pooler_output).float()
            assert emb.shape[-1] == clip.config.projection_dim, emb.shape
            emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
            first.append(emb[0])
            mean.append(emb.mean(0))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array([r["id"] for r in recs]), text=text.astype(np.float32),
             clip_first=np.stack(first), clip_mean=np.stack(mean))
    data_vol.commit()
    return len(recs)


@app.local_entrypoint()
def embed_all():
    vol = modal.Volume.from_name("vjev-snapugc")
    recs = []
    for split in ("train", "val", "test"):
        body = b"".join(vol.read_file(f"index/{split}.jsonl")).decode()
        recs += [json.loads(line) for line in body.splitlines()]
    shards = [(i, recs[s:s + SHARD]) for i, s in enumerate(range(0, len(recs), SHARD))]
    print(f"done {sum(embed_shard.starmap(shards))} clips")


def _temperature(probs: np.ndarray, levels: np.ndarray) -> float:
    """Temperature that minimises NLL of the class probabilities on held-out clips."""
    from scipy.optimize import minimize_scalar

    logp = np.log(np.clip(probs, 1e-12, 1.0))

    def nll(t: float) -> float:
        z = logp / t
        z -= z.max(axis=1, keepdims=True)
        z -= np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-z[np.arange(len(levels)), levels].mean())

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def _scale(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1.0)) / t
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _fit_one(kind: str, xs: dict, ys: dict, levels: dict, val500: np.ndarray,
             edges: np.ndarray, calibrate: bool = False) -> dict:
    """Fit on train and score on val.

    With calibrate, the classifier fits on 90% of train and a temperature is fit on the
    other 10%, so val stays untouched. The regression twin is skipped.
    """
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier, XGBRegressor

    from vjev.metrics import plcc, report, srocc

    if kind == "xgb":
        params = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6,
                  "subsample": 0.8, "colsample_bytree": 0.5, "tree_method": "hist",
                  "n_jobs": 16, "random_state": 0}
        cls, reg = XGBClassifier(**params), XGBRegressor(**params)
    else:
        params = {"hidden_layer_sizes": (512,), "alpha": 1e-3, "early_stopping": True,
                  "max_iter": 200, "random_state": 0}
        cls = make_pipeline(StandardScaler(), MLPClassifier(**params))
        reg = make_pipeline(StandardScaler(), MLPRegressor(**params))
    out, t = {}, 1.0
    if calibrate:
        hold = np.random.default_rng(0).random(len(levels["train"])) < 0.1
        cls.fit(xs["train"][~hold], levels["train"][~hold])
        t = _temperature(cls.predict_proba(xs["train"][hold]), levels["train"][hold])
        out["temperature"] = t
    else:
        cls.fit(xs["train"], levels["train"])
        reg.fit(xs["train"], ys["train"])

    for tag, mask in (("val500", val500), ("val", np.ones(len(ys["val"]), bool))):
        probs = _scale(cls.predict_proba(xs["val"][mask]), t)
        expected = probs @ np.arange(probs.shape[1])
        m = report(expected, probs, ys["val"][mask], edges)
        if not calibrate:
            r = reg.predict(xs["val"][mask])
            m["reg_srocc"], m["reg_plcc"] = srocc(r, ys["val"][mask]), plcc(r, ys["val"][mask])
            m["reg_final"] = (m["reg_srocc"] + m["reg_plcc"]) / 2
        out[tag] = m
        if tag == "val":
            out["_val_preds"] = (expected, probs)
    if "test" in xs:
        probs = _scale(cls.predict_proba(xs["test"]), t)
        out["_test_preds"] = (probs @ np.arange(probs.shape[1]), probs)
    return out



@app.function(image=baseline_image, cpu=16, memory=65536, timeout=7200,
              volumes={DATA: data_vol}, secrets=[modal.Secret.from_name("wandb")])
def fit_all_remote(calibrate: bool = False, log: bool = True, test: bool = False) -> list[dict]:
    """With test, also write test predictions (once; refuses to overwrite). Fits are seeded,
    so the models match the val-only fit. No test metrics are computed or logged here."""
    import wandb

    from vjev.metrics import ecr_level
    from vjev.train import _records

    root = pathlib.Path(DATA)
    edges = np.array(json.loads((root / "index" / "meta.json").read_text())["level_edges"])
    feats = {}
    for f in sorted((root / FEATS).glob("shard_*.npz")):
        # Read each array once. Indexing z[k] per row re-reads it, and every row view then
        # keeps its own full copy alive (this ran out of 64 GB).
        z = dict(np.load(f))
        for j, vid in enumerate(z["ids"]):
            feats[str(vid)] = {k: z[k][j] for k in ("text", "clip_first", "clip_mean")}

    recs, ys, levels = {}, {}, {}
    splits = ("train", "val", "test") if test else ("train", "val")
    for split in splits:
        lines = (root / "index" / f"{split}.jsonl").read_text().splitlines()
        recs[split] = [r for r in map(json.loads, lines) if r["id"] in feats]
        ys[split] = np.array([r["ecr"] for r in recs[split]])
        levels[split] = ecr_level(ys[split], edges)
    # The same 500 val clips the Omni runs are scored on.
    val500_ids = {r["id"] for r in _records("val", 500, 0)}
    val500 = np.array([r["id"] in val500_ids for r in recs["val"]])

    def x(split: str, keys: tuple[str, ...]) -> np.ndarray:
        return np.stack([np.concatenate([feats[r["id"]][k] for k in keys]) for r in recs[split]])

    setups = [
        ("text-xgb", "xgb", ("text",)),
        ("clip-mlp", "mlp", ("clip_first", "clip_mean")),
        ("clip+text-mlp", "mlp", ("clip_first", "clip_mean", "text")),
    ]
    results = []
    for name, kind, keys in setups:
        xs = {s: x(s, keys) for s in splits}
        name = f"{name}-ts" if calibrate else name
        test_out = root / "preds" / f"baseline-{name}-test.jsonl"
        if test and test_out.exists():
            raise RuntimeError(f"{test_out} exists; test is scored once per final model")
        res = {"model": name, **_fit_one(kind, xs, ys, levels, val500, edges, calibrate)}
        expected, probs = res.pop("_val_preds")
        (root / "preds").mkdir(exist_ok=True)
        (root / "preds" / f"baseline-{name}-val.jsonl").write_text("".join(
            json.dumps({"id": r["id"], "ecr": r["ecr"], "expected": float(e), "probs": p.tolist()})
            + "\n" for r, e, p in zip(recs["val"], expected, probs, strict=True)))
        if test:
            expected, probs = res.pop("_test_preds")
            test_out.write_text("".join(
                json.dumps({"id": r["id"], "ecr": r["ecr"], "expected": float(e),
                            "probs": p.tolist()})
                + "\n" for r, e, p in zip(recs["test"], expected, probs, strict=True)))
        data_vol.commit()
        res["n_train"] = len(ys["train"])
        results.append(res)
        print(json.dumps(res), flush=True)
        if not log:
            continue
        run = wandb.init(project="vjev", entity="sp1d5r", name=f"baseline-{name}",
                         config={"model": name, "features": keys, "kind": kind,
                                 "calibrate": calibrate}, reinit=True)
        run.summary.update({f"val/{k}": v for k, v in res["val500"].items()})
        run.summary.update({f"val_full/{k}": v for k, v in res["val"].items()})
        run.finish()
    return results


@app.local_entrypoint()
def fit_all(calibrate: bool = False, log: bool = True, test: bool = False):
    results = fit_all_remote.remote(calibrate, log and not test, test)
    if test:
        print("wrote test predictions to /data/preds/baseline-*-test.jsonl")
        return
    name = "simple_baselines_val_ts.json" if calibrate else "simple_baselines_val.json"
    out = pathlib.Path(__file__).resolve().parent.parent / "results" / name
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")
