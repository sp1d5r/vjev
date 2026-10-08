"""Equal-weight ensemble of saved val predictions. Nothing is fit on val.

    uv run python scripts/ensemble.py pilot-omni-av-10k-val500 baseline-clip-mlp-ts-val

Ranking: mean of the z-scored expected levels. Probabilities: mean of the distributions.
Scored on the clips all inputs share. With --out, also writes a paired bootstrap of the
final-score gaps (2,000 resamples) and every row to a JSON file.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import pearsonr, spearmanr

from vjev.metrics import report

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fetch(name: str, tmp: pathlib.Path) -> dict[str, dict]:
    dst = tmp / f"{name}.jsonl"
    subprocess.run(["modal", "volume", "get", "vjev-snapugc", f"preds/{name}.jsonl", str(dst)],
                   check=True, capture_output=True)
    return {r["id"]: r for r in map(json.loads, dst.read_text().splitlines())}


def edges(tmp: pathlib.Path) -> np.ndarray:
    dst = tmp / "meta.json"
    subprocess.run(["modal", "volume", "get", "vjev-snapugc", "index/meta.json", str(dst)],
                   check=True, capture_output=True)
    return np.array(json.loads(dst.read_text())["level_edges"])


def final(x: np.ndarray, y: np.ndarray) -> float:
    return (spearmanr(x, y)[0] + pearsonr(x, y)[0]) / 2


def bootstrap(scores: dict[str, np.ndarray], ecr: np.ndarray, n: int = 2000,
              seed: int = 0) -> dict[str, list[float]]:
    """Paired bootstrap of final-score gaps: [mean, 2.5%, 97.5%] for each pair."""
    rng = np.random.default_rng(seed)
    names = list(scores)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    gaps = {p: [] for p in pairs}
    for _ in range(n):
        idx = rng.integers(0, len(ecr), len(ecr))
        f = {k: final(v[idx], ecr[idx]) for k, v in scores.items()}
        for a, b in pairs:
            gaps[(a, b)].append(f[b] - f[a])
    return {f"{b} minus {a}": [round(float(np.mean(g)), 3),
                               *np.round(np.percentile(g, [2.5, 97.5]), 3).tolist()]
            for (a, b), g in gaps.items()}


def main(names: list[str], out: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        preds = [fetch(n, tmp) for n in names]
        e = edges(tmp)
    ids = sorted(set.intersection(*(set(p) for p in preds)))
    ecr = np.array([preds[0][i]["ecr"] for i in ids])
    rows = {}
    zs, probs = [], []
    for name, p in zip(names, preds, strict=True):
        x = np.array([p[i]["expected"] for i in ids])
        pr = np.array([p[i]["probs"] for i in ids])
        rows[name] = report(x, pr, ecr, e)
        zs.append((x - x.mean()) / x.std())
        probs.append(pr)
    rank = report(np.mean(zs, axis=0), np.mean(probs, axis=0), ecr, e)
    mixed = np.mean(probs, axis=0)
    rows["ensemble (mean z / mean probs)"] = rank
    rows["ensemble (expected of mean probs)"] = report(mixed @ np.arange(mixed.shape[1]), mixed,
                                                      ecr, e)
    print(f"n = {len(ids)}")
    print(f"{'model':45s} final  srocc  plcc   brier  ece")
    for name, m in rows.items():
        print(f"{name:45s} {m['final']:.3f}  {m['srocc']:.3f}  {m['plcc']:.3f}  "
              f"{m['brier']:.3f}  {m['ece']:.3f}")
    a, b = zs[0], zs[1] if len(zs) > 1 else zs[0]
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"correlation between the first two models' predictions: {corr:.3f}")
    if out:
        boot = bootstrap({**dict(zip(names, zs, strict=True)), "ensemble": np.mean(zs, axis=0)},
                         ecr)
        for k, v in boot.items():
            print(f"{k:60s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
        (ROOT / out).write_text(json.dumps({
            "n": len(ids), "models": names,
            "rows": {k: {m: round(float(v[m]), 3) for m in ("final", "srocc", "plcc", "brier", "ece")}
                     for k, v in rows.items()},
            "prediction_correlation": round(corr, 3),
            "paired_bootstrap_2000": boot}, indent=2) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("--out") + 1] if "--out" in args else None
    main([a for a in args if a not in ("--out", out)], out)
