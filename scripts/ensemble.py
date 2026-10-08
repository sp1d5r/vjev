"""Equal-weight ensemble of saved val predictions. Nothing is fit on val.

    uv run python scripts/ensemble.py pilot-omni-av-10k-val500 baseline-clip-mlp-ts-val

Ranking: mean of the z-scored expected levels. Probabilities: mean of the distributions.
Scored on the clips all inputs share.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

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


def main(names: list[str]) -> None:
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
    print(f"correlation between the first two models' predictions: {np.corrcoef(a, b)[0, 1]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
