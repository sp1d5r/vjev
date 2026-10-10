"""VGGSound (Chen et al. 2020, CC BY 4.0): 10 s YouTube clips labelled by their sound. We keep
eight classes of people sounds, where the answer is mostly in the audio: the clean test of
whether the model uses sound.

    modal run vjev/vggsound.py::fetch_vs --limit 20   # one shard, first 20 matches (test)
    modal run vjev/vggsound.py::fetch_vs              # stream all 20 shards, keep the 8 classes
    modal run vjev/vggsound.py::index_vs              # index/vs_{train,val,test}.jsonl

Each 16 GB shard is streamed and only matching clips are decoded (2 fps, 10 s, 336 px, audio).
The official test split is our test; 10% of train videos (by hash) are val.
"""

from __future__ import annotations

import json
import pathlib

from vjev.common import DATA, app, data_vol, gpu_image

REPO = "Loie/VGGSound"
SHARDS = [f"vggsound_{i:02d}.tar.gz" for i in range(20)]
CSV = f"https://huggingface.co/datasets/{REPO}/resolve/main/vggsound.csv"
# Chosen before any training: people sounds that look alike but sound different.
CLASSES = {
    "people belly laughing": "adults laughing",
    "people giggling": "people giggling",
    "baby laughter": "a baby laughing",
    "people cheering": "people cheering",
    "people crowd": "crowd noise",
    "female speech, woman speaking": "a woman speaking",
    "male speech, man speaking": "a man speaking",
    "child speech, kid speaking": "a child speaking",
}
FPS = 2.0
WINDOW_S = 10
MAX_SIDE = 336


def _labels() -> dict[str, tuple[str, str]]:
    """clip name -> (label, split) for the kept classes."""
    import csv
    import io
    import urllib.request

    with urllib.request.urlopen(CSV) as r:
        rows = list(csv.reader(io.StringIO(r.read().decode())))
    return {f"{yt}_{int(start):06d}": (label, split)
            for yt, start, label, split in rows if label in CLASSES}


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=8, memory=16384, timeout=4 * 3600,
              max_containers=20, retries=1)
def vs_fetch_shard(shard: str, limit: int = 0) -> dict:
    import os
    import subprocess
    import tarfile
    import tempfile
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from vjev.preprocess import extract, pre_path

    root = pathlib.Path(DATA)
    keep = _labels()
    tmp = pathlib.Path(tempfile.mkdtemp())
    t0 = time.time()

    def one(path: pathlib.Path, name: str) -> str:
        try:
            arrays = extract(str(path), fps=FPS, window_s=WINDOW_S, max_side=MAX_SIDE)
        except subprocess.CalledProcessError:
            return "failed"
        finally:
            path.unlink(missing_ok=True)
        if len(arrays["offsets"]) < 3:
            return "failed"
        out = pre_path(root, f"vs{name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        return "ok"

    futures, seen, matched = [], 0, 0
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{shard}"
    with urllib.request.urlopen(url) as resp, ThreadPoolExecutor(6) as ex, \
            tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".mp4"):
                continue
            seen += 1
            name = os.path.basename(m.name)[:-4]
            if name not in keep or pre_path(root, f"vs{name}").exists():
                continue
            matched += 1
            dst = tmp / f"{seen}.mp4"
            with tar.extractfile(m) as src:
                dst.write_bytes(src.read())
            futures.append(ex.submit(one, dst, name))
            if limit and matched >= limit:
                break
    results = [f.result() for f in futures]
    data_vol.commit()
    return {"shard": shard, "seen": seen, "ok": results.count("ok"),
            "failed": results.count("failed"), "seconds": round(time.time() - t0)}


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=2, timeout=1800)
def vs_build_index() -> dict:
    import collections
    import hashlib

    from vjev.preprocess import pre_path

    root = pathlib.Path(DATA)
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for name, (label, split) in _labels().items():
        vid = f"vs{name}"
        if not pre_path(root, vid).exists():
            continue
        if split == "train" and int(hashlib.sha256(name.encode()).hexdigest(), 16) % 100 < 10:
            split = "val"
        rows[split].append({"id": vid, "label": label})
    for split, recs in rows.items():
        (root / "index" / f"vs_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
    data_vol.commit()
    return {s: dict(collections.Counter(r["label"] for r in v)) for s, v in rows.items()}


@app.local_entrypoint()
def fetch_vs(limit: int = 0):
    shards = SHARDS[:1] if limit else SHARDS
    for r in vs_fetch_shard.map(shards, kwargs={"limit": limit}, return_exceptions=True):
        print(r)


@app.local_entrypoint()
def index_vs():
    print(json.dumps(vs_build_index.remote(), indent=1))
