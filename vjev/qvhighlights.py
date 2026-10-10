"""QVHighlights (Lei et al. 2021): text queries over ~150 s YouTube clips, with the moments
that match each query and a 0-4 highlight score for every 2 s of video.

    modal run vjev/qvhighlights.py::fetch_qv --limit 20   # stream the video tar, decode, test
    modal run vjev/qvhighlights.py::fetch_qv              # all videos (144 GB tar, streamed)
    modal run vjev/qvhighlights.py::index_qv              # index/qv_{train,val,test}.jsonl

Annotations are CC BY-NC-SA 4.0 (research use). The official test labels are withheld, so
the standard val split is our test, and we hold out 5% of train videos as val.
"""

from __future__ import annotations

import json
import pathlib

from vjev.common import DATA, app, data_vol, gpu_image

VIDEOS = "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"
ANN = "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_{}_release.jsonl"
# One frame pair per 2 s, which is exactly one QVHighlights clip. Small frames keep a
# 150 s video near 3k video tokens.
FPS = 1.0
WINDOW_S = 150
MAX_SIDE = 224


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=16, memory=32768, timeout=6 * 3600)
def fetch(limit: int = 0) -> dict:
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
    tmp = pathlib.Path(tempfile.mkdtemp())
    counts = {"ok": 0, "cached": 0, "failed": 0}
    t0 = time.time()

    def one(path: pathlib.Path, vid: str) -> str:
        out = pre_path(root, f"qv{vid}")
        try:
            arrays = extract(str(path), fps=FPS, window_s=WINDOW_S, max_side=MAX_SIDE)
        except subprocess.CalledProcessError:
            return "failed"
        finally:
            path.unlink(missing_ok=True)
        if len(arrays["offsets"]) < 3:
            return "failed"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        return "ok"

    futures, seen = [], 0
    with urllib.request.urlopen(VIDEOS) as resp, ThreadPoolExecutor(14) as ex, \
            tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".mp4"):
                continue
            vid = os.path.basename(m.name)[:-4]
            seen += 1
            if pre_path(root, f"qv{vid}").exists():
                counts["cached"] += 1
            else:
                dst = tmp / f"{seen}.mp4"
                with tar.extractfile(m) as src:
                    dst.write_bytes(src.read())
                futures.append(ex.submit(one, dst, vid))
            # Keep the queue short, so temp files do not pile up while ffmpeg catches up.
            while sum(not f.done() for f in futures) > 60:
                time.sleep(0.2)
            if seen % 500 == 0:
                done = [f.result() for f in futures if f.done()]
                print(f"{seen} seen, {done.count('ok')} ok, {time.time() - t0:.0f} s", flush=True)
                data_vol.commit()
            if limit and seen >= limit:
                break
    for f in futures:
        counts[f.result()] += 1
    data_vol.commit()
    return {**counts, "seconds": round(time.time() - t0)}


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=2, timeout=1800)
def build_index() -> dict:
    import hashlib
    import urllib.request

    from vjev.preprocess import pre_path

    root = pathlib.Path(DATA)
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing = 0
    for src in ("train", "val"):
        with urllib.request.urlopen(ANN.format(src)) as r:
            lines = r.read().decode().splitlines()
        for line in lines:
            a = json.loads(line)
            vid = f"qv{a['vid']}"
            if not pre_path(root, vid).exists():
                missing += 1
                continue
            rec = {"id": vid, "qid": a["qid"], "query": a["query"], "duration": a["duration"],
                   "windows": a["relevant_windows"], "clip_ids": a["relevant_clip_ids"],
                   "saliency_scores": a["saliency_scores"]}
            if src == "val":
                split = "test"
            else:  # hold out whole videos, so no val video appears in train
                h = int(hashlib.sha256(a["vid"].encode()).hexdigest(), 16) % 100
                split = "val" if h < 5 else "train"
            rows[split].append(rec)
    (root / "index").mkdir(exist_ok=True)
    for split, recs in rows.items():
        (root / "index" / f"qv_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
    data_vol.commit()
    return {"missing": missing, **{k: len(v) for k, v in rows.items()}}


@app.local_entrypoint()
def fetch_qv(limit: int = 0):
    print(fetch.remote(limit))


@app.local_entrypoint()
def index_qv():
    print(build_index.remote())


