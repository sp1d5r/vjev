"""VideoFeedback (TIGER-Lab, Apache-2.0): 37.6k AI-generated videos with human scores.

    modal run vjev/videofeedback.py::pull_vf     # download and unzip videos + labels
    modal run vjev/videofeedback.py::inspect_vf  # counts, score spread, clip lengths
    modal run vjev/videofeedback.py::index_vf    # index/vf_{train,val,test}.jsonl
    modal run vjev/videofeedback.py::preprocess_vf  # decode frames at 8 fps, once

Five questions, each scored 1 to 4 by raters: visual quality, temporal consistency,
dynamic degree, text-to-video alignment, factual consistency. The official test split
(680 annotated clips) is the one VideoScore reports on.
"""

from __future__ import annotations

import json
import pathlib

import modal

from vjev.common import DATA, app, data_vol, gpu_image

REPO = "TIGER-Lab/VideoFeedback"
VF = "vf"
# The parquet files hold the labels; the zips hold mp4s. Frame zips (42 GB) are skipped:
# we decode our own frames.
FILES = [
    "annotated/train-00000-of-00001.parquet", "annotated/test-00000-of-00001.parquet",
    "real/train-00000-of-00001.parquet", "real/test-00000-of-00001.parquet",
    "annotated/videos_annotated_train.zip", "annotated/videos_annotated_test.zip",
    "real/videos_real_train.zip", "real/videos_real_test.zip",
]
# Clips are 1 to 3 s at 8 fps. Keep every frame: temporal consistency needs them.
FPS = 8
WINDOW_S = 6
QUESTIONS = ["visual quality", "temporal consistency", "dynamic degree",
             "text-to-video alignment", "factual consistency"]

vf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("huggingface_hub[hf_transfer]", "pandas", "pyarrow")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("vjev")
)


@app.function(image=vf_image, volumes={DATA: data_vol}, timeout=3 * 3600, cpu=8,
              memory=16384)
def pull(files: list[str]) -> dict:
    import zipfile

    from huggingface_hub import hf_hub_download

    root = pathlib.Path(DATA) / VF
    out = {}
    for f in files:
        local = hf_hub_download(REPO, f, repo_type="dataset", cache_dir="/tmp/hf")
        dst = root / f
        if f.endswith(".zip"):
            target = root / "videos" / pathlib.Path(f).stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local) as z:
                z.extractall(target)
            out[f] = sum(1 for p in target.rglob("*.mp4"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(pathlib.Path(local).read_bytes())
            out[f] = dst.stat().st_size
        data_vol.commit()
        print(f, out[f], flush=True)
    return out


@app.function(image=vf_image, volumes={DATA: data_vol}, timeout=1800, cpu=8)
def inspect() -> dict:
    import random
    import subprocess

    import pandas as pd

    root = pathlib.Path(DATA) / VF
    res = {}
    for sub in ("annotated", "real"):
        for split in ("train", "test"):
            df = pd.read_parquet(root / sub / f"{split}-00000-of-00001.parquet")
            res[f"{sub}/{split}"] = {
                "n": len(df), "columns": list(df.columns),
                "example": {k: str(v)[:120] for k, v in df.iloc[0].items()
                            if k not in ("conversations", "images")},
                "score_counts": {q: df[q].value_counts().sort_index().to_dict()
                                 for q in QUESTIONS if q in df.columns},
            }
    mp4s = sorted((root / "videos").rglob("*.mp4"))
    res["mp4_count"] = len(mp4s)
    res["mp4_dirs"] = sorted({str(p.parent.relative_to(root)) for p in mp4s})[:20]
    probe = []
    for p in random.Random(0).sample(mp4s, min(30, len(mp4s))):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate,nb_frames:format=duration",
                            "-of", "json", str(p)], capture_output=True, text=True, check=False)
        j = json.loads(r.stdout or "{}")
        s = (j.get("streams") or [{}])[0]
        probe.append({"file": p.name, "w": s.get("width"), "h": s.get("height"),
                      "fps": s.get("r_frame_rate"),
                      "dur": float(j.get("format", {}).get("duration", 0))})
    res["probe"] = probe
    return res


@app.function(image=vf_image, volumes={DATA: data_vol}, timeout=1800, cpu=4, memory=8192)
def build_index() -> dict:
    """Train is split by sha256(id) % 100 < 5 into val; test is the official test split."""
    import hashlib

    import pandas as pd

    from vjev.questions import VF_QUESTIONS

    root = pathlib.Path(DATA)
    mp4 = {p.stem: p for p in (root / VF / "videos").rglob("*.mp4")}
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing = 0
    for sub in ("annotated", "real"):
        for split in ("train", "test"):
            df = pd.read_parquet(root / VF / sub / f"{split}-00000-of-00001.parquet")
            for r in df.to_dict("records"):
                raw = str(r["id"])
                if raw not in mp4:
                    missing += 1
                    continue
                rec = {"id": f"vf{raw}", "path": str(mp4[raw].relative_to(root)),
                       "prompt": str(r["text prompt"]).strip(), "real": sub == "real",
                       "scores": {k: int(r[col]) for k, (col, _i, _c) in VF_QUESTIONS.items()}}
                to = split
                if split == "train" and int(hashlib.sha256(raw.encode()).hexdigest(), 16) % 100 < 5:
                    to = "val"
                rows[to].append(rec)
    (root / "index").mkdir(exist_ok=True)
    for split, recs in rows.items():
        (root / "index" / f"vf_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
    data_vol.commit()
    return {"missing": missing, **{k: len(v) for k, v in rows.items()}}


@app.local_entrypoint()
def pull_vf():
    # One container per file, so the 7.9 GB train zip does not hold up the rest.
    for r in pull.map([[f] for f in FILES]):
        print(r)


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=4, memory=4096, timeout=3600,
              max_containers=40, retries=1)
def preprocess_shard(recs: list[dict]) -> dict:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from vjev.preprocess import extract, pre_path

    root = pathlib.Path(DATA)

    def one(rec: dict) -> str:
        out = pre_path(root, rec["id"])
        if out.exists():
            return "cached"
        try:
            arrays = extract(str(root / rec["path"]), fps=FPS, window_s=WINDOW_S)
        except subprocess.CalledProcessError:
            return "failed"
        if len(arrays["offsets"]) < 3:  # fewer than 2 frames
            return "failed"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        return "ok"

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(one, recs))
    data_vol.commit()
    return {k: results.count(k) for k in ("ok", "cached", "failed")}


@app.local_entrypoint()
def inspect_vf():
    print(json.dumps(inspect.remote(), indent=2, default=str))


@app.local_entrypoint()
def index_vf():
    print(build_index.remote())


@app.local_entrypoint()
def preprocess_vf(limit: int = 0):
    vol = modal.Volume.from_name("vjev-snapugc")
    recs = []
    for split in ("train", "val", "test"):
        body = b"".join(vol.read_file(f"index/vf_{split}.jsonl")).decode()
        recs += [json.loads(line) for line in body.splitlines()]
    recs = recs[:limit] if limit else recs
    shards = [recs[i:i + 400] for i in range(0, len(recs), 400)]
    total = {"ok": 0, "cached": 0, "failed": 0}
    for res in preprocess_shard.map(shards, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"shard failed: {res!r}")
            continue
        for k in total:
            total[k] += res[k]
    print(total)
