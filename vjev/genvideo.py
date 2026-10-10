"""GenVideo (DeMamba), via the MIT-tagged mirror sshaiton/GenVideo-50k-shard: real and
AI-generated clips, each labelled with its generator.

    modal run vjev/genvideo.py::pull_gv --shards 4   # download and unpack the first N shards
    modal run vjev/genvideo.py::index_gv             # counts per generator, index/gv_*.jsonl
    modal run vjev/genvideo.py::preprocess_gv        # decode frames at 8 fps, once

Each shard is a ~1 GB tar of `<n>_<name>.mp4` + `<n>_<name>.json` pairs, where the json holds
`label` (real / fake) and `generator`.
"""

from __future__ import annotations

import json
import pathlib

import modal

from vjev.common import DATA, app, data_vol, gpu_image
from vjev.videofeedback import vf_image

REPO = "sshaiton/GenVideo-50k-shard"
GV = "gv"
# Same decode as VideoFeedback, so the two can share one model.
FPS = 8
WINDOW_S = 6
# Generators kept out of train and val entirely: the test asks whether the model spots
# generators it has never seen. Chosen by name before looking at any results.
HELD_OUT = ("OpenSora", "SVD")


@app.function(image=vf_image, volumes={DATA: data_vol}, timeout=3600, cpu=4, memory=8192)
def gv_pull(shard: int) -> dict:
    import tarfile

    from huggingface_hub import hf_hub_download

    name = f"dataset_shard_{shard:06d}.tar"
    local = hf_hub_download(REPO, name, repo_type="dataset", cache_dir="/tmp/hf")
    dst = pathlib.Path(DATA) / GV / f"{shard:06d}"
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local) as t:
        t.extractall(dst, filter="data")
    data_vol.commit()
    return {"shard": shard, "mp4": sum(1 for _ in dst.glob("*.mp4"))}


@app.function(image=vf_image, volumes={DATA: data_vol}, timeout=1800, cpu=4, memory=8192)
def gv_build_index() -> dict:
    """Held-out generators go to test only. The rest split by sha256(id) % 100: < 10 test,
    < 15 val, else train."""
    import collections
    import hashlib

    root = pathlib.Path(DATA)
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    per_gen: collections.Counter = collections.Counter()
    for meta in sorted((root / GV).glob("*/*.json")):
        mp4 = meta.with_suffix(".mp4")
        if not mp4.exists():
            continue
        j = json.loads(meta.read_text())
        gen = str(j.get("generator", "unknown"))
        per_gen[gen] += 1
        rec = {"id": f"gv{meta.parent.name}{meta.stem.split('_')[0]}",
               "path": str(mp4.relative_to(root)), "real": j.get("label") == "real",
               "generator": gen}
        h = int(hashlib.sha256(rec["id"].encode()).hexdigest(), 16) % 100
        split = "test" if gen in HELD_OUT or h < 10 else "val" if h < 15 else "train"
        rows[split].append(rec)
    (root / "index").mkdir(exist_ok=True)
    for split, recs in rows.items():
        (root / "index" / f"gv_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
    data_vol.commit()
    return {"per_generator": dict(per_gen.most_common()),
            **{k: len(v) for k, v in rows.items()},
            "real_share": {k: round(sum(r["real"] for r in v) / max(1, len(v)), 3)
                           for k, v in rows.items()}}


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=4, memory=4096, timeout=3600,
              max_containers=20, retries=1)
def gv_preprocess_shard(recs: list[dict]) -> dict:
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
def pull_gv(shards: int = 4):
    for r in gv_pull.map(range(shards)):
        print(r)


@app.local_entrypoint()
def index_gv():
    print(json.dumps(gv_build_index.remote(), indent=2))


@app.local_entrypoint()
def preprocess_gv():
    vol = modal.Volume.from_name("vjev-snapugc")
    recs = []
    for split in ("train", "val", "test"):
        body = b"".join(vol.read_file(f"index/gv_{split}.jsonl")).decode()
        recs += [json.loads(line) for line in body.splitlines()]
    shards = [recs[i:i + 300] for i in range(0, len(recs), 300)]
    total = {"ok": 0, "cached": 0, "failed": 0}
    for res in gv_preprocess_shard.map(shards, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"shard failed: {res!r}")
            continue
        for k in total:
            total[k] += res[k]
    print(total)
