"""FunQA (Xie et al. 2023): surprising videos (humour, creative, magic) with the timestamp of
the key moment. We use the three localisation tasks: H1 funny, C1 creative, M1 magic.

    modal run vjev/funqa.py::pull_fq        # download and unzip videos + annotations
    modal run vjev/funqa.py::index_fq       # index/fq_{train,val,test}.jsonl
    modal run vjev/funqa.py::preprocess_fq  # decode frames at 2 fps + audio, once

Licence CC BY-NC-SA 4.0 (research use). Answers are frame numbers within each clip; the
index converts them to seconds with the clip's own frame rate.
"""

from __future__ import annotations

import json
import pathlib

import modal

from vjev.common import DATA, app, data_vol, gpu_image
from vjev.videofeedback import vf_image

REPO = "fesvhtr/FunQA"
FQ = "fq"
SPLITS = {"train": "train", "val": "val", "test": "test"}
TASKS = ("H1", "C1", "M1")
FPS = 2.0
WINDOW_S = 60
MAX_SIDE = 336


@app.function(image=vf_image, volumes={DATA: data_vol}, timeout=3 * 3600, cpu=8, memory=16384)
def fq_pull(split: str) -> dict:
    import zipfile

    from huggingface_hub import hf_hub_download

    root = pathlib.Path(DATA) / FQ
    ann = hf_hub_download(REPO, f"raw/FunQA_{split}.json", repo_type="dataset",
                          cache_dir="/tmp/hf")
    (root / "ann").mkdir(parents=True, exist_ok=True)
    (root / "ann" / f"{split}.json").write_bytes(pathlib.Path(ann).read_bytes())
    z = hf_hub_download(REPO, f"raw/{split}.zip", repo_type="dataset", cache_dir="/tmp/hf")
    dst = root / "videos" / split
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(dst)
    data_vol.commit()
    return {"split": split, "mp4": sum(1 for _ in dst.rglob("*.mp4"))}


@app.function(image=vf_image, volumes={DATA: data_vol}, timeout=3600, cpu=8, memory=8192)
def fq_build_index() -> dict:
    import collections
    import re
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    root = pathlib.Path(DATA)
    mp4 = {p.name: p for p in (root / FQ / "videos").rglob("*.mp4")}

    def fps_of(p: pathlib.Path) -> float:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, check=False).stdout.strip()
        a, _, b = r.partition("/")
        return float(a) / float(b or 1) if a else 30.0

    out, missing = {}, 0
    for split in SPLITS:
        anns = json.loads((root / FQ / "ann" / f"{split}.json").read_text())
        per_vid: dict[str, dict] = collections.OrderedDict()
        for a in anns:
            if a["task"] not in TASKS:
                continue
            name = a["visual_input"]
            if name not in mp4:
                missing += 1
                continue
            rec = per_vid.setdefault(name, {"id": f"fq{name[:-4]}",
                                            "path": str(mp4[name].relative_to(root)),
                                            "kind": a["task"], "instructions": [],
                                            "frames": [int(x) for x in re.findall(r"\d+", a["output"])[:2]]})
            if a["instruction"] not in rec["instructions"]:
                rec["instructions"].append(a["instruction"])
        recs = list(per_vid.values())
        with ThreadPoolExecutor(16) as ex:
            fpss = list(ex.map(lambda r: fps_of(root / r["path"]), recs))
        for r, f in zip(recs, fpss, strict=True):
            r["video_fps"] = f
            r["windows"] = [[r["frames"][0] / f, r["frames"][1] / f]]
        (root / "index" / f"fq_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
        out[split] = dict(collections.Counter(r["kind"] for r in recs))
    data_vol.commit()
    return {"missing": missing, **out}


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=4, memory=8192, timeout=3600,
              max_containers=20, retries=1)
def fq_preprocess_shard(recs: list[dict]) -> dict:
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
            arrays = extract(str(root / rec["path"]), fps=FPS, window_s=WINDOW_S,
                             max_side=MAX_SIDE)
        except subprocess.CalledProcessError:
            return "failed"
        if len(arrays["offsets"]) < 3:
            return "failed"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        return "ok"

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(one, recs))
    data_vol.commit()
    return {k: results.count(k) for k in ("ok", "cached", "failed")}


@app.local_entrypoint()
def pull_fq():
    for r in fq_pull.map(list(SPLITS)):
        print(r)


@app.local_entrypoint()
def index_fq():
    print(json.dumps(fq_build_index.remote(), indent=2))


@app.local_entrypoint()
def preprocess_fq():
    vol = modal.Volume.from_name("vjev-snapugc")
    recs = []
    for split in SPLITS:
        body = b"".join(vol.read_file(f"index/fq_{split}.jsonl")).decode()
        recs += [json.loads(line) for line in body.splitlines()]
    shards = [recs[i:i + 100] for i in range(0, len(recs), 100)]
    total = {"ok": 0, "cached": 0, "failed": 0}
    for res in fq_preprocess_shard.map(shards, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"shard failed: {res!r}")
            continue
        for k in total:
            total[k] += res[k]
    print(total)
