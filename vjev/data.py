"""Get SnapUGC onto the vjev-snapugc volume and build a fixed split.

The official val and test labels are withheld, so every split here comes from the
labelled train_data.csv (106,192 rows). The split is a hash of the video Id, so it never
changes between runs or machines.

    modal run vjev/data.py::upload_labels --csv-path ~/Downloads/train_data.csv
    modal run vjev/data.py::pull_videos
    modal run vjev/data.py::build_split
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

from vjev.common import DATA, app, cpu_image, data_vol

# Google Drive "train set" in the SnapUGC repo is only this CSV. Videos come from the
# Snap CDN links in it, and many links have expired (about 35% alive in August 2026).
MIN_BYTES = 50_000  # a 404 body or a truncated fetch never reaches this
SHARD = 500
# The CDN refuses most requests with the default "Python-urllib" agent.
UA = "curl/8.7.1"
LEVELS = 10  # score questions allow 2 to 10 levels


def split_of(video_id: str) -> str:
    """90 / 5 / 5 split by a stable hash of the Id."""
    bucket = int(hashlib.sha256(video_id.encode()).hexdigest(), 16) % 100
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "val"
    return "train"


@app.local_entrypoint()
def upload_labels(csv_path: str = "~/Downloads/train_data.csv"):
    src = pathlib.Path(csv_path).expanduser()
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(str(src), "/labels/train_data.csv")
    print(f"uploaded {src} -> vjev-snapugc:/labels/train_data.csv")


def _rows() -> list[dict]:
    with open(pathlib.Path(DATA) / "labels" / "train_data.csv", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("ECR")]


@app.function(image=cpu_image, volumes={DATA: data_vol}, timeout=3600, cpu=2,
              max_containers=40, retries=1)
def pull_shard(rows: list[dict]) -> dict:
    """Fetch one shard of videos from the CDN with a thread pool."""
    import http.client
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    out = pathlib.Path(DATA) / "raw" / "videos"
    out.mkdir(parents=True, exist_ok=True)

    def fetch(row: dict) -> str:
        dest = out / f"{row['Id']}.mp4"
        if dest.exists() and dest.stat().st_size >= MIN_BYTES:
            return "cached"
        try:
            req = urllib.request.Request(row["Download_link"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = resp.read()
        except (OSError, ValueError, http.client.HTTPException):
            return "dead"
        if len(body) < MIN_BYTES:
            return "dead"
        dest.write_bytes(body)
        return "ok"

    with ThreadPoolExecutor(32) as ex:
        results = list(ex.map(fetch, rows))
    data_vol.commit()
    return {k: results.count(k) for k in ("ok", "cached", "dead")}


@app.function(image=cpu_image, volumes={DATA: data_vol}, timeout=12 * 3600)
def pull_videos(limit: int = 0):
    """Fan the CDN pull out over many containers. limit=0 means every labelled row."""
    rows = _rows()
    keep = ("Id", "Download_link")
    rows = [{k: r[k] for k in keep} for r in (rows[:limit] if limit else rows)]
    shards = [rows[i:i + SHARD] for i in range(0, len(rows), SHARD)]
    total = {"ok": 0, "cached": 0, "dead": 0}
    for i, res in enumerate(pull_shard.map(shards, return_exceptions=True)):
        if isinstance(res, Exception):
            print(f"shard {i} failed: {res!r}")
            continue
        for k in total:
            total[k] += res[k]
        if i % 10 == 0:
            print(f"{i + 1}/{len(shards)} shards {total}", flush=True)
    print(f"done {total}")
    return total


@app.function(image=cpu_image, volumes={DATA: data_vol}, timeout=3600, memory=8192)
def build_split():
    import numpy as np

    root = pathlib.Path(DATA)
    videos = {p.stem: str(p.relative_to(root)) for p in (root / "raw").rglob("*.mp4")}
    rows = _rows()

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing = 0
    for r in rows:
        path = videos.get(r["Id"])
        if path is None:
            missing += 1
            continue
        splits[split_of(r["Id"])].append({
            "id": r["Id"],
            "ecr": float(r["ECR"]),
            "path": path,
            "title": (r.get("Title") or "").strip(),
            "description": (r.get("Description") or "").strip(),
        })

    # Level edges are deciles over all labelled rows, not just clips whose links survived.
    # Low-ECR videos are deleted more often, so survivor deciles would shift the task.
    all_ecr = np.array([float(r["ECR"]) for r in rows])
    edges = np.quantile(all_ecr, np.linspace(0, 1, LEVELS + 1)[1:-1]).tolist()

    index = root / "index"
    index.mkdir(exist_ok=True)
    for name, recs in splits.items():
        with open(index / f"{name}.jsonl", "w") as fh:
            fh.writelines(json.dumps(rec) + "\n" for rec in recs)
    meta = {
        "labelled_rows": len(rows),
        "videos_found": len(videos),
        "rows_without_video": missing,
        "sizes": {k: len(v) for k, v in splits.items()},
        "mean_ecr_all": float(all_ecr.mean()),
        "mean_ecr_found": {k: float(np.mean([r["ecr"] for r in v])) if v else None
                           for k, v in splits.items()},
        "level_edges": edges,
    }
    (index / "meta.json").write_text(json.dumps(meta, indent=2))
    data_vol.commit()
    print(json.dumps(meta, indent=2))
