"""Decode every clip once into JPEG frames plus 16 kHz mono audio.

Training reads these small files instead of decoding mp4 on the GPU box.
Window: the first WINDOW_S seconds at FPS frames per second. ECR is about the first
5 seconds, so the start of the clip carries most of the signal.

    modal run vjev/preprocess.py::preprocess_all
"""

from __future__ import annotations

import io
import itertools
import json
import pathlib
import subprocess
import tempfile

import numpy as np

from vjev.common import DATA, app, data_vol, gpu_image

FPS = 2
WINDOW_S = 16
MAX_SIDE = 448
SR = 16_000
SHARD = 200


def pre_path(root: pathlib.Path, video_id: str) -> pathlib.Path:
    if video_id.startswith("qv"):  # QVHighlights ids all end in ".0", so shard by a hash
        import hashlib

        return root / "pre" / "qv" / hashlib.md5(video_id.encode()).hexdigest()[:2] / f"{video_id}.npz"
    if video_id[:2] in ("vf", "gv", "fq"):  # these ids share prefixes, so shard by the end
        return root / "pre" / video_id[:2] / video_id[-2:] / f"{video_id}.npz"
    return root / "pre" / video_id[:2] / f"{video_id}.npz"


def _seconds(text: str) -> float | None:
    """ffprobe prints "N/A" when a container has no duration."""
    try:
        return float(text)
    except ValueError:
        return None


def extract(src: str, fps: float = FPS, window_s: int = WINDOW_S, max_side: int = MAX_SIDE) -> dict:
    """Frames as concatenated JPEG bytes plus offsets, audio as int16 PCM."""
    scale = (f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-t", str(window_s), "-i", src,
             "-vf", f"fps={fps},{scale}", "-frames:v", str(int(fps * window_s)),
             "-q:v", "4", f"{tmp}/%03d.jpg"],
            check=True,
        )
        jpgs = [p.read_bytes() for p in sorted(pathlib.Path(tmp).glob("*.jpg"))]
    audio = subprocess.run(
        ["ffmpeg", "-v", "error", "-t", str(window_s), "-i", src, "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "s16le", "-"],
        capture_output=True, check=False,
    ).stdout
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    offsets = np.cumsum([0, *map(len, jpgs)]).astype(np.int64)
    return {
        "jpeg": np.frombuffer(b"".join(jpgs), dtype=np.uint8),
        "offsets": offsets,
        "audio": np.frombuffer(audio, dtype=np.int16),
        "duration": np.float32(_seconds(duration) or len(jpgs) / fps),
    }


def load(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return frames (T, H, W, 3) uint8, audio float32 in [-1, 1], clip duration in seconds."""
    from PIL import Image

    z = np.load(path)
    jpeg, offsets = z["jpeg"].tobytes(), z["offsets"]
    frames = [np.asarray(Image.open(io.BytesIO(jpeg[a:b])).convert("RGB"))
              for a, b in itertools.pairwise(offsets)]
    audio = z["audio"].astype(np.float32) / 32768.0
    return np.stack(frames), audio, float(z["duration"])


@app.function(image=gpu_image, volumes={DATA: data_vol}, cpu=4, memory=4096, timeout=3600,
              max_containers=60, retries=1)
def run_shard(recs: list[dict]) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    root = pathlib.Path(DATA)

    def one(rec: dict) -> str:
        out = pre_path(root, rec["id"])
        if out.exists():
            return "cached"
        try:
            arrays = extract(str(root / rec["path"]))
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
def preprocess_all(limit: int = 0):
    import modal

    vol = modal.Volume.from_name("vjev-snapugc")
    recs = []
    for split in ("train", "val", "test"):
        body = b"".join(vol.read_file(f"index/{split}.jsonl")).decode()
        recs += [json.loads(line) for line in body.splitlines()]
    recs = recs[:limit] if limit else recs
    shards = [recs[i:i + SHARD] for i in range(0, len(recs), SHARD)]
    total = {"ok": 0, "cached": 0, "failed": 0}
    for i, res in enumerate(run_shard.map(shards, return_exceptions=True)):
        if isinstance(res, Exception):
            print(f"shard {i} failed: {res!r}")
            continue
        for k in total:
            total[k] += res[k]
        if i % 20 == 0:
            print(f"{i + 1}/{len(shards)} {total}", flush=True)
    print(f"done {total}")
