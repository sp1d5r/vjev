"""Ask typed questions about a video with Video Jev. Needs one GPU with 80 GB or more.

    pip install torch==2.11.0 transformers==5.10.2 peft accelerate safetensors pillow numpy
    python run.py --video clip.mp4 --record questions.json

`questions.json` is a Jev record; see README.md. ffmpeg must be on the PATH.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file

from vjev import clef_schema as cs
from vjev import model as vm

HERE = pathlib.Path(__file__).resolve().parent


def decode(path: str, fps: float, seconds: int, max_side: int = 448) -> tuple[np.ndarray, np.ndarray]:
    """Frames (T, H, W, 3) at `fps` for the first `seconds`, and 16 kHz mono audio."""
    scale = f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-t", str(seconds), "-i", path, "-vf",
                        f"fps={fps},{scale}", "-q:v", "4", f"{tmp}/%04d.jpg"], check=True)
        frames = [np.asarray(Image.open(p).convert("RGB"))
                  for p in sorted(pathlib.Path(tmp).glob("*.jpg"))]
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-t", str(seconds), "-i", path, "-vn", "-ac",
                          "1", "-ar", str(vm.SR), "-f", "s16le", "-"],
                         capture_output=True, check=False).stdout
    return np.stack(frames), np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def load(weights: pathlib.Path):
    cfg = json.loads((weights / "vjev_config.json").read_text())
    net, proc = vm.load(snapshot_download(cfg["base_model"]), cfg["lora"]["r"], cfg["lora"]["alpha"])
    set_peft_model_state_dict(net.thinker, load_file(weights / "adapter_model.safetensors"))
    net.head.load_state_dict(load_file(weights / "head.safetensors", device="cuda"))
    return net.eval(), proc


def answer(net, proc, record: dict, frames: np.ndarray, audio: np.ndarray | None,
           fps: float) -> dict:
    enc = vm.encode(proc, record, frames, audio, fps=fps)
    with torch.no_grad():
        outs = [o.float().softmax(-1).cpu().numpy() for o in net(enc)]
    result, step = {}, 2 / fps
    for (name, q), p in zip(record["questions"].items(), outs, strict=True):
        if q["type"] == "moment":
            result[name] = [{"from_s": round(i * step, 1), "to_s": round((i + 1) * step, 1),
                             "p": round(float(p[i]), 3)} for i in np.argsort(-p)[:5]]
        else:
            labels = [str(text if q["type"] == "score" else key)
                      for key, text in cs.question_options(q)]
            result[name] = {k: round(float(x), 3) for k, x in zip(labels, p, strict=True)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--weights", default=str(HERE), help="folder with the adapter and head")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    frames, audio = decode(a.video, a.fps, a.seconds)
    net, proc = load(pathlib.Path(a.weights))
    record = json.loads(pathlib.Path(a.record).read_text())
    t0 = time.perf_counter()
    out = answer(net, proc, record, frames, None if a.no_audio else audio, a.fps)
    print(json.dumps({"answers": out, "frames": len(frames),
                      "seconds": round(time.perf_counter() - t0, 2)}, indent=1))


if __name__ == "__main__":
    main()
