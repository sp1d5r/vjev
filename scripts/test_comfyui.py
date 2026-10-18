"""Test the ComfyUI nodes on Modal before publishing them.

    uv run modal run scripts/test_comfyui.py --what precision   # bf16 vs nf4 vs offload
    uv run modal run scripts/test_comfyui.py --what e2e         # a real ComfyUI server + workflow

precision: offload mode answers the Judge questions on VideoFeedback test clips and the funny
moment on FunQA test clips, from frames decoded at the clip's own frame rate (as ComfyUI's
Load Video gives them). Reports peak VRAM, seconds per clip and accuracy against human labels.
"""

from __future__ import annotations

import json
import pathlib
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE = ROOT / "comfyui" / "ComfyUI-VideoJev"
COMFY_COMMIT = "18fe3e6"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.11.0", "torchvision", "torchaudio", "transformers==5.10.2", "peft",
                 "accelerate", "bitsandbytes", "safetensors", "huggingface_hub", "hf_transfer",
                 "pillow", "numpy>=2.0", "requests")
    .run_commands(f"git clone https://github.com/comfyanonymous/ComfyUI /comfy && cd /comfy && "
                  f"git checkout {COMFY_COMMIT} && pip install -r requirements.txt")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(NODE, "/comfy/custom_nodes/ComfyUI-VideoJev")
)
app = modal.App("vjev-comfyui-test")
data = modal.Volume.from_name("vjev-snapugc")
hf = modal.Volume.from_name("vjev-hf-cache")
GPU = {"image": image, "gpu": "H200", "volumes": {"/data": data, "/hf": hf}, "timeout": 3 * 3600,
       "memory": 196608, "cpu": 8}


def _package():
    import importlib.util
    import sys

    sys.path.insert(0, "/comfy/custom_nodes/ComfyUI-VideoJev")
    spec = importlib.util.spec_from_file_location(
        "videojev", "/comfy/custom_nodes/ComfyUI-VideoJev/__init__.py",
        submodule_search_locations=["/comfy/custom_nodes/ComfyUI-VideoJev"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["videojev"] = mod
    spec.loader.exec_module(mod)
    from videojev import engine, presets

    return engine, presets


def _decode(path: str):
    """All frames at the clip's own rate, plus 44.1 kHz audio, like ComfyUI's Load Video."""
    import subprocess

    import numpy as np

    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps, w, h = float(num) / float(den), info["width"], info["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-t", "60", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-t", "60", "-i", path, "-vn", "-ac", "1",
                          "-ar", "44100", "-f", "s16le", "-"], capture_output=True, check=False).stdout
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768 if pcm else None
    return frames, fps, audio


@app.function(**GPU)
def precision(mode: str, n_vf: int = 60, n_fq: int = 20) -> dict:
    import torch

    engine, presets = _package()
    t0 = time.time()
    net, proc = engine.load("sp1d5r/video-jev", mode, 20)
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for split, n in (("vf", n_vf), ("fq", n_fq)):
        recs = [json.loads(x) for x in pathlib.Path(f"/data/index/{split}_test.jsonl").read_text().splitlines()][:n]
        for rec in recs:
            frames, fps, audio = _decode(f"/data/{rec['path']}")
            if split == "vf":
                groups = presets.build(["AI-generated?", "Video quality (5 scores)"], "",
                                       rec.get("prompt", ""))
            else:
                groups = presets.build(["Funniest moment" if rec["kind"] == "H1"
                                        else "Magic trick moment"])
            t = time.time()
            res = engine.ask(net, proc, groups, frames, fps, audio, 44100)
            rows.append({"id": rec["id"], "split": split, "s": round(time.time() - t, 2),
                         "res": res, "label": rec.get("scores") or rec.get("windows"),
                         "real": rec.get("real")})
    return {"mode": mode, "load_s": round(load_s, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "rows": rows}


@app.function(**GPU)
def e2e() -> dict:
    import shutil
    import subprocess

    import requests

    fq = "/data/fq/videos/test/test/test_humor/H_A_101_1433_1631.mp4"
    vf = "/data/vf/videos/videos_annotated_test/7001917.mp4"
    shutil.copy(fq, "/comfy/input/funny.mp4")
    shutil.copy(vf, "/comfy/input/aivideo.mp4")
    log_f = open("/tmp/comfy.log", "w")  # noqa: SIM115 - the server writes to it until the end
    server = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                              cwd="/comfy", stdout=log_f, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            requests.get("http://127.0.0.1:8188/object_info/VideoJevAsk", timeout=2).raise_for_status()
            break
        except requests.RequestException:
            time.sleep(2)
    nodes = requests.get("http://127.0.0.1:8188/object_info/VideoJevAsk").json()
    wf = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": "funny.mp4"}},
        "2": {"class_type": "VideoJevLoader",
              "inputs": {"weights": "sp1d5r/video-jev", "precision": "bf16", "max_gpu_gib": 20}},
        "3": {"class_type": "VideoJevAsk", "inputs": {
            "model": ["2", 0], "video": ["1", 0], "funniest_moment": True, "magic_moment": False,
            "laughing": True, "which_sound": True, "ai_generated": True, "quality": False,
            "described_moment": "a person falls over", "prompt": "", "custom_decode": "moment",
            "custom_record": json.dumps({"questions": {"dog": {
                "type": "noul", "instructions": "Is there a dog in this video?"}}})}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["3", 1], "filename_prefix": "ask"}},
        "6": {"class_type": "VideoJevFindMoment", "inputs": {
            "model": ["2", 0], "video": ["1", 0], "mode": "event (2 fps, first 60 s)",
            "question": "At what point in the video does the funniest thing happen?"}},
        "7": {"class_type": "PreviewAny", "inputs": {"source": ["6", 4]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["6", 3], "filename_prefix": "clip"}},
        "9": {"class_type": "LoadVideo", "inputs": {"file": "aivideo.mp4"}},
        "10": {"class_type": "VideoJevJudge", "inputs": {
            "model": ["2", 0], "video": ["9", 0],
            "prompt": "crowd surfing at a Pink Floid rock concert before the show"}},
        "11": {"class_type": "PreviewAny", "inputs": {"source": ["10", 7]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["10", 8], "filename_prefix": "judge"}},
        "13": {"class_type": "PreviewAny", "inputs": {"source": ["10", 0]}},
    }
    t0 = time.time()
    r = requests.post("http://127.0.0.1:8188/prompt", json={"prompt": wf}).json()
    pid = r.get("prompt_id")
    hist = {}
    while pid and time.time() - t0 < 1800:
        hist = requests.get(f"http://127.0.0.1:8188/history/{pid}").json()
        if pid in hist and hist[pid].get("status", {}).get("completed") is not None:
            break
        if pid in hist and hist[pid].get("status", {}).get("status_str") == "error":
            break
        time.sleep(5)
    server.terminate()
    out = pathlib.Path("/data/comfy_test")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree("/comfy/output", out)
    data.commit()
    log = pathlib.Path("/tmp/comfy.log").read_text()
    return {"submit": r, "seconds": round(time.time() - t0, 1), "node_found": "VideoJevAsk" in nodes,
            "history": hist.get(pid, {}), "log_tail": log[-4000:],
            "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*.png"))}


@app.local_entrypoint()
def main(what: str = "e2e"):
    if what == "e2e":
        res = e2e.remote()
        h = res.pop("history")
        res["status"] = h.get("status")
        res["ui"] = {k: v for k, v in h.get("outputs", {}).items()}
        print(json.dumps(res, indent=1)[:15000])
        return
    # bf16 was measured on 60 + 20 clips; offload is slow, so it answers fewer.
    out = list(precision.starmap([("offload", 15, 5)],
                                 return_exceptions=True))
    path = pathlib.Path("/tmp/vjev_precision.json")
    path.write_text(json.dumps([o if isinstance(o, dict) else {"error": str(o)} for o in out]))
    for o in out:
        if isinstance(o, dict):
            print(o["mode"], "load", o["load_s"], "peak GiB", o["peak_gib"],
                  "s/clip", round(sum(r["s"] for r in o["rows"]) / len(o["rows"]), 2))
        else:
            print("error", str(o)[:2000])
    print("saved", path)
