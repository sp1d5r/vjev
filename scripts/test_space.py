"""Run the Space app on a Modal H200 before publishing it (spaces.GPU is a no-op off ZeroGPU).

    uv run modal run scripts/test_space.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(str(ROOT / "space" / "requirements.txt"))
    .pip_install("hf_transfer", "gradio==5.49.1")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(ROOT / "space", "/root/space")
    .add_local_dir(ROOT / "release" / "vjev", "/root/space/vjev")
)
app = modal.App("vjev-space-test")
data = modal.Volume.from_name("vjev-snapugc")
hf = modal.Volume.from_name("vjev-hf-cache")


@app.function(image=image, gpu="H200", volumes={"/data": data, "/hf": hf}, timeout=3600,
              memory=131072, cpu=8)
def check() -> dict:
    sys.path.insert(0, "/root/space")
    t0 = time.time()
    import app as space

    load_s = time.time() - t0
    fq = min(pathlib.Path("/data/fq/videos/test").rglob("H*.mp4"))
    vf = min(pathlib.Path("/data/vf/videos/videos_annotated_test").rglob("*.mp4"))
    out = {"load_s": round(load_s, 1), "runs": []}
    for video, choices, query, prompt, custom in [
        (fq, space.CHOICES, "a person falls over", "", space.EXAMPLE),
        (vf, ["AI-generated?", "Video quality (5 scores)"], "", "a cat playing piano", ""),
    ]:
        t = time.time()
        html, res = space.ask(str(video), choices, query, prompt, custom, "moment")
        out["runs"].append({"video": str(video), "seconds": round(time.time() - t, 1),
                            "results": res})
        pathlib.Path(f"/data/space_test_{video.stem}.html").write_text(html)
    data.commit()
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=1)[:12000])
