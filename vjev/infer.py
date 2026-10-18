"""Ask your own typed questions about any video file.

    modal run vjev/infer.py::ask --video clip.mp4 --record questions.json [--release g4-joint]

`questions.json` is a Jev record: {"state": {...}, "questions": {name: question}}. Question
types: "noul" (yes/no), "choice" (named options in "criteria"), "score" (ordered levels in
"criteria", worst first) and "moment" (no options: one per second of video). Example:

    {"state": {"title": "my dog vs the hoover"},
     "questions": {
       "laugh": {"type": "noul", "instructions": "Can you hear someone laughing?",
                 "criteria": {"true": "yes", "false": "no"}},
       "funniest": {"type": "moment", "instructions": "When is the funniest moment?"}}}

The first 60 s are read at 2 fps with audio (set --fps, --seconds, --no-audio). Every question
is answered in one forward pass. Prints a probability per option; for moments, the five most
likely seconds.
"""

from __future__ import annotations

import json
import pathlib

import modal

from vjev.common import DATA, HF, app, data_vol, hf_vol, train_image


@app.function(image=train_image, gpu="H200", volumes={DATA: data_vol, HF: hf_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800, cpu=8,
              memory=65536)
def answer(video: bytes, record: dict, release: str, fps: float, seconds: int,
           audio: bool) -> dict:
    import io
    import itertools
    import tempfile
    import time

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    from vjev import clef_schema as cs
    from vjev import model as vm
    from vjev.preprocess import extract
    from vjev.release import load_release

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(video)
        f.flush()
        arrays = extract(f.name, fps=fps, window_s=seconds, max_side=448)
    jpeg, offsets = arrays["jpeg"].tobytes(), arrays["offsets"]
    frames = np.stack([np.asarray(Image.open(io.BytesIO(jpeg[a:b])).convert("RGB"))
                       for a, b in itertools.pairwise(offsets)])
    wav = arrays["audio"].astype(np.float32) / 32768.0
    net, proc = load_release(str(pathlib.Path(DATA) / "release" / release),
                             snapshot_download(vm.OMNI_ID))
    t0 = time.perf_counter()
    enc = vm.encode(proc, record, frames, wav if audio else None, fps=fps)
    with torch.no_grad():
        outs = [o.float().softmax(-1).cpu().numpy() for o in net(enc)]
    ms = (time.perf_counter() - t0) * 1e3
    result = {}
    step = 2 / fps
    for (name, q), p in zip(record["questions"].items(), outs, strict=True):
        if q["type"] == "moment":
            top = np.argsort(-p)[:5]
            result[name] = [{"from_s": round(float(i * step), 1), "to_s": round(float((i + 1) * step), 1),
                             "p": round(float(p[i]), 3)} for i in top]
        else:
            labels = [str(label) for label, _ in cs.question_options(q)]
            result[name] = {label: round(float(x), 3) for label, x in zip(labels, p, strict=True)}
    return {"answers": result, "frames": len(frames), "forward_ms": round(ms)}


@app.local_entrypoint()
def ask(video: str, record: str, release: str = "g4-joint", fps: float = 2.0,
         seconds: int = 60, audio: bool = True):
    rec = json.loads(pathlib.Path(record).read_text())
    out = answer.remote(pathlib.Path(video).read_bytes(), rec, release, fps, seconds, audio)
    print(json.dumps(out, indent=1))
