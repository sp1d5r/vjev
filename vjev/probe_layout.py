"""Print where video and audio tokens sit in the thinker's input, for the moment question type.

    modal run vjev/probe_layout.py
"""

from __future__ import annotations

import numpy as np

from vjev.common import HF, app, gpu_image, hf_vol
from vjev.model import OMNI_ID, SR


@app.function(image=gpu_image, volumes={HF: hf_vol}, cpu=4, memory=16384, timeout=1200)
def probe() -> dict:
    from transformers import Qwen3OmniMoeProcessor

    proc = Qwen3OmniMoeProcessor.from_pretrained(OMNI_ID)
    tok = proc.tokenizer
    vid = tok.convert_tokens_to_ids("<|video_pad|>")
    aud = tok.convert_tokens_to_ids("<|audio_pad|>")
    out = {}
    for fps, n, audio in ((2.0, 32, True), (2.0, 32, False), (8.0, 24, False)):
        frames = np.random.randint(0, 255, (n, 252, 448, 3), dtype=np.uint8)
        kw = {"fps": fps, "do_sample_frames": False, "return_tensors": "pt"}
        text = ["<|vision_start|><|video_pad|><|vision_end|>\n"]
        if audio:
            wav = np.zeros(int(n / fps * SR), dtype=np.float32)
            o = proc(text=text, videos=[frames], audio=[wav], use_audio_in_video=True,
                     sampling_rate=SR, **kw)
        else:
            o = proc(text=text, videos=[frames], **kw)
        ids = o["input_ids"][0].tolist()
        runs, cur, start = [], None, 0
        for i, t in enumerate([*ids, -1]):
            kind = "v" if t == vid else "a" if t == aud else "o"
            if kind != cur:
                if cur is not None:
                    runs.append((cur, start, i - start))
                cur, start = kind, i
        out[f"fps{fps}-n{n}-audio{audio}"] = {
            "grid_thw": o["video_grid_thw"].tolist(), "n_tokens": len(ids),
            "n_video": ids.count(vid), "n_audio": ids.count(aud),
            "runs": runs[:40], "other_tokens": [tok.decode([t]) for t in ids if t not in (vid, aud)][:40],
        }
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(probe.remote(), indent=1))
