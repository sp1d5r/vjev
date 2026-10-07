"""Zero-shot Clef-flash on our SnapUGC split. No training.

    modal run vjev/baseline.py --split test --limit 500
    modal run vjev/baseline.py --split test --limit 500 --no-video   # text-only ablation
    modal run vjev/baseline.py --split test --limit 500 --no-text    # video-only ablation
"""

from __future__ import annotations

import json
import pathlib
import time

import modal

from vjev.common import DATA, HF, app, data_vol, gpu_image, hf_vol

MODEL_ID = "Cloudflare/clef-flash"


def read_frames(path: str, fps: float, max_frames: int, max_side: int):
    """Decode about `fps` frames per second, capped at max_frames (kept even for Qwen)."""
    import av
    import numpy as np
    from PIL import Image

    with av.open(path) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        frames, next_t = [], 0.0
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else next_t
            if t + 1e-6 < next_t:
                continue
            img = frame.to_image()
            scale = max_side / max(img.size)
            if scale < 1:
                img = img.resize((round(img.width * scale), round(img.height * scale)),
                                 Image.BILINEAR)
            frames.append(np.asarray(img.convert("RGB")))
            next_t = t + 1.0 / fps
            if len(frames) >= max_frames:
                break
    if duration is None and frames:
        duration = len(frames) / fps
    if len(frames) % 2:
        frames.append(frames[-1])
    return np.stack(frames), duration


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={DATA: data_vol, HF: hf_vol},
    secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")],
    timeout=4 * 3600,
)
def zero_shot(split: str, limit: int, fps: float, max_frames: int, max_side: int,
              video: bool, text: bool) -> dict:
    import sys

    import numpy as np
    import torch
    import wandb
    from huggingface_hub import snapshot_download

    from vjev import metrics
    from vjev.questions import record_for

    path = snapshot_download(MODEL_ID)
    hf_vol.commit()
    sys.path.insert(0, path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    model, processor = load_release_model(path, device="cuda")

    root = pathlib.Path(DATA)
    meta = json.loads((root / "index" / "meta.json").read_text())
    edges = meta["level_edges"]
    lines = (root / "index" / f"{split}.jsonl").read_text().splitlines()
    recs = [json.loads(line) for line in lines[:limit]]

    config = {"model": MODEL_ID, "split": split, "limit": limit, "fps": fps,
              "max_frames": max_frames, "max_side": max_side, "video": video, "text": text,
              "trained": False}
    mode = "+".join(m for m, on in (("video", video), ("text", text)) if on)
    run = wandb.init(project="vjev", job_type="zero-shot", config=config,
                     name=f"clef-flash-zs-{mode}-{split}{limit}")

    rows, decode_s, forward_s, n_tokens = [], [], [], []
    for i, rec in enumerate(recs):
        t0 = time.perf_counter()
        frames, duration = (read_frames(str(root / rec["path"]), fps, max_frames, max_side)
                            if video else (None, None))
        record = record_for(rec, edges, duration, text)
        if video:
            record["videos"] = [frames]
            record["media_kwargs"] = {"do_sample_frames": False}
        enc = encode_record(processor.tokenizer, record, processor=processor)
        t1 = time.perf_counter()
        batch = collate_records([enc], processor.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            logits = model(batch)[0][0]
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        probs = logits.float().softmax(-1).cpu().numpy()
        rows.append({"id": rec["id"], "ecr": rec["ecr"], "probs": probs.tolist(),
                     "expected": float((probs * np.arange(len(probs))).sum())})
        decode_s.append(t1 - t0)
        forward_s.append(t2 - t1)
        n_tokens.append(int(batch["input_ids"].shape[1]))
        if i % 50 == 0:
            print(f"{i}/{len(recs)} tokens={n_tokens[-1]} fwd={forward_s[-1]*1e3:.0f}ms")

    probs = np.array([r["probs"] for r in rows])
    expected = np.array([r["expected"] for r in rows])
    ecr = np.array([r["ecr"] for r in rows])
    result = metrics.report(expected, probs, ecr, np.array(edges))
    result |= {
        "forward_ms_p50": float(np.median(forward_s) * 1e3),
        "forward_ms_p95": float(np.percentile(forward_s, 95) * 1e3),
        "decode_ms_p50": float(np.median(decode_s) * 1e3),
        "tokens_p50": float(np.median(n_tokens)),
        "clips_per_hour": 3600 / float(np.mean(np.add(decode_s, forward_s))),
        # Spread of predictions; a collapsed model predicts one level for everything.
        "expected_std": float(expected.std()),
    }
    run.summary.update(result)
    run.log({"predictions": wandb.Table(
        columns=["id", "ecr", "expected"],
        data=[[r["id"], r["ecr"], r["expected"]] for r in rows])})
    run.finish()

    out = root / "preds" / f"{run.name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data_vol.commit()
    return result


@app.local_entrypoint()
def main(split: str = "test", limit: int = 500, fps: float = 1.0, max_frames: int = 16,
         max_side: int = 448, video: bool = True, text: bool = True):
    result = zero_shot.remote(split, limit, fps, max_frames, max_side, video, text)
    print(json.dumps(result, indent=2))
