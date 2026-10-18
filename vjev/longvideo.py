"""Long-video test (G5, small): can a model trained on clips of a minute or less find a moment
in a video of several minutes, by scoring overlapping windows?

    modal run vjev/longvideo.py::long_video --run g4-joint --n 60

Each test video is built from val clips: one FunQA clip (the "needle", a funny or magic
moment with a known window) placed at a random point among VGGSound speech and crowd clips,
up to 1, 3, 5 or 10 minutes. The model sees 60 s windows with 30 s stride. Per second, the
combined score is the highest logit any window gives it, and a softmax over the whole video
gives one distribution. HIT@1: the top second is inside the needle's moment.
"""

from __future__ import annotations

import json
import pathlib
import random

import modal
import numpy as np

from vjev.common import DATA, HF, app, data_vol, hf_vol, train_image

FPS = 2.0
WINDOW_S, STRIDE_S = 60, 30
SIZE = (336, 252)  # (W, H), multiples of 28
FILLER = {"female speech, woman speaking", "male speech, man speaking",
          "child speech, kid speaking", "people crowd"}


def combine(windows: list[tuple[int, np.ndarray]], total_s: int) -> tuple[np.ndarray, np.ndarray]:
    """(start second, logits per second) per window -> softmax over max-logit, and over the
    max of per-window probabilities, for every second of the video."""
    best_logit = np.full(total_s, -np.inf)
    best_prob = np.zeros(total_s)
    for start, logits in windows:
        p = np.exp(logits - logits.max())
        p /= p.sum()
        n = min(len(logits), total_s - start)
        best_logit[start:start + n] = np.maximum(best_logit[start:start + n], logits[:n])
        best_prob[start:start + n] = np.maximum(best_prob[start:start + n], p[:n])
    best_logit[np.isinf(best_logit)] = best_logit[~np.isinf(best_logit)].min()
    q = np.exp(best_logit - best_logit.max())
    return q / q.sum(), best_prob / best_prob.sum()


def hit(p: np.ndarray, truth: list[tuple[float, float]], k: int = 1) -> float:
    """1 if any of the k most likely seconds overlaps a true window."""
    top = np.argsort(-p)[:k]
    return float(any(any(t < b and t + 1 > a for a, b in truth) for t in top))


def window_starts(total_s: int) -> list[int]:
    if total_s <= WINDOW_S:
        return [0]
    starts = list(range(0, total_s - WINDOW_S + 1, STRIDE_S))
    if starts[-1] + WINDOW_S < total_s:
        starts.append(total_s - WINDOW_S)
    return starts


@app.function(image=train_image, gpu="H200", volumes={DATA: data_vol, HF: hf_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=4 * 3600, cpu=12,
              memory=65536)
def lv_run(ckpt_run: str, n: int, minutes: list[int], seed: int = 0) -> dict:
    import time

    import torch
    from huggingface_hub import snapshot_download
    from peft import set_peft_model_state_dict
    from PIL import Image
    from transformers import Qwen3OmniMoeProcessor

    from vjev import model as vm
    from vjev.preprocess import load, pre_path
    from vjev.tasks import make_item
    from vjev.train import _records

    root = pathlib.Path(DATA)
    ck = torch.load(root / "ckpt" / ckpt_run / "best.pt", map_location="cuda", weights_only=False)
    path = snapshot_download(vm.OMNI_ID)
    net, _ = vm.load(path, ck["cfg"]["lora_r"], ck["cfg"]["lora_alpha"])
    set_peft_model_state_dict(net.thinker, ck["lora"])
    net.head.load_state_dict(ck["head"])
    net.eval()
    proc = Qwen3OmniMoeProcessor.from_pretrained(path)

    def clip(rec_id: str) -> tuple[np.ndarray, np.ndarray]:
        frames, wav, _ = load(pre_path(root, rec_id))
        frames = frames[: len(frames) // 2 * 2]
        frames = np.stack([np.asarray(Image.fromarray(f).resize(SIZE, Image.BICUBIC)) for f in frames])
        m = int(len(frames) / FPS * vm.SR)
        return frames, np.pad(wav[:m], (0, max(0, m - len(wav))))

    rng = random.Random(seed)
    needles = [r for r in _records("val", 0, seed, "fq") if r["kind"] in ("H1", "M1")][:n]
    fillers = [r for r in _records("val", 0, seed, "vs") if r["label"] in FILLER]
    rows = []
    for rec in needles:
        nf, nw = clip(rec["id"])
        needle_s = len(nf) / FPS
        instr = make_item("fq", rec, len(nf), needle_s, []).record["questions"]["moment"]
        for mins in minutes:
            # Fillers before and after the needle, the needle at a random clip boundary.
            parts, total = [], needle_s
            while total < mins * 60:
                f = clip(rng.choice(fillers)["id"])
                parts.append(f)
                total += len(f[0]) / FPS
            k = rng.randint(0, len(parts))
            seq = parts[:k] + [(nf, nw)] + parts[k:]
            frames = np.concatenate([p[0] for p in seq])
            wav = np.concatenate([p[1] for p in seq])
            offset = sum(len(p[0]) for p in parts[:k]) / FPS
            total_s = int(len(frames) / FPS)
            t0 = time.perf_counter()
            windows = []
            for s in window_starts(total_s):
                wf = frames[int(s * FPS): int((s + WINDOW_S) * FPS)]
                wf = wf[: len(wf) // 2 * 2]
                ww = wav[int(s * vm.SR): int(s * vm.SR) + int(len(wf) / FPS * vm.SR)]
                record = {"state": {"duration_seconds": round(len(wf) / FPS, 1)},
                          "questions": {"moment": instr}}
                enc = vm.encode(proc, record, wf, ww, fps=FPS)
                with torch.no_grad():
                    logits = net(enc)[0].float().cpu().numpy()
                windows.append((s, logits))
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            p_logit, p_prob = combine(windows, total_s)
            truth = [(a + offset, b + offset) for a, b in rec["windows"]]

            rows.append({"id": rec["id"], "minutes": mins, "total_s": total_s,
                         "windows": len(windows), "seconds": secs,
                         "hit1_logit": hit(p_logit, truth), "hit1_prob": hit(p_prob, truth),
                         "hit3_logit": hit(p_logit, truth, 3),
                         "chance": sum(b - a for a, b in truth) / total_s})
    out = root / "preds" / f"longvideo-{ckpt_run}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data_vol.commit()
    summary = {}
    for mins in minutes:
        rs = [r for r in rows if r["minutes"] == mins]
        summary[mins] = {k: float(np.mean([r[k] for r in rs]))
                         for k in ("total_s", "windows", "seconds", "hit1_logit", "hit1_prob",
                                   "hit3_logit", "chance")}
    return {"run": ckpt_run, "n": len(needles), "summary": summary}


@app.local_entrypoint()
def long_video(run: str = "g3a-fq-av", n: int = 60, minutes: str = "1,3,5,10"):
    print(json.dumps(lv_run.remote(run, n, [int(m) for m in minutes.split(",")]), indent=1))
