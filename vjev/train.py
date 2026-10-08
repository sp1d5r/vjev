"""Fine-tune Qwen3-Omni + typed head on SnapUGC ECR, logged to W&B.

    modal run vjev/train.py --train-limit 200 --val-limit 50 --max-steps 20   # smoke test
    modal run vjev/train.py --train-limit 10000 --val-limit 500               # pilot
    modal run vjev/train.py --no-audio ...                                    # video-only ablation
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import time

import modal

from vjev.common import DATA, HF, app, data_vol, hf_vol, train_image


def _records(split: str, limit: int, seed: int) -> list[dict]:
    from vjev.preprocess import pre_path

    root = pathlib.Path(DATA)
    lines = (root / "index" / f"{split}.jsonl").read_text().splitlines()
    recs = [json.loads(line) for line in lines]
    recs = [r for r in recs if pre_path(root, r["id"]).exists()]
    random.Random(seed).shuffle(recs)
    return recs[:limit] if limit else recs


class ClipDataset:
    """Map-style dataset. Each worker loads its own processor on first use."""

    def __init__(self, recs: list[dict], edges: list[float], model_path: str, audio: bool):
        self.recs, self.edges, self.model_path, self.audio = recs, edges, model_path, audio
        self.processor = None

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int):
        import numpy as np
        from transformers import Qwen3OmniMoeProcessor

        from vjev import model, preprocess
        from vjev.metrics import ecr_level
        from vjev.questions import record_for

        if self.processor is None:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        rec = self.recs[i]
        frames, audio, duration = preprocess.load(preprocess.pre_path(pathlib.Path(DATA), rec["id"]))
        record = record_for(rec, self.edges, duration, text=True)
        enc = model.encode(self.processor, record, frames, audio if self.audio else None)
        level = int(ecr_level(np.array([rec["ecr"]]), np.array(self.edges))[0])
        return enc, level, rec["ecr"], rec["id"]


def _loader(ds, shuffle: bool, workers: int = 8):
    import torch

    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=shuffle, num_workers=workers,
                                       prefetch_factor=4, persistent_workers=False)


def evaluate(net, loader, edges, limit_s: float | None = None) -> tuple[dict, list[dict]]:
    import numpy as np
    import torch

    from vjev import metrics

    net.eval()
    rows, fwd = [], []
    with torch.no_grad():
        for enc, _level, ecr, vid in loader:
            t0 = time.perf_counter()
            probs = net(enc)[0].float().softmax(-1).cpu().numpy()
            torch.cuda.synchronize()
            fwd.append(time.perf_counter() - t0)
            rows.append({"id": vid, "ecr": ecr, "probs": probs.tolist(),
                         "expected": float((probs * np.arange(len(probs))).sum())})
            if limit_s and sum(fwd) > limit_s:
                break
    net.train()
    probs = np.array([r["probs"] for r in rows])
    expected = np.array([r["expected"] for r in rows])
    result = metrics.report(expected, probs, np.array([r["ecr"] for r in rows]), np.array(edges))
    result["forward_ms_p50"] = float(np.median(fwd) * 1e3)
    result["expected_std"] = float(expected.std())
    return result, rows


@app.function(
    image=train_image,
    gpu="H200",
    volumes={DATA: data_vol, HF: hf_vol},
    secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")],
    timeout=24 * 3600,
    cpu=12,
    memory=65536,
)
def train(cfg: dict) -> dict:
    import torch
    import wandb
    from huggingface_hub import snapshot_download
    from peft import get_peft_model_state_dict

    from vjev import model
    from vjev.losses import typed_loss

    torch.manual_seed(cfg["seed"])
    path = snapshot_download(model.OMNI_ID)
    hf_vol.commit()
    root = pathlib.Path(DATA)
    edges = json.loads((root / "index" / "meta.json").read_text())["level_edges"]

    net, _ = model.load(path, cfg["lora_r"], cfg["lora_alpha"],
                        vision_lora=cfg.get("vision_lora", False))
    lora = [p for p in net.thinker.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora, "lr": cfg["lr_lora"]},
        {"params": net.head.parameters(), "lr": cfg["lr_head"]},
    ], weight_decay=0.01)

    train_recs = _records("train", cfg["train_limit"], cfg["seed"])
    val_recs = _records("val", cfg["val_limit"], cfg["seed"])
    train_dl = _loader(ClipDataset(train_recs, edges, path, cfg["audio"]), shuffle=True)
    val_dl = _loader(ClipDataset(val_recs, edges, path, cfg["audio"]), shuffle=False, workers=4)

    total_steps = cfg["max_steps"] or math.ceil(len(train_recs) * cfg["epochs"] / cfg["accum"])
    warmup = min(50, total_steps // 10)

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    notes = (f"Qwen3-Omni with {'video and audio' if cfg['audio'] else 'video only (no audio)'}, "
             f"typed head trained from scratch, {len(train_recs):,} training clips, "
             f"LoRA rank {cfg['lora_r']}. Checked on {len(val_recs)} held-out clips "
             f"every {cfg['eval_every']} steps.")
    run = wandb.init(project="vjev", job_type="train", name=cfg["run_name"], notes=notes,
                     config=cfg | {"train_clips": len(train_recs), "val_clips": len(val_recs),
                                   "total_steps": total_steps,
                                   "lora_params": sum(p.numel() for p in lora),
                                   "head_params": sum(p.numel() for p in net.head.parameters())})
    ckpt = root / "ckpt" / cfg["run_name"]
    ckpt.mkdir(parents=True, exist_ok=True)

    step, micro, best, t_start = 0, 0, -1.0, time.perf_counter()
    skipped_clips = skipped_steps = 0
    grad_norm = torch.tensor(0.0)
    window: dict[str, float] = {"loss": 0.0, "ce": 0.0, "brier": 0.0, "emd": 0.0, "n": 0}
    net.train()
    done = False
    for _epoch in range(math.ceil(cfg["epochs"])):
        for enc, level, _ecr, _vid in train_dl:
            out = typed_loss(net(enc)[0], level, cfg["smoothing"], cfg["brier_w"], cfg["emd_w"])
            micro += 1
            # One bad clip must not poison the weights: skip non-finite losses and grads.
            if torch.isfinite(out["loss"]):
                (out["loss"] / cfg["accum"]).backward()
                for k in ("loss", "ce", "brier", "emd"):
                    window[k] += out[k].detach().item()
                window["n"] += 1
            else:
                skipped_clips += 1
                print(f"non-finite loss on {_vid}, skipped", flush=True)
            if micro % cfg["accum"]:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(lora + list(net.head.parameters()), 1.0)
            if torch.isfinite(grad_norm):
                opt.step()
            else:
                skipped_steps += 1
                print(f"non-finite grad norm at step {step}, skipped", flush=True)
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                elapsed = time.perf_counter() - t_start
                n = max(window.pop("n"), 1)
                run.log({f"train/{k}": v / n for k, v in window.items()}
                        | {"train/lr": sched.get_last_lr()[0], "train/clips_per_hour":
                           micro / elapsed * 3600, "train/grad_norm": grad_norm.item(),
                           "train/skipped_clips": skipped_clips,
                           "train/skipped_steps": skipped_steps, "step": step})
                window = {"loss": 0.0, "ce": 0.0, "brier": 0.0, "emd": 0.0, "n": 0}
            if step % cfg["eval_every"] == 0 or step == total_steps:
                res, _ = evaluate(net, val_dl, edges)
                run.log({f"val/{k}": v for k, v in res.items()} | {"step": step})
                print(f"step {step}/{total_steps} val {json.dumps(res)}", flush=True)
                if res["final"] > best:
                    best = res["final"]
                    torch.save({"lora": get_peft_model_state_dict(net.thinker),
                                "head": net.head.state_dict(), "cfg": cfg, "step": step,
                                "val": res}, ckpt / "best.pt")
                    data_vol.commit()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    # LoRA B starts at zero, so a non-zero norm proves the vision adapters got gradients.
    run.summary["vision_lora_b_norm"] = sum(
        p.detach().float().norm().item() for n, p in net.thinker.named_parameters()
        if "visual" in n and "lora_B" in n)
    run.summary["best_val_final"] = best
    run.summary["train_hours"] = (time.perf_counter() - t_start) / 3600
    run.finish()
    return {"best_val_final": best, "steps": step, "clips_seen": micro,
            "hours": (time.perf_counter() - t_start) / 3600}


@app.local_entrypoint()
def main(train_limit: int = 200, val_limit: int = 50, epochs: float = 1.0, max_steps: int = 0,
         accum: int = 8, lr_lora: float = 1e-4, lr_head: float = 3e-4, lora_r: int = 64,
         lora_alpha: int = 128, smoothing: float = 0.05, brier_w: float = 1.0,
         emd_w: float = 1.0, eval_every: int = 100, audio: bool = True, seed: int = 0,
         vision_lora: bool = False, run_name: str = ""):
    cfg = {k: v for k, v in locals().items()}
    cfg["run_name"] = run_name or (
        f"omni-{'av' if audio else 'v'}-n{train_limit}-r{lora_r}-{time.strftime('%m%d-%H%M')}")
    print(json.dumps(train.remote(cfg), indent=2))
