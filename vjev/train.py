"""Fine-tune Qwen3-Omni + typed head, logged to W&B.

Tasks are defined in `vjev/tasks.py`: `snapugc`, `vf`, `gv`, `qv`, or a mix such as
`vf+gv+qv`. All questions about a clip are answered in one forward pass.

    modal run vjev/train.py --train-limit 200 --val-limit 50 --max-steps 20   # smoke test
    modal run vjev/train.py --train-limit 10000 --val-limit 500               # pilot
    modal run vjev/train.py --no-audio ...                                    # video-only ablation
    modal run vjev/train.py --task vf --no-audio ...                          # VideoFeedback
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import time

import modal

from vjev.common import DATA, HF, app, data_vol, hf_vol, train_image


def _index_name(task: str, split: str) -> str:
    from vjev.tasks import INDEX_OF

    task = INDEX_OF.get(task, task)
    return split if task == "snapugc" else f"{task}_{split}"


def _records(split: str, limit: int, seed: int, task: str = "snapugc") -> list[dict]:
    """`task` may join several with "+", e.g. "vf+gv+qv". `limit` applies to each task."""
    from vjev.preprocess import pre_path

    root = pathlib.Path(DATA)
    out = []
    for t in task.split("+"):
        lines = (root / "index" / f"{_index_name(t, split)}.jsonl").read_text().splitlines()
        recs = [json.loads(line) | {"task": t} for line in lines]
        recs = [r for r in recs if pre_path(root, r["id"]).exists()]
        random.Random(seed).shuffle(recs)
        out += recs[:limit] if limit else recs
    random.Random(seed).shuffle(out)
    return out


class ClipDataset:
    """Map-style dataset. Each worker loads its own processor on first use."""

    def __init__(self, recs: list[dict], edges: list[float], model_path: str, audio: bool,
                 task: str = "snapugc"):
        self.recs, self.edges, self.model_path, self.audio = recs, edges, model_path, audio
        self.task = task
        self.processor = None

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int):
        from transformers import Qwen3OmniMoeProcessor

        from vjev import model, preprocess
        from vjev.tasks import make_item, prepare_frames, task_of

        if self.processor is None:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        rec = self.recs[i]
        frames, audio, duration = preprocess.load(preprocess.pre_path(pathlib.Path(DATA), rec["id"]))
        task = task_of(rec, self.task)
        frames = prepare_frames(task, frames)
        item = make_item(task, rec, len(frames), duration, self.edges, self.audio)
        enc = model.encode(self.processor, item.record, frames, audio if item.audio else None,
                           fps=item.fps)
        return enc, item, rec["id"]


def _loader(ds, shuffle: bool, workers: int = 8):
    import torch

    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=shuffle, num_workers=workers,
                                       prefetch_factor=4, persistent_workers=False)


def evaluate(net, loader, edges=None, limit_s: float | None = None,
             task: str = "snapugc") -> tuple[dict, list[dict]]:
    """Metrics per question, keyed `{question}/{metric}` (or `{task}/{question}/{metric}`
    when tasks are mixed), plus their means. A single question keeps the plain keys. `final`
    is the mean of each question's headline number: SROCC/PLCC mean for scores, AUC for
    yes/no, HIT@1 for moments."""
    import numpy as np
    import torch

    from vjev.tasks import question_report

    net.eval()
    groups: dict[str, dict] = {}
    rows, fwd = [], []
    with torch.no_grad():
        for enc, item, vid in loader:
            t0 = time.perf_counter()
            outs = [o.float().softmax(-1).cpu().numpy() for o in net(enc)]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd.append(time.perf_counter() - t0)
            t = item.task or task
            row = {"id": vid, "scores": {}}
            for q, probs in zip(item.record["questions"], outs, strict=True):
                key = f"{t}/{q}"
                g = groups.setdefault(key, {"kind": item.kinds[q], "probs": [], "target": [],
                                            "extra": []})
                g["probs"].append(probs)
                g["target"].append(item.targets[q])
                g["extra"].append(item.extra.get(q, {}))
                row["scores"][q] = {"target": item.targets[q], "probs": probs.tolist()}
            rows.append(row)
            if limit_s and sum(fwd) > limit_s:
                break
    net.train()
    reports = {k: question_report(g["kind"], g["probs"], g["target"], g["extra"])
               for k, g in groups.items()}
    tasks_seen = {k.split("/")[0] for k in reports}
    if len(reports) == 1:
        ((key, result),) = reports.items()
        g = groups[key]
        if g["kind"] == "score" and key.startswith("snapugc/"):  # keep the SnapUGC row format
            rows = [{"id": r["id"], "ecr": e["raw"], "probs": p.tolist(),
                     "expected": float(p @ np.arange(len(p)))}
                    for r, e, p in zip(rows, g["extra"], g["probs"], strict=True)]
        result = dict(result)
    else:
        result = {}
        for k, r in reports.items():
            name = k if len(tasks_seen) > 1 else k.split("/", 1)[1]
            result |= {f"{name}/{m}": v for m, v in r.items() if m != "n"}
        common = set.intersection(*(set(r) for r in reports.values())) - {"n"}
        for m in sorted(common):
            result[m] = float(np.nanmean([r[m] for r in reports.values()]))
        result["n"] = len(rows)
    result["forward_ms_p50"] = float(np.median(fwd) * 1e3)
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

    task = cfg.get("task", "snapugc")
    train_recs = _records("train", cfg["train_limit"], cfg["seed"], task)
    val_recs = _records("val", cfg["val_limit"], cfg["seed"], task)
    train_dl = _loader(ClipDataset(train_recs, edges, path, cfg["audio"], task), shuffle=True)
    val_dl = _loader(ClipDataset(val_recs, edges, path, cfg["audio"], task), shuffle=False,
                     workers=4)

    total_steps = cfg["max_steps"] or math.ceil(len(train_recs) * cfg["epochs"] / cfg["accum"])
    warmup = min(50, total_steps // 10)

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    names = {"snapugc": "SnapUGC retention", "vf": "VideoFeedback (5 quality scores)",
             "gv": "GenVideo (real or AI)", "qv": "QVHighlights (moments)"}
    what = " + ".join(names.get(t, t) for t in task.split("+"))
    notes = (f"{what}. Qwen3-Omni with "
             f"{'video and audio' if cfg['audio'] else 'video only (no audio)'}, "
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
        for enc, item, _vid in train_dl:
            # One loss per question, averaged, so adding questions does not scale the rate.
            parts = [typed_loss(o, item.targets[q], cfg["smoothing"], cfg["brier_w"],
                                cfg["emd_w"], ordinal=item.kinds[q] != "noul")
                     for o, q in zip(net(enc), item.record["questions"], strict=True)]
            out = {k: sum(p[k] for p in parts) / len(parts) for k in parts[0]}
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
                res, _ = evaluate(net, val_dl, edges, task=task)
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
         vision_lora: bool = False, task: str = "snapugc", run_name: str = ""):
    cfg = {k: v for k, v in locals().items()}
    cfg["run_name"] = run_name or (
        f"omni-{'av' if audio else 'v'}-n{train_limit}-r{lora_r}-{time.strftime('%m%d-%H%M')}")
    print(json.dumps(train.remote(cfg), indent=2))
