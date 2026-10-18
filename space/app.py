"""Video Jev demo: ask typed questions about a video, get a probability for every answer.

Runs on a ZeroGPU Space (full RTX Pro 6000, 96 GB). Each group of questions is decoded with
the settings its task was trained on (see presets.py), so one click can run a few passes.
"""

from __future__ import annotations

import html
import json
import os
import pathlib
import subprocess
import tempfile

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download
from peft import set_peft_model_state_dict
from PIL import Image
from presets import CHOICES, MATCHED_FRAMES, MATCHED_SIDE, PROFILES, Profile, build
from safetensors.torch import load_file

from vjev import clef_schema as cs
from vjev import model as vm

WEIGHTS = os.environ.get("VJEV_WEIGHTS", "sp1d5r/video-jev")


def load():
    """Model on cuda at import time, as ZeroGPU expects."""
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    weights = pathlib.Path(snapshot_download(
        WEIGHTS, allow_patterns=["adapter_model.safetensors", "head.safetensors",
                                 "vjev_config.json"]))
    cfg = json.loads((weights / "vjev_config.json").read_text())
    base = snapshot_download(cfg["base_model"])
    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa")
    full.disable_talker()
    thinker = full.thinker
    thinker.requires_grad_(False)
    out_weight = thinker.lm_head.weight.detach().float()
    thinker.lm_head = torch.nn.Identity()
    from peft import LoraConfig, get_peft_model

    thinker = get_peft_model(thinker, LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                                                 lora_dropout=0.0, target_modules=vm.LORA_TARGETS))
    set_peft_model_state_dict(thinker, load_file(weights / "adapter_model.safetensors"))
    hidden = full.config.thinker_config.text_config.hidden_size
    head = cs.JointSchemaHead(hidden_size=hidden, dropout=0.0, **vm.HEAD_CONFIG)
    head.load_state_dict(load_file(weights / "head.safetensors"))
    net = vm.TypedOmni(thinker.to("cuda"), head.to("cuda"), out_weight.to("cuda")).eval()
    return net, Qwen3OmniMoeProcessor.from_pretrained(base)


NET, PROC = load()


def probe(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", path], capture_output=True, text=True, check=False).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def decode(path: str, p: Profile) -> tuple[np.ndarray, np.ndarray | None]:
    scale = f"scale='if(gt(iw,ih),{p.max_side},-2)':'if(gt(iw,ih),-2,{p.max_side})'"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-t", str(p.seconds), "-i", path, "-vf",
                        f"fps={p.fps},{scale}", "-q:v", "4", f"{tmp}/%04d.jpg"], check=True)
        frames = [np.asarray(Image.open(f).convert("RGB"))
                  for f in sorted(pathlib.Path(tmp).glob("*.jpg"))]
    if not frames:
        raise gr.Error("Could not read any frames from this video.")
    frames = np.stack(frames)
    if p.matched:
        frames = frames[:MATCHED_FRAMES]
        if len(frames) < MATCHED_FRAMES:
            frames = np.concatenate([frames, np.repeat(frames[-1:], MATCHED_FRAMES - len(frames), 0)])
        h, w = frames.shape[1:3]
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        frames = np.stack([np.asarray(Image.fromarray(f[top:top + s, left:left + s]).resize(
            (MATCHED_SIDE, MATCHED_SIDE), Image.BICUBIC)) for f in frames])
    if not p.audio:
        return frames, None
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-t", str(p.seconds), "-i", path, "-vn", "-ac",
                          "1", "-ar", str(vm.SR), "-f", "s16le", "-"],
                         capture_output=True, check=False).stdout
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    return frames, audio if len(audio) else None


@spaces.GPU(size="xlarge", duration=90)
def run_model(jobs: list[tuple[dict, np.ndarray, np.ndarray | None, float]]) -> list[list[np.ndarray]]:
    out = []
    for record, frames, audio, fps in jobs:
        enc = vm.encode(PROC, record, frames, audio, fps=fps)
        with torch.no_grad():
            out.append([o.float().softmax(-1).cpu().numpy() for o in NET(enc)])
    return out


def summarise(record: dict, probs: list[np.ndarray], fps: float, n_frames: int) -> dict:
    result = {}
    for (name, q), p in zip(record["questions"].items(), probs, strict=True):
        if q["type"] == "moment":
            bins = vm.moment_bins(n_frames, fps)
            result[name] = {"type": "moment", "question": q["instructions"],
                            "bins": [[round(a, 1), round(b, 1), round(float(x), 4)]
                                     for (a, b), x in zip(bins, p, strict=True)]}
        else:
            opts = cs.question_options(q)
            if q["type"] == "noul" and "criteria" not in q:
                labels = ["yes", "no"]  # Clef's default noul options are long sentences
            else:
                labels = [str(text).split(":")[0] if q["type"] == "score" else str(text)
                          for _key, text in opts]
            result[name] = {"type": q["type"], "question": q["instructions"],
                            "answers": {lab: round(float(x), 4)
                                        for lab, x in zip(labels, p, strict=True)}}
            if q["type"] == "score":
                result[name]["expected"] = round(float((np.arange(len(p)) + 1) @ p), 2)
    return result


TITLES = {"ai_generated": "AI-generated?", "laughing": "Laughing?", "sound": "Which sound?"}


def bar(label: str, p: float, top: bool) -> str:
    return (f'<div class="vj-row"><span class="vj-lab">{html.escape(label)}</span>'
            f'<span class="vj-track"><span class="vj-fill{" vj-top" if top else ""}" '
            f'style="width:{p * 100:.1f}%"></span></span><span class="vj-p">{p:.0%}</span></div>')


def render(results: dict, notes: list[str]) -> str:
    parts = ["""<style>
.vj-card{border:1px solid var(--border-color-primary);border-radius:10px;padding:14px 16px;margin:0 0 12px}
.vj-q{font-size:13px;opacity:.75;margin:0 0 10px}.vj-name{font-weight:600;margin:0 0 2px}
.vj-row{display:grid;grid-template-columns:minmax(90px,38%) 1fr 44px;gap:10px;align-items:center;margin:4px 0;font-size:14px}
.vj-track{height:10px;border-radius:5px;background:var(--neutral-200,#e5e7eb);overflow:hidden}
.dark .vj-track{background:var(--neutral-700,#374151)}
.vj-fill{display:block;height:100%;background:#94a3b8}.vj-top{background:#2f55d4}
.vj-p{text-align:right;font-variant-numeric:tabular-nums}
.vj-hist{display:flex;align-items:flex-end;gap:1px;height:90px;margin:6px 0 2px}
.vj-hist span{flex:1;background:#94a3b8;min-height:1px;border-radius:2px 2px 0 0}
.vj-hist span.vj-top{background:#2f55d4}
.vj-axis{display:flex;justify-content:space-between;font-size:12px;opacity:.7}
.vj-note{font-size:12px;opacity:.7;margin:4px 0 0}
</style>"""]
    for name, r in results.items():
        title = TITLES.get(name, name.replace("_", " ").capitalize())
        parts.append(f'<div class="vj-card"><div class="vj-name">{html.escape(title)}</div>'
                     f'<div class="vj-q">{html.escape(r["question"])}</div>')
        if r["type"] == "moment":
            bins = r["bins"]
            ps = np.array([b[2] for b in bins])
            top3 = set(np.argsort(-ps)[:3].tolist())
            hi = ps.max() or 1.0
            parts.append('<div class="vj-hist">' + "".join(
                f'<span class="{"vj-top" if i in top3 else ""}" style="height:{p / hi * 100:.1f}%" '
                f'title="{b[0]:g} to {b[1]:g} s: {p:.1%}"></span>'
                for i, (b, p) in enumerate(zip(bins, ps, strict=True))) + "</div>")
            parts.append(f'<div class="vj-axis"><span>0 s</span><span>{bins[-1][1]:g} s</span></div>')
            for i in sorted(top3, key=lambda i: -ps[i]):
                parts.append(bar(f"{bins[i][0]:g} to {bins[i][1]:g} s", float(ps[i]), i == int(ps.argmax())))
        else:
            best = max(r["answers"], key=r["answers"].get)
            for lab, p in sorted(r["answers"].items(), key=lambda kv: -kv[1]) if r["type"] == "choice" \
                    else r["answers"].items():
                parts.append(bar(lab, p, lab == best))
            if "expected" in r:
                parts.append(f'<div class="vj-note">Expected score {r["expected"]:.2f} of 4</div>')
        parts.append("</div>")
    parts += [f'<div class="vj-note">{html.escape(n)}</div>' for n in notes]
    return "".join(parts)


def ask(video: str | None, choices: list[str], query: str, prompt: str, custom: str,
        custom_profile: str):
    if not video:
        raise gr.Error("Upload a video first.")
    groups = build(choices, query, prompt)
    if custom.strip():
        try:
            rec = json.loads(custom)
            assert isinstance(rec.get("questions"), dict) and rec["questions"]
        except (json.JSONDecodeError, AssertionError, AttributeError) as e:
            raise gr.Error("The custom record must be JSON with a non-empty \"questions\" object.") from e
        rec.setdefault("state", {})
        if custom_profile in groups:  # keep both: rename the custom group
            groups[f"{custom_profile}:custom"] = rec
        else:
            groups[custom_profile] = rec
    if not groups:
        raise gr.Error("Pick at least one question (a described moment needs its text).")
    duration = probe(video)
    jobs, metas, notes = [], [], []
    for key, rec in groups.items():
        p = PROFILES[key.split(":")[0]]
        frames, audio = decode(video, p)
        rec = json.loads(json.dumps(rec))
        if p.duration_in_state and duration:
            rec["state"].setdefault("duration_seconds", round(min(duration, p.seconds), 1))
        jobs.append((rec, frames, audio, p.fps))
        metas.append((rec, p, len(frames)))
        if duration > p.seconds and not p.matched:
            notes.append(f"{key.split(':')[0]} questions read the first {p.seconds} s of the video.")
    probs = run_model(jobs)
    results = {}
    for (rec, p, n), pr in zip(metas, probs, strict=True):
        results.update(summarise(rec, pr, p.fps, n))
    notes.append("Probabilities are calibrated on the test sets in the paper; treat them as "
                 "estimates on other kinds of video.")
    return render(results, notes), results


EXAMPLE = json.dumps({"state": {"title": "my dog vs the hoover"}, "questions": {
    "dog_scared": {"type": "noul", "instructions": "Does the dog look scared?"},
    "best_moment": {"type": "moment", "instructions": "When does the dog react most strongly?"}}},
    indent=1)

INTRO = """# Video Jev
Ask a video typed questions and get a probability for **every** possible answer, from one model that reads the frames and the sound together.
A `moment` question scores every second of the video, so you see *when*, not just *whether*.

[Model](https://huggingface.co/sp1d5r/video-jev) · [Code and paper](https://github.com/sp1d5r/vjev) · Research release, non-commercial (CC BY-NC-SA 4.0)
"""

with gr.Blocks(title="Video Jev", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=5):
            video = gr.Video(label="Video (mp4, up to 150 s is read)", sources=["upload"])
            choices = gr.CheckboxGroup(CHOICES, value=["Funniest moment", "Laughing?", "AI-generated?"],
                                       label="Questions")
            query = gr.Textbox(label="Described moment", placeholder="a man opens the fridge",
                               info="Used by 'Find a described moment'.")
            prompt = gr.Textbox(label="Generation prompt (optional)",
                                info="For AI videos: lets the quality questions check text alignment.")
            with gr.Accordion("Your own questions (Jev record)", open=False):
                custom = gr.Code(EXAMPLE, language="json", label="Record")
                custom_profile = gr.Dropdown(
                    list(PROFILES), value="moment", label="Decode like",
                    info="moment: 2 fps, 60 s, audio · query: 1 fps, 150 s, audio · sound: 2 fps, "
                         "10 s, audio · ai: first 1.5 s, square · quality: 8 fps, 6 s, no audio")
                use_custom = gr.Checkbox(False, label="Include these questions")
            go = gr.Button("Ask", variant="primary")
        with gr.Column(scale=6):
            out_html = gr.HTML()
            with gr.Accordion("Raw output", open=False):
                out_json = gr.JSON()
    go.click(lambda v, c, q, p, cu, cp, use: ask(v, c, q, p, cu if use else "", cp),
             [video, choices, query, prompt, custom, custom_profile, use_custom], [out_html, out_json])
    gr.Markdown("Trained on VideoFeedback, QVHighlights, FunQA and VGGSound. "
                "Questions outside those kinds of video and question work, but are less reliable. "
                "Your video is processed for this request only.")

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
