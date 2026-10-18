"""The same Qwen3-Omni, untrained, asked each question as plain text. This is the "general
video model" baseline for every task: what our training adds over prompting alone.

    modal run vjev/zeroshot.py::zero_shot --task vf --split val --limit 300

Scores, yes/no and choice questions read the next-token probabilities of the allowed answers
("1" to "4", "yes"/"no", "A" to "H"), so they get the same metrics as the trained model.
Moment questions ask for "start-end" in seconds and parse the reply.
"""

from __future__ import annotations

import json
import pathlib
import re

import modal
import numpy as np

from vjev.common import DATA, HF, app, data_vol, hf_vol, train_image

SYSTEM = "You are a careful video analyst. Answer exactly in the format asked."


def _prompt(question: dict, kind: str) -> tuple[str, list[str]]:
    """Question text and the answer strings to score (empty for moments)."""
    ins = question["instructions"]
    if kind == "score":
        crit = "\n".join(f"{i + 1}: {c}" for i, c in enumerate(question["criteria"]))
        return f"{ins}\n{crit}\nAnswer with one number only.", [str(i + 1) for i in range(len(question["criteria"]))]
    if kind == "noul":
        return f"{ins}\nAnswer yes or no only.", ["yes", "no"]
    if kind == "choice":
        keys = sorted(question["criteria"])
        letters = [chr(65 + i) for i in range(len(keys))]
        opts = "\n".join(f"{ltr}: {question['criteria'][k]}" for ltr, k in zip(letters, keys, strict=True))
        return f"{ins}\n{opts}\nAnswer with one letter only.", letters
    text = (f"{ins}\nGive the start and end time in seconds from the start of the video, "
            f"as two numbers like 12.0-18.5. Answer with the times only.")
    return text, []


def _moment_probs(reply: str, n_bins: int, bin_s: float) -> np.ndarray:
    """A parsed span becomes a flat distribution over its bins; no answer is uniform."""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", reply)[:2]]
    p = np.full(n_bins, 1e-6)
    if len(nums) == 2:
        a, b = sorted(nums)
        lo = min(int(a // bin_s), n_bins - 1)
        hi = max(lo + 1, min(int(np.ceil(b / bin_s)), n_bins))
        p[lo:hi] = 1.0
    else:
        p[:] = 1.0
    return p / p.sum()


@app.function(image=train_image, gpu="H200", volumes={DATA: data_vol, HF: hf_vol},
              secrets=[modal.Secret.from_name("huggingface")],
              timeout=4 * 3600, cpu=12, memory=65536)
def zs_run(task: str, split: str, limit: int, audio: bool = True, wording: str = "canonical",
           held_out: bool = False) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    from vjev import model as vm
    from vjev.preprocess import load, pre_path
    from vjev.tasks import make_item, prepare_frames, question_report
    from vjev.train import _records

    path = snapshot_download(vm.OMNI_ID)
    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa")
    full.disable_talker()
    thinker = full.thinker.eval()
    proc = Qwen3OmniMoeProcessor.from_pretrained(path)
    tok = proc.tokenizer
    root = pathlib.Path(DATA)
    edges = json.loads((root / "index" / "meta.json").read_text())["level_edges"]

    groups: dict[str, dict] = {}
    rows = []
    for rec in _records(split, limit, 0, task, "", held_out):
        frames, wav, duration = load(pre_path(root, rec["id"]))
        frames = prepare_frames(task, frames)
        if len(frames) % 2:
            frames = frames[:-1]
        item = make_item(task, rec, len(frames), duration, edges, audio, wording)
        row = {"id": rec["id"], "scores": {}}
        for q, question in item.record["questions"].items():
            kind = item.kinds[q]
            if kind == "moment":
                bins = vm.moment_bins(len(frames), item.fps)
            text, answers = _prompt(question, kind)
            state = json.dumps(item.record["state"]) if item.record["state"] else ""
            content = (f"<|vision_start|><|video_pad|><|vision_end|>"
                       f"{'Context: ' + state + chr(10) if state else ''}{text}")
            chat = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{content}"
                    f"<|im_end|>\n<|im_start|>assistant\n")
            kw = {"fps": item.fps, "do_sample_frames": False, "return_tensors": "pt"}
            if item.audio:
                n = int(len(frames) / item.fps * vm.SR)
                a = np.pad(wav[:n], (0, max(0, n - len(wav))))
                inputs = proc(text=[chat], videos=[frames], audio=[a], use_audio_in_video=True,
                              sampling_rate=vm.SR, **kw)
            else:
                inputs = proc(text=[chat], videos=[frames], **kw)
            inputs = {k: (v.to("cuda").to(torch.bfloat16) if v.is_floating_point() else v.to("cuda"))
                      for k, v in inputs.items() if hasattr(v, "to")}
            with torch.no_grad():
                if kind == "moment":
                    out = thinker.generate(**inputs, use_audio_in_video=item.audio,
                                           max_new_tokens=24, do_sample=False)
                    reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    probs = _moment_probs(reply, len(bins), bins[0][1] - bins[0][0])
                    row["scores"][q] = {"reply": reply}
                else:
                    logits = thinker(**inputs, use_audio_in_video=item.audio).logits[0, -1].float()
                    # Each answer may start in lower or upper case; pool both spellings.
                    scores = []
                    for ans in answers:
                        ids = {tok.encode(v, add_special_tokens=False)[0]
                               for v in (ans, ans.capitalize(), " " + ans)}
                        scores.append(torch.logsumexp(logits[list(ids)], 0))
                    probs = torch.stack(scores).softmax(-1).cpu().numpy()
            g = groups.setdefault(q, {"kind": kind, "probs": [], "target": [], "extra": []})
            g["probs"].append(probs)
            g["target"].append(item.targets[q])
            g["extra"].append(item.extra.get(q, {}))
            row["scores"][q] = row["scores"].get(q, {}) | {"probs": probs.tolist(),
                                                           "target": item.targets[q]}
        rows.append(row)
    reports = {q: question_report(g["kind"], g["probs"], g["target"], g["extra"])
               for q, g in groups.items()}
    tag = ("" if audio else "-noaudio") + ("" if wording == "canonical" else f"-{wording}")
    tag += "-noC1" if held_out else ""
    out = root / "preds" / f"zeroshot-{task}-{split}{limit or ''}{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data_vol.commit()
    final = float(np.mean([r["final"] for r in reports.values()]))
    return {"task": task, "split": split, "n": len(rows), "final": final, **reports,
            "examples": [r["scores"] for r in rows[:3]]}


@app.local_entrypoint()
def zero_shot(task: str = "vf", split: str = "val", limit: int = 300, audio: bool = True,
              wording: str = "canonical", held_out: bool = False):
    print(json.dumps(zs_run.remote(task, split, limit, audio, wording, held_out), indent=1,
                     default=float))
