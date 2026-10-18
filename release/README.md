---
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3-Omni-30B-A3B-Instruct
library_name: peft
pipeline_tag: video-classification
tags:
  - video
  - audio
  - video-understanding
  - moment-retrieval
  - video-quality
  - calibration
  - jev
datasets:
  - TIGER-Lab/VideoFeedback
  - Loie/VGGSound
---

# Video Jev

Ask a video a typed question and get a calibrated probability for every possible answer.

Video Jev is a LoRA adapter and a typed answer head for [Qwen3-Omni-30B-A3B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).
It reads raw video frames and the sound together.
You write questions as a Jev record, and one forward pass answers all of them.

| Type | You get | Example |
|---|---|---|
| `noul` | P(yes), P(no) | "Can you hear someone laughing?" "Is this AI-generated?" |
| `choice` | P per named option | "Which sound is this: laughing, cheering, speech...?" |
| `score` | P per ordered level | "Visual quality, bad / average / good / perfect?" |
| `moment` | P per second of video | "When is the funniest moment?" "When does the man open the fridge?" |

For `moment` questions, each second of video is an answer option, scored on that second's own video and audio tokens.
There is no detector and no generated timestamp text.

Built by [Retina](https://retina.dev/coeus).
Code and working paper: [github.com/sp1d5r/vjev](https://github.com/sp1d5r/vjev).

**Status:** research release, one training seed, non-commercial use only.

## Quick start

Needs one GPU with 80 GB or more (bf16) and ffmpeg.

```bash
pip install torch==2.11.0 transformers==5.10.2 peft accelerate safetensors pillow numpy huggingface_hub
git clone https://huggingface.co/sp1d5r/video-jev && cd video-jev
python run.py --video clip.mp4 --record questions.json
```

`questions.json`:

```json
{"state": {"title": "my dog vs the hoover"},
 "questions": {
  "funny_moment": {"type": "moment", "instructions": "When is the funniest moment in this video?"},
  "laugh": {"type": "noul", "instructions": "Can you hear someone laughing in this clip?",
            "criteria": {"true": "yes, someone laughs", "false": "no laughter"}},
  "quality": {"type": "score", "instructions": "Rate how clean and sharp this video looks.",
              "criteria": ["bad", "average", "good", "perfect"]}}}
```

Output on a 7 s FunQA test clip whose funny moment is 0 to 6.7 s (with one more question, "Is this clip AI-generated?"):

```json
{"funny_moment": [{"from_s": 6.0, "to_s": 7.0, "p": 0.181}, {"from_s": 5.0, "to_s": 6.0, "p": 0.18},
                  {"from_s": 4.0, "to_s": 5.0, "p": 0.171}, ...],
 "laugh": {"true": 0.198, "false": 0.802},
 "ai": {"true": 0.022, "false": 0.978},
 "quality": {"bad": 0.012, "average": 0.043, "good": 0.234, "perfect": 0.71}}
```

By default `run.py` reads the first 60 s at 2 frames per second with audio (`--seconds`, `--fps`, `--no-audio`).

## Results

Official test splits, each scored once.
"Untrained" is the same Qwen3-Omni asked the question as text, reading the probability of each answer.

| Task | Metric | Untrained | **Video Jev** | Reference |
|---|---|---|---|---|
| VideoFeedback, 5 quality questions | mean Spearman | 0.340 | 0.743 | VideoScore 0.771 |
| VideoFeedback, real or AI | AUC | 0.905 | 0.996 | - |
| GenVideo, real or AI (unseen generators, never trained on) | AUC | - | 0.942 | - |
| QVHighlights moments (val split, 1,550 queries) | R1@0.5 / R1@0.7 | 0.528 / 0.354 | 0.679 / 0.463 | Moment-DETR 53.94 / 34.84 |
| QVHighlights | HIT@1 / mAP avg | 0.181 / 0.043 | 0.358 / 0.055 | Moment-DETR 55.55 / 32.20 |
| FunQA funny and magic moments | HIT@1 | 0.504 | 0.776 | - |
| VGGSound, 8 people sounds | accuracy | 0.503 | 0.590 | chance 0.125 |

**Questions it was never trained on**

| Test | Video Jev | Untrained |
|---|---|---|
| Every question in two wordings never seen in training | no drop (0.728 against 0.730) | - |
| "Can you hear someone laughing?" on VGGSound test | AUC 0.970 | AUC 0.873 |

**Calibration.** Expected calibration error is 0.02 to 0.05 on quality and real-or-AI questions (untrained: 0.29 to 0.56), so the probabilities can be used as thresholds.

**Long video.** Score 60 s windows with 30 s overlap and combine them.
Finding one moment in a 10-minute video: right second 13% of the time (chance 0.9%), 22.5 s on one H200.

**Speed.** 130 to 290 ms per clip of up to 60 s on one H200, after the first call.

A VideoFeedback-only adapter is in `videofeedback-only/`: 0.775 mean Spearman on the same test (VideoScore 0.771).

## What it does not do well

- Text-query moments: R1 is strong, but span ranking (mAP) is weak.
- VGGSound official test is much harder than its val for every model we tried.
- Real-or-AI was trained on 2023 to 2024 generators. Do not rely on it for modern ones.
- Long videos: accuracy falls with length.
- One seed; no confidence intervals on these numbers yet.

## Training

- Base: Qwen3-Omni-30B-A3B thinker, frozen, bf16. LoRA rank 64 on the language model's attention.
- Head: Clef's joint schema head ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), Apache 2.0), trained from scratch.
- Loss: soft-target cross-entropy + Brier score + ordinal distance (EMD) for score and moment questions.
- Data: 26,564 clips, 1 epoch, 10 h on one H200. Every question asked in one of three wordings at random.

| Dataset | Used for | Licence |
|---|---|---|
| [VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback) (Mantis/VideoScore) | 5 quality scores, real or AI | Apache 2.0 |
| [QVHighlights](https://github.com/jayleicn/moment_detr) | text-query moments | CC BY-NC-SA 4.0 |
| [FunQA](https://github.com/Jingkang50/FunQA) | funny and magic moments | CC BY-NC-SA 4.0 |
| [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) | which sound | CC BY 4.0 |

No videos are distributed. Video copyright stays with its owners.

## Licence

CC BY-NC-SA 4.0, because two training sets carry share-alike, non-commercial terms.
The base model is Apache 2.0; its licence and notices apply to it.
`vjev/clef_schema.py` is vendored from Cloudflare's Clef under Apache 2.0.

## Citation

Please cite the datasets above, Qwen3-Omni, and Clef. A paper citation will be added when it is out.

## About

Video Jev is made by [Retina](https://retina.dev/coeus). Learn more at [retina.dev/coeus](https://retina.dev/coeus).
