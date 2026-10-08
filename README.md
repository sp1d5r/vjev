# vjev

Video Jev: a typed, calibrated decision model for short videos.
It answers Jev / SystemOne questions (`choice`, `score`, `noul`) about a video in one forward pass.
It returns a probability for every allowed answer, so code can act on confidence.
The first benchmark is SnapUGC engagement prediction (ECR).

## Status

| Step | State |
|---|---|
| Data pipeline (34,081 SnapUGC clips on Modal) | Done |
| Zero-shot Clef-flash baseline | Done |
| Preprocessing (16 s at 2 fps + 16 kHz audio per clip) | Done |
| Qwen3-Omni + typed head trainer (video + audio) | Done |
| Pilot: 10k clips, video + audio | Done: val final 0.654 |
| Cheap baselines on our split (text XGBoost, CLIP MLP) | Done: CLIP MLP + temperature 0.673 |
| Video-only ablation (same backbone, no audio) | Done: 0.628, audio gap inside noise |
| Omni + CLIP ensemble | Done: 0.695 |
| VideoLLaMA2-AV retrained on our split | Planned |
| Full run (30.6k clips, E006) | Running: weak so far (0.24 at step 1,200), LR too high at peak |
| Learning-rate sweep + vision LoRA test (2k clips, E007) | Running |

## Results

Higher SROCC, PLCC and Final is better.
Lower Brier and ECE is better.
A uniform guess over 10 levels has Brier 0.900.

### Trained (val split, the same 500 clips for every row)

| Model | Training clips | Input | SROCC | PLCC | Final | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Qwen3-Omni + typed head | 10,000 | video + audio + text | 0.655 | 0.652 | 0.654 | **0.839** | **0.035** |
| **Omni + CLIP-MLP-temperature ensemble** | 10,000 + ~27,600 | video + audio + text | 0.696 | 0.694 | **0.695** | **0.829** | **0.030** |
| Qwen3-Omni + typed head, no audio | 10,000 | video + text | 0.627 | 0.626 | 0.628 | 0.847 | 0.029 |
| CLIP + MLP + temperature | ~27,600 | 8 frames | 0.673 | 0.673 | **0.673** | 0.845 | 0.046 |
| CLIP + MLP | 30,631 | 8 frames | 0.666 | 0.658 | 0.662 | 0.876 | 0.144 |
| CLIP + text + MLP + temperature | ~27,600 | 8 frames + text | 0.663 | 0.659 | 0.661 | 0.852 | 0.037 |
| CLIP + text + MLP | 30,631 | 8 frames + text | 0.613 | 0.612 | 0.612 | 1.136 | 0.465 |
| Text + XGBoost | 30,631 | text | 0.336 | 0.316 | 0.326 | 0.899 | 0.055 |

These are val numbers from one seed, and val chose the Omni checkpoint.
The test split stays untouched until the final model.

### Zero-shot (test split, 1,736 clips)

| Model | Trained | Input | SROCC | PLCC | Final | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Clef-flash | No | video + text | 0.128 | 0.127 | 0.127 | 0.912 | 0.064 |
| Clef-flash | No | text only | 0.175 | 0.168 | 0.172 | 1.037 | 0.278 |
| Clef-flash | No | video only | 0.142 | 0.139 | 0.140 | 0.910 | 0.051 |

For context, trained models in the VQualA 2025 challenge reach a final score of 0.66 to 0.71 on the official test set.
Those numbers are not directly comparable, because the official test labels are withheld.
We retrain the baselines on our exact split for a fair comparison. See [`EVALUATION.md`](EVALUATION.md).

Full log of every experiment: [`results/README.md`](results/README.md).
Live charts in plain language: [W&B report](https://wandb.ai/sp1d5r/vjev/reports/vjev:-can-a-typed-model-predict-video-retention?--VmlldzoxODA3MzM0NQ==) (private).
Runs page in plain language: ["Plain language" view](https://wandb.ai/sp1d5r/vjev?nw=hon9b3fyzh6) (private).
Rebuild them with `uv run python scripts/wandb_report.py` and `uv run python scripts/wandb_workspace.py`.

## Findings so far

1. A typed head on Qwen3-Omni, trained on 10,000 clips with audio, reaches val final 0.654. That passes our pre-set bar of 0.50.
2. A cheap CLIP + MLP baseline with temperature scaling ranks slightly better (0.673) and is nearly as well calibrated (Brier 0.845 against 0.839). It trained on about 3x the data. It is the real bar.
3. Without audio, the Omni model scores 0.628. The gap to the pilot (0.026) is inside the noise, so audio helps at most a little.
4. An equal-weight ensemble of Omni and CLIP + MLP reaches 0.695, beating both. Omni carries signal CLIP misses (their predictions correlate at 0.82).
5. Visual features carry most of the signal. Text alone reaches only 0.33.
6. Zero-shot Clef-flash barely ranks engagement (final 0.13 to 0.17).
7. Without training, video does not help. The zero-shot video runs predict close to uniform.
8. Zero-shot text only is the most confident but badly calibrated (ECE 0.28).
9. Only 32% of SnapUGC videos are still downloadable, and low-engagement videos are deleted more often. Mean ECR is 0.50 over all labels and 0.47 over the clips we have.
10. Decoding video on the fly was 3.5x slower than the forward pass, so we decode every clip once up front.

## How it works

```mermaid
flowchart LR
  V[Video clip] --> P[Preprocess once<br/>first 16 s, 2 fps<br/>+ 16 kHz audio]
  P --> O[Qwen3-Omni-30B-A3B thinker<br/>video + audio interleaved<br/>+ LoRA on text attention]
  T[Title + description] --> O
  Q[Typed questions<br/>score / choice / noul] --> O
  O --> H[Joint schema head<br/>one logit per option]
  H --> L[Loss: CE + Brier + ordinal EMD]
  H --> E[Metrics<br/>SROCC, PLCC, Brier, ECE]
  E --> W[(W&B)]
```

The backbone is [Qwen3-Omni-30B-A3B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) (Apache-2.0, 3B active parameters).
It reads video frames and the clip's audio on one interleaved timeline.
The head is Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef-flash) joint schema head, trained from scratch on this backbone.
Training follows the Clef recipe: LoRA plus the head, with label-smoothed cross-entropy, a Brier term, and an ordinal term for `score` questions.
ECR is asked as a 10-level `score` question.
The prediction is the probability-weighted score.

## Setup

```sh
uv sync
uv run pytest
```

Modal needs two secrets: `wandb` (`WANDB_API_KEY`) and `huggingface` (`HF_TOKEN`).

## Data

The official SnapUGC val and test labels are withheld.
All splits come from the labelled `train_data.csv` (106,192 rows).
Videos come from the Snap CDN links in that CSV, and we keep the 34,081 links that still work.
The split is `sha256(Id) % 100`: under 5 is test, under 10 is val, the rest is train.

| Split | Clips | Mean ECR |
|---|---|---|
| Train | 30,631 | 0.472 |
| Val | 1,714 | 0.482 |
| Test | 1,736 | 0.463 |

The 10 level edges are ECR deciles over all 106,192 labels, not only the surviving clips.
They are stored in `index/meta.json` on the volume.

```sh
uv run modal run vjev/data.py::upload_labels --csv-path ~/Downloads/train_data.csv
uv run modal run vjev/data.py::pull_videos
uv run modal run vjev/data.py::build_split
```

## Experiments

```sh
uv run modal run vjev/baseline.py --split test --limit 1736             # zero-shot Clef-flash, video + text
uv run modal run vjev/baseline.py --split test --limit 1736 --no-video  # text only
uv run modal run vjev/preprocess.py::preprocess_all                     # decode clips once
uv run modal run --detach vjev/train.py --train-limit 10000 --val-limit 500 --eval-every 200
uv run modal run --detach vjev/train.py --no-audio ...                  # video-only ablation
uv run modal run --detach vjev/train.py --vision-lora --lr-lora 3e-5 ... # also adapt the vision tower
uv run modal run vjev/simple_baselines.py::embed_all                    # text + CLIP features, once
uv run modal run vjev/simple_baselines.py::fit_all [--calibrate]        # cheap baselines (+ temperature)
uv run modal run vjev/predict.py::predict_ckpt --run pilot-omni-av-10k  # per-clip val predictions
uv run python scripts/ensemble.py pilot-omni-av-10k-val500 baseline-clip-mlp-ts-val
```

Rule for spend: test settings on 2,000 clips (about $4) before any run over about $20.

Training runs on one H200 at about 2,860 clips per hour (about $1.70 per 1,000 clips).

## Metrics

- SROCC and PLCC between the expected score and true ECR, as in the VQualA 2025 challenge. Final is their mean.
- Brier and ECE of the 10-level distribution, for calibration.
- Forward latency per clip and clips per hour, for cost.

## License

Code: Apache-2.0.
SnapUGC has no stated license.
Check the dataset terms before redistributing videos.
