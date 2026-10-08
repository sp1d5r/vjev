# Experiment log

Newest first.
Every entry gives the setup, the numbers, what we learned and the cost.
All numbers are on our test split (1,736 clips, `sha256(Id) % 100 < 5`) unless stated.
They are not comparable to the VQualA 2025 leaderboard, whose test labels are withheld.

---

## 2026-10-08 · E002 · Qwen3-Omni + typed head, pilot

**Question:** does training a typed head on an audio-native backbone beat zero-shot, and by how much?

**Setup**

- Backbone: Qwen3-Omni-30B-A3B thinker, bf16, talker disabled.
- Input: first 16 s at 2 fps (up to 32 frames) plus the clip's 16 kHz audio, interleaved (`use_audio_in_video=True`), plus title and description.
- Trainable: LoRA r=64, alpha=128 on text attention (q, k, v, o), plus a new joint schema head.
- Loss: CE (label smoothing 0.05) + 1.0 x Brier + 1.0 x ordinal EMD.
- Optimiser: AdamW, LR 1e-4 (LoRA) and 3e-4 (head), 8-clip accumulation, warmup then cosine.
- Data: 10,000 train clips, 1 epoch, eval on 500 val clips every 200 steps.
- Hardware: 1x H200 on Modal.

**Throughput:** about 2,860 clips per hour over the full run (3.55 h for 10,000 clips).

**Success criteria (set before results):**

| Outcome | Val final | Meaning |
|---|---|---|
| No | ≤ 0.25 | Training barely helps over zero-shot. Rethink inputs or backbone. |
| Yes, weakly | 0.25 to 0.50 | Real signal. Tune and scale. |
| Yes | ≥ 0.50 | In the range of prior trained models. |
| Strong | ≥ Li24 re-run on our split | Competitive with published work. |

**Results** (val, 500 clips, 1 seed)

| Step | SROCC | PLCC | Final | Brier | ECE | Level acc | Spread (levels) |
|---|---|---|---|---|---|---|---|
| 200 | 0.490 | 0.500 | 0.495 | 0.894 | 0.002 | 0.116 | 0.15 |
| 400 | 0.548 | 0.515 | 0.532 | 0.874 | 0.038 | 0.188 | 0.95 |
| 600 | 0.523 | 0.528 | 0.525 | 0.884 | 0.022 | 0.132 | 0.48 |
| 800 | 0.634 | 0.630 | 0.632 | 0.856 | 0.029 | 0.182 | 1.44 |
| 1000 | 0.653 | 0.648 | 0.650 | 0.843 | 0.029 | 0.222 | 1.69 |
| 1200 | 0.655 | 0.652 | 0.653 | 0.839 | 0.031 | 0.232 | 1.73 |
| 1250 (end) | **0.655** | **0.652** | **0.654** | **0.839** | 0.035 | **0.236** | 1.73 |

Forward pass: about 290 ms per clip (p50, batch size 1, H200).

**Verdict against the success criteria: Yes (val final 0.654, bar is 0.50).**

**What we learned**

- Training works. Val final went from 0.13 to 0.17 (zero-shot Clef-flash, test) to 0.65 after 10,000 clips.
- The model ranks well before it is confident. At step 200 it already ranks at 0.50, but its spread is only 0.15 levels, so its probabilities stay near uniform.
- Brier fell from 0.894 to 0.839, so the probabilities beat a blind guess (0.900), but only modestly. Retention is noisy, and a 10-level answer is hard to get exactly right.
- Calibration stays good (ECE 0.03 to 0.04) while the model becomes more decisive.
- The curve flattens after step 1000, at the end of the cosine schedule. More data, not more steps on this data, is the next lever.

**Caveats**

- This is val, not test. Val was used to choose the checkpoint, so the number is slightly optimistic.
- With n=500, the standard error of SROCC is about 0.03.
- One seed only.
- The 2025 challenge models scored 0.66 to 0.71 on a different test set. We cannot claim parity until the baselines are retrained on our split.

**Cost:** about $17 (3.55 h on one H200).

**W&B run:** `pilot-omni-av-10k`. Checkpoint: `/data/ckpt/pilot-omni-av-10k/best.pt` on the `vjev-snapugc` volume.

---

## 2026-10-07 · E001 · Zero-shot Clef-flash

**Question:** how well does Clef-flash rank engagement with no training, and does video help?

**Setup**

- Model: `Cloudflare/clef-flash` (Qwen3.5-9B backbone + joint schema head), no training.
- Question: ECR as a 10-level `score`, levels are ECR deciles over all labels.
- Video: 1 fps, max 16 frames, longest side 448 px.
- Text: title and description in `state`.
- Hardware: 1x H100 on Modal, batch size 1.

**Results**

| Input | SROCC | PLCC | Final | Brier | ECE | Level acc | Forward p50 | Decode p50 |
|---|---|---|---|---|---|---|---|---|
| video + text | 0.128 | 0.127 | 0.127 | 0.912 | 0.064 | 0.112 | 188 ms | 618 ms |
| text only | 0.175 | 0.168 | 0.172 | 1.037 | 0.278 | 0.099 | 151 ms | 2 ms |
| video only | 0.142 | 0.139 | 0.140 | 0.910 | 0.051 | 0.117 | 202 ms | 658 ms |

**What we learned**

- Without training, the model barely ranks engagement.
- Video does not help yet. The video runs stay close to uniform (Brier 0.910 against 0.900 for a uniform guess), so their low ECE means hedging, not skill.
- Text only is the most confident and the worst calibrated.
- The standard error of SROCC at n=1,736 is about 0.024, so the gaps between rows are small.
- Decoding is the bottleneck at about 3.5x the forward pass time.

**Cost:** about $7 of H100 time for all three runs.

**Raw data:** [`zero_shot_clef_flash_test.json`](zero_shot_clef_flash_test.json).
W&B runs: `clef-flash-zs-video+text-test1736`, `clef-flash-zs-text-test1736`, `clef-flash-zs-video-test1736`.

---

## 2026-10-07 · D001 · Dataset pull

**Question:** how much of SnapUGC can we still get?

**Results**

- 106,192 labelled rows, 34,081 videos downloadable (32%).
- The CSV is sorted by ECR, and liveness rises with ECR: about 15% of the first rows are alive, against about 33% later.
- Mean ECR is 0.498 over all labels and 0.472 over the train clips we have.

**What we learned**

- Survivorship bias is real and must be reported in the paper.
- Level edges come from all labels, so the task does not depend on which clips survived.
- The Google Drive "train set" is only the label CSV; there is no video bundle.

**Cost:** a few dollars of CPU time on Modal.
