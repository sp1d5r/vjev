# Experiment log

Newest first.
Every entry gives the setup, the numbers, what we learned and the cost.
All numbers are on our test split (1,736 clips, `sha256(Id) % 100 < 5`) unless stated.
They are not comparable to the VQualA 2025 leaderboard, whose test labels are withheld.

---

## 2026-10-08 · E004 · Video only, no audio (ablation)

**Question:** does audio help?

**Setup:** same as E002 (same 10,000 train clips, same 500 val clips, same hyperparameters) but with no audio input.
One change from E002: the typed head now runs in fp32 (see the NaN incident below). So this is not a pure audio ablation.

**Results** (val, 500 clips, 1 seed)

| Step | Final | Brier | ECE | Spread (levels) |
|---|---|---|---|---|
| 200 | 0.208 | 0.900 | 0.021 | 0.00 |
| 400 | 0.312 | 0.898 | 0.034 | 0.02 |
| 600 | 0.395 | 0.898 | 0.030 | 0.07 |
| 800 | 0.531 | 0.890 | 0.063 | 0.39 |
| 1000 | 0.617 | 0.852 | 0.032 | 1.41 |
| 1200 (best) | **0.628** | **0.847** | 0.029 | 1.72 |
| 1250 (end) | 0.627 | 0.847 | 0.044 | 1.72 |

| Run | Audio | Head | Final | Brier | ECE |
|---|---|---|---|---|---|
| E002 pilot | yes | bf16 | 0.654 | 0.839 | 0.035 |
| E004 ablation | no | fp32 | 0.628 | 0.847 | 0.029 |

**What we learned**

- Without audio, the model ends 0.026 lower. That is inside the noise (standard error about 0.03 at n=500), and the head precision also changed. So audio helps at most a little. This is not yet a clean result.
- With the fp32 head, the model learns almost nothing while the learning rate is near its peak (1e-4). Its predictions stay flat until about step 700, then it catches up quickly as the cosine schedule decays. The bf16 pilot learned earlier. A lower peak learning rate is the next thing to try.
- Without audio, training is about 1.8x faster (5,200 clips per hour against 2,860).

**Cost:** about $9 (1.95 h on one H200).

**W&B run:** `ablation-omni-v-10k-v2`.

---

## 2026-10-08 · E003 · Cheap baselines on our split

**Question:** how far do cheap, standard features get on the same split, and is the 30B model worth it?

**Setup**

- Text: title + description embedded with `paraphrase-multilingual-mpnet-base-v2`.
- Visual: CLIP ViT-L/14 embeddings of 8 frames spread over the first 16 s. Features are the first frame plus the frame mean.
- Models: XGBoost on text. MLP (one hidden layer of 512, early stopping on a train holdout) on CLIP, and on CLIP + text.
- Each model predicts the 10 ECR levels as classes, so it gets Brier and ECE like the typed model. A regression twin is fit for ranking only.
- Trained on all 30,631 train clips. Hyperparameters fixed up front, nothing tuned on val, no test.
- Scored on the same 500 val clips as the Omni runs, and on all 1,714 val clips.

**Results** (500 val clips, same as E002)

| Model | Train clips | Final | Brier | ECE | Level acc | Regression twin final |
|---|---|---|---|---|---|---|
| Text + XGBoost | 30,631 | 0.326 | 0.899 | 0.055 | 0.192 | 0.341 |
| CLIP + MLP | 30,631 | **0.662** | 0.876 | 0.144 | **0.246** | 0.621 |
| CLIP + text + MLP | 30,631 | 0.612 | 1.136 | 0.465 | 0.232 | 0.657 |
| Qwen3-Omni + typed head (E002) | 10,000 | 0.654 | **0.839** | **0.035** | 0.236 | - |

On all 1,714 val clips: text 0.360, CLIP 0.652, CLIP + text 0.625.

**With temperature scaling** (classifier fit on 90% of train, one temperature fit on the other 10%, val untouched)

| Model | Temperature | Final | Brier | ECE | Level acc |
|---|---|---|---|---|---|
| Text + XGBoost | 1.32 | 0.329 | 0.891 | 0.044 | 0.184 |
| CLIP + MLP | 2.04 | **0.673** | 0.845 | 0.046 | 0.224 |
| CLIP + text + MLP | 5.88 | 0.661 | 0.852 | 0.037 | 0.230 |
| Qwen3-Omni + typed head (E002, no scaling) | - | 0.654 | **0.839** | **0.035** | **0.236** |

On all 1,714 val clips: text 0.364, CLIP 0.656, CLIP + text 0.656.
Raw data: [`simple_baselines_val_ts.json`](simple_baselines_val_ts.json). W&B runs end in `-ts`.

**What we learned**

- CLIP + MLP is a strong baseline. It ranks as well as the Omni pilot, which saw a third of the data.
- The visual signal dominates. Text alone reaches only 0.33 to 0.36.
- The typed model's edge is honest probabilities: Brier 0.839 against 0.876, ECE 0.035 against 0.144. The MLPs are overconfident, and adding text makes it worse (ECE 0.47).
- So the real bar for the paper is "beat CLIP + MLP on ranking", not 0.50. The full 30k run (E004) is the fair test.

**Caveats**

- The first fit ran out of memory and Modal retried it, so each W&B run appears twice. The classifier numbers matched exactly. The CLIP + MLP regression twin gave 0.498 once and 0.621 once, so it is unstable and we report the classifier. The memory bug is fixed.

**Cost:** about $5 (feature extraction on L40S, fitting on CPU, including the failed attempt).

**Raw data:** [`simple_baselines_val.json`](simple_baselines_val.json). Code: `vjev/simple_baselines.py`. W&B runs: `baseline-text-xgb`, `baseline-clip-mlp`, `baseline-clip+text-mlp`.

---

## 2026-10-08 · Incident · Two runs went NaN

The first video-only ablation and the first full run both hit a NaN loss near step 720, about 2.5 h in.
A NaN step corrupts the weights for good, so both were stopped. About $23 was lost.
The video-only run had already degraded before that (val final 0.47 at step 400, then 0.19 at step 600).
Fix: the typed head now runs in fp32 (the 30B thinker stays in bf16). Steps with a non-finite loss or gradient norm are skipped and counted in W&B (`train/skipped_*`, `train/grad_norm`).
The pilot (E002) used the old bf16 head and did not hit this. Its numbers stand.
Both runs restarted as `ablation-omni-v-10k-v2` (E004) and `full-omni-av-30k-v2` (running).
With the fp32 head, both v2 runs learn slowly while the learning rate is near its peak (see E004).

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
