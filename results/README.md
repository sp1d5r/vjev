# Experiment log

Newest first.
Every entry gives the setup, the numbers, what we learned and the cost.
All numbers are on our test split (1,736 clips, `sha256(Id) % 100 < 5`) unless stated.
They are not comparable to the VQualA 2025 leaderboard, whose test labels are withheld.

---

## 2026-10-07 · E002 · Qwen3-Omni + typed head, pilot (running)

**Question:** does training a typed head on an audio-native backbone beat zero-shot, and by how much?

**Setup**

- Backbone: Qwen3-Omni-30B-A3B thinker, bf16, talker disabled.
- Input: first 16 s at 2 fps (up to 32 frames) plus the clip's 16 kHz audio, interleaved (`use_audio_in_video=True`), plus title and description.
- Trainable: LoRA r=64, alpha=128 on text attention (q, k, v, o), plus a new joint schema head.
- Loss: CE (label smoothing 0.05) + 1.0 x Brier + 1.0 x ordinal EMD.
- Optimiser: AdamW, LR 1e-4 (LoRA) and 3e-4 (head), 8-clip accumulation, warmup then cosine.
- Data: 10,000 train clips, 1 epoch, eval on 500 val clips every 200 steps.
- Hardware: 1x H200 on Modal.

**Throughput:** about 2,500 clips per hour, measured on a 160-clip run.

**Results:** pending.

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
