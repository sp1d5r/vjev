# SnapUGC engagement: findings (parked)

Status as of 10 October 2026: **parked**.
The core result is done.
The remaining items need about $90 to $120 of GPU time, so we come back to them after the generalisation work.
This file stands alone: it holds everything needed to pick the work up again or write it up as a paper section.
Full run-by-run detail is in [`results/README.md`](../results/README.md) (E001 to E010).

## 1. Claim

A single Qwen3-Omni model with LoRA and a typed answer head predicts short-video engagement as well as the best 2025 challenge ensemble, and its probabilities are calibrated.

- Test final score **0.709** (SROCC 0.708, PLCC 0.709) on 1,736 held-out clips, scored once.
- **+0.065** over CLIP + MLP trained on the same clips (paired bootstrap 95% CI 0.041 to 0.088).
- Brier 0.818, ECE 0.028, with no calibration step after training.
- The VQualA 2025 winner scored 0.710 on the official test set, with an ensemble of four models.
  Our split is different, so this is "level with", not "beats".

## 2. Task and data

- **Dataset:** SnapUGC (VQualA 2025 EVQA-SnapUGC challenge).
  Short user-made Snapchat videos, each labelled with its engagement continuation rate (ECR).
  ECR is the share of viewers who keep watching past the first 5 seconds.
- **What we could get:** 106,192 labelled rows, of which 34,081 videos were still downloadable (32%).
- **Survivorship bias:** low-engagement videos were deleted more often.
  Mean ECR is 0.472 on our train clips against 0.498 over all labels.
- **Split** (by `sha256(Id) % 100`):

  | Split | Clips | Rule |
  |---|---|---|
  | Test | 1,736 | `< 5`, touched once per final model |
  | Val | 1,714 | next 5%, used for all choices |
  | Train | 30,631 | the rest |

- **Question:** ECR as a 10-level `score` question.
  Level edges are ECR deciles over all labels, so they do not depend on which clips survived.
- **Metrics:** SROCC, PLCC and their mean ("final", the challenge score) on the expected level.
  Brier and ECE on the 10-level distribution.
  A uniform guess has Brier 0.900.

## 3. Model

- **Backbone:** Qwen3-Omni-30B-A3B thinker (MoE, about 3B active parameters per token), bf16, talker off.
- **Input:** the first 16 s at 2 fps (up to 32 frames, longest side 448 px), plus 16 kHz mono audio.
  Frames and audio are interleaved by timestamp (`use_audio_in_video=True`).
  Title and description go into the Jev `state`.
- **Head:** Clef's joint schema head (vendored from `Cloudflare/clef-flash`, Apache-2.0), in fp32, trained from scratch.
  It gives one logit per allowed answer.
- **Trainable:** LoRA r=64, alpha=128 on the text attention (q, k, v, o), plus the head.
- **Loss:** cross-entropy (label smoothing 0.05) + 1.0 × Brier + 1.0 × ordinal EMD.

## 4. Final recipe (E009)

| Setting | Value |
|---|---|
| Train clips | 30,631, 1 epoch (3,829 steps) |
| Batch | 8-clip gradient accumulation |
| LoRA LR | 3e-5 |
| Head LR | 1e-4 |
| Schedule | warmup, then cosine to zero |
| Vision LoRA | off |
| Checkpoint | best val final, step 3,200 (val 0.718) |
| Hardware | 1 × H200, 7.5 h, about $43 |

Command:

```
uv run modal run --detach vjev/train.py --train-limit 0 --val-limit 500 --eval-every 200 \
  --lr-lora 3e-5 --lr-head 1e-4 --run-name full-lr3e-5-30k
uv run modal run vjev/predict.py::predict_ckpt --run full-lr3e-5-30k --split test --limit 0 --final
```

Checkpoint: `/data/ckpt/full-lr3e-5-30k/best.pt` on the `vjev-snapugc` Modal volume.
Test predictions: `/data/preds/full-lr3e-5-30k-test.jsonl`.

## 5. Results

### Test split (1,736 clips, scored once)

| Model | Train clips | Input | Final | Brier | ECE |
|---|---|---|---|---|---|
| **Qwen3-Omni + typed head** | 30,631 | video + audio + text | **0.709** | **0.818** | **0.028** |
| Omni + CLIP-MLP ensemble (recipe fixed before test) | 30,631 | video + audio + text | 0.709 | 0.816 | 0.033 |
| CLIP + MLP + temperature | 30,631 | 8 frames | 0.643 | 0.843 | 0.034 |
| CLIP + text + MLP + temperature | 30,631 | 8 frames + text | 0.639 | 0.859 | 0.029 |
| Text + XGBoost + temperature | 30,631 | text | 0.347 | 0.884 | 0.029 |
| Clef-flash, no training | 0 | video + text | 0.127 | 0.912 | 0.064 |

Paired bootstrap, 2,000 resamples:

- Omni minus CLIP + MLP: +0.065 [0.041, 0.088].
- Ensemble minus Omni: +0.001 [-0.010, +0.012].

Raw data: [`results/test_final.json`](../results/test_final.json), [`results/test_baselines.json`](../results/test_baselines.json).

### VQualA 2025 leaderboard (official test set, not our split)

| Rank | Team | Final |
|---|---|---|
| 1 | ECNU-SJTU VQA (3 × VideoLLaMA2 + Qwen2.5-VL ensemble, with audio) | 0.710 |
| 2 | IMCL-DAMO | 0.698 |
| 3 | HKUST-Cardiff-MI-BAAI | 0.680 |
| 4 | MCCE / EasyVQA | 0.667 |
| - | Challenge baseline | about 0.66 |

Source: [arXiv 2509.02969](https://arxiv.org/abs/2509.02969), [LMM-EVQA](https://github.com/sunwei925/LMM-EVQA).

## 6. What we learned

1. **Training is the whole game.**
   Untrained Clef-flash scores 0.13.
   10,000 clips get 0.654 (val).
   30,631 clips get 0.718 (val), 0.709 (test).
2. **The learning rate decides stability, not the data size.**
   A LoRA rate near 1e-4 held for 400 or more steps collapsed twice (E006 fell to 0.19).
   At 3e-5 the full epoch trained cleanly (E008, E009).
   Short 250-step sweeps (E007) could not show this: the failure appears only after 400 to 800 steps.
3. **The low-rate tail of the schedule matters.**
   Steps 2,400 to 3,200 added 0.014 to final and lowered Brier from 0.821 to 0.808.
4. **CLIP + MLP is a strong baseline.**
   It matched the 10k-clip Omni pilot.
   Only the full-data Omni model beat it clearly.
5. **Ensembling stops helping as the model gets stronger.**
   With CLIP: +0.041 for the pilot, +0.037 at 6k clips, +0.009 at full data (val), +0.001 on test.
6. **Calibration comes from the loss.**
   ECE stays near 0.03 throughout, with no temperature step.
   The MLP baselines need temperature scaling to get there (ECE 0.144 before).
7. **Audio helps a little at most.**
   Removing audio cost 0.026 at 10k clips (E004), inside the noise, and the head precision also changed between the runs.
   This is not a clean result yet.
8. **Vision LoRA did not help.**
   Promising at n=200 (E007), 0.014 behind at n=500 (E008), and 20% slower.
9. **Text alone is weak.**
   Title and description reach only 0.35.
   The visual signal dominates.
10. **fp32 head.**
    A bf16 head gave NaN losses around step 720 in two runs.
    The head now runs in fp32 and non-finite steps are skipped and counted.

## 7. Cost

| Item | Cost |
|---|---|
| Data pull and preprocessing | a few dollars |
| Zero-shot Clef-flash (E001) | about $7 |
| Pilot (E002) | about $17 |
| Baselines (E003) | about $5 |
| Audio ablation (E004) | about $9 |
| Failed runs (NaN incident, E006) | about $46 |
| LR sweep (E007) | about $17 |
| Stability test (E008) | about $18 |
| Full run (E009) | about $43 |
| Test scoring (E010) | about $3 |

These are per-run estimates, rounded up, so they add to more than the bill.
Modal billed $156.75 in total by 9 October 2026, which also includes the start of the VideoFeedback work.
The final model alone costs about $43 to reproduce.

## 8. Limitations

- **Different test split.** The challenge test labels are withheld and the server closed on 6 July 2025.
  So we cannot place ourselves on the official leaderboard.
- **Published checkpoints are not usable as baselines on our split.**
  Li24 and LMM-EVQA trained on all of `train_data.csv`, which contains our test clips.
- **One seed.** Seed variance is unknown.
- **Survivorship bias.** Only 32% of videos are still online, skewed to higher ECR.
- **Checkpoint chosen on val.** Val numbers are slightly optimistic (0.718 val against 0.709 test).

## 9. To do when we come back

| Item | Why | Est. cost |
|---|---|---|
| 3 seeds of the E009 recipe | Show 0.709 is not one lucky run | about $90 |
| Retrain VideoLLaMA2-AV (winner's best single model) on our split | Direct, same-clip comparison with the winner | about $20 to $30 |
| Ask the SnapUGC authors for test labels or offline scoring | Official leaderboard number | free |
| Clean audio ablation (same recipe, no audio, full data) | Settle whether audio helps | about $25 |
| Modality dropout (train on A+V, V, A and text views) | May make the model use audio more | small test first |

## 10. Pointers

- Experiment log: [`results/README.md`](../results/README.md).
- Evaluation rules: [`EVALUATION.md`](../EVALUATION.md).
- Code: `vjev/train.py`, `vjev/predict.py`, `vjev/simple_baselines.py`, `scripts/ensemble.py`.
- W&B project: `sp1d5r/vjev`, runs `full-lr3e-5-30k`, `mid-lr3e-5-6k`, `pilot-omni-av-10k`, `baseline-*-ts`.
