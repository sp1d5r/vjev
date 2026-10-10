# Experiment log

Newest first.
Every entry gives the setup, the numbers, what we learned and the cost.
All numbers are on our test split (1,736 clips, `sha256(Id) % 100 < 5`) unless stated.
They are not comparable to the VQualA 2025 leaderboard, whose test labels are withheld.

---

## 2026-10-10 · G3 · FunQA moments, with and without audio (done)

**Question:** can the new `moment` question type find when the funny, creative or magic moment happens? Does audio help?

**Setup:** FunQA (CC BY-NC-SA 4.0), localisation tasks H1 (funny), C1 (creative), M1 (magic): 3,072 train videos, scored on 400 val videos.
One `moment` question per video, worded as in FunQA (for example "Identify the video's funny moment."), with one option per second of video (2 fps, up to 60 s, 336 px).
Each option's evidence is that second's video tokens (and its audio tokens when audio is on). The head is unchanged.
Target: a distribution over seconds, weighted by overlap with the answer window. 1 epoch (384 steps), LoRA LR 3e-5, head LR 1e-4.
The two runs differ only in audio.

**Metrics:** HIT@1, the top second falls inside the answer window. R1@0.5, the span around the peak (bins at 50% or more of the peak) overlaps the answer with IoU 0.5 or more. mIoU of that span. Mass in moment, the probability on seconds inside the answer.
Content-free baselines on the same clips come from `vjev/moment_baselines.py`.

**Results** (val, 400 videos, final step)

| Method | HIT@1 | R1@0.5 | mIoU | Mass in moment |
|---|---|---|---|---|
| Uniform guess | 0.435 | 0.320 | 0.380 | 0.454 |
| Position prior (mean train target by position) | 0.533 | 0.135 | 0.190 | 0.484 |
| Omni + typed head, video only (`g3b-fq-v2`) | 0.792 | 0.580 | 0.537 | 0.701 |
| **Omni + typed head, video + audio (`g3a-fq-av`)** | **0.807** | **0.580** | **0.538** | **0.705** |

Val by step (HIT@1, with audio / video only): 100: 0.750 / 0.762, 200: 0.785 / 0.800, 300: 0.795 / 0.800, 384: 0.807 / 0.792.

**What we learned**

- The `moment` type works with no change to the head: +27 points HIT@1 and +26 points R1@0.5 over the best content-free guess.
- It gives a probability over time, so it can report the top few moments and how sure it is. A model that writes timestamps as text cannot.
- Audio adds nothing here (all gaps within 0.015, inside the noise at n=400). FunQA moments are visual: pranks, tricks and magic.
- So the clean audio test still needs a task where the answer is in the sound (laughter, speech, music).
- We have not yet compared against published FunQA timestamp numbers; their metric definitions differ and need checking first. Test split untouched.

**Cost:** about $8 for both runs.

---

## 2026-10-10 · G1a · Real or AI, matched input (done)

**Question:** with length, shape and size hidden, does Omni + typed head tell real from generated video better than CLIP + MLP, on generators and sources it has not seen?

**Setup:** GenVideo, 4 shards (4,796 clips, 10 generators), matched input (first 1.5 s, square crop, 256x256, no duration; see the finding below).
One `noul` question. Train 3,652 clips, 1 epoch (454 steps), LoRA LR 3e-5, head LR 1e-4, video only.
Cross-dataset test: the same question on all 1,860 VideoFeedback val clips (11% real), which come from other generators and other real sources (`vfai`).

**Results**

| Model | GenVideo val AUC | Cross-dataset AUC | Acc | Brier | ECE |
|---|---|---|---|---|---|
| CLIP + MLP + temperature | 0.997 | **0.934** | 0.912 | **0.138** | 0.046 |
| Omni + typed head | 0.997 | 0.929 | 0.903 | 0.148 | **0.044** |

**What we learned**

- A tie. On these 2023-era generators the typed model adds nothing over CLIP, in domain or across datasets.
- Real or AI on these sets is mostly a frame-level texture task, which CLIP features already capture.
- This is not a good showcase for the typed model. A useful version needs modern generators (Sora, Veo, Kling class), and we have no open labelled set of them.
- The held-out-generator test split is kept untouched for now.

**Cost:** about $5 (training) plus $1 (cross-dataset scoring).

**W&B run:** `g1a-gv-matched`. Predictions: `/data/preds/g1a-gv-matched-vfai-val.jsonl`.

---

## 2026-10-10 · Finding · GenVideo real-or-AI has a format shortcut

**What happened:** the first real-or-AI run (`g1a-gv-3k`) reached val AUC 0.995 after 50 steps, and the CLIP + MLP baseline reached 0.999.
That is too easy, so we probed every clip (`vjev/gv_stats.py`).

| Source | Clips | Size | Frame rate | Length |
|---|---|---|---|---|
| Real | 2,440 | 640x360 (66%), 360x640, 640x362 | 24 fps | 10.0 to 10.2 s |
| ST2V | 242 | 1280x720 | 10 fps | 7.2 s |
| COG | 251 | 720x480 | 8 fps | 6.0 s |
| SVD (held out) | 251 | 1024x576 | 7 fps | 3.6 s |
| OpenSora (held out) | 245 | 512x512, 256x256 | 8 fps | 2.0 s |
| Other 6 generators | 1,394 | 256x256 to 1088x640 | 8 to 25 fps | 1.6 to 4 s (ZeroScope up to 40 s) |

Every real clip is 10 s long and every generated clip but a few ZeroScope ones is 7.2 s or shorter.
Our record also put `duration_seconds` in the state.
So the answer could be read off the length, and the held-out generators are just as short, so the held-out test was not safe either.

**Fix (before any test scoring):** real-or-AI clips now all use the first 1.5 s (12 frames at 8 fps), a center square crop and 256x256 frames, and the state has no duration (`vjev/tasks.py::prepare_frames`).
Every clip is now exactly 1,022 tokens.
The CLIP baseline uses the same frames.
`g1a-gv-3k` was stopped and is void. The rerun is `g1a-gv-matched`.

**Still open:** resolution and compression differ by source even after resizing (real clips are upscaled from 360p).
The cross-dataset test (train on GenVideo, test on VideoFeedback's real and generated clips, `vfai`) checks whether what the model learns carries over to other sources.

**Lesson:** probe every real-or-AI set for format differences before training. A baseline at 0.999 is a warning, not a result.

**CLIP + MLP with the matched input** (fit on GenVideo train, temperature on 10% of train):

| Test | Clips | AUC | Acc | Brier | ECE |
|---|---|---|---|---|---|
| GenVideo val (seen generators) | 203 | 0.997 | 0.975 | 0.038 | 0.018 |
| VideoFeedback val, cross-dataset (`vfai`, 11% real) | 1,860 | 0.934 | 0.912 | 0.138 | 0.046 |

Even with length, shape and size hidden, these 2023-era generators are easy to tell from real footage.
Seen-generator real-or-AI is saturated, so it cannot rank models.
The cross-dataset number is the useful one.
A hard real-or-AI test needs modern generators, which neither set has.

---

## 2026-10-10 · V001 · VideoFeedback small test (G0, passed)

**Question:** does the SnapUGC recipe transfer to AI-video quality, with five typed questions per clip?

**Setup:** VideoFeedback (TIGER-Lab, Apache-2.0), 2,000 train clips, 250 steps, scored on 300 val clips every 50 steps.
Video only (the clips are silent), 8 fps, up to 6 s, longest side 448 px.
Five `score` questions (1 to 4) per clip, all answered in one forward pass: visual quality, temporal consistency, dynamic degree, text-to-video alignment, factual consistency.
LoRA LR 3e-5, head LR 1e-4, as E009.
Val is 5% of the official train split (`sha256(id) % 100 < 5`), so these are not test numbers.

**Gate (set before results):** average Spearman 0.60 or more on val.

**Results** (val, 300 clips; Spearman per question)

| Step | Visual quality | Temporal | Dynamic | Text align | Factual | Avg SROCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.51 | 0.55 | 0.41 | 0.43 | 0.56 | 0.493 | 0.608 | 0.091 |
| 100 | 0.62 | 0.65 | 0.46 | 0.48 | 0.66 | 0.573 | 0.545 | 0.093 |
| 150 | 0.74 | 0.73 | 0.51 | 0.52 | 0.73 | 0.648 | 0.480 | 0.067 |
| 200 | 0.74 | 0.74 | 0.53 | 0.51 | 0.72 | 0.648 | 0.471 | 0.058 |
| 250 | **0.75** | **0.74** | **0.54** | **0.51** | **0.73** | **0.653** | **0.465** | 0.066 |
| VideoScore (test, 37.6k train clips) | 0.862 | 0.803 | 0.776 | 0.594 | 0.821 | 0.771 | - | - |
| GPT-4o (test, prompted) | | | | | | 0.230 | - | - |

**What we learned**

- Gate passed: 0.653 after 2,000 clips (about 6% of the train split).
- Visual quality, temporal consistency and factual consistency are already within 0.07 to 0.11 of VideoScore.
- Dynamic degree lags most (0.54 against 0.78). The 6 s cap is not the cause (clips are 1 to 3 s), so this is likely data volume.
- Forward pass 170 ms per clip, 4,800 clips per hour in training.
- Different splits: VideoScore numbers are on the official test set; ours are on val.

**Cost:** about $3 (0.46 h on one H200 plus model load).

**W&B run:** `v001-vf-2k`. Next: G0 full run on all 35,041 train clips (`g0-vf-full`).

---

## 2026-10-09 · E010 · Test split, scored once

**Question:** do the val results hold on the untouched test split?

**Setup:** all 1,736 test clips (`sha256(Id) % 100 < 5`), scored once per final model.
The models and the ensemble recipe were fixed before this run: E009 `best.pt` (step 3,200), the temperature-scaled baselines from E003 refit with the same seeds (their val numbers match exactly, T=2.036 for CLIP + MLP), and the equal-weight mean of z-scores.
Commands: `predict_ckpt --run full-lr3e-5-30k --split test --limit 0 --final`, `simple_baselines.py::fit_all --calibrate --test`, `scripts/ensemble.py ... --out results/test_final.json`.

**Results** (test, 1,736 clips)

| Model | SROCC | PLCC | Final | Brier | ECE |
|---|---|---|---|---|---|
| **Qwen3-Omni + typed head (E009)** | 0.708 | 0.709 | **0.709** | **0.818** | **0.028** |
| Omni + CLIP-MLP ensemble (pre-set recipe) | 0.711 | 0.708 | 0.709 | 0.816 | 0.033 |
| CLIP + MLP + temperature | 0.645 | 0.642 | 0.643 | 0.843 | 0.034 |
| CLIP + text + MLP + temperature | 0.642 | 0.636 | 0.639 | 0.859 | 0.029 |
| Text + XGBoost + temperature | 0.348 | 0.347 | 0.347 | 0.884 | 0.029 |
| Zero-shot Clef-flash, video + text (E001) | 0.128 | 0.127 | 0.127 | 0.912 | 0.064 |

Paired bootstrap (2,000 resamples) of the final-score gap:

- Omni minus CLIP + MLP: **+0.065 [0.041, 0.088]**.
- Ensemble minus Omni: +0.001 [-0.010, +0.012].

**What we learned**

- The main claim holds on test. With the same training data, Omni + typed head ranks clearly better than CLIP + MLP, and its probabilities are better (Brier 0.818 against 0.843).
- Val slightly flattered both models (0.717 to 0.709 for Omni, 0.673 to 0.643 for CLIP), as expected from choosing on val.
- On test, the ensemble adds nothing. The E005 and E008 ensemble gains came from weaker Omni models. One model is enough.
- One seed only. Seed variance is the next open question.

**Cost:** about $3.

---

## 2026-10-08 · E009 · Full run, 30,631 clips, LoRA LR 3e-5 (done)

**Question:** with all train clips and the settings that passed E008, does Omni beat CLIP + MLP alone? Does the ensemble improve?

**Setup:** video + audio, LoRA LR 3e-5, head LR 1e-4, no vision LoRA, 1 epoch (3,829 steps, cosine to zero), scored on the same 500 val clips every 200 steps. App `ap-NNSpTEsWhLc0fOvdKo1U1i`. 7.5 h.

**Stop rule (set before results):** at step 800, val final 0.60 or more; after that, stop if two evals in a row are more than 0.05 below the best. Neither triggered.

**Results** (val final, 500 clips; Brier in brackets)

| Step | Final | Step | Final |
|---|---|---|---|
| 200 | 0.428 (0.897) | 2,200 | 0.699 (0.818) |
| 400 | 0.557 (0.877) | 2,400 | 0.704 (0.821) |
| 600 | 0.642 (0.856) | 2,600 | 0.709 (0.816) |
| 800 | 0.641 (0.847) | 2,800 | 0.714 (0.812) |
| 1,000 | 0.660 (0.846) | 3,000 | 0.715 (0.808) |
| 1,200 | 0.682 (0.829) | 3,200 | **0.718** (0.808), kept as `best.pt` |
| 1,400 | 0.686 (0.825) | 3,400 | 0.717 (0.808) |
| 1,600 | 0.672 (0.852) | 3,600 | 0.716 (0.808) |
| 1,800 | 0.697 (0.829) | 3,800 | 0.716 (0.809) |
| 2,000 | 0.700 (0.824) | | |

No step or clip was skipped.

Ensemble with CLIP + MLP (temperature-scaled) on the same 500 val clips (`results/ensemble_full_val500.json`):

| Model | Final | Brier | ECE |
|---|---|---|---|
| Omni E009 | 0.717 | **0.808** | 0.032 |
| CLIP + MLP, temperature-scaled | 0.673 | 0.845 | 0.046 |
| Equal-weight ensemble | 0.726 | 0.813 | 0.032 |

Paired bootstrap: Omni minus CLIP +0.044 [0.005, 0.082]; ensemble minus Omni +0.009 [-0.009, +0.028].

**What we learned**

- At LoRA LR 3e-5, a full epoch is stable. The rate, not the data size, caused the E006 collapse.
- With the same training data, Omni beats CLIP + MLP alone on val.
- The low-rate tail helped: steps 2,400 to 3,200 added 0.014 to final and lowered Brier from 0.821 to 0.808.
- The ensemble gain shrinks as Omni gets stronger (+0.009, inside the noise).

**Cost:** about $43 including scoring (billed $113.21 before, $155.91 after).

**W&B run:** `full-lr3e-5-30k`.

---

## 2026-10-08 · E008 · Medium stability test (passed)

**Question:** with a lower peak LoRA rate (3e-5, head 1e-4), does the model stay stable through steps 400 to 800? And does vision LoRA help at n=500?

**Why:** E007 showed that short tests cannot reveal the failure. It appears only after 400 to 800 steps at a high rate. A full run with vision LoRA would cost about $70, so this test comes first.

**Setup:** 6,400 train clips (800 steps), video + audio, scored on the same 500 val clips as E002 every 100 steps.

| Run | LoRA LR | Head LR | Vision LoRA | Est. cost |
|---|---|---|---|---|
| `mid-lr3e-5-vis-6k` | 3e-5 | 1e-4 | yes | about $15 |
| `mid-lr3e-5-6k` | 3e-5 | 1e-4 | no | about $8 |

**Pass rule (set before results):** val final keeps rising through steps 400 to 800 with no collapse, and ends at 0.60 or more.

**Results** (val final, 500 clips; Brier in brackets)

| Step | No vision LoRA | Vision LoRA |
|---|---|---|
| 100 | 0.448 (0.899) | 0.470 (0.897) |
| 200 | 0.469 (0.888) | 0.564 (0.873) |
| 300 | 0.582 (0.871) | 0.582 (0.882) |
| 400 | 0.655 (0.847) | 0.645 (0.859) |
| 500 | 0.671 (0.837) | 0.646 (0.852) |
| 600 | 0.676 (0.836) | 0.665 (0.839) |
| 700 | 0.675 (0.834) | 0.666 (0.839) |
| 800 | **0.677** (**0.833**) | 0.663 (0.839) |
| ECE at 800 | 0.042 | 0.023 |
| Clips per hour | 3,790 | 3,090 |

Both runs pass. Neither skipped a step.

Ensemble of the no-vision run with CLIP + MLP (temperature-scaled), same 500 clips, from `scripts/ensemble.py --out results/ensemble_mid_val500.json`:

| Model | Final | Brier | ECE |
|---|---|---|---|
| Omni E008, no vision LoRA | 0.677 | 0.833 | 0.042 |
| CLIP + MLP, temperature-scaled | 0.673 | 0.845 | 0.046 |
| **Equal-weight ensemble** | **0.714** | **0.826** | 0.054 |

Paired bootstrap (2,000 resamples) of the final-score gap:

- Ensemble minus Omni: +0.037 [0.016, 0.060].
- Ensemble minus CLIP: +0.041 [0.017, 0.066].
- CLIP minus Omni: -0.004 [-0.048, +0.041].

Prediction correlation between Omni and CLIP: 0.79 (0.82 for the pilot).

**What we learned**

- A lower LoRA rate fixes the collapse at this length. The no-vision run rose every eval to step 600 and then held.
- With 6,400 clips, Omni beats the 10,000-clip pilot (0.677 against 0.654) and ties CLIP + MLP trained on 30,631 clips.
- Both ensemble gaps now exclude zero. Omni and CLIP make different mistakes.
- Vision LoRA does not help at n=500. It is 0.014 behind and 20% slower, but it has the lowest ECE. The E007 gain did not hold up.
- Caveat: the checkpoint (best of 8 evals) and the ensemble were chosen on these val clips, so 0.714 is slightly optimistic. The test split will give the honest number.

**Cost:** about $18 for both runs and scoring (billed $95.34 before, $113.21 after).

---

## 2026-10-08 · E007 · Learning-rate sweep and vision LoRA

**Question:** which peak learning rate lets the model learn from the first steps without collapsing, and does adapting the vision tower help?

**Why:** with the fp32 head, E004 and E006 learned almost nothing while the LoRA learning rate was near its 1e-4 peak. The full run (E006) should have been preceded by a small test like this one.

**Setup:** 4 runs on the same 2,000 train clips (video + audio), scored on 200 val clips every 50 steps. 250 steps each, cosine schedule. About $4 each.
Vision LoRA adds rank-64 adapters on `visual.blocks.N.attn.{qkv,proj}`. A 10-step smoke test confirmed they receive gradients (LoRA B norm 5.96, from 0). It trains about 30% slower.

**Results** (val final, 200 clips; spread in levels in brackets)

| Step | LoRA 1e-4, head 3e-4 (control) | LoRA 3e-5 | LoRA 1e-5 | LoRA 3e-5 + vision LoRA |
|---|---|---|---|---|
| 50 | 0.536 (0.01) | 0.475 (0.01) | 0.470 (0.01) | 0.507 (0.01) |
| 100 | 0.588 (0.00) | 0.498 (0.01) | 0.475 (0.01) | 0.538 (0.05) |
| 150 | 0.578 (0.18) | 0.516 (0.06) | 0.497 (0.04) | 0.450 (0.05) |
| 200 | 0.588 (0.04) | 0.524 (0.26) | 0.493 (0.10) | 0.545 (0.80) |
| 250 | **0.586** (0.21) | 0.518 (0.47) | 0.490 (0.18) | 0.575 (1.02) |
| Brier at 250 | 0.888 | 0.885 | 0.894 | **0.872** |

**What we learned**

- My hypothesis was wrong at this scale. Over 250 steps, the highest learning rate learns fastest. Lower rates are slower, not more stable.
- So the failures in E004 and E006 are not "too high from step 1". They come after 400 to 800 steps at a still-high rate (see the table below). A 250-step test cannot show this, so it was the wrong test for that question.
- Vision LoRA at 3e-5 beats plain 3e-5 by 0.057 and has the best Brier and spread. At n=200 (standard error about 0.05) this is suggestive, not proven.
- The control also changes the head LR (3e-4 against 1e-4), so LoRA LR and head LR are confounded here.

Where each longer run went wrong (val final by step):

| Run | Head | 400 | 600 | 800 | Then |
|---|---|---|---|---|---|
| E002 pilot, 1,250 steps | bf16 | 0.532 | 0.525 | 0.632 | 0.654 |
| E004 v1 no audio | bf16 | 0.461 | 0.191 | NaN | stopped |
| E004 v2 no audio, 1,250 steps | fp32 | 0.312 | 0.395 | 0.531 | 0.628 |
| E006 v1 full | bf16 | 0.389 | - | NaN | stopped |
| E006 v2 full, 3,829 steps | fp32 | 0.437 | - | 0.185 | 0.21 to 0.26 to step 2,000 |

Every run is weak or unstable in the 400 to 800 step range while the LoRA rate is above about 7e-5. The short runs recover because their schedule decays quickly. The full run's schedule keeps the rate high for about 2,000 steps.

**Cost:** about $17 for the four runs.

**W&B runs:** `sweep-lr1e-4-2k`, `sweep-lr3e-5-2k`, `sweep-lr1e-5-2k`, `sweep-lr3e-5-vis-2k`.

---

## 2026-10-08 · E006 · Full run, 30,631 clips (stopped)

**Question:** with the same data as CLIP + MLP, does the Omni model rank better?

**Setup:** as E002 (video + audio, LoRA LR 1e-4, head LR 3e-4) on all 30,631 train clips, 1 epoch, 3,829 steps planned, scored on the same 500 val clips every 400 steps. fp32 head.

**Results** (500 val clips)

| Step | Final | Brier | ECE | Spread (levels) |
|---|---|---|---|---|
| 400 | 0.437 | 0.898 | 0.033 | 0.03 |
| 800 | 0.185 | 0.899 | 0.014 | 0.04 |
| 1200 | 0.238 | 0.898 | 0.012 | 0.10 |
| 1600 | 0.259 | 0.897 | 0.015 | 0.29 |
| 2000 | 0.213 | 0.895 | 0.015 | 0.32 |
| 2400 | 0.261 | 0.894 | 0.037 | 0.35 |

**Verdict:** stopped at step 2,440 by the stop rule set in advance (under 0.4 at step 2,400).

**What we learned**

- The model collapsed after step 400 and never recovered while the rate stayed high. It did not rescue itself as the rate fell from 1e-4 to about 6e-5.
- A long, high-rate schedule is the failure, not the amount of data (see E007).
- This run should have been preceded by a medium-length test (E008). That rule now applies to every run over about $20.

**Cost:** about $23 lost (4.9 h on one H200).

**W&B run:** `full-omni-av-30k-v2`.

---

## 2026-10-08 · E005 · Ensemble of Omni and CLIP

**Question:** does the Omni model know something CLIP + MLP does not?

**Setup**

- Inputs: the E002 pilot checkpoint, re-scored on the same 500 val clips (`vjev/predict.py`), and CLIP + MLP with temperature scaling (E003).
- Re-scoring with the fp32 head gives 0.653, matching the saved 0.654, so the checkpoint and predict path are sound.
- Ensemble: equal weights, fixed before looking. Ranking uses the mean of each model's z-scored expected level. Probabilities use the mean of the two distributions. Nothing is fit on val.
- Paired bootstrap over clips, 2,000 resamples.

**Results** (500 val clips)

| Model | Final | SROCC | PLCC | Brier | ECE |
|---|---|---|---|---|---|
| Omni pilot (E002) | 0.653 | 0.654 | 0.652 | 0.838 | 0.030 |
| CLIP + MLP + temperature | 0.673 | 0.673 | 0.673 | 0.845 | 0.046 |
| **Omni + CLIP ensemble** | **0.695** | **0.696** | **0.694** | **0.829** | **0.030** |
| Omni + CLIP-and-text ensemble | 0.697 | 0.698 | 0.695 | 0.832 | 0.052 |

| Paired difference in final | Mean | 95% CI |
|---|---|---|
| Ensemble - CLIP | +0.022 | -0.001 to +0.045 |
| Ensemble - Omni | +0.041 | +0.020 to +0.064 |
| CLIP - Omni | +0.019 | -0.025 to +0.063 |

The two models' predictions correlate at 0.82.

**What we learned**

- The Omni model carries signal CLIP does not. Together they beat both, on ranking and on Brier.
- The gain over CLIP alone is borderline at n=500 (the interval just touches 0). A larger eval set would settle it.
- On their own, Omni and CLIP are not distinguishable at this sample size.

**Cost:** about $2 (one H200 re-score).

**Raw data:** [`ensemble_val500.json`](ensemble_val500.json). Script: `scripts/ensemble.py`.

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
