# Evaluation plan

## The problem

The VQualA 2025 EVQA-SnapUGC challenge withheld its val and test labels.
The challenge server closed on 6 July 2025, with no post-challenge phase.
Published checkpoints (Li24, LMM-EVQA) were trained on all of `train_data.csv`.
Our test clips are part of that file, so those checkpoints have seen our test set.
Scoring them on our split would inflate their numbers, so we never do that.

## The plan

### 1. A frozen test split, touched once

- Test: 1,736 clips, `sha256(Id) % 100 < 5`.
- No model trains or tunes on it.
- All tuning and model choice use val (1,714 clips).
- We run the test set once per final model.
- We report 95% bootstrap confidence intervals.
- Main results use 3 seeds.

### 2. Baselines retrained on our exact split

Every model trains on the same 30,631 clips and is scored on the same test clips.

| Baseline | Why | Source |
|---|---|---|
| VideoLLaMA2-AV + regression head | Challenge winner's best single model, uses audio | `sunwei925/LMM-EVQA` (Apache-2.0, training code open) |
| Sentence embeddings + XGBoost | Text-only floor (7th-place method) | Our implementation |
| CLIP features + MLP | Cheap visual floor | Our implementation |
| Zero-shot Clef-flash | Typed model, no training | Done (E001) |

### 3. Official numbers, if possible

We will ask the SnapUGC authors (Dasong Li et al.) for the test labels or offline scoring.
If they agree, we also report on the official split.

### 4. What we measure

- Ranking: SROCC, PLCC, and their mean (the challenge's final score).
- Calibration: Brier and ECE of the 10-level distribution. No prior SnapUGC work reports these.
- Ablations: audio vs no audio, video vs text, frame budget.
- Cost: forward latency and clips per hour against generative VLMs.

### 5. Limitations we report

- Only 32% of SnapUGC videos are still downloadable.
- Low-engagement videos were deleted more often, so our clips skew to higher ECR (mean 0.472 against 0.498 overall).
- Our numbers are not directly comparable to the challenge leaderboard.
