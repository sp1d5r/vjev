# vjev

Video Jev: a typed, calibrated decision model for short videos.
It answers Jev / SystemOne questions (`choice`, `score`, `noul`) about a video in one forward pass.
The first benchmark is SnapUGC engagement prediction (ECR).

## Setup

```sh
uv sync
uv run pytest
```

Modal needs two secrets: `wandb` (`WANDB_API_KEY`) and `huggingface` (`HF_TOKEN`).

## Data

The official SnapUGC val and test labels are withheld.
All splits come from the labelled `train_data.csv` (106,192 rows).
Videos come from the Snap CDN links in that CSV.
Many links have expired, so we keep whatever is still alive.
The split is `sha256(Id) % 100`: under 5 is test, under 10 is val, the rest is train.

```sh
uv run modal run vjev/data.py::upload_labels --csv-path ~/Downloads/train_data.csv
uv run modal run vjev/data.py::pull_videos
uv run modal run vjev/data.py::build_split
```

ECR is asked as a 10-level `score` question.
The level edges are train-set ECR deciles, stored in `index/meta.json`.

## Experiments

```sh
uv run modal run vjev/baseline.py --split test --limit 500             # zero-shot Clef-flash
uv run modal run vjev/baseline.py --split test --limit 500 --no-video  # text-only ablation
```

Every run logs to the W&B project `vjev`.

## Metrics

- SROCC and PLCC between the expected score and true ECR, as in the VQualA 2025 challenge. `final` is their mean.
- Brier and ECE of the 10-level distribution, for calibration.
- Forward latency per clip and clips per hour, for cost.

## License

Code: Apache-2.0.
SnapUGC has no stated license; check the dataset terms before redistributing videos.
