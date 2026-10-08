"""Build the "Plain language" saved view of the W&B runs page.

    uv run python scripts/wandb_workspace.py

Sections are named as questions, charts have plain-English titles, and test runs are
greyed out (not deleted). The W&B API cannot edit personal views, so this is a saved
view; its URL is kept in results/wandb_view_url.txt and reruns update it in place.
"""

from __future__ import annotations

import pathlib

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY, PROJECT = "sp1d5r", "vjev"
URL_FILE = pathlib.Path(__file__).resolve().parent.parent / "results" / "wandb_view_url.txt"
# Runs that only test the plumbing. Greyed out so they don't clutter the charts.
TEST_RUNS = {"clef-flash-zs-video+text-train20", "smoke-omni-av", "tput-omni-av",
             "smoke-fp32-head", "ablation-omni-v-10k", "full-omni-av-30k"}

READ_ME = """### What this project tests
Can a typed model predict whether viewers keep watching a short video past the first 5 seconds?

### How to read the numbers
- **Final score**: how well clips are ranked by retention. 0 = random, 1 = perfect. Trained models in the 2025 challenge scored 0.66 to 0.71 (different test set).
- **Brier**: how close the probabilities are to the truth. Lower is better. **0.900 = blind guess.**
- **Calibration error (ECE)**: is "70% sure" right 70% of the time? Lower is better.
- **Level accuracy**: top level exactly right. 10% = chance.

### Success bar (set before results)
0.25 or less = no · 0.25 to 0.50 = weak signal · 0.50 or more = competitive

### Runs
`zs-` = zero-shot Clef-flash, no training · `pilot-` / `omni-` = Qwen3-Omni training runs · `baseline-` = cheap baselines · greyed out = plumbing tests and failed runs
"""


def line(title: str, metrics: dict[str, str], w: int = 12, h: int = 8,
         smooth: float | None = None) -> wr.LinePlot:
    return wr.LinePlot(title=title, x="step", y=list(metrics), title_x="Training step",
                       line_titles=metrics, smoothing_factor=smooth,
                       layout=wr.Layout(w=w, h=h))


def sections() -> list[ws.Section]:
    return [
        ws.Section(name="Read me first", is_open=True, panels=[
            wr.MarkdownPanel(markdown=READ_ME, layout=wr.Layout(w=24, h=10)),
        ]),
        ws.Section(name="1. Is it learning to predict retention?", is_open=True, panels=[
            line("Final score on held-out clips (higher is better)",
                 {"val/final": "Final score"}, w=24),
            line("Rank and linear correlation (the two halves of the final score)",
                 {"val/srocc": "Rank correlation (SROCC)",
                  "val/plcc": "Linear correlation (PLCC)"}),
            line("Top level exactly right (10% = chance)", {"val/level_acc": "Level accuracy"}),
        ]),
        ws.Section(name="2. Are its probabilities honest?", is_open=True, panels=[
            line("Brier score (lower is better, 0.900 = blind guess)",
                 {"val/brier": "Brier score"}),
            line("Calibration error (lower is better)", {"val/ece": "Calibration error"}),
            line("Spread of predictions (0 = same answer for every clip)",
                 {"val/expected_std": "Prediction spread (levels)"}, w=24, h=7),
        ]),
        ws.Section(name="3. Is training healthy?", is_open=False, panels=[
            line("Training loss (lower is better)", {"train/loss": "Total loss"}, smooth=0.6),
            line("Loss parts", {"train/ce": "Cross-entropy", "train/brier": "Brier",
                                "train/emd": "Ordinal distance"}, smooth=0.6),
            line("Learning rate", {"train/lr": "Learning rate"}, h=7),
            line("Speed (clips per hour, about $1.80 per 1,000 clips)",
                 {"train/clips_per_hour": "Clips per hour"}, h=7),
        ]),
        ws.Section(name="4. Cheap baselines vs the pilot (same 500 val clips)", is_open=True,
                   panels=[
            wr.BarPlot(title="Final score on val (higher is better)", metrics=["val/final"],
                       orientation="h", layout=wr.Layout(w=8, h=8)),
            wr.BarPlot(title="Brier on val (lower is better)", metrics=["val/brier"],
                       orientation="h", layout=wr.Layout(w=8, h=8)),
            wr.BarPlot(title="Calibration error on val (lower is better)", metrics=["val/ece"],
                       orientation="h", layout=wr.Layout(w=8, h=8)),
        ]),
        ws.Section(name="5. Zero-shot baselines (no training)", is_open=True, panels=[
            wr.BarPlot(title="Final score with no training (higher is better)",
                       metrics=["final"], orientation="h", layout=wr.Layout(w=12, h=8)),
            wr.BarPlot(title="Brier with no training (0.900 = blind guess)",
                       metrics=["brier"], orientation="h", layout=wr.Layout(w=12, h=8)),
            wr.BarPlot(title="Calibration error with no training (lower is better)",
                       metrics=["ece"], orientation="h", layout=wr.Layout(w=12, h=8)),
            wr.BarPlot(title="Forward time per clip, ms (lower is faster)",
                       metrics=["forward_ms_p50"], orientation="h",
                       layout=wr.Layout(w=12, h=8)),
        ]),
    ]


def main() -> None:
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    settings = ws.RunsetSettings(run_settings={
        r.id: ws.RunSettings(disabled=True) for r in runs if r.name in TEST_RUNS})
    if URL_FILE.exists():
        view = ws.Workspace.from_url(URL_FILE.read_text().strip())
        view.sections, view.runset_settings = sections(), settings
        view.save()
    else:
        view = ws.Workspace(entity=ENTITY, project=PROJECT, name="Plain language",
                            sections=sections(), runset_settings=settings)
        view.save_as_new_view()
    URL_FILE.write_text(view.url + "\n")
    print(view.url)


if __name__ == "__main__":
    main()
