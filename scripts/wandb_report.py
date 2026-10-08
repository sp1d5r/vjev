"""Build (or rebuild) the plain-language W&B report and per-run notes.

    uv run python scripts/wandb_report.py

The report URL is kept in results/wandb_report_url.txt, so reruns update the same page.
"""

from __future__ import annotations

import pathlib

import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY, PROJECT = "sp1d5r", "vjev"
URL_FILE = pathlib.Path(__file__).resolve().parent.parent / "results" / "wandb_report_url.txt"

# Plain-language notes shown on each run's page and in the runs table.
RUN_NOTES = {
    "clef-flash-zs-video+text-test1736":
        "Zero-shot Clef-flash, no training. Sees video frames plus title and description. "
        "Final 0.127: it barely ranks retention.",
    "clef-flash-zs-text-test1736":
        "Zero-shot Clef-flash, no training. Sees only the title and description. "
        "Final 0.172, but overconfident (calibration error 0.28).",
    "clef-flash-zs-video-test1736":
        "Zero-shot Clef-flash, no training. Sees only video frames. "
        "Final 0.140. Predicts close to a uniform guess.",
    "clef-flash-zs-video+text-train20": "Pipeline smoke test on 20 clips. Ignore the numbers.",
    "smoke-omni-av": "Trainer smoke test: 3 steps on 12 clips. Ignore the numbers.",
    "tput-omni-av": "Speed test: 20 steps on 160 clips. Measured about 2,500 clips per hour.",
    "pilot-omni-av-10k":
        "Pilot. Qwen3-Omni with video and audio, typed head trained from scratch, "
        "10,000 training clips, checked on 500 held-out clips every 200 steps. "
        "Result: final 0.654, passing the 0.50 bar. Probabilities stay honest (ECE 0.035).",
}

TRAIN_RUNS = "pilot|omni-"  # name filter for real training runs
ZERO_SHOT_RUNS = "zs-.*test1736"


def set_notes() -> None:
    api = wandb.Api()
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        note = RUN_NOTES.get(run.name)
        if note and run.notes != note:
            run.notes = note
            run.update()
            print(f"noted {run.name}")


def blocks() -> list:
    train = wr.Runset(ENTITY, PROJECT, name="Training runs", query=TRAIN_RUNS)
    zero = wr.Runset(ENTITY, PROJECT, name="Zero-shot baselines", query=ZERO_SHOT_RUNS)
    return [
        wr.P("We are testing one question: can a typed model predict whether viewers keep "
             "watching a short video past the first 5 seconds?"),
        wr.P("The model watches the first 16 seconds of a SnapUGC clip, with its audio. "
             "It answers one typed question with 10 levels, from almost everyone swipes away "
             "to almost everyone keeps watching. Each level gets a probability."),
        wr.CalloutBlock(text=[
            ("Current answer: yes. Zero-shot models cannot do it (final score 0.13 to 0.17). "
            "After training on 10,000 clips with audio, the pilot scores 0.654 on held-out "
            "clips, above our 0.50 bar. Its curve is in the first chart below.")]),
        wr.H2("How to read the numbers"),
        wr.UnorderedList(items=[
            ("Final score: how well the model ranks clips by retention, from 0 (random) to 1 "
            "(perfect). It is the mean of SROCC and PLCC. Trained models in the 2025 challenge "
            "scored 0.66 to 0.71 on a different test set."),
            ("Brier score: how close the probabilities are to the truth. Lower is better. "
            "0.900 means the model just guessed every level equally."),
            ("Calibration error (ECE): when the model says 70% sure, is it right 70% of the time? "
            "Lower is better. 0 means perfectly honest confidence."),
            "Level accuracy: how often the top level is exactly right. 10% is chance.",
        ]),
        wr.H2("Success criteria, set before any results"),
        wr.MarkdownBlock(text=(
            "| Outcome | Val final score | Meaning |\n|---|---|---|\n"
            "| No | 0.25 or less | Training barely helps. Rethink inputs or backbone. |\n"
            "| Yes, weakly | 0.25 to 0.50 | Real signal. Tune and scale. |\n"
            "| Yes | 0.50 or more | In the range of prior trained models. |\n"
            "| Strong | matches Li24 on our split | Competitive with published work. |")),
        wr.H2("1. Is the model learning to predict retention?"),
        wr.P("Each point is a check on 500 held-out clips the model never trained on. "
             "Up is better. Above 0.25 means real signal, above 0.50 means competitive."),
        wr.PanelGrid(runsets=[train], panels=[
            wr.LinePlot(title="Ranking quality on held-out clips (higher is better)",
                        x="step", y=["val/final"], title_x="Training step",
                        title_y="Final score", line_titles={"val/final": "Final score"},
                        layout=wr.Layout(w=24, h=9)),
            wr.LinePlot(title="The two halves of the final score", x="step",
                        y=["val/srocc", "val/plcc"], title_x="Training step",
                        line_titles={"val/srocc": "Rank correlation (SROCC)",
                                     "val/plcc": "Linear correlation (PLCC)"},
                        layout=wr.Layout(w=12, h=8)),
            wr.LinePlot(title="Top level exactly right (10% = chance)", x="step",
                        y=["val/level_acc"], title_x="Training step",
                        line_titles={"val/level_acc": "Level accuracy"},
                        layout=wr.Layout(w=12, h=8)),
        ]),
        wr.H2("2. Are its probabilities honest?"),
        wr.P("A useful reviewer must know when it is unsure. Both lines should go down. "
             "Brier below 0.900 means the model beats a blind guess."),
        wr.PanelGrid(runsets=[train], panels=[
            wr.LinePlot(title="Brier score (lower is better, 0.900 = blind guess)", x="step",
                        y=["val/brier"], title_x="Training step",
                        line_titles={"val/brier": "Brier score"}, layout=wr.Layout(w=12, h=8)),
            wr.LinePlot(title="Calibration error (lower is better)", x="step", y=["val/ece"],
                        title_x="Training step", line_titles={"val/ece": "Calibration error"},
                        layout=wr.Layout(w=12, h=8)),
            wr.LinePlot(title="Spread of predictions (0 = same answer for every clip)",
                        x="step", y=["val/expected_std"], title_x="Training step",
                        line_titles={"val/expected_std": "Prediction spread (levels)"},
                        layout=wr.Layout(w=24, h=7)),
        ]),
        wr.H2("3. Is training healthy?"),
        wr.P("Loss should fall and then flatten. The learning rate warms up, then decays. "
             "Speed tells us the cost: about 2,500 clips per hour is about $1.80 per 1,000 clips."),
        wr.PanelGrid(runsets=[train], panels=[
            wr.LinePlot(title="Training loss (lower is better)", x="step", y=["train/loss"],
                        title_x="Training step", smoothing_factor=0.6,
                        line_titles={"train/loss": "Total loss"}, layout=wr.Layout(w=12, h=8)),
            wr.LinePlot(title="Loss parts", x="step",
                        y=["train/ce", "train/brier", "train/emd"], title_x="Training step",
                        smoothing_factor=0.6,
                        line_titles={"train/ce": "Cross-entropy", "train/brier": "Brier",
                                     "train/emd": "Ordinal distance"},
                        layout=wr.Layout(w=12, h=8)),
            wr.LinePlot(title="Learning rate", x="step", y=["train/lr"],
                        title_x="Training step", line_titles={"train/lr": "Learning rate"},
                        layout=wr.Layout(w=12, h=7)),
            wr.LinePlot(title="Speed (clips per hour)", x="step", y=["train/clips_per_hour"],
                        title_x="Training step",
                        line_titles={"train/clips_per_hour": "Clips per hour"},
                        layout=wr.Layout(w=12, h=7)),
        ]),
        wr.H2("4. Zero-shot baselines (before any training)"),
        wr.P("Cloudflare's Clef-flash with no training, on our 1,736-clip test split. "
             "This is the starting point the trained model must beat."),
        wr.PanelGrid(runsets=[zero], panels=[
            wr.BarPlot(title="Final score with no training (higher is better)",
                       metrics=["final"], orientation="h", layout=wr.Layout(w=12, h=8)),
            wr.BarPlot(title="Brier score with no training (0.900 = blind guess)",
                       metrics=["brier"], orientation="h", layout=wr.Layout(w=12, h=8)),
        ]),
        wr.H2("Data notes"),
        wr.UnorderedList(items=[
            "SnapUGC has 106,192 labelled clips. Only 34,081 links still work (32%).",
            "Low-retention videos were deleted more often, so our clips skew slightly higher.",
            ("The official test labels are withheld, so we use our own fixed split. "
            "Our numbers are not directly comparable to the challenge leaderboard."),
        ]),
    ]


def main() -> None:
    set_notes()
    title = "vjev: can a typed model predict video retention?"
    desc = "Plain-language results for the video Jev project. Rebuilt by scripts/wandb_report.py."
    if URL_FILE.exists():
        # The "?" in the title ends up raw in the URL, which hides the report id from the parser.
        report = wr.Report.from_url(URL_FILE.read_text().strip().replace("?", "%3F"))
        report.title, report.description, report.blocks = title, desc, blocks()
    else:
        report = wr.Report(project=PROJECT, entity=ENTITY, title=title, description=desc,
                           blocks=blocks())
    report.save()
    URL_FILE.write_text(report.url + "\n")
    print(report.url)


if __name__ == "__main__":
    main()
