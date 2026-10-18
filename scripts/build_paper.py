"""Builds docs/video_jev_paper.html; print to PDF with headless Chrome. Charts are drawn from the numbers below so they stay to scale."""

import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "video_jev_paper.html"
INK, MUTE, RULE, ACC, REF, ZS = "#1c2230", "#5b6475", "#d9dde5", "#1f5fbf", "#c9822b", "#9aa3b2"


def hbars(rows, lo=0.0, hi=1.0, w=640, label_w=210, row_h=46, ticks=(0, .25, .5, .75, 1.0), legend=()):
    """rows: (label, sublabel, [(value, color, text)])"""
    plot_w = w - label_w - 125
    x = lambda v: label_w + (v - lo) / (hi - lo) * plot_w
    h = 26 + row_h * len(rows) + 24
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for t in ticks:
        s.append(f'<line x1="{x(t):.1f}" y1="18" x2="{x(t):.1f}" y2="{h - 22}" stroke="{RULE}"/>')
        s.append(f'<text x="{x(t):.1f}" y="{h - 6}" font-size="11" fill="{MUTE}" text-anchor="middle">{t:g}</text>')
    lx = label_w
    for name, col in legend:
        s.append(f'<rect x="{lx}" y="2" width="10" height="10" fill="{col}"/>'
                 f'<text x="{lx + 14}" y="11" font-size="11" fill="{MUTE}">{name}</text>')
        lx += 16 + 6.3 * len(name) + 18
    for i, (name, sub, bars) in enumerate(rows):
        y0 = 26 + i * row_h
        s.append(f'<text x="{label_w - 10}" y="{y0 + 15}" font-size="12.5" fill="{INK}" text-anchor="end" font-weight="600">{name}</text>')
        s.append(f'<text x="{label_w - 10}" y="{y0 + 30}" font-size="10.5" fill="{MUTE}" text-anchor="end">{sub}</text>')
        bh = (row_h - 10) / len(bars)
        for j, (v, col, txt) in enumerate(bars):
            yb = y0 + j * bh
            s.append(f'<rect x="{x(lo):.1f}" y="{yb:.1f}" width="{x(v) - x(lo):.1f}" height="{bh - 2:.1f}" fill="{col}"/>')
            s.append(f'<text x="{x(v) + 4:.1f}" y="{yb + bh - 4:.1f}" font-size="10.5" fill="{INK}" font-family="IBM Plex Mono, monospace">{txt}</text>')
    s.append("</svg>")
    return "".join(s)


def f3(v):
    return f"{v:.3f}"


# Main results: the released joint model (G4) on official test splits.
main_rows = [
    ("AI-video quality", "VideoFeedback · Spearman", 0.340, 0.743, (0.771, "VideoScore")),
    ("Real or AI", "VideoFeedback · AUC", 0.905, 0.996, None),
    ("Text-query moment", "QVHighlights · R1@0.5", 0.528, 0.679, (0.539, "Moment-DETR")),
    ("Funny or magic moment", "FunQA · HIT@1", 0.504, 0.776, (0.767, "single-task")),
    ("Which sound", "VGGSound · accuracy", 0.503, 0.590, (0.125, "chance")),
    ("Laughing? (never trained)", "VGGSound · AUC", 0.873, 0.970, None),
]
chart_main = hbars(
    [(n, sub, [(z, ZS, f3(z)), (o, ACC, f3(o))] + ([(r[0], REF, f"{f3(r[0])} {r[1]}")] if r else []))
     for n, sub, z, o, r in main_rows],
    legend=(("Untrained Qwen3-Omni", ZS), ("Video Jev (G4)", ACC), ("Reference", REF)), row_h=54)

# AI-video quality per question, official test split.
vf_rows = [("Visual quality", .312, .822, .847, .862), ("Temporal consistency", .302, .786, .822, .803),
           ("Dynamic degree", .226, .694, .706, .776), ("Text alignment", .407, .599, .660, .594),
           ("Factual consistency", .453, .813, .838, .821), ("Mean", .340, .743, .775, .771)]
chart_vf = hbars([(n, "", [(z, ZS, f3(z)), (j, ACC, f3(j)), (s, "#6f93d6", f3(s)), (r, REF, f3(r))])
                  for n, z, j, s, r in vf_rows],
                 legend=(("Untrained", ZS), ("Joint", ACC), ("VideoFeedback only", "#6f93d6"),
                         ("VideoScore", REF)), row_h=62, label_w=170)

# Audio on or off, same recipe, single-task runs (val).
audio = [("Which sound", "VGGSound, accuracy", .753, .817), ("Funny moment", "FunQA, HIT@1", .792, .807),
         ("Engagement", "SnapUGC 10k, final", .628, .654)]
chart_audio = hbars([(n, sub, [(v, ZS, f3(v)), (a, ACC, f3(a))]) for n, sub, v, a in audio],
                    lo=0.5, hi=0.9, ticks=(.5, .6, .7, .8, .9),
                    legend=(("Video only", ZS), ("Video + audio", ACC)), row_h=44, label_w=170)


def moment_diagram():
    w, h = 640, 190
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    probs = [.02, .03, .04, .05, .10, .26, .30, .12, .04, .02, .01, .01]
    x0, bw = 60, 44
    s.append(f'<text x="10" y="40" font-size="11" fill="{MUTE}">P(moment)</text>')
    for i, p in enumerate(probs):
        x = x0 + i * bw
        hh = p * 300
        col = ACC if 5 <= i <= 6 else ZS
        s.append(f'<rect x="{x + 4}" y="{110 - hh:.1f}" width="{bw - 8}" height="{hh:.1f}" fill="{col}"/>')
        s.append(f'<rect x="{x + 2}" y="118" width="{bw - 4}" height="22" rx="3" fill="#eef1f6" stroke="{RULE}"/>')
        s.append(f'<text x="{x + bw / 2}" y="133" font-size="9.5" fill="{INK}" text-anchor="middle">V A</text>')
        s.append(f'<text x="{x + bw / 2}" y="156" font-size="10" fill="{MUTE}" text-anchor="middle">{i}s</text>')
    s.append(f'<rect x="{x0 + 5 * bw}" y="114" width="{2 * bw}" height="30" fill="none" stroke="{REF}" stroke-width="2" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{x0 + 6 * bw}" y="178" font-size="11" fill="{REF}" text-anchor="middle">true funny moment (5 to 7 s)</text>')
    s.append(f'<text x="10" y="133" font-size="11" fill="{MUTE}">tokens</text>')
    s.append("</svg>")
    return "".join(s)


def pipeline():
    steps = [("Raw mp4", "any length or codec"), ("Decode once", "frames + 16 kHz audio, cached"),
             ("Jev record", "state + typed questions as text"), ("Omni + LoRA", "Qwen3-Omni: frames and sound on one timeline"),
             ("Typed head", "one logit per allowed answer"), ("Probabilities", "per question, calibrated")]
    w, h, bw, gap = 660, 92, 96, 17
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i, (t, sub) in enumerate(steps):
        x = 4 + i * (bw + gap)
        fill = "#eaf1fc" if i in (3, 4) else "#fff"
        s.append(f'<rect x="{x}" y="8" width="{bw}" height="74" rx="5" fill="{fill}" stroke="{INK if i in (3, 4) else RULE}"/>')
        s.append(f'<text x="{x + bw / 2}" y="30" font-size="11" font-weight="600" fill="{INK}" text-anchor="middle">{t}</text>')
        words, line, lines = sub.split(), "", []
        for wd in words:
            if len(line) + len(wd) > 15:
                lines.append(line)
                line = wd
            else:
                line = f"{line} {wd}".strip()
        lines.append(line)
        for k, ln in enumerate(lines):
            s.append(f'<text x="{x + bw / 2}" y="{48 + 12 * k}" font-size="9.5" fill="{MUTE}" text-anchor="middle">{ln}</text>')
        if i < len(steps) - 1:
            s.append(f'<text x="{x + bw + gap / 2}" y="49" font-size="13" fill="{MUTE}" text-anchor="middle">→</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root { --ink:#1c2230; --mute:#5b6475; --rule:#d9dde5; --soft:#f3f5f8; --accent:#1f5fbf; --good:#1e7a4c; --warn:#a4561a; --bad:#a12a2a; }
@page { size: A4; margin: 15mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "IBM Plex Sans", system-ui, sans-serif; color: var(--ink); background: #fff; font-size: 10pt; line-height: 1.45; margin: 0; }
h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif; text-wrap: balance; margin: 0; }
h1 { font-size: 25pt; line-height: 1.1; margin-top: 4pt; }
h2 { font-size: 15pt; margin-top: 16pt; padding-bottom: 3pt; border-bottom: 1.5px solid var(--ink); break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 10pt; break-after: avoid; }
p { margin: 5pt 0; max-width: 72ch; }
ul, ol { margin: 4pt 0; padding-left: 16pt; }
li { margin: 2pt 0; max-width: 74ch; }
.label { font-size: 7.5pt; letter-spacing: .09em; text-transform: uppercase; color: var(--mute); font-weight: 600; }
.sub { color: var(--mute); font-size: 11.5pt; margin-top: 6pt; max-width: 60ch; }
.meta { display: flex; flex-wrap: wrap; gap: 16pt; margin-top: 10pt; font-size: 8.5pt; color: var(--mute); }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.8pt; }
tr { break-inside: avoid; }
th, td { text-align: left; padding: 3.2pt 5pt; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; font-size: 7.6pt; color: var(--mute); text-transform: uppercase; letter-spacing: .05em; border-bottom: 1.5px solid var(--ink); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", monospace; font-size: 8.4pt; white-space: nowrap; }
tr.us td { background: #eaf1fc; font-weight: 600; }
code, .mono { font-family: "IBM Plex Mono", monospace; font-size: 8.4pt; }
pre { font-family: "IBM Plex Mono", monospace; font-size: 8pt; background: var(--soft); padding: 7pt 9pt; border-radius: 4pt; white-space: pre-wrap; margin: 6pt 0; break-inside: avoid; }
.box { background: var(--soft); border-left: 3px solid var(--accent); padding: 8pt 12pt; margin-top: 12pt; break-inside: avoid; }
.box li, .box p { max-width: none; }
.warnbox { border-left-color: var(--warn); }
.note { font-size: 8.3pt; color: var(--mute); }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 14pt; break-inside: avoid; }
.two > div { min-width: 0; }
figure { margin: 8pt 0; break-inside: avoid; }
figcaption { font-size: 8.3pt; color: var(--mute); margin-top: 3pt; }
.tag { display: inline-block; font-size: 7.3pt; padding: 1pt 5pt; border-radius: 3pt; font-weight: 600; white-space: nowrap; }
.ok { background: #e3f3ea; color: var(--good); }
.mid { background: #fbeee2; color: var(--warn); }
.no { background: #f6e3e3; color: var(--bad); }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8pt; margin: 10pt 0; break-inside: avoid; }
.kpi { border: 1px solid var(--rule); border-radius: 5pt; padding: 7pt 9pt; }
.kpi .v { font-family: "Source Serif 4", serif; font-size: 19pt; font-weight: 600; color: var(--accent); line-height: 1.1; }
.kpi .k { font-size: 8pt; color: var(--mute); margin-top: 2pt; }
.pb { break-before: page; }
.uc { border: 1px solid var(--rule); border-radius: 5pt; padding: 7pt 10pt; break-inside: avoid; }
.uc h3 { margin: 0 0 3pt 0; display: flex; justify-content: space-between; gap: 8pt; align-items: baseline; }
.uc q { font-style: italic; color: var(--ink); }
.ucs { display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; }
svg text { font-family: "IBM Plex Sans", sans-serif; }
"""

BODY = f"""
<div class="label">Working paper · 18 October 2026 · not peer reviewed</div>
<h1>Video Jev: one calibrated head for typed questions about video and sound</h1>
<p class="sub">Ask a video a typed question: yes or no, pick one, score it, or "when?". Get back a probability for every possible answer. Many questions in one pass.</p>
<div class="meta"><span>Backbone: Qwen3-Omni-30B-A3B</span><span>Weights: huggingface.co/sp1d5r/video-jev</span><span>Licence: CC BY-NC-SA 4.0</span></div>

<div class="box">
<div class="label">Abstract</div>
<p>We add a LoRA adapter and a typed answer head to Qwen3-Omni, an open model that reads video frames and sound on one timeline.
Questions are written as Jev records: yes/no, choice, ordered score, and a new <code>moment</code> type for "when" questions, where each second of video is an answer option scored on its own video and audio tokens.
One forward pass returns a calibrated probability for every answer of every question.
A single joint model, trained on five public tasks with randomly reworded questions, beats the untrained backbone on every task.
On official test splits it reaches R1@0.5 0.679 on QVHighlights (untrained 0.528, Moment-DETR 0.539), 0.776 HIT@1 on FunQA moments, 0.743 mean Spearman on VideoFeedback (VideoScore 0.771) and 0.996 AUC on real-or-AI.
It loses nothing on question wordings it never saw, and answers a yes/no question it was never trained on with AUC 0.970 (untrained 0.873).
A task-specific model trained the same way reaches 0.775 on VideoFeedback, and 0.709 on SnapUGC engagement, level with the 2025 challenge winner's four-model ensemble.
We release the joint model for non-commercial use.</p>
</div>

<div class="kpis">
<div class="kpi"><div class="v">0.679</div><div class="k">Text-query moment R1@0.5<br>untrained 0.528 · Moment-DETR 0.539</div></div>
<div class="kpi"><div class="v">0.743</div><div class="k">AI-video quality, Spearman<br>VideoScore 0.771</div></div>
<div class="kpi"><div class="v">0.970</div><div class="k">Never-trained question, AUC<br>untrained 0.873</div></div>
<div class="kpi"><div class="v">0.03 to 0.05</div><div class="k">Calibration error, quality<br>untrained 0.29 to 0.56</div></div>
</div>

<h2>1. Introduction</h2>
<p><b>Jev</b> is a format for decisions. A record holds some <i>state</i> (context) and a set of typed <i>questions</i>, each with its allowed answers. A Jev model returns a probability for each answer, and code acts on those numbers: thresholds, rankings, or sending unsure cases to a person. That only works if the probabilities are calibrated, so that "84%" is right about 84% of the time.</p>
<p>Clef, the existing Jev model, reads text and images. Video models today mostly answer in free text, report accuracy only, and give timestamps as generated text with no confidence. We ask whether one head on an open audio-visual model can answer typed questions about raw video, including "when" questions, with probabilities that can be trusted.</p>
<p><b>Contributions.</b></p>
<ul>
<li>A <code>moment</code> question type that reuses the video's own tokens as answer options, so "when" questions need no detector and no timestamp decoding, and return a probability per second.</li>
<li>One joint model for score, yes/no, choice and moment questions across five public tasks, trained with reworded questions, and tested on wordings and a question it never saw.</li>
<li>Test-split results against the untrained backbone and published task-specific systems, with calibration reported for every question type.</li>
<li>A released adapter and head, with a standalone inference script.</li>
</ul>

<h2>2. Method</h2>
<figure>{pipeline()}
<figcaption>Figure 1. Only the LoRA adapters and the head are trained. Frames and audio share one timeline inside Qwen3-Omni.</figcaption></figure>

<h3>2.1 Questions</h3>
<table>
<tr><th>Type</th><th>Answer</th><th>Example</th></tr>
<tr><td><code>noul</code></td><td>P(yes), P(no)</td><td>"Can you hear someone laughing?" "Is this AI-generated?"</td></tr>
<tr><td><code>choice</code></td><td>P per named option</td><td>"Which sound is most prominent: laughing, giggling, cheering, speech...?"</td></tr>
<tr><td><code>score</code></td><td>P per ordered level</td><td>"How good is the visual quality: bad, average, good, perfect?"</td></tr>
<tr><td><code>moment</code></td><td>P per second</td><td>"When is the funny moment?" "When does this happen: a man opens a fridge?"</td></tr>
</table>

<h3>2.2 Typed head</h3>
<p>The record is written as text after the video and audio tokens, with each question's options listed in the prompt. We use the joint schema head from Clef: it reads the backbone's final hidden states over each option's tokens and gives that option one logit. A softmax over a question's options gives its probabilities. All questions in a record are answered in one forward pass.</p>

<h3>2.3 Moment questions</h3>
<p>Qwen3-Omni places video tokens in blocks of two frames, with each block's audio tokens next to it. For a moment question, each time bin is an option, and its tokens are that bin's own video and audio tokens. The head is unchanged; it scores every bin against the question. The answer is a distribution over seconds, from which we take the top moment, a span around the peak, or the top few candidates.</p>
<figure>{moment_diagram()}
<figcaption>Figure 2. Each block of video (V) and audio (A) tokens is one answer option. Schematic.</figcaption></figure>

<h3>2.4 Training</h3>
<ul>
<li><b>Trainable:</b> LoRA (rank 64, alpha 128) on the language model's attention projections, plus the head in fp32. The rest of Qwen3-Omni-30B-A3B is frozen in bf16.</li>
<li><b>Loss:</b> cross-entropy on soft targets (label smoothing 0.05) + Brier score + an ordinal earth mover's distance for score and moment questions. Moment targets weight each bin by its overlap with the answer window, or by human highlight scores where they exist.</li>
<li><b>Reworded questions:</b> every question has five wordings. Training draws one of the first three at random each time a clip is seen; the last two are kept for testing.</li>
<li><b>Recipe:</b> LoRA learning rate 3e-5, head 1e-4, 1 epoch, warmup then cosine, 8-clip gradient accumulation. Chosen on SnapUGC and reused unchanged for every task.</li>
</ul>

<h2>3. Experimental setup</h2>
<table>
<tr><th>Dataset</th><th>Question</th><th>Type</th><th class="n">Train</th><th>Test split</th><th>Input</th></tr>
<tr><td>VideoFeedback</td><td>5 quality scores, 1 to 4</td><td>score</td><td class="n">10,000</td><td>official test, 760</td><td>8 fps, video only</td></tr>
<tr><td>VideoFeedback</td><td>real or AI-generated</td><td>noul</td><td class="n">2,000</td><td>official test, 760</td><td>first 1.5 s, 256 px square</td></tr>
<tr><td>QVHighlights</td><td>when does the query happen</td><td>moment</td><td class="n">6,862</td><td>official val, 1,550</td><td>1 fps, 150 s, audio</td></tr>
<tr><td>FunQA</td><td>funny or magic moment</td><td>moment</td><td class="n">2,135</td><td>official test, 339</td><td>2 fps, 60 s, audio</td></tr>
<tr><td>VGGSound (8 people-sound classes)</td><td>which sound</td><td>choice</td><td class="n">5,506</td><td>official test, 398</td><td>2 fps, 10 s, audio</td></tr>
<tr><td>GenVideo</td><td>real or AI (test only)</td><td>noul</td><td class="n">0</td><td>test, 961</td><td>as above</td></tr>
<tr><td>SnapUGC</td><td>share still watching at 5 s</td><td>score</td><td class="n">30,631*</td><td>held-out 5%, 1,736</td><td>2 fps, 16 s, audio</td></tr>
</table>
<p class="note">*SnapUGC trains a separate single-task model and is not in the joint model, because the dataset has no stated licence. QVHighlights test labels are withheld, so its official val split is our test. Real-or-AI clips all use the same length, crop and size, because in GenVideo every real clip is 10 s long and every generated clip is shorter (section 5).</p>

<p><b>Metrics.</b> Scores: Spearman and Pearson correlation of the expected level with the human score. Yes/no: AUC. Choice: accuracy. Moments: HIT@1 (the top second is inside the answer, or rated "very good" in QVHighlights), R1@0.5 and R1@0.7 (the span of bins at 50% or more of the peak overlaps the answer by that IoU), mIoU and mAP. Calibration: Brier score and expected calibration error (ECE, 10 bins).</p>
<p><b>Baselines.</b> (1) The same Qwen3-Omni, untrained, asked each question as text: we read the probability of each answer word ("1" to "4", "yes"/"no", option letters); for moments it writes a start and end time. (2) Single-task models trained with the same recipe on one dataset each. (3) Published task-specific systems on the same split.</p>
<p><b>Protocol.</b> All choices are made on validation data. Each final model is scored once on each test split. Every number is from one training run.</p>

<h2>4. Results</h2>
<h3>4.1 One joint model</h3>
<figure>{chart_main}
<figcaption>Figure 3. The released joint model (G4) on official test splits, against the untrained backbone and a reference per task. Each task has its own metric on a 0 to 1 scale.</figcaption></figure>
<table>
<tr><th>Task (test clips)</th><th>Metric</th><th class="n">Untrained</th><th class="n">Single-task</th><th class="n">Joint</th><th>Published</th></tr>
<tr><td>AI-video quality (760)</td><td>mean Spearman</td><td class="n">0.340</td><td class="n">0.775</td><td class="n">0.743</td><td>VideoScore 0.771</td></tr>
<tr><td>Real or AI, VideoFeedback (760)</td><td>AUC</td><td class="n">0.905</td><td class="n">-</td><td class="n">0.996</td><td>-</td></tr>
<tr><td>Real or AI, GenVideo (961), unseen generators</td><td>AUC</td><td class="n">-</td><td class="n">-</td><td class="n">0.942</td><td>-</td></tr>
<tr class="us"><td>Text-query moment, QVHighlights (1,550)</td><td>R1@0.5 / R1@0.7</td><td class="n">0.528 / 0.354</td><td class="n">-</td><td class="n">0.679 / 0.463</td><td>Moment-DETR 0.539 / 0.348</td></tr>
<tr><td>QVHighlights</td><td>HIT@1 / mAP</td><td class="n">0.181 / 0.043</td><td class="n">-</td><td class="n">0.358 / 0.055</td><td>Moment-DETR 0.556 / 0.322</td></tr>
<tr><td>Funny or magic moment, FunQA (339)</td><td>HIT@1 / R1@0.5</td><td class="n">0.504 / 0.398</td><td class="n">0.767 / 0.507</td><td class="n">0.776 / 0.504</td><td>-</td></tr>
<tr><td>Which sound, VGGSound (398)</td><td>accuracy</td><td class="n">0.503</td><td class="n">0.598</td><td class="n">0.590</td><td>chance 0.125</td></tr>
</table>
<p class="note">Single-task models use all of their dataset; the joint model uses 10,000 of 35,041 VideoFeedback clips. Moment-DETR numbers are its published single-run val result (QVHighlights supplement, 10 queries).</p>
<ul>
<li>The joint model beats the untrained backbone on every task, by 0.09 to 0.40.</li>
<li>Joint training costs almost nothing where data is equal: FunQA +0.009, VGGSound -0.008. VideoFeedback is 0.032 lower, with a third of the data.</li>
<li>On QVHighlights, R1 is above Moment-DETR, but HIT@1 and mAP are well below it: the model finds the right span but ranks alternative spans and single seconds poorly.</li>
<li>Real-or-AI trained on VideoFeedback carries over to GenVideo's generators (0.942).</li>
</ul>

<h3>4.2 AI-video quality per question</h3>
<figure>{chart_vf}<figcaption>Figure 4. Spearman correlation per question, official test split. VideoScore is a dedicated model trained on all of VideoFeedback.</figcaption></figure>
<ul>
<li>The VideoFeedback-only model is level with VideoScore (0.775 against 0.771): ahead on temporal consistency, text alignment and factual consistency, behind on visual quality and dynamic degree.</li>
<li>The joint model trails it most on text alignment (0.599 against 0.660).</li>
<li>Untrained, the backbone reaches 0.340, already above GPT-4o's published 0.230.</li>
</ul>

<h3>4.3 Questions it was never trained on</h3>
<table>
<tr><th>Test</th><th class="n">Joint</th><th class="n">Untrained</th></tr>
<tr><td>Every question in two wordings never seen in training (mean of headline metrics; trained wordings: 0.730)</td><td class="n">0.728</td><td class="n">-</td></tr>
<tr class="us"><td>"Can you hear someone laughing in this clip?" on VGGSound test (AUC)</td><td class="n">0.970</td><td class="n">0.873</td></tr>
</table>
<p>The laughing question is a new yes/no question about clips seen only with the 8-way "which sound" question. Its labels come from the class names (adult laughter, giggling, baby laughter against speech, crowd and cheering). The model answers it far better than the untrained backbone, so the head follows the question, not only the task. We also held out FunQA's creative moments as an unseen question family, but that test is void: every answer window in that subset starts within the first 3 seconds, so it measures position, not content.</p>

<h3>4.4 Calibration</h3>
<table>
<tr><th>Question (test)</th><th class="n">Untrained ECE</th><th class="n">Joint ECE</th></tr>
<tr><td>VideoFeedback, 5 scores</td><td class="n">0.29 to 0.56</td><td class="n">0.031 to 0.045</td></tr>
<tr><td>Real or AI</td><td class="n">0.034</td><td class="n">0.023</td></tr>
<tr><td>Which sound</td><td class="n">-</td><td class="n">0.157</td></tr>
<tr><td>Laughing (never trained)</td><td class="n">0.112</td><td class="n">0.101</td></tr>
</table>
<p>The training loss alone, with no temperature scaling, brings score questions from badly calibrated to ECE under 0.05. VGGSound is the exception: its official test is harder than its val for every model (the untrained backbone drops from 0.66 to 0.50 accuracy), and calibration suffers with it.</p>

<h3>4.5 Does audio help?</h3>
<div class="two">
<div><figure>{chart_audio}<figcaption>Figure 5. Single-task models, audio on or off, val. Axis starts at 0.5.</figcaption></figure></div>
<div>
<ul>
<li>When the answer is in the sound (VGGSound), audio adds 6.4 points and cuts ECE from 0.058 to 0.021.</li>
<li>When it is visual (FunQA pranks and tricks), audio adds little.</li>
<li>For engagement, the gain is inside the noise.</li>
</ul>
</div>
</div>

<h3>4.6 Long video</h3>
<div class="two">
<div>
<table>
<tr><th>Length</th><th class="n">HIT@1</th><th class="n">HIT@3</th><th class="n">Chance</th><th class="n">Time</th></tr>
<tr><td>1 min</td><td class="n">0.32</td><td class="n">0.42</td><td class="n">0.08</td><td class="n">2.2 s</td></tr>
<tr><td>3 min</td><td class="n">0.23</td><td class="n">0.25</td><td class="n">0.03</td><td class="n">6.6 s</td></tr>
<tr><td>5 min</td><td class="n">0.18</td><td class="n">0.23</td><td class="n">0.018</td><td class="n">11.0 s</td></tr>
<tr><td>10 min</td><td class="n">0.13</td><td class="n">0.20</td><td class="n">0.009</td><td class="n">22.5 s</td></tr>
</table>
</div>
<div>
<p>We hide one FunQA moment among VGGSound speech and crowd clips, score 60 s windows with 30 s overlap, and take each second's highest logit across windows (60 val videos per length, one H200). The method works, but accuracy falls with length: at 10 minutes the right second is found about 14 times more often than chance, against 0.78 for single clips.</p>
</div>
</div>

<h3>4.7 Engagement (single-task, not released)</h3>
<div class="two">
<div>
<table>
<tr><th>SnapUGC (test, 1,736)</th><th class="n">Final</th><th class="n">Brier</th><th class="n">ECE</th></tr>
<tr class="us"><td>Qwen3-Omni + typed head</td><td class="n">0.709</td><td class="n">0.818</td><td class="n">0.028</td></tr>
<tr><td>CLIP frames + MLP</td><td class="n">0.643</td><td class="n">0.843</td><td class="n">0.034</td></tr>
<tr><td>Text only (XGBoost)</td><td class="n">0.347</td><td class="n">0.884</td><td class="n">0.029</td></tr>
<tr><td>Clef-flash, untrained</td><td class="n">0.127</td><td class="n">0.912</td><td class="n">0.064</td></tr>
</table>
</div>
<div>
<p>The question asks what share of viewers keep watching past 5 seconds, as a 10-level score, with title and description in the state. The 2025 challenge winner, a four-model ensemble, scored 0.710 on the official test, whose labels are withheld, so ours is on a held-out 5% of the public training labels: "level with", not "beats". The gap to CLIP is +0.065 (95% CI 0.041 to 0.088).</p>
</div>
</div>

<h2>5. Analysis</h2>
<ul>
<li><b>Dataset shortcuts.</b> In GenVideo every real clip is 10 s at 640x360 and every generated clip is 7.2 s or shorter, and our first record put the duration in the state. A first model and a CLIP baseline both reached 0.999. All real-or-AI clips now use the first 1.5 s, a square crop and 256 px, with no duration. A baseline near 1.0 is a warning, not a result.</li>
<li><b>Learning rate decides stability.</b> A LoRA rate near 1e-4 held for 400 or more steps collapsed twice; 3e-5 trained cleanly on every task. Short sweeps cannot show this failure.</li>
<li><b>Data volume matters more than task mixing.</b> The joint model matches single-task models where it sees the same data, and trails only where it sees less.</li>
<li><b>Text-query moments need data.</b> With 2,000 training queries the model was below the untrained backbone (R1@0.5 0.26 on val); with all 6,862 it was well above (0.68 on test).</li>
<li><b>Benchmarks can be void.</b> FunQA's creative subset answers are all at the clip start, so a position prior solves it.</li>
</ul>

<h2>6. Related work</h2>
<p>Each part of this design exists in prior work; we found no work that combines them.</p>
<table>
<tr><th>Work</th><th>What it does</th><th>Difference</th></tr>
<tr><td>Q-Align (2023)</td><td>Quality scores from probabilities over rating words</td><td>Uses the language model's vocabulary; scores only</td></tr>
<tr><td>VideoScore (2024)</td><td>AI-video quality on 5 fixed questions; source of VideoFeedback</td><td>Fixed questions, no audio, no probabilities</td></tr>
<tr><td>DPC-VQA (2026)</td><td>Logits over ordered quality labels, plus calibration</td><td>One task</td></tr>
<tr><td>UVE (2025)</td><td>Yes/no probabilities to score AI video</td><td>No learned head, no "when"</td></tr>
<tr><td>Engagement with LMMs (2025)</td><td>SnapUGC engagement; audio helps</td><td>Numbers as generated text; no calibration</td></tr>
<tr><td>Moment-DETR, UniVTG (2021, 2023)</td><td>Dedicated moment retrieval models</td><td>Task-specific heads, not typed questions</td></tr>
<tr><td>TimeChat, VTimeLLM (2023)</td><td>Video LLMs that write timestamps as text</td><td>One answer, no probability over time</td></tr>
<tr><td>"Your VLM Already Knows When" (2026)</td><td>Moments by yes/no readout, no training</td><td>Untrained</td></tr>
<tr><td>ReVisionLLM (2024)</td><td>Calibration for moment confidence</td><td>Grounding only</td></tr>
<tr><td>Clef (Cloudflare)</td><td>Typed head for text and images</td><td>No video timeline, no audio, no moments</td></tr>
</table>

<h2>7. Limitations</h2>
<ul>
<li><b>One seed</b> per result; no run-to-run spread yet.</li>
<li><b>QVHighlights ranking:</b> HIT@1 and mAP are well below dedicated models.</li>
<li><b>Real-or-AI</b> uses generators from 2023 and 2024. It says nothing about modern generators.</li>
<li><b>Long video</b> loses accuracy with length; the test videos are synthetic splices.</li>
<li><b>Short inputs:</b> 60 s per forward pass by default, 150 s at most in training.</li>
<li><b>Engagement</b> is a separate model on our own split, and is not released.</li>
<li><b>Licences:</b> research and non-commercial use only.</li>
</ul>

<h2>8. Release</h2>
<p>The joint model is released as a LoRA adapter and head (714 MB) at <b>huggingface.co/sp1d5r/video-jev</b>, with a standalone <code>run.py</code>, under CC BY-NC-SA 4.0. A VideoFeedback-only adapter is in <code>videofeedback-only/</code>. It is trained only on datasets with a stated licence; no videos are distributed.</p>
<pre>{{"state": {{}},
 "questions": {{
  "funny_moment": {{"type": "moment", "instructions": "When is the funniest moment in this video?"}},
  "laugh":   {{"type": "noul",  "instructions": "Can you hear someone laughing in this clip?"}},
  "ai":      {{"type": "noul",  "instructions": "Is this clip AI-generated?"}},
  "quality": {{"type": "score", "instructions": "Rate how clean and sharp this video looks.",
              "criteria": ["bad", "average", "good", "perfect"]}}}}}}</pre>
<pre>funny_moment  6-7 s: 0.181   5-6 s: 0.180   4-5 s: 0.171   ...
laugh         yes: 0.198   no: 0.802
ai            yes: 0.022   no: 0.978
quality       bad: 0.012   average: 0.043   good: 0.234   perfect: 0.710</pre>
<p class="note">Real output of the released <code>run.py</code> on a 7 s FunQA test clip whose funny moment is 0 to 6.7 s.</p>
<table>
<tr><th>Input (one H200, batch 1)</th><th class="n">Forward pass</th></tr>
<tr><td>1.5 s clip, video only</td><td class="n">133 ms</td></tr>
<tr><td>10 s clip, video + audio</td><td class="n">206 ms</td></tr>
<tr><td>Up to 60 s, video + audio</td><td class="n">230 ms</td></tr>
<tr><td>150 s, video + audio at 1 fps</td><td class="n">1,140 ms</td></tr>
</table>
<p class="note">Measured during evaluation, after warm-up; decoding not included. bf16 weights need an 80 GB GPU. Every question in the record shares the one forward pass.</p>

<h2>Appendix A. Training details</h2>
<table>
<tr><th>Setting</th><th>Joint model (G4)</th></tr>
<tr><td>Backbone</td><td>Qwen/Qwen3-Omni-30B-A3B-Instruct thinker, bf16, talker off</td></tr>
<tr><td>Adapter</td><td>LoRA r 64, alpha 128, dropout 0.05, q/k/v/o of the language model</td></tr>
<tr><td>Head</td><td>Clef joint schema head, width 1024, 2 routing + 4 layers, 16 heads, fp32, from scratch</td></tr>
<tr><td>Optimiser</td><td>AdamW, weight decay 0.01, LoRA 3e-5, head 1e-4, warmup 50 steps, cosine to 0</td></tr>
<tr><td>Batch</td><td>1 clip, 8-step accumulation, gradient clip 1.0</td></tr>
<tr><td>Data</td><td>26,564 clips, 1 epoch, 3,348 steps, 10 h on one H200</td></tr>
<tr><td>Checkpoint</td><td>best mean val metric (step 3,200)</td></tr>
</table>

<h2>Appendix B. Datasets and licences</h2>
<table>
<tr><th>Dataset</th><th>Licence</th><th>In release</th></tr>
<tr><td>VideoFeedback (TIGER-Lab)</td><td>Apache 2.0</td><td>yes</td></tr>
<tr><td>QVHighlights</td><td>CC BY-NC-SA 4.0 (annotations)</td><td>yes</td></tr>
<tr><td>FunQA</td><td>CC BY-NC-SA 4.0</td><td>yes</td></tr>
<tr><td>VGGSound</td><td>CC BY 4.0</td><td>yes</td></tr>
<tr><td>GenVideo</td><td>none stated</td><td>test only</td></tr>
<tr><td>SnapUGC</td><td>none stated</td><td>no (separate model)</td></tr>
</table>
<p class="note">Base model Qwen3-Omni: Apache 2.0. Clef's head code: Apache 2.0 (Cloudflare). No private data is used.</p>
"""


OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Video Jev working paper</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style></head><body>{BODY}</body></html>""")
print(OUT)
