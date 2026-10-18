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


# 1. Training against no training, per task.
tasks = [
    ("Engagement", "SnapUGC · test · final", 0.127, 0.709, (0.643, "CLIP + MLP")),
    ("AI-video quality", "VideoFeedback · test · Spearman", 0.366, 0.775, (0.771, "VideoScore")),
    ("Funny moment", "FunQA · val · HIT@1", 0.545, 0.807, (0.533, "position prior")),
    ("Which sound", "VGGSound · val · accuracy", 0.659, 0.817, (0.125, "chance")),
    ("Real or AI, new sources", "VideoFeedback · val · AUC", 0.916, 0.929, (0.934, "CLIP + MLP")),
    ("Text-query moment", "QVHighlights · G4 · R1@0.5", 0.528, 0.679, (0.539, "Moment-DETR")),
]
chart_tasks = hbars(
    [(n, sub, [(z, ZS, f3(z)), (o, ACC, f3(o)), (r, REF, f"{f3(r)} {rn}")]) for n, sub, z, o, (r, rn) in tasks],
    legend=(("No training", ZS), ("Ours, trained", ACC), ("Reference", REF)), row_h=54)

# 2. G0 per question.
g0 = [("Visual quality", .341, .847, .862), ("Temporal consistency", .379, .822, .803),
      ("Dynamic degree", .218, .706, .776), ("Text alignment", .452, .660, .594),
      ("Factual consistency", .442, .838, .821), ("Average", .366, .775, .771)]
chart_g0 = hbars([(n, "", [(z, ZS, f3(z)), (o, ACC, f3(o)), (r, REF, f3(r))]) for n, z, o, r in g0],
                 legend=(("No training (val)", ZS), ("Ours (test)", ACC), ("VideoScore (test)", REF)),
                 row_h=44, label_w=170)

# 3. Audio effect.
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
<div class="label">Working paper · draft of 18 October 2026 · not peer reviewed</div>
<h1>Video Jev: one calibrated head for typed questions about video and sound</h1>
<p class="sub">Ask a video a typed question: yes or no, pick one, score it, or "when?". Get back a probability for every possible answer. Many questions in one pass.</p>
<div class="meta"><span>Backbone: Qwen3-Omni-30B-A3B (Apache 2.0)</span><span>Compute: Modal, 1 × H200</span><span>Spend to date: $337 billed</span><span>Weights: huggingface.co/sp1d5r/video-jev</span><span>Code: multimodal-jev</span></div>

<div class="box">
<div class="label">Summary</div>
<ul>
<li><b>What it is.</b> We add a small trained head to an open audio-visual model, Qwen3-Omni. The head reads questions written in Jev form and returns a calibrated probability for every allowed answer. One model, one forward pass, many questions.</li>
<li><b>One model for everything.</b> A single set of weights (G4) answers score, yes/no, choice and moment questions across five tasks, follows wordings it never saw, and answers a yes/no question it was never trained on (AUC 0.970 against 0.873 untrained).</li>
<li><b>Engagement (SnapUGC):</b> 0.709 on held-out test, level with the 2025 challenge winner's 4-model ensemble (0.710, different split).</li>
<li><b>AI-video quality (VideoFeedback):</b> 0.775 average Spearman on the official test split, against VideoScore's 0.771. Ahead on 3 of 5 questions, from a general recipe with no task tuning.</li>
<li><b>"When" questions:</b> a new <code>moment</code> type, with no new parts, finds the funny moment in FunQA 81% of the time, against 53% for the best blind guess.</li>
<li><b>Sound:</b> on "which sound is this?", audio adds 6.4 points and cuts calibration error by 64%.</li>
<li><b>Text-query moments:</b> QVHighlights R1@0.5 0.679 on its val split, against 0.528 untrained and 0.539 for Moment-DETR.</li>
<li><b>What does not work yet.</b> Span ranking (mAP) on QVHighlights; modern AI-video detection; long videos lose accuracy with length (13% at 10 minutes, 15 times chance).</li>
<li><b>Released.</b> The joint model is on Hugging Face (CC BY-NC-SA 4.0), trained only on datasets with a stated licence. $87 of the $150 was spent.</li>
</ul>
</div>

<div class="kpis">
<div class="kpi"><div class="v">0.709</div><div class="k">Engagement, test<br>2025 winner: 0.710</div></div>
<div class="kpi"><div class="v">0.775</div><div class="k">AI-video quality, test<br>VideoScore: 0.771</div></div>
<div class="kpi"><div class="v">0.679</div><div class="k">Text-query moment R1@0.5<br>untrained: 0.528</div></div>
<div class="kpi"><div class="v">0.970</div><div class="k">Never-trained question, AUC<br>untrained: 0.873</div></div>
</div>

<h2>1. What we are building</h2>
<p><b>Jev</b> is a format for decisions. A record holds some <i>state</i> (context) and a set of typed <i>questions</i>. Each question lists its allowed answers. A Jev model returns a probability for each answer, and code acts on those numbers.</p>
<p>Clef, the existing Jev model, reads text and images. <b>Video Jev</b> reads raw video with its sound. The goal is a general head: write a new question in Jev form, and the model answers it with a probability you can trust.</p>
<figure>{pipeline()}
<figcaption>The pipeline. Only the LoRA adapters and the head are trained. The rest of Qwen3-Omni is frozen.</figcaption></figure>

<h3>Four question types</h3>
<table>
<tr><th>Type</th><th>Answer</th><th>Example questions</th><th>Shown on</th></tr>
<tr><td><code>noul</code></td><td>P(yes), P(no)</td><td>"Does this look AI-generated?" "Did the crowd cheer?"</td><td>GenVideo, VideoFeedback</td></tr>
<tr><td><code>choice</code></td><td>P per named option</td><td>"Which sound is this: laughing, giggling, cheering, speech...?"</td><td>VGGSound</td></tr>
<tr><td><code>score</code></td><td>P per ordered level</td><td>"How good is the visual quality, 1 to 4?" "What share of viewers keep watching?"</td><td>SnapUGC, VideoFeedback</td></tr>
<tr><td><code>moment</code> <span class="tag mid">new</span></td><td>P per second</td><td>"When is the funny moment?" "When does the goal go in?"</td><td>FunQA, QVHighlights</td></tr>
</table>

<h3>An example record (illustrative)</h3>
<div class="two">
<div><pre>{{"state": {{"title": "my dog vs the hoover"}},
 "questions": {{
  "keeps_watching": {{"type": "score",
     "levels": ["0-10%", ..., "90-100%"]}},
  "got_a_laugh": {{"type": "noul",
     "instructions": "Did this get a laugh?"}},
  "funniest": {{"type": "moment",
     "instructions": "When is the funniest moment?"}},
  "sound": {{"type": "choice", "options":
     ["laughing", "cheering", "speech", ...]}}}}}}</pre></div>
<div><pre>keeps_watching  60-70%: 0.31  70-80%: 0.27 ...
got_a_laugh     yes: 0.84   no: 0.16
funniest        5s: 0.26  6s: 0.30  7s: 0.12 ...
sound           laughing: 0.71  speech: 0.18 ...

one forward pass, about 0.2 s on one H200</pre>
<p class="note">Illustrative only. This shows the shape of the input and output, not a real model output. Each question here has been trained separately; G4 trains them together.</p></div>
</div>

<h3>Why probabilities, and why calibration</h3>
<p>A model that says "yes" is hard to build on. A model that says "yes, 84%" lets code set thresholds, rank clips, and send unsure cases to a person. That only works if 84% means right about 84% of the time. We measure this with <b>Brier score</b> (squared error of the probabilities) and <b>ECE</b> (expected calibration error: the gap between stated confidence and actual accuracy). Most video models report accuracy only.</p>

<h2>2. How it works</h2>
<h3>The head</h3>
<p>The answer options are written in the prompt. The head reads the model's hidden states over each option's tokens and gives that option one logit. A softmax over a question's options gives its probabilities. All questions in a record are answered in one forward pass, so ten questions cost about the same as one.</p>

<h3>The <code>moment</code> trick</h3>
<p>Qwen3-Omni lays video and audio tokens on one timeline, in 2-frame blocks. For a "when" question, each time bin <i>is</i> an option. The option's tokens are that bin's own video and audio tokens. The same head scores every bin, so the answer is a probability for every second. This needs no detector and no generated timestamps, and it can report the top few moments and how sure it is.</p>
<figure>{moment_diagram()}
<figcaption>Each block of video (V) and audio (A) tokens is one answer option. The head gives a probability per second (bars). Illustrative shape; real FunQA results are in section 3.</figcaption></figure>

<h3>Training</h3>
<ul>
<li><b>Trainable:</b> LoRA (rank 64) on the language model's attention, plus the head. About 0.2% of parameters.</li>
<li><b>Loss:</b> cross-entropy with soft targets, plus Brier score, plus an ordinal distance (EMD) for score and moment questions. Calibration comes from this loss; no step after training is needed.</li>
<li><b>Recipe:</b> LoRA learning rate 3e-5, head 1e-4, 1 epoch, cosine schedule, 8-clip gradient accumulation. Found on SnapUGC and reused unchanged on every new task.</li>
<li><b>Splits:</b> official test sets where they exist, scored once per final model. Choices are made on val only.</li>
</ul>

<h2>3. Results</h2>
<figure>{chart_tasks}
<figcaption>Every task on one 0 to 1 scale, each with its own metric. "No training" is the same Qwen3-Omni asked the question as text, reading the probability of each answer word (SnapUGC uses untrained Clef-flash). SnapUGC and VideoFeedback on test; the rest on val. QVHighlights is the joint model (G4) on its official val split, 1,550 queries.</figcaption></figure>

<h3>3.0 One model for every question type (G4, the release model)</h3>
<p>One set of weights, trained once on five tasks with every question asked in three wordings. Official test splits, each scored once. Single-task runs and untrained Omni are scored on the same clips.</p>
<table>
<tr><th>Task (test clips)</th><th>Metric</th><th class="n">Untrained</th><th class="n">Single-task</th><th class="n">Joint (G4)</th><th>Reference</th></tr>
<tr><td>AI-video quality (760)</td><td>mean Spearman</td><td class="n">0.340</td><td class="n">0.775</td><td class="n">0.743</td><td>VideoScore 0.771</td></tr>
<tr><td>Real or AI, VideoFeedback (760)</td><td>AUC</td><td class="n">0.905</td><td class="n">-</td><td class="n">0.996</td><td>-</td></tr>
<tr><td>Real or AI, GenVideo, never trained on (961)</td><td>AUC</td><td class="n">-</td><td class="n">-</td><td class="n">0.942</td><td>-</td></tr>
<tr class="us"><td>Text-query moment, QVHighlights (1,550)</td><td>R1@0.5 / R1@0.7</td><td class="n">0.528 / 0.354</td><td class="n">-</td><td class="n">0.679 / 0.463</td><td>Moment-DETR 0.539 / 0.348</td></tr>
<tr><td>Funny and magic moment, FunQA (339)</td><td>HIT@1</td><td class="n">0.504</td><td class="n">0.767</td><td class="n">0.776</td><td>-</td></tr>
<tr><td>Which sound, VGGSound (398)</td><td>accuracy</td><td class="n">0.503</td><td class="n">0.598</td><td class="n">0.590</td><td>chance 0.125</td></tr>
</table>
<p class="note">QVHighlights "test" is its official val split; Moment-DETR numbers are its published val result. Our mAP is far lower (0.055 against 0.322): the span ranking is weak. VGGSound's official test is much harder than its val for every model (untrained 0.50 against 0.66).</p>
<div class="two">
<div>
<table>
<tr><th>Generality test</th><th class="n">Joint</th><th class="n">Untrained</th></tr>
<tr><td>Every question in 2 wordings never trained on (trained wordings: 0.730)</td><td class="n">0.728</td><td class="n">-</td></tr>
<tr class="us"><td>"Can you hear someone laughing?", never trained (AUC)</td><td class="n">0.970</td><td class="n">0.873</td></tr>
<tr><td>FunQA creative moments, held out</td><td class="n">void</td><td class="n">void</td></tr>
</table>
<p class="note">The creative-moment test is void: every answer window in that FunQA subset starts in the first 3 s, so it measures position, not content.</p>
</div>
<div>
<table>
<tr><th>Long video (60 per length)</th><th class="n">HIT@1</th><th class="n">Chance</th><th class="n">Time</th></tr>
<tr><td>1 min</td><td class="n">0.32</td><td class="n">0.08</td><td class="n">2.2 s</td></tr>
<tr><td>3 min</td><td class="n">0.23</td><td class="n">0.03</td><td class="n">6.6 s</td></tr>
<tr><td>5 min</td><td class="n">0.18</td><td class="n">0.018</td><td class="n">11.0 s</td></tr>
<tr><td>10 min</td><td class="n">0.13</td><td class="n">0.009</td><td class="n">22.5 s</td></tr>
</table>
<p class="note">One FunQA moment hidden among speech and crowd clips; 60 s windows, 30 s overlap, one H200.</p>
</div>
</div>
<ul>
<li><b>One model answers all four question types.</b> Joint training cost almost nothing where data was equal (FunQA +0.009, VGGSound -0.008). VideoFeedback is 0.032 below the single-task model, which saw 3.5 times more of its clips.</li>
<li><b>QVHighlights went from our weakest task to a clear win</b> once it had full data: R1@0.5 0.679 against 0.528 untrained.</li>
<li><b>The head follows the question, not just the task:</b> new wordings cost nothing, and a never-trained yes/no question beats the untrained model (0.970 against 0.873).</li>
</ul>

<h3>3.1 Engagement: SnapUGC (test, 1,736 clips, scored once)</h3>
<div class="two">
<div>
<table>
<tr><th>Model</th><th class="n">Final</th><th class="n">Brier</th><th class="n">ECE</th></tr>
<tr class="us"><td>Omni + typed head</td><td class="n">0.709</td><td class="n">0.818</td><td class="n">0.028</td></tr>
<tr><td>CLIP frames + MLP</td><td class="n">0.643</td><td class="n">0.843</td><td class="n">0.034</td></tr>
<tr><td>Text only (XGBoost)</td><td class="n">0.347</td><td class="n">0.884</td><td class="n">0.029</td></tr>
<tr><td>Clef-flash, untrained</td><td class="n">0.127</td><td class="n">0.912</td><td class="n">0.064</td></tr>
<tr><td>2025 winner, 4-model ensemble*</td><td class="n">0.710</td><td class="n">-</td><td class="n">-</td></tr>
</table>
<p class="note">Final = mean of Spearman and Pearson. *Official test split; ours is a different held-out split, so "level with", not "beats". Gap to CLIP: +0.065, 95% CI 0.041 to 0.088.</p>
</div>
<div>
<p>The question: what share of viewers keep watching past 5 seconds, as a 10-level score. 30,631 training clips, video + audio + title.</p>
<p>One model matches a four-model ensemble. Its probabilities are calibrated with no extra step.</p>
</div>
</div>

<h3>3.2 AI-video quality: VideoFeedback</h3>
<figure style="max-width:80%">{chart_g0}</figure>
<div>
<p>Five score questions per clip, all in one pass. 35,041 training clips, 4,000 steps.</p>
<ul>
<li><b>Official test split, 760 clips, scored once:</b> 0.775 against VideoScore's 0.771.</li>
<li>Ahead on temporal consistency, text alignment and factual consistency.</li>
<li>Behind on visual quality (0.847 against 0.862) and dynamic degree, the amount of motion (0.706 against 0.776).</li>
<li>Calibration error 0.043 overall. VideoScore gives no probabilities.</li>
<li>Test (0.775) came in above val (0.755). One seed.</li>
</ul>
</div>

<h3>3.3 "When" questions: FunQA and QVHighlights</h3>
<div class="two">
<div>
<table>
<tr><th>FunQA (400 val videos)</th><th class="n">HIT@1</th><th class="n">R1@0.5</th><th class="n">mIoU</th></tr>
<tr><td>Uniform guess</td><td class="n">0.435</td><td class="n">0.320</td><td class="n">0.380</td></tr>
<tr><td>Position prior</td><td class="n">0.533</td><td class="n">0.135</td><td class="n">0.190</td></tr>
<tr><td>Untrained Omni (text answer)</td><td class="n">0.545</td><td class="n">0.295</td><td class="n">0.350</td></tr>
<tr class="us"><td>Ours, video + audio</td><td class="n">0.807</td><td class="n">0.580</td><td class="n">0.538</td></tr>
</table>
<p class="note">HIT@1: the most likely second is inside the true moment. R1@0.5: the predicted span overlaps the true one by IoU 0.5 or more. Funny, creative and magic moments.</p>
</div>
<div>
<table>
<tr><th>QVHighlights (356 val)</th><th class="n">HIT@1</th><th class="n">R1@0.5</th><th class="n">mIoU</th></tr>
<tr><td>Uniform guess</td><td class="n">0.089</td><td class="n">0.121</td><td class="n">0.263</td></tr>
<tr><td>Untrained Omni (text answer)</td><td class="n">0.195</td><td class="n">0.539</td><td class="n">0.520</td></tr>
<tr><td>Ours, small run (2k queries)</td><td class="n">0.213</td><td class="n">0.264</td><td class="n">0.264</td></tr>
<tr class="us"><td>Ours, joint model G4 (all queries, 1,550 test)</td><td class="n">0.358</td><td class="n">0.679</td><td class="n">0.601</td></tr>
</table>
<p class="note">Text queries like "a woman walks her dog". The untrained model was likely trained on this kind of data already. The small run was undertrained; with all the training data (G4) it clearly beats the untrained model.</p>
</div>
</div>

<h3>3.4 Sound: does audio help?</h3>
<figure style="max-width:80%">{chart_audio}<figcaption>Same model and recipe, audio on or off. Axis starts at 0.5.</figcaption></figure>
<div>
<ul>
<li><b>VGGSound</b> (8 people-sound classes: laughing, giggling, baby laughter, cheering, crowd, woman, man and child speech): audio adds 6.4 points, and ECE falls from 0.058 to 0.021. This is the clean audio result.</li>
<li><b>FunQA</b>: no clear gain. Its moments are visual: pranks, tricks, magic.</li>
<li><b>SnapUGC</b>: a small gain inside the noise.</li>
<li>So the model uses sound when the answer is in the sound.</li>
</ul>
</div>

<h3>3.5 Real or AI</h3>
<p>On GenVideo (10 generators from 2023 to 2024) and across to VideoFeedback's sources, our model ties a CLIP baseline (AUC 0.929 against 0.934). These older generators are easy to spot from single frames. A useful test needs modern generators (Sora or Veo class), and no open labelled set of them exists yet.</p>

<h2>4. What we tried, and what we learned</h2>
<table>
<tr><th>Date</th><th>What we tried</th><th>What happened</th><th>Lesson</th></tr>
<tr><td>7 Oct</td><td>Untrained Clef-flash on SnapUGC</td><td>0.127</td><td>Prompting is not enough; training is the whole game.</td></tr>
<tr><td>8 Oct</td><td>Pilot, 10k clips, Qwen3-Omni + head</td><td>0.654 val</td><td>The design works; CLIP + MLP was level, so more data was needed.</td></tr>
<tr><td>8 Oct</td><td>Full run at LoRA rate 1e-4</td><td><span class="tag no">collapsed</span> to 0.19 after 400+ steps</td><td>Short sweeps cannot see this failure. 3e-5 is stable.</td></tr>
<tr><td>8 Oct</td><td>Head in bf16</td><td><span class="tag no">NaN</span> around step 720, twice</td><td>Head now runs in fp32; bad steps are skipped and counted.</td></tr>
<tr><td>9 Oct</td><td>Full run at 3e-5, 30k clips; test once</td><td><span class="tag ok">0.709 test</span></td><td>One model is level with a 4-model ensemble.</td></tr>
<tr><td>9 Oct</td><td>Omni + CLIP ensemble</td><td>+0.001 on test</td><td>Ensembling stops helping as the model gets stronger.</td></tr>
<tr><td>10 Oct</td><td>Same recipe on VideoFeedback</td><td><span class="tag ok">0.653</span> after 2k clips, 0.775 test at full</td><td>The recipe transfers with no tuning.</td></tr>
<tr><td>10 Oct</td><td>Real or AI on GenVideo</td><td><span class="tag no">0.999 shortcut</span></td><td>Real clips were all 10 s, fakes shorter. We now hide length, size and shape. Probe every dataset first.</td></tr>
<tr><td>10 Oct</td><td>New <code>moment</code> type on FunQA</td><td><span class="tag ok">0.807 HIT@1</span></td><td>"When" questions work with no change to the head.</td></tr>
<tr><td>10 Oct</td><td>VGGSound with and without audio</td><td><span class="tag ok">+6.4 pts</span></td><td>First clean audio gain.</td></tr>
<tr><td>10 Oct</td><td>QVHighlights, small run</td><td><span class="tag mid">behind on spans</span></td><td>Long clips (150 s) and a strong untrained model: needs full data and sharper spans.</td></tr>
<tr><td>10 Oct</td><td>Untrained Omni on every task</td><td>0.37 to 0.92</td><td>Training doubles VideoFeedback agreement; untrained probabilities are badly calibrated.</td></tr>
</table>

<h3>What surprised us</h3>
<ul>
<li><b>The recipe did not need retuning.</b> Settings found on engagement worked on quality, moments, sound and real-or-AI.</li>
<li><b>Calibration is close to free.</b> ECE stayed between 0.02 and 0.06 on every trained task, with no temperature scaling. Untrained, it was 0.18 to 0.56.</li>
<li><b>Datasets have traps.</b> One real-or-AI set could be solved from clip length alone. A baseline at 0.999 is a warning, not a result.</li>
</ul>

<h2>5. Where we are now</h2>
<table>
<tr><th>Phase</th><th>Task</th><th>Status</th><th>Result</th></tr>
<tr><td>SnapUGC</td><td>Engagement</td><td><span class="tag ok">done, parked</span></td><td>0.709 test; 3 seeds later</td></tr>
<tr><td>G0</td><td>AI-video quality</td><td><span class="tag ok">done</span></td><td>0.775 test, VideoScore 0.771</td></tr>
<tr><td>G1</td><td>Real or AI</td><td><span class="tag mid">done, tie</span></td><td>Ties CLIP; old generators too easy</td></tr>
<tr><td>G2</td><td>Text-query moments</td><td><span class="tag ok">done in G4</span></td><td>R1@0.5 0.679, untrained 0.528</td></tr>
<tr><td>G3</td><td>Funny moments, sound</td><td><span class="tag ok">done</span></td><td>0.807 HIT@1; audio +6.4 pts</td></tr>
<tr><td>G4</td><td>One joint model, unseen questions</td><td><span class="tag ok">done, released</span></td><td>All tasks in one model; new wordings free</td></tr>
<tr><td>G5</td><td>Long video</td><td><span class="tag mid">small test</span></td><td>10 min: 13% HIT@1, 15x chance</td></tr>
</table>

<h3>Spend so far: $337 billed</h3>
<div class="two">
<div>
<table>
<tr><th>Item</th><th class="n">Approx. cost</th></tr>
<tr><td>SnapUGC: data, pilot, failed runs, full run, test</td><td class="n">$157</td></tr>
<tr><td>Generalisation G0 to G3, zero-shot baselines</td><td class="n">$62</td></tr>
<tr><td>G4 joint model, test scoring, baselines, long video, release (the $150)</td><td class="n">$87</td></tr>
<tr><td>Storage (volumes: videos, caches, checkpoints)</td><td class="n">$32</td></tr>
<tr><td><b>Billed total</b> (after $30 credit and free storage)</td><td class="n"><b>$337</b></td></tr>
</table>
</div>
<div>
<p>About $50 of that was runs we learned from and then replaced: the learning-rate collapse, the NaN incident and the GenVideo shortcut. Those lessons are now built into the code, which is why later phases cost $5 to $25 each.</p>
<p class="note">Storage keeps billing monthly. We can delete raw downloads we no longer need.</p>
</div>
</div>

<h2>6. What the $150 bought</h2>
<p>The plan below was written before the results. Actual cost: $87.</p>
<table>
<tr><th>#</th><th>Step</th><th class="n">Cost</th><th>What you get</th><th>Gate (decided before results)</th></tr>
<tr><td>1</td><td>Score G0 on the official VideoFeedback test split, once</td><td class="n">done</td><td><b>0.775 against VideoScore's 0.771</b> (paid from the raised limit, 17 Oct)</td><td>-</td></tr>
<tr><td>2</td><td>Small joint test: all tasks, 10% of data</td><td class="n">$2</td><td>Killed twice by a dropped client at step 100; every task was learning, so we went straight to the full run</td><td>Replaced by a stop rule at step 800</td></tr>
<tr><td>3</td><td><b>G4 joint model</b>: VideoFeedback, FunQA, QVHighlights (all of it), VGGSound, real-or-AI, with questions reworded in many ways</td><td class="n">$50</td><td><b>Done.</b> One set of weights for all four question types.</td><td>FunQA <span class="tag ok">pass</span> (+0.009); VideoFeedback <span class="tag no">fail</span> (-0.032, a third of the data)</td></tr>
<tr><td>4</td><td>Unseen-question test</td><td class="n">in 5</td><td>New wordings: no drop. New laughing question: 0.970 AUC</td><td><span class="tag ok">pass</span>; creative family void</td></tr>
<tr><td>5</td><td>Score every test split once; untrained baseline on the same clips</td><td class="n">$17</td><td>Section 3.0 and the model card</td><td>-</td></tr>
<tr><td>6</td><td>Long-video small test, release end-to-end checks</td><td class="n">$3</td><td>Section 3.0; a release verified by running it as a user would</td><td>-</td></tr>
<tr><td>7</td><td>Release on Hugging Face</td><td class="n">$0</td><td>sp1d5r/video-jev: adapter + head (714 MB), standalone run.py, model card</td><td>SnapUGC and GenVideo left out: no stated licence</td></tr>
<tr class="us"><td></td><td>Total</td><td class="n">$87 (incl. ~$15 idle storage)</td><td>About $63 of the $150 unspent</td><td></td></tr>
</table>

<h3>How we test "general"</h3>
<ul>
<li><b>New wordings.</b> Each question is trained with several paraphrases, and tested with paraphrases it never saw.</li>
<li><b>New question families.</b> Hold one out completely. For example, train on funny and magic moments, then ask for the <i>creative</i> moment. Or train four VideoFeedback questions, then ask the fifth.</li>
<li>If the model beats its untrained self on these, the head generalises beyond its training questions. That is what makes it a general Jev head, not five task models in one file.</li>
</ul>

<div class="box warnbox">
<div class="label">Still not done</div>
<ul>
<li>More seeds (about $140 for two more), so headline numbers come from one run each.</li>
<li>Better span ranking on QVHighlights (mAP).</li>
<li>Detection of modern AI video and sports goal timing: these need data we do not have.</li>
<li>Long video beyond a small test: accuracy falls with length.</li>
<li>Engagement in the release: SnapUGC has no stated licence.</li>
<li>Commercial use. FunQA is CC BY-NC-SA, so the release is non-commercial.</li>
</ul>
</div>

<h3>What it costs to use</h3>
<table>
<tr><th>Input (one H200, batch 1, measured)</th><th class="n">Forward pass</th><th class="n">Per 1,000 clips</th></tr>
<tr><td>1.5 s clip, video only (real or AI)</td><td class="n">133 ms</td><td class="n">~$0.18</td></tr>
<tr><td>10 s clip, video + audio (VGGSound)</td><td class="n">206 ms</td><td class="n">~$0.27</td></tr>
<tr><td>16 s clip, video + audio + title (SnapUGC)</td><td class="n">290 ms</td><td class="n">~$0.39</td></tr>
<tr><td>Up to 60 s, video + audio (FunQA)</td><td class="n">230 ms</td><td class="n">~$0.31</td></tr>
<tr><td>150 s, video + audio at 1 fps (QVHighlights)</td><td class="n">1,140 ms</td><td class="n">~$1.52</td></tr>
</table>
<p class="note">At about $4.80 per GPU hour; decoding not included. The bf16 model needs an 80 GB GPU. A 4-bit version (about 17 GB, untested) could fit a 24 GB consumer card. Each forward pass answers every question in the record.</p>

<h2>7. Use cases</h2>
<p>Ready means the question type is trained and measured. Partly means one part is shown and the rest needs data. Each use case is a Jev record: you write the questions and get probabilities back.</p>
<div class="ucs">
<div class="uc"><h3>Clip finder for creators <span class="tag ok">ready</span></h3>
<p><q>When is the funniest moment?</q> <q>Will viewers keep watching?</q> Find the best seconds of a video. Engagement scoring works too (section 3.1), but is not in the release because SnapUGC has no stated licence. This follows directly from the clipping-for-distribution work.</p></div>
<div class="uc"><h3>Reviewer for AI video <span class="tag ok">ready</span></h3>
<p><q>Visual quality 1 to 4?</q> <q>Does it match the prompt?</q> <q>Does the physics hold?</q> Score every generation, drop bad takes, and give a generator a reward signal (the generate-then-review loop).</p></div>
<div class="uc"><h3>Sound-aware tagging <span class="tag ok">ready</span></h3>
<p><q>Is that laughing, cheering or speech?</q> <q>Did this get a laugh?</q> Tag clips by what you hear, not just what you see, with a probability per tag.</p></div>
<div class="uc"><h3>Triage with a human in the loop <span class="tag ok">ready</span></h3>
<p>Because the probabilities are calibrated, auto-accept above 95%, auto-reject below 5%, and send the rest to a person. A plain classifier cannot set these thresholds safely.</p></div>
<div class="uc"><h3>Dataset labelling for research <span class="tag ok">ready</span></h3>
<p>Label a large video set with several typed questions in one pass each. Use the uncertain clips for human labelling next (active learning).</p></div>
<div class="uc"><h3>Find a moment by text <span class="tag ok">ready</span></h3>
<p><q>When does the woman open the door?</q> Search inside videos up to a few minutes. R1@0.5 0.679 on QVHighlights; longer videos lose accuracy.</p></div>
<div class="uc"><h3>Sports and live events <span class="tag mid">partly</span></h3>
<p><q>When does the goal go in?</q> <q>Did the crowd cheer?</q> The moment and sound parts both work. It needs labelled match data (SoccerNet needs a signed agreement).</p></div>
<div class="uc"><h3>Spotting modern AI video <span class="tag no">not yet</span></h3>
<p><q>Was this AI-generated?</q> Works on old generators, but so does CLIP. Modern generators need a new labelled set.</p></div>
</div>

<h2>8. Related work</h2>
<p>Each part has been done before. We found no work that combines them. We do not claim novelty; we want a good general head, compared fairly with what exists.</p>
<table>
<tr><th>Work</th><th>What it does</th><th>How we differ</th></tr>
<tr><td>Q-Align (2023)</td><td>Quality scores from probabilities over rating words</td><td>Uses the language model's vocabulary; scores only</td></tr>
<tr><td>VideoScore (2024)</td><td>AI-video quality on 5 fixed questions; source of VideoFeedback</td><td>Our direct baseline. No audio, fixed questions</td></tr>
<tr><td>DPC-VQA (2026)</td><td>Logits over ordered quality labels, plus calibration</td><td>Closest to our score head; one task only</td></tr>
<tr><td>UVE (2025)</td><td>Yes/no probabilities to score AI video</td><td>No learned head, no "when"</td></tr>
<tr><td>Engagement with LMMs (2025)</td><td>SnapUGC engagement; audio helps</td><td>Numbers as generated text; no calibration</td></tr>
<tr><td>Moment-DETR, UniVTG (2021, 2023)</td><td>Dedicated moment-finding models</td><td>Separate heads per task, not typed questions</td></tr>
<tr><td>TimeChat, VTimeLLM (2023)</td><td>Video models that write timestamps as text</td><td>One answer, no probability over time</td></tr>
<tr><td>"Your VLM Already Knows When" (2026)</td><td>Finds moments by yes/no readout, no training</td><td>Untrained; we train with soft targets</td></tr>
<tr><td>ReVisionLLM (2024)</td><td>Reports calibration for moment confidence</td><td>Grounding only</td></tr>
<tr><td>v-HUB (2025)</td><td>Video humour benchmark with audio</td><td>A benchmark, not a model</td></tr>
</table>
<p class="note">First-pass search from paper summaries; full reading is needed before submission. Details in docs/related_work.md.</p>

<h2>9. Limitations</h2>
<ul>
<li><b>One seed per result.</b> We do not yet know the run-to-run spread.</li>
<li><b>Engagement is not in the release.</b> SnapUGC has no stated licence; its 0.709 is a separate model.</li>
<li><b>SnapUGC split differs from the challenge.</b> The official test labels are withheld.</li>
<li><b>Only 32% of SnapUGC videos are still online,</b> and the deleted ones skew to low engagement.</li>
<li><b>Short clips.</b> Up to 16 s for engagement and 150 s for QVHighlights. Long video is untested.</li>
<li><b>Real-or-AI uses old generators</b> and ties CLIP.</li>
<li><b>Licences:</b> research and non-commercial use only. We release weights, never the videos.</li>
</ul>

<h2>10. Datasets and licences</h2>
<table>
<tr><th>Dataset</th><th>Task</th><th class="n">Train clips</th><th>Licence</th></tr>
<tr><td>SnapUGC (VQualA 2025)</td><td>Engagement score</td><td class="n">30,631</td><td>Challenge terms (research)</td></tr>
<tr><td>VideoFeedback (TIGER-Lab)</td><td>5 quality scores, real or AI</td><td class="n">35,041</td><td>Apache 2.0 (to confirm)</td></tr>
<tr><td>GenVideo</td><td>Real or AI</td><td class="n">3,652</td><td>To confirm before release</td></tr>
<tr><td>FunQA</td><td>Funny, creative, magic moments</td><td class="n">3,072</td><td>CC BY-NC-SA 4.0</td></tr>
<tr><td>QVHighlights</td><td>Text-query moments</td><td class="n">6,862</td><td>CC BY-NC-SA 4.0 (to confirm)</td></tr>
<tr><td>VGGSound</td><td>Which sound</td><td class="n">5,506</td><td>CC BY 4.0</td></tr>
</table>
<p class="note">No private data is used. Full run logs: results/README.md. Runs: W&amp;B project sp1d5r/vjev.</p>
"""

OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Video Jev working paper</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style></head><body>{BODY}</body></html>""")
print(OUT)
