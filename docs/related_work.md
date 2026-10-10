# Related work (first pass, 2026-10-10)

This is a snippet-level search.
Check each primary paper before citing it.

## Closest works

| Work | Overlap | Difference |
|---|---|---|
| [Q-Align](https://arxiv.org/abs/2312.17090) | Scores from token probabilities over rating levels | Uses the LM vocabulary, scores only, no calibration eval |
| [VideoScore](https://arxiv.org/abs/2406.15252) | Video quality metric, source of VideoFeedback | Generated-video quality only, no audio, no typed questions |
| [LMM-VQA](https://arxiv.org/abs/2408.14008) | LMM video quality scoring | Generates the score as text |
| [UVE](https://arxiv.org/abs/2503.09949) | Yes/no token probabilities score AI video | Scores only, no learned head, no timestamps |
| [DPC-VQA](https://arxiv.org/abs/2604.12813) | Logits over ordered quality labels, with calibration | Single task; closest to our score head |
| [Engagement with LMMs](https://arxiv.org/abs/2508.02516) | SnapUGC engagement, audio helps | Generated-text numbers, no head, no calibration |
| [v-HUB](https://arxiv.org/abs/2509.25773) | Video humour with audio | Benchmark, not a calibrated model |
| [Your VLM Already Knows When](https://arxiv.org/abs/2608.08315) | Grounding by yes/no token readout | Training-free, no soft targets |
| [UniVTG](https://arxiv.org/abs/2307.16715) | Unified grounding across tasks | Task-specific heads, not one LLM answering typed questions |
| [ReVisionLLM](https://arxiv.org/abs/2411.14901) | ECE on grounding confidence | Hierarchical grounding model, no option-span head |
| [TimeChat](https://arxiv.org/abs/2312.02051), [VTimeLLM](https://arxiv.org/abs/2311.18445) | Video LLM timestamps | Timestamps generated as text |
| [Hidden Calibration](https://arxiv.org/abs/2406.16535) | Reads hidden states instead of label tokens | Text only |

## What is already done

- Scoring from token probabilities (Q-Align, VideoScore, DPC-VQA, UVE).
- Yes/no readout for grounding (2608.08315).
- Calibrated grounding confidence (ReVisionLLM).
- Audio-aware engagement prediction on SnapUGC (2508.02516).

## What looks new

- One shared head that answers mixed question types in one forward pass.
- Moment questions where each option is a video/audio token block, scored by that head with no decoder.
- Soft-target training (CE, Brier, EMD) with calibration reported for every question type.

No single work found covers all three.
The contribution is the combination, so the paper must show the joint model (G4) works.

## Still to check

- FunQA, E.T. Bench, GitHub and Papers with Code.
- VideoScore's head design and Q-Align's softmax details, from the primary papers.
