---
title: Video Jev
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
models:
  - sp1d5r/video-jev
  - Qwen/Qwen3-Omni-30B-A3B-Instruct
short_description: Typed questions about video and sound, with probabilities
---

# Video Jev demo

Upload a video and ask it typed questions.
Each answer comes back as a probability for every option.
`moment` questions return a probability for every second of the video.

Model: [sp1d5r/video-jev](https://huggingface.co/sp1d5r/video-jev).
Code and paper: [github.com/sp1d5r/vjev](https://github.com/sp1d5r/vjev).
Research release, non-commercial use only (CC BY-NC-SA 4.0).
