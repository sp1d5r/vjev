"""Shared Modal app, images and volumes."""

from __future__ import annotations

import modal

app = modal.App("vjev")

# Videos, labels and split indexes. Kept apart from the older snapugc-* volumes.
data_vol = modal.Volume.from_name("vjev-snapugc", create_if_missing=True)
# Hugging Face weights, so each run does not re-download 20 GB.
hf_vol = modal.Volume.from_name("vjev-hf-cache", create_if_missing=True)

DATA = "/data"
HF = "/hf"

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip")
    .uv_pip_install("gdown==5.2.0", "numpy>=2.0")
    .add_local_python_source("vjev")
)

_gpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    # Versions from the Clef-flash model card ("tested with torch 2.11 and transformers 5.10.2").
    .uv_pip_install(
        "torch==2.11.0",
        "transformers==5.10.2",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "pillow",
        "av",
        "numpy>=2.0",
        "scipy>=1.14",
        "wandb",
        "torchvision",
    )
    .env({"HF_HOME": HF, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
# Local source goes last, so code edits never trigger an image rebuild.
gpu_image = _gpu_base.add_local_python_source("vjev")
train_image = _gpu_base.uv_pip_install("peft").add_local_python_source("vjev")
