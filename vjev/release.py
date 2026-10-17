"""Export a trained run as a small, standalone release: LoRA adapter + typed head + config.

    modal run vjev/release.py::export --run g4-joint                       # writes /data/release/g4-joint
    modal run vjev/release.py::upload --run g4-joint --repo sp1d5r/video-jev   # pushes it to the Hub

The release holds no videos and no copy of Qwen3-Omni. Users load the base model from the
Hub and apply these weights with `vjev/infer.py` (or `load_release` below).
"""

from __future__ import annotations

import json
import pathlib

import modal

from vjev.common import DATA, app, data_vol, train_image


@app.function(image=train_image, volumes={DATA: data_vol}, cpu=4, memory=32768, timeout=1800)
def export_run(run: str) -> dict:
    import torch
    from safetensors.torch import save_file

    from vjev import model as vm
    from vjev.tasks import TASKS

    root = pathlib.Path(DATA)
    ck = torch.load(root / "ckpt" / run / "best.pt", map_location="cpu", weights_only=False)
    out = root / "release" / run
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ck["lora"].items()}, out / "adapter_model.safetensors")
    save_file({k: v.contiguous() for k, v in ck["head"].items()}, out / "head.safetensors")
    cfg = ck["cfg"]
    (out / "vjev_config.json").write_text(json.dumps({
        "base_model": vm.OMNI_ID,
        "lora": {"r": cfg["lora_r"], "alpha": cfg["lora_alpha"], "targets": vm.LORA_TARGETS},
        "head": vm.HEAD_CONFIG,
        "decode": {t: {"fps": f, "audio": a} for t, (f, a) in TASKS.items()},
        "trained_on": cfg.get("task"),
        "step": ck["step"],
        "val": ck["val"],
    }, indent=1, default=float))
    data_vol.commit()
    sizes = {p.name: p.stat().st_size for p in out.iterdir()}
    return {"out": str(out), "bytes": sizes}


@app.function(image=train_image, volumes={DATA: data_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def upload_run(run: str, repo: str, private: bool) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    src = pathlib.Path(DATA) / "release" / run
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(src), commit_message=f"Release {run}")
    return {"repo": repo, "files": sorted(p.name for p in src.iterdir()), "private": private}


def load_release(release_dir: str, base_path: str):
    """(TypedOmni, processor) with the release weights applied, ready for inference."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from vjev import model as vm

    cfg = json.loads((pathlib.Path(release_dir) / "vjev_config.json").read_text())
    net, proc = vm.load(base_path, cfg["lora"]["r"], cfg["lora"]["alpha"])
    set_peft_model_state_dict(net.thinker, load_file(f"{release_dir}/adapter_model.safetensors"))
    net.head.load_state_dict(load_file(f"{release_dir}/head.safetensors", device="cuda"))
    return net.eval(), proc


@app.local_entrypoint()
def export(run: str):
    print(json.dumps(export_run.remote(run), indent=1))


@app.local_entrypoint()
def upload(run: str, repo: str, private: bool = True):
    print(json.dumps(upload_run.remote(run, repo, private), indent=1))

