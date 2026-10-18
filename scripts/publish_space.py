"""Publish the demo Space (space/ plus the release's vjev/ package) and read its logs.

    uv run modal run scripts/publish_space.py              # upload and request ZeroGPU
    uv run modal run scripts/publish_space.py --logs run   # or build; status only: --logs none
    uv run modal run scripts/publish_space.py --card       # model card only

The HF token lives in the Modal secret "huggingface", so this runs there.
"""

from __future__ import annotations

import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPACE = "sp1d5r/video-jev"
FILES = {f"{n}": ROOT / "space" / n
         for n in ("app.py", "presets.py", "requirements.txt", "packages.txt", "README.md")}
FILES |= {f"vjev/{n}": ROOT / "release" / "vjev" / n
          for n in ("__init__.py", "clef_schema.py", "model.py")}

app = modal.App("vjev-publish-space")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub", "requests")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=900)
def publish(files: dict[str, bytes]) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    # Ask for ZeroGPU at creation: free accounts may host ZeroGPU Spaces, but not cpu-basic
    # Gradio Spaces (402 without PRO).
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
                    private=False, exist_ok=True)
    api.create_commit(SPACE, repo_type="space", commit_message="Update demo",
                      operations=[CommitOperationAdd(k, v) for k, v in files.items()])
    try:
        api.request_space_hardware(SPACE, "zero-a10g")
    except Exception as e:  # noqa: BLE001 - report the refusal, the upload still stands
        return {"hardware_error": str(e)[:500], "runtime": str(api.get_space_runtime(SPACE))}
    return {"runtime": str(api.get_space_runtime(SPACE))}


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=300)
def space_logs(kind: str) -> dict:
    import os

    import requests
    from huggingface_hub import HfApi

    rt = HfApi().get_space_runtime(SPACE)
    out = {"stage": rt.stage, "hardware": rt.hardware, "requested": rt.requested_hardware}
    if kind in ("build", "run"):
        lines = []
        try:
            with requests.get(f"https://huggingface.co/api/spaces/{SPACE}/logs/{kind}",
                              headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
                              stream=True, timeout=(10, 20)) as r:
                for raw in r.iter_lines():
                    if raw.startswith(b"data:"):
                        try:
                            lines.append(json.loads(raw[5:]).get("data", "").rstrip())
                        except json.JSONDecodeError:
                            lines.append(raw[5:].decode(errors="replace"))
        except requests.exceptions.RequestException:
            pass  # the log stream stays open; a read timeout ends it
        out["log"] = "\n".join(lines[-80:])
    return out


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=300)
def model_card(card: bytes) -> str:
    from huggingface_hub import HfApi

    info = HfApi().upload_file(path_or_fileobj=card, path_in_repo="README.md",
                               repo_id="sp1d5r/video-jev", commit_message="Update model card")
    return str(info)


@app.local_entrypoint()
def main(logs: str = "", card: bool = False):
    if card:  # only the model card: release/README.md -> sp1d5r/video-jev
        print(model_card.remote((ROOT / "release" / "README.md").read_bytes()))
        return
    if logs:
        print(json.dumps(space_logs.remote(logs), indent=1))
        return
    print(json.dumps(publish.remote({k: p.read_bytes() for k, p in FILES.items()}), indent=1))
