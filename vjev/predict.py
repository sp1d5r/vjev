"""Score a saved checkpoint on a split and keep the per-clip predictions.

    modal run vjev/predict.py::predict_ckpt --run pilot-omni-av-10k --split val --limit 500

Writes /data/preds/{run}-{split}{limit}.jsonl with id, ecr, expected and probs per clip.

The test split is touched once per final model: pass --split test --limit 0 --final. It
refuses to run if test predictions for that run already exist.
"""

from __future__ import annotations

import json
import pathlib

import modal

from vjev.common import DATA, HF, app, data_vol, hf_vol, train_image


@app.function(image=train_image, gpu="H200", volumes={DATA: data_vol, HF: hf_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=4 * 3600, cpu=12,
              memory=65536)
def predict(run: str, split: str, limit: int) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import set_peft_model_state_dict

    from vjev import model
    from vjev.train import ClipDataset, _loader, _records, evaluate

    root = pathlib.Path(DATA)
    out = root / "preds" / f"{run}-{split}{limit or ''}.jsonl"
    if split == "test" and out.exists():
        raise RuntimeError(f"{out} exists; test is scored once per final model")
    ckpt = torch.load(root / "ckpt" / run / "best.pt", map_location="cuda", weights_only=False)
    cfg = ckpt["cfg"]
    path = snapshot_download(model.OMNI_ID)
    net, _ = model.load(path, cfg["lora_r"], cfg["lora_alpha"],
                        vision_lora=cfg.get("vision_lora", False))
    set_peft_model_state_dict(net.thinker, ckpt["lora"])
    net.head.load_state_dict(ckpt["head"])

    edges = json.loads((root / "index" / "meta.json").read_text())["level_edges"]
    recs = _records(split, limit, cfg["seed"])
    loader = _loader(ClipDataset(recs, edges, path, cfg["audio"]), shuffle=False, workers=8)
    res, rows = evaluate(net, loader, edges)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data_vol.commit()
    return {"run": run, "step": ckpt["step"], "saved_val": ckpt["val"], "rescored": res,
            "out": str(out)}


@app.local_entrypoint()
def predict_ckpt(run: str, split: str = "val", limit: int = 500, final: bool = False):
    if split == "test" and not final:
        raise SystemExit("test is touched once per final model; pass --final to mean it")
    print(json.dumps(predict.remote(run, split, limit), indent=2))
