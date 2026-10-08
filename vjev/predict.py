"""Score a saved checkpoint on a split and keep the per-clip predictions.

    modal run vjev/predict.py --run pilot-omni-av-10k --split val --limit 500

Writes /data/preds/{run}-{split}{limit}.jsonl with id, ecr, expected and probs per clip.
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
    ckpt = torch.load(root / "ckpt" / run / "best.pt", map_location="cuda", weights_only=False)
    cfg = ckpt["cfg"]
    path = snapshot_download(model.OMNI_ID)
    net, _ = model.load(path, cfg["lora_r"], cfg["lora_alpha"])
    set_peft_model_state_dict(net.thinker, ckpt["lora"])
    net.head.load_state_dict(ckpt["head"])

    edges = json.loads((root / "index" / "meta.json").read_text())["level_edges"]
    recs = _records(split, limit, cfg["seed"])
    loader = _loader(ClipDataset(recs, edges, path, cfg["audio"]), shuffle=False, workers=8)
    res, rows = evaluate(net, loader, edges)
    out = root / "preds" / f"{run}-{split}{limit or ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data_vol.commit()
    return {"run": run, "step": ckpt["step"], "saved_val": ckpt["val"], "rescored": res,
            "out": str(out)}


@app.local_entrypoint()
def main(run: str, split: str = "val", limit: int = 500):
    if split == "test":
        raise SystemExit("test is touched once per final model; run it deliberately, not here")
    print(json.dumps(predict.remote(run, split, limit), indent=2))
