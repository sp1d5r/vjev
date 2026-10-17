"""Start a training run on the deployed `vjev` app, so it keeps going if this machine drops.

    uv run modal deploy vjev/train.py
    uv run python scripts/spawn_train.py --run-name g4-joint --task vf+vfai+qv+fq+vs ...

Takes the same options as `vjev/train.py::main`.
"""

from __future__ import annotations

import argparse
import inspect

import modal

from vjev.train import main


def build_cfg(argv: list[str] | None = None) -> dict:
    params = inspect.signature(main.info.raw_f).parameters
    ap = argparse.ArgumentParser()
    for name, p in params.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(p.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=p.default)
        else:
            ap.add_argument(flag, type=type(p.default), default=p.default)
    cfg = vars(ap.parse_args(argv))
    if not cfg["run_name"]:
        raise SystemExit("--run-name is required")
    return cfg


if __name__ == "__main__":
    cfg = build_cfg()
    call = modal.Function.from_name("vjev", "train").spawn(cfg)
    print(cfg["run_name"], call.object_id)
