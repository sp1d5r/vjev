"""Start any function of the deployed `vjev` app and return at once.

    uv run modal deploy vjev/deploy.py
    uv run python scripts/spawn.py predict '{"run": "g4-joint", "split": "val", "limit": 100}'
"""

from __future__ import annotations

import json
import sys

import modal

if __name__ == "__main__":
    name, kwargs = sys.argv[1], json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}")
    call = modal.Function.from_name("vjev", name).spawn(**kwargs)
    print(name, kwargs, call.object_id)
