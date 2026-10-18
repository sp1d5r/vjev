"""Every GPU job in one deployed app, so jobs started with `scripts/spawn.py` keep running
when the client disconnects.

    uv run modal deploy vjev/deploy.py
"""

from vjev import infer, longvideo, predict, release, train, zeroshot  # noqa: F401
from vjev.common import app  # noqa: F401
