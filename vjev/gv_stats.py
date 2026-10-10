"""Shortcut check for GenVideo: do real and generated clips differ in length, size or frame
rate? If they do, a model can tell them apart without looking at content.

    modal run vjev/gv_stats.py::main_gv_stats
"""

from __future__ import annotations

import json
import pathlib

from vjev.common import DATA, app, data_vol
from vjev.videofeedback import vf_image


@app.function(image=vf_image, volumes={DATA: data_vol}, cpu=8, timeout=1800)
def gv_stats() -> dict:
    import collections
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    root = pathlib.Path(DATA)
    recs = []
    for s in ("train", "val", "test"):
        recs += [json.loads(x) for x in (root / "index" / f"gv_{s}.jsonl").read_text().splitlines()]

    def probe(r: dict) -> dict:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,r_frame_rate:format=duration",
                              "-of", "json", str(root / r["path"])],
                             capture_output=True, text=True, check=False).stdout
        j = json.loads(out or "{}")
        st = (j.get("streams") or [{}])[0]
        a, _, b = st.get("r_frame_rate", "0/1").partition("/")
        try:
            dur = round(float(j.get("format", {}).get("duration", 0)), 1)
        except ValueError:
            dur = -1.0
        return {"gen": r["generator"], "size": f"{st.get('width')}x{st.get('height')}",
                "fps": round(float(a) / float(b or 1), 1), "dur": dur}

    with ThreadPoolExecutor(32) as ex:
        rows = list(ex.map(probe, recs))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["gen"]].append(r)
    return {g: {"n": len(rs),
                "size": collections.Counter(r["size"] for r in rs).most_common(3),
                "fps": collections.Counter(r["fps"] for r in rs).most_common(3),
                "dur": collections.Counter(r["dur"] for r in rs).most_common(3)}
            for g, rs in by.items()}


@app.local_entrypoint()
def main_gv_stats():
    print(json.dumps(gv_stats.remote()))
