#!/bin/sh
# Copy the shared code into the ComfyUI node package. tests/test_comfyui.py checks they match.
set -e
cd "$(dirname "$0")/.."
D=comfyui/ComfyUI-VideoJev
mkdir -p $D/vjev
cp space/presets.py $D/presets.py
cp release/vjev/__init__.py release/vjev/clef_schema.py release/vjev/model.py $D/vjev/
