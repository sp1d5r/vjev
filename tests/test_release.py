import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_release_code_matches_the_package():
    """release/vjev holds copies for the Hugging Face repo; they must not drift."""
    for name in ("model.py", "clef_schema.py"):
        assert (ROOT / "release" / "vjev" / name).read_bytes() == (ROOT / "vjev" / name).read_bytes()
