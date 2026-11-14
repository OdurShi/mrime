#!/usr/bin/env python
"""Write or check ``data/SHA256SUMS``, the SHA-256 manifest of the shipped benchmark data.

The manifest covers the CEC2017 / CEC2022 definition data (``matlab/input_data``,
``matlab/input_data22``) and the two reference C sources that read them. It uses the format of
``sha256sum`` with paths relative to the repository root, so it can also be checked with
``shasum -a 256 -c data/SHA256SUMS``.

Examples::

    python scripts/data_checksums.py --check      # verify the files against the manifest
    python scripts/data_checksums.py --write      # regenerate the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "SHA256SUMS"
#: Directories and single files covered by the manifest, relative to the repository root.
COVERED = ("matlab/input_data", "matlab/input_data22", "matlab/cec17_func.cpp",
           "matlab/cec22_func.cpp")


def covered_files() -> list[Path]:
    """All covered files in a stable order."""
    files: list[Path] = []
    for entry in COVERED:
        path = REPO / entry
        files.extend(sorted(path.iterdir()) if path.is_dir() else [path])
    return files


def manifest_lines() -> list[str]:
    """``<sha256>  <relative path>`` for every covered file."""
    return [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(REPO).as_posix()}"
        for path in covered_files()
    ]


def check() -> list[str]:
    """Return the differences between the manifest and the files (empty when in sync)."""
    recorded = MANIFEST.read_text(encoding="ascii").splitlines()
    current = manifest_lines()
    return sorted(set(recorded) ^ set(current))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate data/SHA256SUMS")
    group.add_argument("--check", action="store_true", help="verify the files")
    args = parser.parse_args()
    if args.write:
        lines = manifest_lines()
        MANIFEST.write_text("\n".join(lines) + "\n", encoding="ascii")
        print(f"wrote {len(lines)} checksums to {MANIFEST.relative_to(REPO)}")
        return 0
    differences = check()
    for line in differences:
        print(line, file=sys.stderr)
    print("OK" if not differences else f"{len(differences)} differences")
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
