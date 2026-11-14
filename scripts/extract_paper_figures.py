#!/usr/bin/env python
"""Crop the numbered figures of the published article out of its PDF.

The crop boxes below (PDF points, origin at the top-left corner of the page) were found from
the caption blocks ("Fig. N ...") and the graphics above them and then checked by eye. Each
figure is rendered without its caption. The PDF is only read.

Example::

    python scripts/extract_paper_figures.py /path/to/s10586-025-05348-3.pdf

Needs PyMuPDF (``pip install -e ".[figures]"``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple


class Figure(NamedTuple):
    """One figure: output name, 1-based page, crop box (x0, y0, x1, y1) and rendering dpi."""

    name: str
    page: int
    box: tuple[float, float, float, float]
    dpi: int = 200


#: Margin added around every crop box, in PDF points.
PADDING = 3.0

FIGURES: tuple[Figure, ...] = (
    Figure("fig01.png", 4, (53.9, 59.2, 541.4, 228.4)),
    Figure("fig02.png", 7, (53.5, 59.2, 542.2, 361.9)),
    Figure("fig03.png", 11, (67.0, 245.0, 528.4, 519.6)),
    Figure("fig04.png", 12, (85.2, 59.2, 510.2, 200.7)),
    Figure("fig05.png", 12, (53.3, 259.4, 542.0, 423.2)),
    Figure("fig06.png", 13, (59.1, 60.1, 536.2, 338.8)),
    Figure("fig07.png", 14, (53.8, 59.2, 541.6, 272.5)),
    Figure("fig08.png", 15, (67.0, 59.2, 528.4, 332.8)),
    Figure("fig09.png", 20, (59.1, 60.1, 536.2, 285.3)),
    Figure("fig10.png", 20, (85.0, 323.3, 510.3, 467.4)),
    Figure("fig11.png", 23, (53.8, 59.2, 541.7, 289.1)),
    Figure("fig12.png", 24, (67.0, 59.2, 528.4, 333.6)),
    Figure("fig13.png", 43, (59.2, 60.1, 536.2, 361.7)),
    Figure("fig14.png", 44, (59.2, 60.1, 536.2, 361.7)),
    Figure("fig15.png", 45, (59.2, 60.1, 536.2, 361.9)),
    Figure("fig16.png", 46, (59.2, 60.1, 536.2, 362.8)),
    Figure("fig17.png", 46, (59.2, 398.9, 536.2, 550.0)),
    Figure("fig18.png", 47, (59.2, 60.1, 536.2, 211.2)),
)


def extract(pdf: Path, output_dir: Path) -> list[Path]:
    """Render every entry of :data:`FIGURES` from ``pdf`` into ``output_dir``."""
    try:
        import pymupdf
    except ImportError:  # PyMuPDF older than 1.24.3 only provides the name "fitz"
        import fitz as pymupdf

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with pymupdf.open(pdf) as document:
        for figure in FIGURES:
            page = document[figure.page - 1]
            x0, y0, x1, y1 = figure.box
            box = pymupdf.Rect(x0 - PADDING, y0 - PADDING, x1 + PADDING, y1 + PADDING)
            clip = box & page.rect
            pixmap = page.get_pixmap(clip=clip, dpi=figure.dpi, alpha=False)
            target = output_dir / figure.name
            pixmap.save(target)
            written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path, help="path to the article PDF (read only)")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "paper_figures",
    )
    args = parser.parse_args()
    for path in extract(args.pdf, args.output_dir):
        print(f"{path.name}  {path.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
