"""Shipped assets: paper figures, data manifest and the links of the Markdown documents."""

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO / "assets" / "paper_figures"
DOCUMENTS = ("README.md", "data/README.md", "assets/paper_figures/README.md",
             "docs/implementation_notes.md", "THIRD_PARTY_NOTICES.md")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_numbered_figure_of_the_paper_is_present_and_listed():
    figures = _load_script("extract_paper_figures").FIGURES
    names = [figure.name for figure in figures]
    assert names == [f"fig{n:02d}.png" for n in range(1, 19)]  # Figs. 1-18
    listing = (FIGURE_DIR / "README.md").read_text(encoding="utf-8")
    total = 0
    for number, name in enumerate(names, start=1):
        path = FIGURE_DIR / name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size < 1_500_000
        total += path.stat().st_size
        assert f"| `{name}` | Fig. {number} |" in listing
    assert total < 25_000_000
    assert sorted(p.name for p in FIGURE_DIR.glob("*.png")) == names
    assert "10.1007/s10586-025-05348-3" in listing


def test_figure_crop_boxes_are_well_formed():
    for figure in _load_script("extract_paper_figures").FIGURES:
        x0, y0, x1, y1 = figure.box
        assert 0 <= x0 < x1 <= 596 and 0 <= y0 < y1 <= 792  # inside the 595 x 791 pt page
        assert 1 <= figure.page <= 50 and figure.dpi >= 100


def test_data_manifest_matches_the_shipped_files():
    checksums = _load_script("data_checksums")
    assert checksums.check() == []
    lines = checksums.MANIFEST.read_text(encoding="ascii").splitlines()
    assert len(lines) == 385
    assert sum("matlab/input_data/" in line for line in lines) == 329
    assert sum("matlab/input_data22/" in line for line in lines) == 54
    digest = hashlib.sha256(checksums.MANIFEST.read_bytes()).hexdigest()
    assert digest in (REPO / "data" / "README.md").read_text(encoding="utf-8")
    assert max(path.stat().st_size for path in checksums.covered_files()) < 50_000_000


@pytest.mark.parametrize("document", DOCUMENTS)
def test_relative_links_resolve(document):
    path = REPO / document
    text = path.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    assert targets or document != "README.md"
    for target in targets:
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (path.parent / target.split("#")[0]).resolve()
        assert resolved.exists(), f"{document}: broken link {target}"


def test_readme_shows_the_method_figure_and_result_figures():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "assets/paper_figures/fig02.png" in text
    assert len(set(re.findall(r"assets/paper_figures/fig\d\d\.png", text))) >= 4
