"""PLAN 10.3 / 10.5: leaderboard and paper assets are pasted from committed tables."""

import importlib.util
import json

from pharmachecked.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "release_assets", REPO_ROOT / "scripts" / "release_assets.py"
)
release_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_assets)


def test_every_paper_source_exists() -> None:
    for stem, _ in release_assets.FIGURES.values():
        assert (release_assets.F / f"{stem}.png").is_file()
        assert (release_assets.F / f"{stem}.pdf").is_file()
    for stem, _ in release_assets.TABLES.values():
        if stem == "final_test" and not (release_assets.T / "final_test.csv").is_file():
            continue
        assert not release_assets.read_table(stem).empty


def test_numbers_are_cells_of_their_source() -> None:
    path = REPO_ROOT / "results" / "paper" / "numbers.json"
    numbers = json.loads(path.read_text(encoding="utf-8"))
    tables = {}
    for entry in numbers.values():
        src = entry["source"]
        stem = src.rsplit("/", 1)[1].rsplit(".", 1)[0]
        df = tables.setdefault(stem, release_assets.read_table(stem))
        row_key, col = entry["row"], entry["column"]
        assert entry["value"] in set(df.loc[df[df.columns[0]] == row_key, col])


def test_leaderboard_is_sorted() -> None:
    lines = [
        ln
        for ln in release_assets.leaderboard().splitlines()
        if ln.startswith("| ") and "|---" not in ln
    ]
    ladder = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if not cells[0].isdigit():
            break
        ladder.append(float(cells[3].split(" ")[0]))
    assert ladder == sorted(ladder, reverse=True)
