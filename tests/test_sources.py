"""Source registry: roles, folder/constant labels, membership filtering."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.data.sources import SourceSpec, list_raw_files, load_specs

EXT = (".jpg", ".png")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    _touch(root / "dev" / "dataset" / "Fake" / "a.png")
    _touch(root / "dev" / "dataset" / "train" / "Real" / "b.jpg")
    _touch(root / "dev" / "dataset" / "Real" / "desktop.ini")
    _touch(root / "ext" / "c.jpg")
    _touch(root / "ext" / "d.jpg")
    (tmp_path / "members.csv").write_text("image_stem,keep\nc,true\nd,false\n")
    return root


def _specs(tmp_path: Path) -> list[SourceSpec]:
    cfg = {
        "sources": {
            "dev": {"role": "development", "folder_labels": {"Fake": 1, "Real": 0}},
            "ext": {
                "role": "external",
                "label": 1,
                "membership": "members.csv",
                "include_column": "keep",
            },
        }
    }
    return load_specs(cfg, tmp_path)


def test_labels_roles_and_membership(raw: Path, tmp_path: Path) -> None:
    files = list_raw_files(raw, _specs(tmp_path), EXT)
    got = {(f.path.name, f.source, f.role, f.label, f.included) for f in files}
    assert got == {
        ("a.png", "dev", "development", 1, True),
        ("b.jpg", "dev", "development", 0, True),
        ("c.jpg", "ext", "external", 1, True),
        ("d.jpg", "ext", "external", 1, False),
    }
    assert [f.path for f in files] == sorted(f.path for f in files)


def test_labelled_role_requires_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs"):
        load_specs({"sources": {"x": {"role": "external"}}}, tmp_path)


def test_unknown_role_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="role"):
        load_specs({"sources": {"x": {"role": "train", "label": 0}}}, tmp_path)


def test_missing_source_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_raw_files(tmp_path / "nope", _specs(tmp_path), EXT)


def test_repository_config_is_valid() -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "data" / "sources.yaml"))
    specs = load_specs(cfg, REPO_ROOT)
    roles = {s.name: s.role for s in specs}
    assert [n for n, r in roles.items() if r == "development"] == ["kaggle_fake_real"]
    for spec in specs:
        if spec.membership is not None:
            assert spec.membership.is_file()
