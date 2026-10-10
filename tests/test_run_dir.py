"""Run ids follow the CLAUDE.md format and run folders get config.yaml + env.json."""

from datetime import datetime
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pharmachecked.utils.io import read_json, read_yaml
from pharmachecked.utils.run_dir import (
    COMPLETE_FILES,
    DirtyTreeError,
    create_run_dir,
    make_run_id,
    missing_files,
    parse_run_id,
)

NOW = datetime(2026, 10, 9, 14, 3, 5)
CLEAN_ENV = {"git_commit": "abc123", "git_dirty": False, "python": "3.13"}
DIRTY_ENV = {**CLEAN_ENV, "git_dirty": True}


def test_run_id_format() -> None:
    run_id = make_run_id("e01", "resnet50", "full_ft", 2, 1, NOW)
    assert run_id == "e01_resnet50_full-ft_f2_s1_20261009-140305"


def test_run_id_roundtrip() -> None:
    run_id = make_run_id("e04", "Multi Branch", "no_local", 4, 2, NOW, dirty=True)
    fields = parse_run_id(run_id)
    assert fields == {
        "exp": "e04",
        "model": "multi-branch",
        "variant": "no-local",
        "fold": 4,
        "seed": 2,
        "stamp": "20261009-140305",
        "dirty": True,
    }


def test_parse_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_run_id("e01_resnet50_f0_s0")


def test_create_run_dir_writes_config_and_env(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"model": {"name": "resnet50"}, "lr": 1e-3, "ref": "${lr}"})
    run = create_run_dir(tmp_path, cfg, CLEAN_ENV, "e01", "resnet50", "default", 3, 2, NOW)
    assert run.root == tmp_path / "e01_resnet50_default_f3_s2_20261009-140305"
    assert not run.dirty
    saved = read_yaml(run.file("config.yaml"))
    assert saved.model.name == "resnet50"
    assert saved.ref == 1e-3  # interpolations are resolved on save
    assert read_json(run.file("env.json")) == CLEAN_ENV
    assert sorted(set(COMPLETE_FILES) - set(missing_files(run.root))) == [
        "config.yaml",
        "env.json",
    ]


def test_dirty_tree_allowed_only_for_fold0_seed0(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, {}, DIRTY_ENV, "e01", "m", "v", 0, 0, NOW)
    assert run.dirty
    assert run.run_id.endswith("_dirty")
    with pytest.raises(DirtyTreeError):
        create_run_dir(tmp_path, {}, DIRTY_ENV, "e01", "m", "v", 1, 0, NOW)
    with pytest.raises(DirtyTreeError):
        create_run_dir(tmp_path, {}, DIRTY_ENV, "e01", "m", "v", 0, 1, NOW)


def test_existing_run_dir_is_not_overwritten(tmp_path: Path) -> None:
    create_run_dir(tmp_path, {}, CLEAN_ENV, "e01", "m", "v", 0, 0, NOW)
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, {}, CLEAN_ENV, "e01", "m", "v", 0, 0, NOW)


def test_missing_files_lists_everything_for_empty_folder(tmp_path: Path) -> None:
    assert missing_files(tmp_path) == list(COMPLETE_FILES)


def test_find_resumable_only_matches_identical_incomplete_runs(tmp_path: Path) -> None:
    from pharmachecked.utils.run_dir import find_resumable

    cfg = {"lr": 0.1, "model": {"name": "m"}}
    run = create_run_dir(tmp_path, cfg, CLEAN_ENV, "e01", "m", "v", 2, 1, NOW)
    found = find_resumable(tmp_path, cfg, "e01", "m", "v", 2, 1)
    assert found is not None and found.root == run.root
    assert find_resumable(tmp_path, {**cfg, "lr": 0.2}, "e01", "m", "v", 2, 1) is None
    assert find_resumable(tmp_path, cfg, "e01", "m", "v", 3, 1) is None
    for name in COMPLETE_FILES:
        run.file(name).touch()
    assert find_resumable(tmp_path, cfg, "e01", "m", "v", 2, 1) is None  # finished runs are final
