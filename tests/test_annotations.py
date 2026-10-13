"""Annotation stores, app navigation, app construction and concept-subset selection."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pharmachecked.config import REPO_ROOT
from pharmachecked.data.annotation_app import (
    build_concept_app,
    build_packaging_app,
    first_todo,
    step,
)
from pharmachecked.data.annotations import (
    CONCEPTS,
    PACKAGING_COLUMNS,
    ConceptStore,
    PackagingStore,
)
from pharmachecked.data.clip_prelabel import prompts
from pharmachecked.data.manifest import read_manifest
from pharmachecked.data.toy import ToyData


def _prelabels(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"image_id": ids, "clip_label": "packaging", "clip_conf": 0.5})


def test_packaging_store_roundtrip(tmp_path: Path) -> None:
    store = PackagingStore.create(tmp_path / "p.csv", _prelabels(["b", "a"]))
    assert store.ids() == ["a", "b"]
    store.set_label("a", "blister", "Sophie")
    again = PackagingStore(tmp_path / "p.csv")
    assert again.row("a")["final_label"] == "blister"
    assert again.row("a")["annotator"] == "Sophie"
    assert again.row("b")["final_label"] == ""
    assert again.n_done() == 1
    assert tuple(pd.read_csv(tmp_path / "p.csv").columns) == PACKAGING_COLUMNS


def test_packaging_store_validation(tmp_path: Path) -> None:
    store = PackagingStore.create(tmp_path / "p.csv", _prelabels(["a"]))
    with pytest.raises(ValueError):
        store.set_label("a", "jar", "Sophie")
    with pytest.raises(ValueError):
        store.set_label("a", "packaging", " ")
    with pytest.raises(FileExistsError):
        PackagingStore.create(tmp_path / "p.csv", _prelabels(["a"]))


def test_concept_store_roundtrip(tmp_path: Path) -> None:
    store = ConceptStore.create(tmp_path / "c.csv", ["x", "y"])
    values = dict.fromkeys(CONCEPTS, "absent") | {"print_blur": "present"}
    store.set_concepts("x", values, "Sophie")
    again = ConceptStore(tmp_path / "c.csv")
    assert again.row("x")["print_blur"] == "present"
    assert again.n_done() == 1
    with pytest.raises(ValueError):
        store.set_concepts("y", {"print_blur": "present"}, "Sophie")
    with pytest.raises(ValueError):
        store.set_concepts("y", dict.fromkeys(CONCEPTS, "maybe"), "Sophie")


def test_navigation_skips_done_images() -> None:
    ids = ["a", "b", "c", "d"]
    done = {"b", "c"}.__contains__
    assert step(ids, 0, +1, done, todo_only=True) == 3
    assert step(ids, 0, +1, done, todo_only=False) == 1
    assert step(ids, 3, +1, done, todo_only=True) == 3  # nothing further: stay
    assert step(ids, 3, -1, done, todo_only=True) == 0
    assert first_todo(ids, done) == 0
    assert first_todo(ids, lambda _: True) == 0


def test_prompts_follow_the_plan() -> None:
    assert prompts() == [
        "a photo of a medicine blister",
        "a photo of a medicine packaging",
        "a photo of a medicine blister pack together with its box",
        "a photo of something that is not a medicine",
        "a photo of a medicine other",
    ]


def test_refresh_clip_keeps_human_labels(tmp_path: Path) -> None:
    store = PackagingStore.create(tmp_path / "p.csv", _prelabels(["a", "b"]))
    store.set_label("a", "both", "Sophie")
    fresh = pd.DataFrame({"image_id": ["a", "b"], "clip_label": "not_a_drug", "clip_conf": 0.9})
    store.refresh_clip(fresh)
    again = PackagingStore(tmp_path / "p.csv")
    assert again.row("a")["final_label"] == "both"
    assert again.row("b")["clip_label"] == "not_a_drug"
    with pytest.raises(ValueError, match="cover"):
        store.refresh_clip(fresh.iloc[:1])


def test_apps_build(tmp_path: Path, toy: ToyData) -> None:
    ids = sorted(read_manifest(toy.manifest)["image_id"])[:3]
    packaging = build_packaging_app(
        PackagingStore.create(tmp_path / "p.csv", _prelabels(ids)), toy.images_dir
    )
    concepts = build_concept_app(ConceptStore.create(tmp_path / "c.csv", ids), toy.images_dir)
    assert packaging.title == "Packaging type review"
    assert concepts.title == "Concept labelling"


def _select_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "select_concept_subset", REPO_ROOT / "scripts" / "select_concept_subset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concept_subset_is_stratified_and_deterministic() -> None:
    select = _select_module().select
    ids = [f"{k:016x}" for k in range(500)]
    labels = [int(k % 8 < 3) for k in range(500)]  # 37.5% positive
    a = select(ids, labels, 300)
    assert a == select(ids, labels, 300)
    assert len(a) == 300
    share = sum(labels[ids.index(i)] for i in a) / 300
    assert abs(share - 0.375) < 0.01


def test_audit_store_and_app(tmp_path: Path, toy: ToyData) -> None:
    from pharmachecked.data.annotation_app import build_audit_app
    from pharmachecked.data.annotations import AUDIT_COLUMNS, AuditStore

    ids = sorted(read_manifest(toy.manifest)["image_id"])[:2]
    pd.DataFrame(
        [{c: "" for c in AUDIT_COLUMNS} | {"image_id": i, "original_label": "1"} for i in ids]
    ).to_csv(tmp_path / "audit.csv", index=False)
    store = AuditStore(tmp_path / "audit.csv")
    store.set_decision(ids[0], "relabel", "authentic packaging", "Sophie")
    again = AuditStore(tmp_path / "audit.csv")
    assert again.row(ids[0])["new_label"] == "0"
    assert again.n_done() == 1
    with pytest.raises(ValueError):
        store.set_decision(ids[1], "remove", " ", "Sophie")
    assert build_audit_app(store, toy.images_dir).title == "Label audit"
