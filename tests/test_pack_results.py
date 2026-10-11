"""Results bundles round-trip with hashes intact, never carry weights, and reject tampering."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from pharmachecked.utils.bundle import MANIFEST, BundleError, pack, unpack, verify
from pharmachecked.utils.io import sha256_file

RUN_A = "e01_resnet50_default_f0_s0_20261009-120000"
RUN_B = "e01_resnet50_default_f1_s0_20261009-130000"


def _make_run(root: Path, run_id: str) -> Path:
    run = root / run_id
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("model: resnet50\n")
    (run / "metrics.json").write_text('{"auroc": null}\n')
    (run / "oof_predictions.csv").write_text("image_id,prob\nabc,0.5\n")
    (run / "checkpoint_path.txt").write_text("checkpoints/x.pt\n")
    (run / "sub").mkdir()
    (run / "sub" / "extra.txt").write_text("x\n")
    (run / "stray_weights.pt").write_bytes(b"\x00" * 64)
    return run


@pytest.fixture
def source(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    runs_root = tmp_path / "gpu_pc" / "results" / "runs"
    runs = [_make_run(runs_root, RUN_A), _make_run(runs_root, RUN_B)]
    table = tmp_path / "gpu_pc" / "results" / "tables" / "bench_gpu.md"
    table.parent.mkdir(parents=True)
    table.write_text("| model | img/s |\n")
    return runs, [table]


def test_roundtrip_preserves_hashes_and_drops_weights(
    tmp_path: Path, source: tuple[list[Path], list[Path]]
) -> None:
    runs, tables = source
    result = pack(runs, tables, tmp_path / "b.zip", created_by="gpu")
    assert result.n_runs == 2
    assert result.skipped_weights == (
        f"runs/{RUN_A}/stray_weights.pt",
        f"runs/{RUN_B}/stray_weights.pt",
    )
    results = tmp_path / "laptop" / "results"
    assert unpack(result.path, results) == [RUN_A, RUN_B]
    for run in runs:
        for src in run.rglob("*"):
            if src.is_file() and src.suffix != ".pt":
                dst = results / "runs" / run.name / src.relative_to(run)
                assert sha256_file(dst) == sha256_file(src)
        assert not (results / "runs" / run.name / "stray_weights.pt").exists()
    assert (results / "tables" / "bench_gpu.md").read_text() == "| model | img/s |\n"


def test_unpack_twice_is_idempotent(tmp_path: Path, source: tuple[list[Path], list[Path]]) -> None:
    runs, tables = source
    bundle = pack(runs, tables, tmp_path / "b.zip").path
    results = tmp_path / "results"
    unpack(bundle, results)
    unpack(bundle, results)


def test_unpack_refuses_to_overwrite_different_content(
    tmp_path: Path, source: tuple[list[Path], list[Path]]
) -> None:
    runs, tables = source
    bundle = pack(runs, tables, tmp_path / "b.zip").path
    results = tmp_path / "results"
    existing = results / "runs" / RUN_A / "metrics.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"auroc": 0.5}\n')
    with pytest.raises(BundleError, match="overwrite"):
        unpack(bundle, results)
    assert not (results / "runs" / RUN_B).exists()  # nothing written on refusal


def _rewrite(bundle: Path, out: Path, replace: dict[str, bytes], manifest: dict | None) -> None:
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == MANIFEST and manifest is not None:
                dst.writestr(name, json.dumps(manifest))
            else:
                dst.writestr(name, replace.get(name, src.read(name)))
        for name, data in replace.items():
            if name not in src.namelist():
                dst.writestr(name, data)


def test_tampered_file_is_rejected(tmp_path: Path, source: tuple[list[Path], list[Path]]) -> None:
    runs, tables = source
    bundle = pack(runs, tables, tmp_path / "b.zip").path
    bad = tmp_path / "bad.zip"
    _rewrite(bundle, bad, {f"runs/{RUN_A}/metrics.json": b'{"auroc": 0.99}\n'}, None)
    with pytest.raises(BundleError, match="hash mismatch"):
        verify(bad)


def test_unlisted_file_is_rejected(tmp_path: Path, source: tuple[list[Path], list[Path]]) -> None:
    runs, tables = source
    bundle = pack(runs, tables, tmp_path / "b.zip").path
    bad = tmp_path / "bad.zip"
    _rewrite(bundle, bad, {f"runs/{RUN_A}/new.txt": b"x"}, None)
    with pytest.raises(BundleError, match="manifest mismatch"):
        verify(bad)


@pytest.mark.parametrize(
    "name", ["../evil.txt", "runs/../../evil.txt", "/abs.txt", "other/x.txt", "runs/notarun/x"]
)
def test_unsafe_paths_are_rejected(tmp_path: Path, name: str) -> None:
    data = b"x"
    manifest = {"files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": 1}}}
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr(name, data)
        zf.writestr(MANIFEST, json.dumps(manifest))
    with pytest.raises(BundleError):
        unpack(bad, tmp_path / "results")
    assert not (tmp_path / "evil.txt").exists()


def test_pack_rejects_non_run_folder(tmp_path: Path) -> None:
    folder = tmp_path / "not_a_run"
    folder.mkdir()
    with pytest.raises(BundleError, match="not a run folder"):
        pack([folder], [], tmp_path / "b.zip")


def test_pipeline_state_is_replaced_but_runs_are_not(
    tmp_path: Path, source: tuple[list[Path], list[Path]]
) -> None:
    runs, tables = source
    state = tmp_path / "decisions.json"
    state.write_text('{"stage": 1}')
    first = pack(runs, tables, tmp_path / "a.zip", extra_files={"pipeline": [state]}).path
    results = tmp_path / "results"
    unpack(first, results)
    state.write_text('{"stage": 2}')
    second = pack(runs, tables, tmp_path / "b.zip", extra_files={"pipeline": [state]}).path
    unpack(second, results)
    assert (results / "pipeline" / "decisions.json").read_text() == '{"stage": 2}'
    with pytest.raises(BundleError):
        pack(runs, tables, tmp_path / "c.zip", extra_files={"runs": [state]})
