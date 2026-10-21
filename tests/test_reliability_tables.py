"""E10/E11 tables on synthetic reliability predictions (PLAN 5.4, 5.6, 5.7)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmachecked.evaluation import reliability as rel
from pharmachecked.evaluation.aggregate import RunRecord
from pharmachecked.evaluation.reliability_pass import COLUMNS, RELIABILITY_FILE

SIZES = {"train": 60, "val": 20, "cal": 30, "heldout": 30}


def _ood_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    for name in ("near", "far"):
        folder = data_dir / "ood" / name
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "image_id": [f"{name}{k:02d}" for k in range(20)],
                "set": name,
                "source_dataset": f"src_{name}",
                "source_class": ["a", "b"] * 10,
            }
        ).to_csv(folder / "manifest.csv", index=False)
    return data_dir


def _run(tmp_path: Path, model: str, fold: int, seed: int, sharp: float) -> RunRecord:
    """Logits ``sharp`` x (class mean +-1.5 + unit noise); OOD images get small |logit|
    and large feature distances, so every score separates them."""
    rng = np.random.default_rng(100 * fold + seed + (7 if model == "b" else 0))
    frames = []
    for split, n in SIZES.items():
        y = np.array([0, 1] * (n // 2))
        z = sharp * ((2 * y - 1) * 1.5 + rng.normal(0, 1, n))
        ids = [f"f{fold}{split}{k:02d}" for k in range(n)]
        frames.append(
            pd.DataFrame(
                {"image_id": ids, "split": split, "fold": fold, "y_true": y, "logit": z}
            ).assign(maha=rng.normal(10, 2, n), knn=rng.normal(0.5, 0.05, n))
        )
    for name in ("near", "far"):
        ids = [f"{name}{k:02d}" for k in range(20)]
        frames.append(
            pd.DataFrame(
                {
                    "image_id": ids,
                    "split": f"ood_{name}",
                    "fold": -1,
                    "y_true": -1,
                    "logit": rng.normal(0, 0.3, 20),
                    "maha": rng.normal(30, 2, 20),
                    "knn": rng.normal(0.9, 0.05, 20),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.assign(seed=seed, prob=1 / (1 + np.exp(-df["logit"])), source_dataset="x")[
        list(COLUMNS)
    ]
    root = tmp_path / f"p4ref_{model}_ft_f{fold}_s{seed}_20261020-000000"
    root.mkdir()
    df.to_csv(root / RELIABILITY_FILE, index=False)
    return RunRecord(root, model, "ft", fold, seed, {}, 0.5)


@pytest.fixture
def data(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    runs = [
        _run(tmp_path, m, f, s, sharp)
        for m, sharp in (("a", 3.0), ("b", 1.0))
        for f in (0, 1)
        for s in (0, 1, 2)
    ]
    return rel.load(runs), _ood_dir(tmp_path)


def test_temperature_and_calibration_table(data: tuple[pd.DataFrame, Path]) -> None:
    df, _ = data
    temps = df.groupby("config")["temperature"].mean()
    # same generative process, logits of "a" 3x sharper: its temperature is about 3x larger
    assert temps["a/ft"] > 2 * temps["b/ft"]
    table = rel.calibration_table(df, n_boot=50).set_index("config")
    assert table.loc["a/ft", "runs"] == 6 and table.loc["a/ft", "images"] == 60
    assert table.loc["a/ft", "log_loss after"] < table.loc["a/ft", "log_loss before"]
    bins = rel.reliability_diagram_data(df)
    assert set(bins["stage"]) == {"before", "after"} and bins["count"].sum() == 2 * 2 * 60


def test_selective_and_conformal(data: tuple[pd.DataFrame, Path]) -> None:
    df, _ = data
    table, curves = rel.selective_table(df)
    assert list(table["config"]) == ["a/ft", "b/ft"]
    assert set(curves["kind"]) == {"single", "ensemble"}
    per_run = rel.conformal_runs(df)
    assert len(per_run) == 12 * 8  # 12 fold models x (2 scores x 2 modes x 2 alphas)
    summary_table = rel.conformal_table(per_run)
    assert set(summary_table["runs"]) == {6}
    score, mode, met, summary = rel.choose_conformal(per_run, "a/ft")
    assert score in ("lac", "aps") and mode in ("marginal", "mondrian")
    assert len(summary) == 4 and isinstance(met, bool)


def test_ood_guard_and_end_to_end(data: tuple[pd.DataFrame, Path]) -> None:
    df, data_dir = data
    table = rel.ood_table(df)
    assert set(table["score"]) == {"msp", "energy", "maha", "knn", "disagreement"}
    row = table[(table["config"] == "a/ft") & (table["score"] == "maha")].iloc[0]
    assert row["units"] == 6 and row["near AUROC"].startswith("1.000")
    dis = table[(table["config"] == "a/ft") & (table["score"] == "disagreement")].iloc[0]
    assert dis["units"] == 2  # one per fold
    guard, summary = rel.choose_guard(df, "a/ft", data_dir)
    assert guard in ("maha", "knn") and len(summary) == 4
    e2e = rel.end_to_end_table(df, "a/ft", guard, "lac", "mondrian", data_dir).set_index("system")
    bare = e2e.loc["bare classifier"]
    assert bare["issued"].startswith("1.000") and bare["ID rejected"].startswith("0.000")
    full = e2e.loc["full system (alpha 0.1)"]
    assert float(full["OOD rejected"].split()[0]) > 0.9
    assert float(full["error on issued"].split()[0]) < float(bare["error on issued"].split()[0])
