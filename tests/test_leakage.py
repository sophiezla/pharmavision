"""Rule 3 for derived images: synthetic and evasive images inherit their source's group and fold,
come only from development images, and enter a run's training set only when their source is in
that run's training partition (PLAN 7.2 acceptance criterion)."""

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from pharmachecked.data.toy import ToyData
from pharmachecked.jobs import compose_job
from pharmachecked.synthesis.generate import generate, load_synthesis_config, synthetic_store


@pytest.fixture(scope="module")
def run_cfg(tmp_path_factory: pytest.TempPathFactory) -> object:
    root = tmp_path_factory.mktemp("leak")
    cfg = compose_job(
        [
            "experiment=e14_synthetic",
            "model=efficientnetv2_s",
            "data.dataset=toy",
            f"paths.results_dir={root.as_posix()}",
            f"paths.toy_dir={(root / 'toy').as_posix()}",
            "synthetic.ratio=2.0",
        ]
    )
    syn = load_synthesis_config()
    for key, value in {"ocr_backend": "stub", "inpaint.backend": "stub", "per_source": 2}.items():
        OmegaConf.update(syn, key, value)
    generate(cfg, syn, limit=8)
    return cfg


def test_synthetic_images_inherit_group_and_fold(run_cfg: object) -> None:
    from pharmachecked.training.task import load_data

    manifest, _, splits, _ = load_data(run_cfg)
    table = synthetic_store(run_cfg).read()
    assert len(table) == 8 * 2 * 2  # sources x per_source x generators
    src = manifest.set_index("image_id").loc[table["source_image_id"]]
    assert (table["group_id"].to_numpy() == src["group_id"].to_numpy()).all()
    assert (table["fold"].to_numpy() == src["fold"].astype(int).to_numpy()).all()
    assert (src["label"] == 0).all() and (src["split"] == "dev").all()
    assert set(table["source_image_id"]) <= splits.all_ids()
    assert (table["label"] == 1).all() and (table["split"] == "synthetic").all()


def test_training_set_takes_synthetic_images_only_from_the_training_partition(
    run_cfg: object,
) -> None:
    from pharmachecked.training.task import fold_datasets, load_data

    _, _, splits, _ = load_data(run_cfg)
    data = fold_datasets(run_cfg)
    _real, *extra = data.train.datasets
    assert extra, "ratio 2 adds synthetic images"
    table = synthetic_store(run_cfg).read().set_index("image_id")
    train = set(splits.ids(int(run_cfg.fold), "train"))
    for ds in extra:
        assert set(table.loc[list(ds.image_ids), "source_image_id"]) <= train
    for part in ("val", "heldout"):
        ids = set(getattr(data, part).image_ids)
        assert not ids & set(table.index)  # evaluation stays real
    assert data.info["n_train_extra"] == sum(len(ds) for ds in extra)


def test_synthetic_only_drops_real_counterfeits(run_cfg: object) -> None:
    from pharmachecked.training.task import fold_datasets

    cfg = OmegaConf.merge(run_cfg, {"synthetic": {"only": True, "ratio": 1.0}})
    real = fold_datasets(cfg).train.datasets[0]
    assert set(real.labels) == {0}


def test_race_pool_keeps_training_sources_only(tmp_path: Path, toy: ToyData) -> None:
    from pharmachecked.adversarial.race import EVASIVE_COLUMNS, evasive_images

    manifest = pd.read_csv(toy.manifest, dtype={"image_id": str})
    sources = list(manifest["image_id"][:3])
    for k in (1, 2):
        folder = tmp_path / f"round{k}"
        folder.mkdir()
        pd.DataFrame(
            {
                "image_id": [f"r{k}a", f"r{k}b"],
                "source_image_id": sources[:2],
                "label": 1,
                "split": "synthetic",
                "group_id": "g",
                "fold": 0,
                "source_dataset": f"race_round{k}",
                "packaging_type": "",
            }
        )[list(EVASIVE_COLUMNS)].to_csv(folder / "evasive.csv", index=False)
    out = evasive_images(tmp_path, 2, [sources[0], sources[2]])
    assert [ids for _, ids, _ in out] == [["r1a"], ["r2a"]]
