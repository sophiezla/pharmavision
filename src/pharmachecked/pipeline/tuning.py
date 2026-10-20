"""Optuna hyperparameter search for one arm (PLAN 2.7; pre-registered in E01.md).

Fold 0 only: train on fold 0's inner train split, score on its inner val split. Objective:
minimize the val log-loss (BCE) of the epoch selected by the ADR-012 rule; with val AUROC
saturated at 1.0 for many arms, the loss is the informative criterion. TPE sampler (seed 0),
median pruning on per-epoch val loss (5 startup trials, 3 warm-up epochs). The study lives in a
SQLite file, so an interrupted search continues where it stopped.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pharmachecked.utils.io import write_json
from pharmachecked.utils.logging import get_logger

log = get_logger(__name__)
SEARCH_SPACES = {
    # kind: {override: (type, low, high, log)}
    "ft": {
        "train.lr": ("float", 1e-5, 1e-3, True),
        "train.weight_decay": ("float", 1e-4, 0.3, True),
        "train.backbone_lr_mult": ("float", 0.01, 1.0, True),
    },
    "lp": {
        "train.lr": ("float", 1e-4, 3e-2, True),
        "train.weight_decay": ("float", 1e-6, 0.1, True),
    },
    "scratch": {
        "train.lr": ("float", 1e-4, 1e-2, True),
        "train.weight_decay": ("float", 1e-5, 0.1, True),
    },
    "logreg": {"model.C": ("float", 1e-3, 100.0, True)},
    "xgboost": {
        "model.max_depth": ("int", 2, 8, False),
        "model.learning_rate": ("float", 0.01, 0.3, True),
        "model.n_estimators": ("int", 100, 800, False),
        "model.subsample": ("float", 0.6, 1.0, False),
        "model.colsample_bytree": ("float", 0.3, 1.0, False),
    },
}


def kind_of(model: str, variant: str) -> str:
    """Search-space kind of an arm (model config name, variant)."""
    if model == "handcrafted_lr":
        return "logreg"
    if model == "handcrafted_xgb":
        return "xgboost"
    if model == "cnn_small":
        return "scratch"
    return "lp" if variant == "lp" else "ft"


def suggest(trial: Any, kind: str) -> dict[str, Any]:
    """Sample one point of the search space as ``{override key: value}``."""
    out: dict[str, Any] = {}
    for key, (kind_, low, high, log_scale) in SEARCH_SPACES[kind].items():
        if kind_ == "int":
            out[key] = trial.suggest_int(key, int(low), int(high))
        else:
            out[key] = trial.suggest_float(key, low, high, log=log_scale)
    return out


def _torch_objective(cfg: Any, trial: Any, work: Path) -> float:
    import optuna

    from pharmachecked.models.registry import build_model
    from pharmachecked.training.task import device_for, fold_datasets
    from pharmachecked.training.trainer import fit
    from pharmachecked.utils.seed import seed_everything

    seed_everything(int(cfg.seed))
    data = fold_datasets(cfg)
    model = build_model(cfg.model)

    def report(epoch: int, _auroc: float, loss: float) -> None:
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    result = fit(
        model,
        data.train,
        data.val,
        cfg.train,
        device_for(cfg),
        work / "ckpt",
        work / "train_log.csv",
        int(cfg.seed),
        on_epoch=report,
    )
    return float(result.best_val_loss)


def _handcrafted_objective(cfg: Any) -> float:
    from pharmachecked.evaluation.metrics import log_loss
    from pharmachecked.models.handcrafted import build_classifier
    from pharmachecked.training.task import _rows, handcrafted_features, load_data, partitions

    manifest, external, splits, images = load_data(cfg)
    parts = partitions(cfg, splits, external, manifest)
    size = int(cfg.model.img_size)
    cache = Path(cfg.paths.results_dir) / "scratch" / "features" / f"handcrafted_{size}"
    train_rows, val_rows = _rows(manifest, parts.train), _rows(manifest, parts.val)
    pipeline = build_classifier(cfg.model, int(cfg.seed))
    pipeline.fit(handcrafted_features(parts.train, images, cache, size), train_rows["label"])
    prob = pipeline.predict_proba(handcrafted_features(parts.val, images, cache, size))[:, 1]
    return log_loss(val_rows["label"], prob)


def run_study(
    name: str,
    base_overrides: list[str],
    kind: str,
    n_trials: int,
    out_dir: Path,
    scratch: Path,
) -> dict[str, Any]:
    """Run (or continue) the study until ``n_trials`` trials completed or were pruned.

    Returns the best trial as ``{"params": {override: value}, "value": val log-loss, ...}`` and
    writes it to ``out_dir/{name}_best.json`` with all trials in ``{name}_trials.csv``.
    """
    import optuna

    from pharmachecked.jobs import compose_job

    out_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        storage=f"sqlite:///{(out_dir / f'{name}.db').as_posix()}",
        study_name=name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        load_if_exists=True,
    )
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

    def objective(trial: Any) -> float:
        params = suggest(trial, kind)
        cfg = compose_job([*base_overrides, *(f"{k}={v}" for k, v in params.items())])
        work = scratch / name / f"trial_{trial.number}"
        try:
            if kind in ("logreg", "xgboost"):
                return _handcrafted_objective(cfg)
            return _torch_objective(cfg, trial, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)  # results/scratch only

    attempts = 0
    while sum(t.state in finished for t in study.trials) < n_trials and attempts < 2 * n_trials:
        attempts += 1
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"study {name}: no trial completed")
    best = min(complete, key=lambda t: t.value)
    record = {
        "study": name,
        "kind": kind,
        "objective": "fold-0 val log-loss at the ADR-012-selected epoch (minimize)",
        "params": dict(best.params),
        "value": best.value,
        "best_trial": best.number,
        "n_complete": len(complete),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "n_failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
    }
    write_json(out_dir / f"{name}_best.json", record)
    study.trials_dataframe().to_csv(
        out_dir / f"{name}_trials.csv", index=False, lineterminator="\n"
    )
    log.info("study %s: best val log-loss %.4f with %s", name, best.value, best.params)
    return record
