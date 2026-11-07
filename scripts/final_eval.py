"""Final, one-time evaluation on the locked test set (PLAN 10.2). The only code allowed to read it.

Refuses to run unless ``--i-understand-this-is-final`` is passed and ``results/LOCK`` does not
exist; on completion it writes ``results/LOCK`` so it can never run again (Rule 2).

Frozen models (ADR-030): the final model (``e05_efficientnetv2-s_default``), the E1 best model
retrained on data v2 (``p4ref_convnext-tiny_ft``) and the E16 student
(``e16_mobilenetv3-large_kd-t4``), 15 fold models each. Every fold model scores the locked test
images; its threshold (Youden on ``val``), temperature (``cal``; T = 1 when not identifiable)
and LAC Mondrian conformal quantiles (``cal``, alpha 0.10, ADR-028) are fitted on its own dev
partitions, which are predicted here with the same code. Reported: per-run mean ± sd of every
CLAUDE.md section 5 metric, the 15-model ensemble (mean probability) with 2,000-resample
stratified bootstrap CIs, and paired DeLong tests of the final model's ensemble against the
others (Holm).

Before anything is opened, every checkpoint is located; ``--check-only`` stops there.

Usage:
    python scripts/final_eval.py --i-understand-this-is-final [--check-only]
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pharmachecked.config import REPO_ROOT

EXIT_REFUSED = 2
EXIT_NOT_READY = 3
FROZEN = {  # ADR-030
    "final model (EfficientNetV2-S)": ("e05", "efficientnetv2-s", "default"),
    "E1 best (ConvNeXt-T, data v2)": ("p4ref", "convnext-tiny", "ft"),
    "E16 student (MobileNetV3-L, KD T=4)": ("e16", "mobilenetv3-large", "kd-t4"),
}
REFERENCE = "final model (EfficientNetV2-S)"
ALPHA = 0.10
N_BOOT = 2000


def _runs() -> dict[str, list]:
    from pharmachecked.evaluation.aggregate import collect_runs

    out = {}
    for name, (exp, model, variant) in FROZEN.items():
        recs = [
            r
            for r in collect_runs(REPO_ROOT / "results" / "runs", exp)
            if r.model == model and r.variant == variant
        ]
        out[name] = sorted(recs, key=lambda r: (r.fold, r.seed))
    return out


def _checkpoints(runs: dict[str, list]) -> tuple[dict[str, Path], list[str]]:
    from pharmachecked.models.loading import run_checkpoint

    found, missing = {}, []
    for name, recs in runs.items():
        if len(recs) != 15:
            missing.append(f"{name}: {len(recs)} runs, expected 15")
        for r in recs:
            ckpt = run_checkpoint(r.root)
            if ckpt is None:
                missing.append(r.root.name)
            else:
                found[r.root.name] = ckpt
    return found, missing


def _logits(model, transform, ids: list[str], batch: int = 32) -> np.ndarray:
    import torch
    from PIL import Image

    images = REPO_ROOT / "data" / "processed" / "images"
    out = []
    for k in range(0, len(ids), batch):
        xs = []
        for i in ids[k : k + batch]:
            with Image.open(images / f"{i}.png") as im:
                xs.append(transform(im.convert("RGB")))
        with torch.no_grad():
            out.append(model(torch.stack(xs))["logits"].float().reshape(-1).numpy())
    return np.concatenate(out)


def _evaluate_run(record, ckpt: Path, test_ids: list[str]) -> pd.DataFrame:
    from omegaconf import OmegaConf

    from pharmachecked.config import CONFIG_DIR
    from pharmachecked.data.transforms import build_eval_transform
    from pharmachecked.models.loading import load_torch_model, run_config
    from pharmachecked.training.task import load_data

    paths = OmegaConf.load(CONFIG_DIR / "paths" / "local.yaml")
    cfg = run_config(record.root, paths, "cpu")
    model = load_torch_model(cfg, ckpt)
    size = int(cfg.data.get("img_size") or cfg.model.img_size)
    tcfg = OmegaConf.load(CONFIG_DIR / "data" / "transforms.yaml")
    transform = build_eval_transform(tcfg, list(cfg.model.mean), list(cfg.model.std), size)
    manifest, _, splits, _ = load_data(cfg)
    labels = manifest.set_index("image_id")["label"]
    parts = {
        "val": list(splits.ids(int(cfg.fold), "val")),
        "cal": list(splits.ids(int(cfg.fold), "cal")),
        "locked": test_ids,
    }
    frames = [
        pd.DataFrame(
            {
                "image_id": ids,
                "part": part,
                "y_true": [int(labels[i]) for i in ids],
                "logit": _logits(model, transform, ids),
                "run": record.root.name,
            }
        )
        for part, ids in parts.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _per_run(preds: pd.DataFrame) -> pd.DataFrame:
    from pharmachecked.evaluation.calibration import (
        apply_temperature,
        fit_temperature,
        temperature_identifiable,
    )
    from pharmachecked.evaluation.conformal import calibrate, prediction_sets, set_metrics
    from pharmachecked.evaluation.metrics import binary_metrics, youden_threshold

    rows = []
    for run, part in preds.groupby("run"):
        p = {k: g for k, g in part.groupby("part")}
        val, cal, held = p["val"], p["cal"], p["locked"]
        prob = lambda g: 1 / (1 + np.exp(-g["logit"].to_numpy()))  # noqa: E731
        thr = youden_threshold(val["y_true"], prob(val))
        row = {"run": run} | binary_metrics(held["y_true"], prob(held), thr)
        z, y = cal["logit"].to_numpy(), cal["y_true"].to_numpy()
        t = fit_temperature(z, y) if temperature_identifiable(z, y) else 1.0
        pc_cal = apply_temperature(z, t)
        pc = apply_temperature(held["logit"].to_numpy(), t)
        from pharmachecked.evaluation.metrics import brier, expected_calibration_error

        row |= {
            "temperature": t,
            "ece_after_t": expected_calibration_error(held["y_true"], pc),
            "brier_after_t": brier(held["y_true"], pc),
        }
        q = calibrate(y, pc_cal, "lac", "mondrian", ALPHA)
        row |= {
            f"conformal_{k}": v
            for k, v in set_metrics(held["y_true"], prediction_sets(pc, "lac", q)).items()
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _pm(v: pd.Series, digits: int = 3) -> str:
    v = v.dropna()
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}" if len(v) > 1 else "n/a"


def _ci(i) -> str:
    return f"{i.estimate:.3f} [{i.low:.3f}, {i.high:.3f}]"


def _tables(all_preds: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from pharmachecked.evaluation.metrics import (
        auprc,
        auroc,
        expected_calibration_error,
        sensitivity_at_specificity,
    )
    from pharmachecked.evaluation.stats import bootstrap_ci, delong_test, holm

    summary, ensembles = [], {}
    for name, preds in all_preds.items():
        per = _per_run(preds)
        held = preds[preds["part"] == "locked"].assign(prob=lambda d: 1 / (1 + np.exp(-d["logit"])))
        ens = held.groupby("image_id", as_index=False).agg(
            y_true=("y_true", "first"), prob=("prob", "mean")
        )
        ensembles[name] = ens
        y, p = ens["y_true"].to_numpy(), ens["prob"].to_numpy()
        summary.append(
            {
                "model": name,
                "runs": len(per),
                "images": len(ens),
                "counterfeit": int(y.sum()),
                "AUROC (ensemble)": _ci(bootstrap_ci(y, (p,), auroc, N_BOOT)),
                "AUPRC (ensemble)": _ci(bootstrap_ci(y, (p,), auprc, N_BOOT)),
                "sens@95%spec (ensemble)": f"{sensitivity_at_specificity(y, p, 0.95):.3f}",
                "ECE (ensemble, uncalibrated)": f"{expected_calibration_error(y, p):.4f}",
                "AUROC (per run)": _pm(per["auroc"]),
                "balanced acc (per run, val threshold)": _pm(per["balanced_accuracy"]),
                "macro-F1 (per run)": _pm(per["macro_f1"]),
                "sensitivity (per run)": _pm(per["sensitivity"]),
                "specificity (per run)": _pm(per["specificity"]),
                "ECE after T (per run)": _pm(per["ece_after_t"], 4),
                "Brier after T (per run)": _pm(per["brier_after_t"], 4),
                "conformal coverage (per run, alpha 0.10)": _pm(per["conformal_coverage"]),
                "coverage authentic": _pm(per["conformal_coverage_authentic"]),
                "coverage counterfeit": _pm(per["conformal_coverage_counterfeit"]),
                "mean set size": _pm(per["conformal_mean_set_size"]),
            }
        )
    rows, intervals = [], []
    ref = ensembles[REFERENCE]
    for name, ens in ensembles.items():
        if name == REFERENCE:
            continue
        m = ref.merge(ens, on="image_id", suffixes=("_ref", ""))
        y, a, b = m["y_true"].to_numpy(), m["prob"].to_numpy(), m["prob_ref"].to_numpy()
        ci = bootstrap_ci(y, (a, b), lambda yy, q, r: auroc(yy, q) - auroc(yy, r), N_BOOT)
        intervals.append(ci)
        rows.append(
            {
                "model": name,
                "reference": REFERENCE,
                "AUROC diff": _ci(ci),
                "delong_p": delong_test(y, a, b).p_value,
            }
        )
    pair = pd.DataFrame(rows)
    if len(pair):
        reject, corrected = holm(pair["delong_p"].tolist())
        pair["holm_p"] = corrected
        pair["better"] = [bool(r and c.low > 0) for r, c in zip(reject, intervals, strict=True)]
        pair["worse"] = [bool(r and c.high < 0) for r, c in zip(reject, intervals, strict=True)]
    ens_all = pd.concat([e.assign(model=n) for n, e in ensembles.items()], ignore_index=True)
    return pd.DataFrame(summary), pair, ens_all


def main(argv: list[str] | None = None) -> int:
    """Check the guards, then evaluate once; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-understand-this-is-final", dest="final", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true", help="locate checkpoints, open nothing"
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    lock = args.results_dir / "LOCK"
    if not args.final:
        sys.stderr.write("refused: pass --i-understand-this-is-final to open the locked test set\n")
        return EXIT_REFUSED
    if lock.exists():
        sys.stderr.write(f"refused: {lock} exists; the locked test set was already evaluated\n")
        return EXIT_REFUSED
    runs = _runs()
    ckpts, missing = _checkpoints(runs)
    if missing:
        sys.stderr.write(
            f"not ready: {len(missing)} checkpoints missing (install the MODELS bundle), e.g. "
            f"{missing[0]}; the locked test set was not opened\n"
        )
        return EXIT_NOT_READY
    # file existence only: no image is read before every prerequisite is in place
    from pharmachecked.data.splits import load_locked_test

    images = REPO_ROOT / "data" / "processed" / "images"
    ids = load_locked_test(REPO_ROOT / "splits" / "v1.json", acknowledge_final=True)
    absent = [i for i in ids if not (images / f"{i}.png").is_file()]
    if absent:
        sys.stderr.write(
            f"not ready: {len(absent)} of {len(ids)} locked-test images are not in {images} "
            "(normalize them from data/raw); the locked test set was not opened\n"
        )
        return EXIT_NOT_READY
    if args.check_only:
        sys.stderr.write(
            f"ready: {len(ckpts)} checkpoints found; the locked test set was not opened\n"
        )
        return EXIT_NOT_READY

    from pharmachecked.data.splits import load_locked_test
    from pharmachecked.evaluation.aggregate import to_markdown

    test_ids = sorted(load_locked_test(REPO_ROOT / "splits" / "v1.json", acknowledge_final=True))
    all_preds = {
        name: pd.concat(
            [_evaluate_run(r, ckpts[r.root.name], test_ids) for r in recs], ignore_index=True
        )
        for name, recs in runs.items()
    }
    summary, pair, ens = _tables(all_preds)
    tables = args.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "final_test.csv", index=False, lineterminator="\n")
    pair.to_csv(tables / "final_test_pairwise.csv", index=False, lineterminator="\n")
    ens.to_csv(
        tables / "final_test_predictions.csv", index=False, lineterminator="\n", float_format="%.7g"
    )
    note = (
        "Generated once by `scripts/final_eval.py` (PLAN 10.2) on the locked test set "
        f"({len(test_ids)} images); frozen models in ADR-030. Ensemble = mean probability of the "
        "15 fold models with 95% stratified bootstrap CIs (2000 resamples, seed 0); per-run = "
        "mean ± sd over the 15 fold models, each with its own val threshold, cal temperature and "
        "LAC Mondrian conformal quantiles (alpha 0.10).\n\n"
    )
    (tables / "final_test.md").write_text(
        "# Locked test set: final evaluation\n\n"
        + note
        + to_markdown(summary)
        + "\n\n## Paired DeLong vs the final model (ensembles, Holm)\n\n"
        + to_markdown(pair),
        encoding="utf-8",
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    ).stdout.strip()
    lock.write_text(f"locked test set evaluated {dt.datetime.now().isoformat()} at {commit}\n")
    sys.stderr.write(f"done: wrote {tables / 'final_test.md'} and {lock}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
