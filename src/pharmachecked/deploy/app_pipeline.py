"""The decision chain behind the app (PLAN 8.6): quality checks -> OOD guard -> model ->
temperature -> conformal set -> label.

Every fitted quantity comes from one fold model's own ``val`` / ``cal`` predictions
(``reliability_predictions.csv``, Rule 3), with the methods chosen in Phase 5: MSP guard at 95%
acceptance of ``val`` (E11, ``e11_guard_choice``), temperature on ``cal`` (E10; T = 1 when not
identifiable) and LAC Mondrian conformal at alpha 0.10 on ``cal`` (ADR-028). The same
:func:`decide` runs offline on saved logits and online in the app, so both give identical
outputs for the same logit.

Quality checks (``e09_quality_limits``): an input is sent back for a retake when one of three
measurable statistics (mean brightness, RMS contrast, sharpness = variance of the Laplacian) is
worse than that of development images corrupted at the corruption's quality-limit severity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from pharmachecked.evaluation.calibration import (
    apply_temperature,
    fit_temperature,
    temperature_identifiable,
)
from pharmachecked.evaluation.conformal import calibrate, prediction_sets
from pharmachecked.evaluation.ood import msp_score, threshold_at_acceptance

ALPHA = 0.10  # E11.md: alpha of the deployed system
CONFORMAL = ("lac", "mondrian")  # ADR-028
RELIABILITY_FILE = "reliability_predictions.csv"
# measurable statistic per E9 corruption (lower = worse for all three)
QUALITY_STATS = {"brightness": "brightness", "contrast": "contrast", "gaussian_blur": "sharpness"}

AUTHENTIC = "Likely authentic"
COUNTERFEIT = "Likely counterfeit"
UNCERTAIN = "Uncertain — verify manually"


@dataclass(frozen=True)
class AppParams:
    """Fitted parameters of one fold model's decision chain."""

    run: str
    temperature: float
    guard_threshold: float  # on msp_score: accept when score <= threshold
    quantiles: tuple[float, float]  # LAC Mondrian, per class
    quality: dict[str, float] = field(default_factory=dict)  # statistic -> minimum allowed
    alpha: float = ALPHA

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form."""
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    """The app's output for one image."""

    label: str
    reason: str
    prob_counterfeit: float | None  # calibrated; None when not computed (quality / guard)
    prediction_set: tuple[int, ...]


def fit_params(
    reliability: pd.DataFrame, run: str, quality: dict[str, float] | None = None
) -> AppParams:
    """Fit guard, temperature and conformal quantiles from one run's val / cal rows.

    Args:
        reliability: The run's ``reliability_predictions.csv`` (only ``val`` and ``cal`` used).
        run: Run folder name, recorded in the parameters.
        quality: Quality thresholds from :func:`fit_quality_thresholds`.

    Returns:
        The fitted parameters.
    """
    val = reliability[reliability["split"] == "val"]
    cal = reliability[reliability["split"] == "cal"]
    z, y = cal["logit"].to_numpy(), cal["y_true"].to_numpy()
    t = fit_temperature(z, y) if temperature_identifiable(z, y) else 1.0
    guard = threshold_at_acceptance(msp_score(val["logit"].to_numpy()))
    q = calibrate(y, apply_temperature(z, t), *CONFORMAL, ALPHA)
    return AppParams(run, float(t), float(guard), (float(q[0]), float(q[1])), quality or {})


def decide(logit: float, params: AppParams, quality_failures: list[str] | None = None) -> Decision:
    """Map one logit (and failed quality checks) to the app's label.

    Order as in E11's end-to-end system: rejected inputs get no prediction; an accepted input
    gets a label only when its conformal set is a single class, otherwise it abstains.
    """
    if quality_failures:
        return Decision(UNCERTAIN, "retake photo: " + ", ".join(quality_failures), None, ())
    if float(msp_score(np.array([logit]))[0]) > params.guard_threshold:
        return Decision(UNCERTAIN, "unfamiliar image (OOD guard)", None, ())
    p = float(apply_temperature(np.array([logit]), params.temperature)[0])
    members = prediction_sets(np.array([p]), CONFORMAL[0], np.array(params.quantiles))[0]
    pset = tuple(int(c) for c in np.flatnonzero(members))
    if pset == (1,):
        return Decision(COUNTERFEIT, "single-class conformal set", p, pset)
    if pset == (0,):
        return Decision(AUTHENTIC, "single-class conformal set", p, pset)
    return Decision(UNCERTAIN, f"model abstains (conformal set of size {len(pset)})", p, pset)


def offline_decisions(reliability: pd.DataFrame, params: AppParams, split: str) -> pd.DataFrame:
    """:func:`decide` on the saved logits of one split (the offline pipeline)."""
    part = reliability[reliability["split"] == split]
    rows = []
    for image_id, logit in zip(part["image_id"], part["logit"], strict=True):
        d = decide(float(logit), params)
        rows.append({"image_id": image_id, "logit": float(logit)} | asdict(d))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- quality checks


def quality_stats(image: Image.Image) -> dict[str, float]:
    """Brightness (mean), contrast (RMS) and sharpness (Laplacian variance) on 0-1 grey."""
    grey = image.convert("L")
    g = np.asarray(grey, dtype=np.float64) / 255.0
    lap = np.asarray(grey.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128)))
    lap = (lap.astype(np.float64) - 128.0) / 255.0
    return {
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "sharpness": float(lap.var()),
    }


def fit_quality_thresholds(
    images: list[tuple[str, Image.Image]], limits: pd.DataFrame, corruption_cfg: object
) -> dict[str, float]:
    """Median statistic of development images corrupted at each quality-limit severity.

    Args:
        images: ``(image_id, image)`` pairs at model-input geometry (development images only).
        limits: ``e09_quality_limits`` rows (``corruption``, ``max_severity``).
        corruption_cfg: ``configs/eval/corruptions.yaml``.

    Returns:
        Statistic name -> minimum allowed value.
    """
    from pharmachecked.data.corruptions import corrupt

    out = {}
    for row in limits.itertuples():
        stat = QUALITY_STATS.get(str(row.corruption))
        if stat is None:
            continue
        values = [
            quality_stats(
                corrupt(im, iid, str(row.corruption), int(row.max_severity), corruption_cfg)
            )[stat]
            for iid, im in images
        ]
        out[stat] = float(np.median(values))
    return out


def quality_failures(image: Image.Image, params: AppParams) -> list[str]:
    """Names of the statistics below their minimum (empty = passes)."""
    stats = quality_stats(image)
    return [k for k, minimum in params.quality.items() if stats[k] < minimum]


def load_reliability(run_root: Path) -> pd.DataFrame:
    """The run's reliability predictions (string image ids)."""
    return pd.read_csv(run_root / RELIABILITY_FILE, dtype={"image_id": str})
