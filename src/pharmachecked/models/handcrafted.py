"""Handcrafted features + classical classifiers (E1 tier 1).

Features per image (after the evaluation geometry: resize short side + center crop):
    colour histogram, 32 bins per RGB channel, normalized          (96)
    HOG, 9 orientations, 16x16-pixel cells, 2x2-cell blocks (Dalal & Triggs, 2005)
    uniform LBP histogram, P=8, R=1 (Ojala et al., 2002)            (10)
Features are standardized with a scaler fitted on the training split only, then classified by
logistic regression or XGBoost (Chen & Guestrin, 2016).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HIST_BINS = 32
LBP_P, LBP_R = 8, 1


def extract_features(image: Image.Image, size: int = 224) -> np.ndarray:
    """Colour histogram + HOG + uniform-LBP feature vector of one RGB image."""
    w, h = image.size
    scale = size / min(w, h)
    resized = image.resize(
        (max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR
    )
    left, top = (resized.width - size) // 2, (resized.height - size) // 2
    arr = np.asarray(resized.crop((left, top, left + size, top + size)).convert("RGB"))
    hist = np.concatenate(
        [np.histogram(arr[..., c], bins=HIST_BINS, range=(0, 256))[0] for c in range(3)]
    ).astype(float)
    hist /= hist.sum() / 3.0
    gray = rgb2gray(arr)
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), LBP_P, LBP_R, method="uniform")
    lbp_hist = np.bincount(lbp.astype(int).ravel(), minlength=LBP_P + 2).astype(float)
    lbp_hist /= lbp_hist.sum()
    return np.concatenate([hist, hog_vec, lbp_hist]).astype(np.float32)


def build_classifier(cfg: Any, seed: int) -> Pipeline:
    """Scaler + classifier pipeline from a ``configs/model/handcrafted_*.yaml`` config."""
    if cfg.classifier == "logreg":
        clf: Any = LogisticRegression(C=float(cfg.C), max_iter=int(cfg.max_iter))
    elif cfg.classifier == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=int(cfg.n_estimators),
            max_depth=int(cfg.max_depth),
            learning_rate=float(cfg.learning_rate),
            subsample=float(cfg.subsample),
            colsample_bytree=float(cfg.colsample_bytree),
            random_state=seed,
            n_jobs=int(cfg.get("n_jobs", 4)),
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"unknown classifier {cfg.classifier!r}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
