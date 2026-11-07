"""PLAN 10.2: the final-evaluation tables on synthetic predictions (no real data opened)."""

import importlib.util

import numpy as np
import pandas as pd

from pharmachecked.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("final_eval", REPO_ROOT / "scripts" / "final_eval.py")
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)


def _preds(rng: np.random.Generator, strength: float) -> pd.DataFrame:
    frames = []
    for r in range(15):
        for part, n in (("val", 60), ("cal", 60), ("locked", 95)):
            rs = np.random.default_rng(1 if part == "locked" else 100 + r)
            y = rs.integers(0, 2, n)
            z = (2 * y - 1) * strength + rng.normal(0, 1.5, n)
            ids = [f"{part}{i}" for i in range(n)]
            frames.append(
                pd.DataFrame(
                    {"image_id": ids, "part": part, "y_true": y, "logit": z, "run": f"r{r}"}
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_tables_shapes_and_reference() -> None:
    rng = np.random.default_rng(0)
    preds = {name: _preds(rng, 3.0 if name == fe.REFERENCE else 0.0) for name in fe.FROZEN}
    summary, pair, ens = fe._tables(preds)
    assert len(summary) == 3 and set(summary["runs"]) == {15}
    assert len(pair) == 2 and pair["worse"].all()
    assert ens.groupby("model")["image_id"].nunique().eq(95).all()
