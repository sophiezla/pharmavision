"""E13 concept bottleneck: concept targets, masked loss, interventions (PLAN 6.5-6.6)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pharmachecked.data.annotations import CONCEPT_COLUMNS, CONCEPTS
from pharmachecked.training.cbm import ConceptLoss, concept_targets, interventions


def _concepts(path: Path) -> Path:
    rows = [
        {"image_id": "a", **dict.fromkeys(CONCEPTS, "present")},
        {"image_id": "b", **dict.fromkeys(CONCEPTS, "unsure")},
        {"image_id": "c", **dict.fromkeys(CONCEPTS, "")},
        {"image_id": "d", **dict.fromkeys(CONCEPTS, "absent"), CONCEPTS[0]: "unsure"},
    ]
    table = pd.DataFrame(rows).assign(annotator="x", timestamp="")
    table[list(CONCEPT_COLUMNS)].to_csv(path / "concepts.csv", index=False)
    return path / "concepts.csv"


def test_concept_targets_mask_unsure_and_missing(tmp_path: Path) -> None:
    targets, masks = concept_targets(_concepts(tmp_path))
    assert set(targets) == {"a", "d"}  # b (all unsure) and c (unlabelled) carry no label
    assert targets["a"].tolist() == [1.0] * 7 and masks["a"].tolist() == [1.0] * 7
    assert masks["d"][0] == 0 and masks["d"][1:].tolist() == [1.0] * 6
    assert targets["d"].sum() == 0


def test_concept_loss_ignores_masked_values(tmp_path: Path) -> None:
    targets, masks = concept_targets(_concepts(tmp_path))
    loss = ConceptLoss(targets, masks)
    logits = torch.zeros(2, 7, requires_grad=True)
    value = loss(["d", "c"], logits)
    value.backward()
    assert torch.isclose(value, torch.tensor(np.log(2), dtype=torch.float32))
    assert logits.grad[0, 0] == 0 and (logits.grad[1] == 0).all()
    assert (logits.grad[0, 1:] != 0).all()


def test_interventions_replace_the_most_uncertain_annotated_concepts() -> None:
    head = nn.Linear(7, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0, 0, 0, 0, 0, 0]]))
        head.bias.zero_()
    probs = np.array([[0.55, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
    annotated = np.full((1, 7), np.nan)
    annotated[0, 0] = 0.0  # the only annotated concept, and the most uncertain
    out = interventions(probs, annotated, head)
    assert out.shape == (1, 8)
    assert np.isclose(out[0, 0], 1 / (1 + np.exp(-0.55)))
    assert np.allclose(out[0, 1:], 0.5)  # concept 0 set to 0 -> logit 0
