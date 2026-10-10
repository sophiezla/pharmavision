# Architecture decision records

One record per decision that changes the plan, the protocol or the final model. Newest last.
Template:

```
## ADR-{NNN} · {decision in one line}
- Date: YYYY-MM-DD · Status: Proposed | Accepted | Superseded by ADR-{MMM}
- Context: {why a decision was needed}
- Options considered: {list}
- Decision: {what was chosen}
- Evidence: {results file paths}
- Consequences: {what changes downstream}
```

---

## ADR-001 · Phase 0 set-up: data source, repository, tracker, toolchain
- Date: 2026-10-09 · Status: Accepted (Sophie, 2026-10-09)
- Context: PLAN Phase 0 asks where the raw data lives, the repository and remote, and the experiment tracker. The development machine has Python 3.13 only and no `make`.
- Options considered:
    - Data: link the existing raw tree read-only vs copy it; development pool = all sources pooled and re-split vs Kaggle only with the other sources as external sets.
    - Tracker: Weights & Biases (offline) vs MLflow.
    - Toolchain: install Python 3.11 + make vs Python 3.13 with a Python task runner.
- Decision:
    - Raw data is copied into `data/raw/` (git-ignored, read-only by Rule 5) from the earlier project's raw tree.
    - The development pool and locked test set come from `data/raw/kaggle_fake_real` only. The Mendeley sets (`mendeley_iphone11pro`, `mendeley_split_c`, `mendeley_split_d`) and Split E (`split_e_harvested`, `split_e_regulatory`) are external evaluation sets. `roboflow`, `balanced_authentic` and `synthetic_counterfeit` are copied but excluded from every partition until Sophie assigns them (shown in the PLAN 1.1 inventory).
    - Exact duplicates collapse to one manifest row. If identical bytes ever carry conflicting labels, the whole duplicate group is dropped and logged.
    - Repository `github.com/sophiezla/pharmavision`; Python package name `pharmachecked`.
    - Tracker: MLflow with a local SQLite backend; run folders remain the source of truth.
    - Python 3.13; `scripts/tasks.py` implements every `make` target, and the Makefile delegates to it.
    - Heavy runs execute on Sophie's GPU PC through the bundle in `docs/RUNNING_ELSEWHERE.md`. Split E images never enter the repository or a bundle (most are permission-restricted).
- Evidence: `results/tables/backbones.md` (environment works); data counts follow in `results/tables/raw_inventory.md` (PLAN 1.1).
- Consequences: E7 cannot run "train on A, test on B, then reverse", because the external sets are single-class (Mendeley authentic-only, Split E counterfeit-only). E7 will be redefined as Kaggle-trained models scored on each external set (specificity on Mendeley, recall on Split E); that redefinition gets its own ADR before Phase 4.

## ADR-002 · Locked-test membership is fixed across manifest versions; scope of Rule 2
- Date: 2026-10-09 · Status: Accepted (Sophie, 2026-10-09)
- Context: PLAN 2.14 regenerates `splits/v2.json` "with the same seeds and procedure" after the label audit. Removing images from the pool would change the 15% group carve and so the test set. Separately, normalization, hashing, dedup and packaging-type annotation must read every image, including test images.
- Options considered: re-carve the test set per manifest version; freeze test membership from v1.
- Decision:
    - The test groups carved in `splits/v1.json` are copied unchanged into every later split version. Only the development pool is re-folded with the same seeds and procedure. Test images are not label-audited, because the audit uses out-of-fold predictions only.
    - Rule 2 forbids feeding test images to a model, or computing anything fitted or evaluated on them, anywhere except `scripts/final_eval.py`. Unfitted, label-blind preprocessing and metadata (PNG normalization, hashing, dedup grouping, packaging-type annotation) may cover all images.
- Evidence: none yet (protocol decision).
- Consequences: `tests/test_splits.py` must assert that test membership is identical across split versions. `tests/test_test_lock.py` checks model-facing code paths, not preprocessing.

## ADR-003 · Pretrained weights: timm default tags; CLIP uses OpenAI weights
- Date: 2026-10-09 · Status: Proposed (awaiting Sophie at CHECKPOINT 0)
- Context: PLAN 0.5 names architectures, not weights. Several architectures have more than one pretrained tag in timm 1.0.30.
- Options considered: timm's default tag per architecture; ImageNet-1k-only weights everywhere (more uniform, usually weaker); larger-pretraining tags (in21k/in22k) where available.
- Decision: the default timm tag for each architecture; for CLIP ViT-B/16, open_clip `ViT-B-16` with `openai` weights (the standard zero-shot reference). DINOv2 runs at 224 px with interpolated position embeddings, and EfficientNetV2-S (pretrained at 300 px) runs at 224 px, as the protocol fixes 224 px for E1.
- Evidence: `results/tables/backbones.md`.
- Consequences: pretraining data differs across the ladder (in1k, in12k, in21k, LVD-142M, WIT-400M), which E1 must state when comparing tiers.
