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

## ADR-003 · Pretrained weights: largest supervised pretraining fine-tuned on ImageNet-1k; foundation tier as published
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10: "most scientifically defensible and likely to produce strong results")
- Context: PLAN 0.5 names architectures, not weights. Several architectures have more than one pretrained tag in timm 1.0.30.
- Options considered:
    - (a) timm's default tag per architecture: convenient, but follows no stated rule (Swin-T's default is in1k-only while ViT-B's is in21k).
    - (b) ImageNet-1k-only weights everywhere: uniform for the CNN/ViT tiers, but weaker transfer, and uniformity is impossible anyway because the foundation tier is defined by its own pretraining.
    - (c) A stated rule that prefers larger supervised pretraining.
- Decision: rule (c).
    - Supervised backbones (tiers 3–4 and the E16 students) use the timm checkpoint with the largest supervised pretraining (in12k/in21k/in22k) that was then fine-tuned on ImageNet-1k. If none exists in timm, they fall back to timm's default in1k tag. Larger supervised pretraining is the standard recipe for transfer to small downstream datasets (Kolesnikov et al., 2020, BiT; Steiner et al., 2022, "How to train your ViT").
    - Self-supervised or language-supervised web-scale weights are reserved for the foundation tier (tier 5), where the pretraining is the object of study: DINOv2 LVD-142M, and CLIP ViT-B/16 with the OpenAI weights, the standard zero-shot reference (Radford et al., 2021).
    - Resulting tags: ResNet-50 `a1_in1k` (no supervised larger-pretraining tag), EfficientNetV2-S `in21k_ft_in1k`, ConvNeXt-T `in12k_ft_in1k`, ViT-B/16 `augreg2_in21k_ft_in1k`, Swin-T `ms_in22k_ft_in1k` (changed from the proposed `ms_in1k`), MobileNetV3-L `miil_in21k_ft_in1k` (changed from `ra_in1k`; expects mean 0 / std 1 normalization, recorded in its config), EfficientNet-Lite0 `ra_in1k` (no larger tag).
    - DINOv2 runs at 224 px with interpolated position embeddings, and EfficientNetV2-S (pretrained at 300 px) runs at 224 px, because the protocol fixes 224 px for E1.
- Evidence: `results/tables/backbones.md`.
- Consequences: pretraining data differ across the ladder (in1k, in12k, in21k/22k, LVD-142M, WIT-400M). E1 reports each model's pretraining source and must not attribute tier differences to architecture alone. ResNet-50 is the one supervised backbone without larger pretraining, which E1 states. Normalization statistics come from each model config, never hard-coded.

## ADR-004 · Split E membership carried over; duplicate Mendeley folder excluded
- Date: 2026-10-10 · Status: Accepted (engineering decision within ADR-001; reported at the 1.1 pause)
- Context: the 1.1 inventory shows `mendeley_iphone11pro` and `mendeley_split_d` hold byte-identical files. The two Split E folders contain every screened candidate, not only the images retained in Split E.
- Options considered: keep both Mendeley folders (double-counts one set); keep one.
- Decision: `mendeley_iphone11pro` becomes `excluded`; `mendeley_split_d` stays external. Split E membership comes from `data/annotations/split_e_membership.csv`, derived from the earlier project's `split_e_stats.csv` (columns `in_split_e`, `exclusion_code`, source label and organization). Only rows with `in_split_e = true` are used, all labelled falsified in the source records.
- Evidence: `results/tables/raw_inventory.md` (cross-source duplicate table; unique images per source).
- Consequences: external sets are `mendeley_split_c` and `mendeley_split_d` (authentic) and the two Split E folders (counterfeit, membership-filtered).

## ADR-005 · Every experiment table carries an external-set block
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10, option c at the 1.1 pause)
- Context: in the Kaggle development pool, file format, colour mode, resolution and file size each separate the classes completely (`results/tables/raw_inventory.md`: counterfeit = PNG/RGBA, median 454x548 px, 325 KB; authentic = JPEG/RGB, median 225x225 px, 6 KB). In-distribution AUROC may therefore saturate for many models and stop telling them apart.
- Options considered: (a) proceed unchanged; (b) label-blind harmonization (resolution cap + fixed JPEG re-encode) in 1.2; (c) proceed unchanged and add external-set evaluation to every experiment.
- Decision: (c). Data and protocol stay as planned. Every experiment that trains a model also scores its 15 fold models (no retraining) on the external sets and reports specificity on `mendeley_split_c` and `mendeley_split_d` (authentic only) and recall on Split E (counterfeit only) at the val-selected threshold. Each is reported as mean ± std over the 15 runs, plus a 95% bootstrap CI on the pooled predictions. External images are never used for training, selection, thresholds or calibration.
- Evidence: `results/tables/raw_inventory.md`.
- Consequences: CLAUDE.md §5 lists the external block under metrics. Run folders gain `external_predictions.csv`. E4's frequency branch and its 896 px tiling are revisited in the Phase 3 plan (authentic images are at most 312 px wide).

## ADR-006 · Near-duplicate threshold 10; figures showing dataset images stay local
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10, pause 2)
- Context: PLAN 1.4 asks Sophie to confirm the pHash threshold from a contact sheet before splits are made. The contact sheet shows thumbnails of dataset images, the repository is public, and the Kaggle licence is unconfirmed.
- Options considered: thresholds 8 (default), 10, 12, 14 (`results/tables/dedup_summary.md`).
- Decision: threshold 10. At distance 10 several cross-group pairs are the same photograph (missed duplicates would leak across partitions); the cost is one false merge of two different products with different labels. Figures that show dataset images are written to `results/figures/local/` (git-ignored) and regenerated by their scripts; only tables and figures without dataset images are committed.
- Evidence: `results/tables/dedup_summary.md` (groups per threshold; at 10: 569 groups, largest 4, one mixed-label group).
- Consequences: `manifest_v1.csv` group ids use threshold 10; the mixed-label group is assigned to one partition as a unit like any other group.

## ADR-007 · Packaging types: Sophie's three-type scheme, earlier labels imported
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10, option a)
- Context: PLAN 1.7 specified five types (box, blister, bottle, loose_pill, other). Sophie had already tagged 523 of the 646 development-pool images in an earlier project with three types (`data/metadata/manual_modality_tags.csv` there, SHA-256 `cf3fd1c18bc8ba92e5e49f6f17b4a280b338af263a14aa3d5d2e3a89183f6e56`), matched to v1 ids by file hash with no conflicts.
- Options considered: (a) adopt the three-type scheme and import the tags; (b) keep five types and review all 646 images with the tags as suggestions.
- Decision: (a). Types are `blister | packaging | other`. CLIP pre-labels are regenerated with the three prompts so the 1.7 agreement table still compares CLIP with the human labels. The earlier tags are imported as final labels (annotator recorded as imported), and Sophie labels the remaining images in the app.
- Evidence: the earlier tag file (hash above); `results/tables/packaging_label_agreement.md` once labelling is complete.
- Consequences: the E4 router is 3-way with 3 expert heads. In the earlier tags, `other` is almost entirely authentic (144 vs 2), so packaging type is strongly associated with the class label. A router or packaging-conditioned model can partly learn the label through it, which E4 must report. E8 qualification (≥ 30 images per class per type) is decided from the final labels.
- Amendment (2026-10-10, Sophie): two further types, `both` (blister and packaging in one image) and `not_a_drug` (image does not show a medicine), so the scheme is `blister | packaging | both | not_a_drug | other`. The template prompt cannot phrase these, so CLIP uses `a photo of a medicine blister pack together with its box` and `a photo of something that is not a medicine`. Imported `other` labels predate the new types and can be re-reviewed with `--review other`. The router's class set and the treatment of `not_a_drug` images are decided in the Phase 3 plan.
