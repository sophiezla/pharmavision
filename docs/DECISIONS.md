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

## ADR-008 · E1 tier 7 (vision-language model) is skipped
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10, option c at CHECKPOINT 1)
- Context: PLAN 2.6 tier 7 needed a model and API choice before implementation.
- Options considered: (a) a Claude model through the Anthropic API; (b) an open-weight VLM on the GPU PC; (c) skip.
- Decision: (c). E1 runs tiers 1–6.
- Evidence: none (scope decision).
- Consequences: no API key or response caching is needed; E1 tables have no VLM row.

## ADR-009 · E4 router classes: blister, packaging, both, other
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10, CHECKPOINT 1)
- Context: ADR-007 left the router class set open. `not_a_drug` has two images, and `both` and `other` are almost entirely authentic.
- Options considered: all five types; merge `both`, `not_a_drug` and `other`; Sophie's choice below.
- Decision: the router predicts four classes, `blister | packaging | both | other`. `not_a_drug` is not a router class: those images contribute no router loss (they still train the authenticity head and stay flagged for the E3 audit). The E4 model has four expert heads.
- Evidence: `results/tables/data_card_counts.md` (packaging type by class).
- Consequences: E4 must report how much of the router's benefit could come from the type-label association (ADR-007).

## ADR-010 · GPU PC: RTX 5060 (8 GB)
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10)
- Context: Phase 2 is the first phase that needs the GPU PC; the plan's E4 memory criterion assumed 16 GB.
- Decision: full runs target an NVIDIA RTX 5060. It needs PyTorch CUDA 12.8+ builds (`cu128` index). Memory-dependent settings (batch size, E4's memory criterion) are sized for 8 GB, using gradient checkpointing or accumulation where needed and reporting it.
- Evidence: `results/tables/bench_gpu-pc.md` once the benchmark has run there.
- Consequences: batch sizes are configurable per model and are checked against the GPU PC's benchmark before each experiment.

## ADR-011 · GPU runs travel as a no-command kit
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10: unzip plus two double-clicks; uv, Split E in the private kit and Windows approved)
- Context: Sophie asked that nothing sent to the GPU PC require AI, commands or steps beyond unzipping and two clicks.
- Options considered: manual setup with git and pip (the earlier RUNNING_ELSEWHERE procedure); a self-contained kit.
- Decision: `scripts/make_gpu_kit.py` builds `PRIVATE_gpu_kit_<name>_<commit>.zip`. It contains a `git archive` snapshot with `SNAPSHOT.json` (commit + file hashes, so runs record provenance without git), the development images except the locked test set, all external images including Split E (private, Sophie's machines only), uv (hash-verified) and two batch files. `1_SETUP.bat` installs Python 3.13 and the pinned packages (PyTorch cu130: the only CUDA build of torch 2.14.1 for Windows/Python 3.13, and the RTX 5060 needs CUDA >= 12.8) and checks the GPU. `2_RUN.bat` benchmarks, runs the job list with skip/resume, keeps Windows awake and writes a RESULTS zip.
- Evidence: laptop dry run of a self-test kit (CPU, toy data) before the first real kit.
- Consequences: deterministic evaluation geometry is cached per image to avoid re-decoding large external photos in every run. Results return through `unpack_results.py` with hash verification.

## ADR-012 · Model selection breaks validation-AUROC ties with validation loss
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10, option b)
- Context: in the E1 timing batch, validation AUROC reached 1.000 after the first (warm-up) epoch for five full fine-tuning arms. Early stopping on val AUROC alone then keeps the epoch-0 weights and stops at epoch 8, so model selection is arbitrary at the in-distribution ceiling (ADR-005), and the external block would be measured on barely trained checkpoints.
- Options considered: (a) keep AUROC-only selection; (b) highest val AUROC, ties broken by lower val loss; (c) val loss only.
- Decision: (b), `train.selection: auroc_then_loss`. An epoch is an improvement if its val AUROC is higher, or equal within 1e-12 with a strictly lower val loss (BCE). Early-stopping patience (7) counts epochs without an improvement under this rule. Only the fold's val split is used; external data never influence selection.
- Evidence: `results/tables/e01_timing.md`; per-epoch val AUROC in the timing run folders (moved to `results/timing/e01/`).
- Consequences: the 13 timing runs used the old rule. They are kept as timing evidence in `results/timing/e01/` and are not E1 results; the full E1 batch re-runs fold 0 / seed 0. Estimated E1 GPU time rises from 13.5 h to at most 25.7 h (every trained run at 30 epochs). The aggregator refuses duplicate runs for one arm/fold/seed.

## ADR-013 · One GPU kit per phase; decisions inside a phase are pre-registered and automatic
- Date: 2026-10-10 · Status: Accepted (Sophie, 2026-10-10: "go ahead with this plan")
- Context: each GPU round trip needs Sophie to carry kits and results between machines. Most experiments depend on earlier results, so per-experiment kits would need many trips.
- Options considered: one kit per experiment, with a decision on the laptop between kits; one kit per phase, with every intra-phase decision written as a rule before the kit runs.
- Decision: one kit per phase where dependencies allow. Decision rules (which arms advance, tuning objectives and search spaces, which backbones go to the next experiment) are written in the experiment pages before the kit is built and applied by the pipeline on the GPU PC. Each decision is recorded in `results/pipeline/`. Rankings use development-pool OOF predictions only: pooled AUROC, exact ties broken by pooled log-loss. External sets never drive a decision (ADR-005). The phase checkpoint becomes a review of the returned results. Steps that need Sophie (label audit, concept labels, synthetic-image quality review) sit between kits.
- Evidence: `docs/experiments/E01.md` (decision rules), `docs/experiments/E02.md`.
- Consequences: Phase 2 runs as one kit (`scripts/phase2_pipeline.py`: E1 → decisions → Optuna → tuned re-runs → E2), dry-run on the laptop with a self-test profile before handover. Tuned settings come back as JSON and are frozen into configs on the laptop with an ADR.

## ADR-014 · E2 backbones: ConvNeXt-T and EfficientNetV2-S (pre-registered rule)
- Date: 2026-10-13 · Status: Accepted (applied automatically under ADR-013)
- Context: PLAN 2.8 requires an ADR naming the two best E1 backbones for E2.
- Options considered: none at decision time; the rule was fixed in E01.md before the runs.
- Decision: `convnext-tiny` and `efficientnetv2-s`, the two best distinct pretrained backbones (default-settings arms) by pooled OOF AUROC with exact ties broken by pooled OOF log-loss, development pool only.
- Evidence: `results/pipeline/e01_decisions.json`, `results/pipeline/e01_ranking.csv`, `results/tables/e01_summary.md`.
- Consequences: E2 ran on these two backbones in the same kit (`results/tables/e02_summary.md`). Pairwise tests (`results/tables/e01_pairwise.md`) show most pretrained arms are statistically indistinguishable from the best, so the rule picked among near-equals; the external sets played no part.

## ADR-015 · E4 global branch: EfficientNetV2-S with LoRA rank 8
- Date: 2026-10-13 · Status: Accepted (Sophie, 2026-10-13)
- Context: PLAN 2.11 requires an ADR choosing the global-branch backbone and adaptation strategy for E4.
- Options considered: every E2 arm (`results/tables/e02_summary.md`). All except efficientnetv2-s/lp are statistically indistinguishable in the development pool (`results/tables/e02_pairwise.md`).
- Decision: `efficientnetv2_s` with `model.mode=lora`, `model.lora_rank=8` (pointwise-conv targets, ADR/E02.md), backbone LR multiplier 1.0, default training recipe. It is the top arm under the same pre-registered rule used in Phase 2 (pooled OOF AUROC, exact ties → pooled OOF log-loss; development pool only) and trains 0.65 M parameters.
- Evidence: `results/tables/e02_summary.md`, `results/tables/e02_pairwise.md`, `results/figures/e02_auroc_by_strategy.png`.
- Consequences: E5 (training recipe) runs on this global-only model; E4's global branch uses it.

## ADR-016 · Tuned settings recorded, defaults kept downstream
- Date: 2026-10-13 · Status: Accepted (Sophie, 2026-10-13)
- Context: PLAN 2.7 says to freeze the Optuna-chosen settings in the model configs. The tuned re-runs did not improve on the default settings beyond the confidence intervals (`results/tables/e01_summary.md`; cnn-small/ft-tuned is lower), and the E4 global branch (ADR-015) is an untuned E2 arm.
- Options considered: freeze tuned settings into the configs (changing later experiments); record them and keep the default recipe.
- Decision: record and keep defaults. The model configs are unchanged; tuned settings stay in the study files below and in the `*-tuned` run configs.

| Study | Best settings | Fold-0 val log-loss | Trials complete / pruned / failed |
|---|---|---|---|
| `e01_cnn-small_ft` | train.lr=0.007819, train.weight_decay=0.00445 | 0.255 | 8 / 22 / 0 |
| `e01_convnext-tiny_ft` | train.backbone_lr_mult=0.0675, train.lr=0.0007751, train.weight_decay=0.006524 | 1.53e-05 | 12 / 18 / 0 |
| `e01_dinov2-vits14_ft` | train.backbone_lr_mult=0.01102, train.lr=0.000964, train.weight_decay=0.009332 | 1.65e-05 | 14 / 16 / 0 |
| `e01_handcrafted-xgb_default` | model.colsample_bytree=0.8704, model.learning_rate=0.09664, model.max_depth=6, model.n_estimators=542, model.subsample=0.9196 | 0.0611 | 30 / 0 / 0 |
| `e01_swin-tiny_ft` | train.backbone_lr_mult=0.8459, train.lr=7.502e-05, train.weight_decay=0.1261 | 1.82e-05 | 12 / 18 / 0 |

- Observation: for the three fine-tuning studies the best fold-0 val log-loss is of order 1e-5, with many trials near zero (swin-tiny: 8 of 12 completed trials below 1e-3, `results/optuna/e01_swin-tiny_ft_trials.csv`). The objective saturated as AUROC had, so the chosen trial among near-zero losses carries little information. This supports keeping the defaults.
- Evidence: `results/optuna/e01_*_best.json`, `results/optuna/e01_*_trials.csv`, `results/tables/e01_summary.md`.
- Consequences: later phases use `configs/train/default.yaml`.

## ADR-017 · E4 design: 448 px with 2x2 tiles as the main model; frequency branch kept
- Date: 2026-10-13 · Status: Accepted (Sophie, 2026-10-13)
- Context: PLAN 3.5 specifies an 896 px padded-square input with a 4x4 grid of 224 px tiles. Authentic development images are at most 312 px wide and counterfeit ones about 450 px (`results/tables/data_card_counts.md`), so 896 px tiles would be upsampled about 3x for one class and 2x for the other, a class-dependent interpolation cue. The frequency branch (FFT + 8x8 block-DCT) will see JPEG block artifacts present only in authentic images (ADR-005).
- Options considered: tiling (a) 896/4x4 main, 448/2x2 ablation (plan); (b) 448/2x2 main, 896/4x4 ablation. Frequency branch: keep; drop; keep with JPEG re-compression augmentation.
- Decision: (b) 448 px padded square with 2x2 tiles of 224 px is the main E4 model; 896 px / 4x4 is the ablation arm. The frequency branch is kept as planned, computed on the 448 px grayscale input (the planned 512 px centre crop exceeds the input); the ''- frequency'' ablation and the external block show whether it relies on the compression cue.
- Evidence: `results/tables/data_card_counts.md`, `results/tables/raw_inventory.md`.
- Consequences: about 5 backbone passes per image instead of 17; PLAN 3.5/3.6 updated.

## ADR-018 · Data-loader worker processes persist for a whole run
- Date: 2026-10-13 · Status: Accepted (Sophie asked for a faster Phase 3 kit, 2026-10-13)
- Context: Phase 2 runs on the GPU PC took about 15 s per epoch for EfficientNetV2-S, while the GPU benchmark (`results/tables/bench_gpu-pc.md`, 311 train and 1,140 eval img/s) puts the compute at about 1.1 s. On Windows every DataLoader worker is a new Python process that re-imports torch; the trainer created new workers twice per epoch (training and validation). Measured on the laptop, fold 0 of v2 (329 images): 4 restarted workers 22-25 s per pass, no workers 2.4-3.8 s, 4 persistent workers 1.3-1.4 s after the first pass.
- Decision: training workers persist for the whole run (`persistent_workers`), the validation loader is built once per run (at most 2 persistent workers), and one-off prediction passes (held-out, calibration, external sets) load in the main process. The SupCon loader persists its workers as well. Models, data, recipes and decision rules are unchanged.
- Consequences:
    - Phase 3 is several times faster. The kit estimate drops from 3-4 days to roughly 12-24 hours.
    - With workers, the augmentation random stream differs from Phase 2's (a different draw of the same distributions); runs stay reproducible from scratch for a given seed. Comparisons inside E5 and E4 all use the new loader (E5 re-runs its `default` arm); the E3 before/after table compares runs that differ in both data version and augmentation draw, which is noted there.
    - A run resumed after an interruption with workers is no longer bit-identical to an uninterrupted one in its augmentation draws (exact resume still holds with `num_workers: 0`, which the tests use).

## ADR-019 · E5 training recipe: the default (no component kept)
- Date: 2026-10-19 · Status: Accepted (applied automatically under ADR-013)
- Context: PLAN 3.3 / E05.md require an ADR fixing the training recipe used by E4 and later phases.
- Options considered: the 8 single-change arms of E05.md, each compared with a re-run `default` under the pre-registered rule (Holm-corrected DeLong p < 0.05 and a positive AUROC-gain CI).
- Decision: the default recipe (`configs/data/transforms.yaml`, BCE, no MixUp/CutMix, no SupCon). No component met the rule, so no `combo` arm ran.
- Evidence: `results/pipeline/e05_decisions.json`, `results/tables/e05_vs_reference.md`, `results/tables/e05_summary.md`.
- Consequences: E4 trained with the default recipe (`results/pipeline/e04_arms.json`). Six of the eight components raised pooled ECE (CI above 0 in `e05_vs_reference.md`), which is also a reason not to add them for the Phase 5 reliability work.

## ADR-020 · Final Phase 3 model: the global-only EfficientNetV2-S (no multi-branch component kept)
- Date: 2026-10-19 · Status: Proposed (pre-registered rule, to be confirmed by Sophie at CHECKPOINT 3)
- Context: E04.md fixes the rule for the model carried into Phase 4 as the "full E4 model".
- Options considered: `full` and its 7 ablation / design arms; the global-only reference `e05:efficientnetv2-s/default`.
- Decision: `full` does not significantly beat `no-local`, `no-frequency` or `no-router`, so no branch is kept; no design variant significantly beats `full`. Under the fallback the final model is the global-only reference: EfficientNetV2-S + LoRA r8, backbone LR multiplier 1.0, default recipe, 224 px.
- Evidence: `results/tables/e04_vs_reference.md`, `results/tables/e04_pairwise.md`, `results/tables/e04_summary.md`.
- Consequences: Phase 4 compares against this model; the multi-branch code stays in the package and the E4 results are reported as a negative result. On the external sets the multi-branch arms trade Mendeley specificity for Split E recall (descriptive only, ADR-005). The planned gate-weight figure (PLAN 3.7) is not yet produced (needs the E4 checkpoints from the GPU PC).
