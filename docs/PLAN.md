# PharmaChecked v2 — Execution Plan

Read `CLAUDE.md` first. This plan is executed top to bottom. Each step has an **acceptance criterion** (AC); tick the box only when the AC is met, and append the commit hash: `- [x] 1.3 ... (a1b2c3d)`. Stop at every **CHECKPOINT** and follow CLAUDE.md §13.

Notation: `E{N}` = experiment number; "full protocol" = 5 outer folds × 3 seeds with OOF predictions, as defined in CLAUDE.md §5. "Comparison models" (used in Phase 4) = handcrafted best, best pretrained CNN, best ViT, DINOv2 global-only, full E4 model.

---

## Phase 0 · Environment and repository

Before writing any code, ask Sophie: (a) where the raw data lives and how it is organized, (b) the repo name and GitHub remote, (c) whether she prefers Weights & Biases (offline mode) or MLflow for tracking. Record the answers in `docs/DECISIONS.md` as ADR-001.

Answered 2026-10-09 (ADR-001): raw data copied into `data/raw/`; development pool = Kaggle `kaggle_fake_real` only, Mendeley + Split E external; remote `sophiezla/pharmavision`; MLflow; Python 3.13; `scripts/tasks.py` stands in for `make` on Windows.

- [x] **0.1 Scaffold.** (1830ade) Create the directory tree in CLAUDE.md §3, `pyproject.toml` (package `pharmachecked`, `src/` layout), `.gitignore`, `README.md` stub, `Makefile`.
    - Core deps: `torch`, `torchvision`, `timm`, `hydra-core`, `omegaconf`, `numpy`, `pandas`, `scikit-learn`, `scikit-image`, `xgboost`, `imagehash`, `Pillow`, `matplotlib`, `optuna`, `statsmodels`, `torchmetrics`, `open_clip_torch`, `peft`, `cleanlab`, `grad-cam`, `captum`, `onnx`, `onnxruntime`, `gradio`, and the chosen tracker.
    - Extras: `[dev]` = `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`; `[anomaly]` = `anomalib`; `[synthesis]` = `diffusers`, `transformers`, `accelerate`, `easyocr`; `[adversarial]` = `torchattacks`.
    - Pin exact versions after the first successful install (`pip freeze` → constraints file).
    - AC: `make setup` succeeds from a clean clone.
- [x] **0.2 Quality tooling.** (45c6d15) `.pre-commit-config.yaml` (ruff check, ruff format, end-of-file, trailing whitespace, check-added-large-files ≤ 1 MB). `.github/workflows/ci.yml` runs `make lint` and `make test`.
    - AC: a deliberately bad commit is blocked locally; CI passes on a clean commit.
- [x] **0.3 Utilities.** (d94038e; fix 9142d0e) `utils/seed.py`, `utils/logging.py`, `utils/env.py` (git hash, dirty flag, versions, GPU), `utils/run_dir.py` (creates run folder per CLAUDE.md §7), `utils/io.py`.
    - AC: unit tests for seed reproducibility (two runs → identical tensors) and run-folder contents.
- [x] **0.4 Hydra skeleton.** (8aa5628) `configs/config.yaml`, `paths/local.yaml`, `paths/colab.yaml`, and the `pharmachecked.run` entry point that composes config, creates the run folder, saves `config.yaml` + `env.json`.
    - AC: `python -m pharmachecked.run experiment=noop` writes a run folder.
- [x] **0.5 Backbone availability.** (9ff0906) Script `scripts/check_backbones.py` that calls `timm.list_models` / `open_clip.list_pretrained` and instantiates each planned backbone on CPU: ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny, ViT-B/16, Swin-Tiny, DINOv2 ViT-S/14, DINOv2 ViT-B/14, CLIP ViT-B/16, MobileNetV3-Large, EfficientNet-Lite0. Write the verified exact names into `configs/model/*.yaml`.
    - AC: script prints all names resolved and parameter counts; output saved to `results/tables/backbones.md`.
- [x] **0.6 Docs skeleton.** (acc286e) Create `docs/EXPERIMENT_LOG.md`, `docs/DECISIONS.md` (with ADR-001), `docs/experiments/`, `docs/phase_reports/`.
    - AC: files exist with templates from CLAUDE.md §11.
- [x] **0.7 Remote-run bundle.** (ba872d1) `scripts/bench_machine.py` (per-backbone train/eval throughput → `results/tables/bench_<host>.md`), `scripts/pack_results.py` / `scripts/unpack_results.py` (zip run folders without checkpoints, with a SHA-256 manifest; unpack verifies before installing), `docs/RUNNING_ELSEWHERE.md` (clone → CUDA torch → data → job list → pack → bring back).
    - AC: `test_pack_results.py` round trip preserves hashes and excludes checkpoints; benchmark table for this machine committed.

**CHECKPOINT 0.**

---

## Phase 1 · Data pipeline and annotations

- [x] **1.1 Inventory.** (5b6f9aa) Read-only scan of `data/raw/`; report counts per source, class, file format, size distribution, corrupt/unreadable files. Save `results/tables/raw_inventory.md`.
    - AC: inventory reported to Sophie before any processing.
- [x] **1.2 Normalization.** (f876542) `data/normalize.py`: open each image, convert to RGB (alpha composited onto white), apply EXIF orientation, re-encode every image to PNG (lossless) in `data/processed/images/{image_id}.png`. `image_id` = first 16 hex chars of the SHA-256 of the original bytes.
    - AC: test that output mode is RGB for RGBA, L, P and CMYK inputs; processed count = readable raw count.
- [x] **1.3 Manifest.** (e110ce2) `data/manifest.py` builds `data/processed/manifest_v1.csv` with columns `image_id, raw_path, source_dataset, label, width, height, sha256, phash, group_id, packaging_type, split, fold`.
    - AC: schema validated by a test (`pandera` not required; a plain assertion function is fine).
- [x] **1.4 Near-duplicate grouping.** (1b314b0, e2691dc; threshold 10 confirmed, ADR-006) `data/dedup.py`: 64-bit pHash; union-find over pairs with Hamming distance ≤ threshold (default 8; configurable). Exact SHA-256 duplicates collapse to one manifest row. Save a contact sheet of the 50 closest cross-group pairs and the 50 farthest within-group pairs to `results/figures/dedup_check.png`.
    - AC: Sophie confirms the threshold from the contact sheet (ask at the checkpoint); group counts reported.
- [x] **1.5 Splits.** (3c385d2) `data/splits.py` + `scripts/make_splits.py` produce `splits/v1.json`: locked test (15% of groups, stratified), 5 outer folds over the rest via `StratifiedGroupKFold`, inner train/val/cal per fold (75/12.5/12.5).
    - AC: `tests/test_splits.py` and `tests/test_test_lock.py` pass; class balance table per partition saved to `results/tables/split_balance.md`.
- [x] **1.6 Datasets and transforms.** (6520e2c) `data/datasets.py` (`ManifestDataset` that receives indices from a split object; can return raw image, label, metadata), `data/transforms.py` (default recipe: RandomResizedCrop scale 0.7–1.0, rotation ±10°, brightness/contrast jitter 0.1, no flips, ImageNet or backbone-specific normalization), eval transform (resize + center crop/pad).
    - AC: test that the dataset never yields an index from another partition.
- [x] **1.7 Packaging-type labels.** (d8e6143, 0b6d095, 9fed76a + this commit; five-type scheme, ADR-007) `data/annotation_app.py` (Gradio): CLIP zero-shot pre-labels each image as `blister | packaging | other` (three-type scheme, ADR-007; originally `box | blister | bottle | loose_pill | other`) with prompts `"a photo of a medicine {type}"`; Sophie's earlier labels are imported and she labels the rest in the app; writes `data/annotations/packaging_type.csv` (`image_id, clip_label, clip_conf, final_label, annotator, timestamp`).
    - AC: app launches; after Sophie annotates, every image has `final_label`; CLIP-vs-human agreement saved to `results/tables/packaging_label_agreement.md`.
- [ ] **1.8 Concept labels.** Same app, concept mode, on a stratified subset of 300 images from the development pool (never the test set): `text_error, color_mismatch, print_blur, logo_distortion, layout_misalignment, missing_security_feature, seal_or_package_damage`, each `present | absent | unsure`. Writes `data/annotations/concepts.csv`.
    - AC: subset selection script committed; annotations saved (Sophie may complete this during Phases 2–5; it gates only E13).
- [x] **1.9 OOD reference sets.** (c6a4e21, 0b6d095 + this commit; Places365 replaced by Open Images V7 scenes, approved 2026-10-10) Assemble `data/ood/far/` (~500 natural-scene and texture images from a public dataset available through `torchvision.datasets` or the Hugging Face Hub, e.g. DTD textures + a scene dataset) and `data/ood/near/` (~500 non-medicine product-packaging images from a public grocery/retail product dataset). Propose specific datasets and their licences to Sophie before downloading.
    - AC: manifests `data/ood/*/manifest.csv` with source and licence columns.
- [x] **1.10 Data card.** (b9c42a0 + this commit) `docs/DATA_CARD.md`: sources + licences, counts per class / source / packaging type / partition, preprocessing steps, dedup threshold, known limitations.
    - AC: every number in it produced by a script and pasted from its output.
- [x] **1.11 Smoke test.** (bc9bd6a) `make smoke`: tiny CNN, 1 epoch, 64 training images from fold 0, CPU; complete run folder.
    - AC: passes locally and in CI in < 3 minutes.

**CHECKPOINT 1.**

---

## Phase 2 · Baselines (E1–E3)

### Shared infrastructure (build before E1)

- [x] **2.1 Trainer.** (d25fe25) `training/trainer.py`: plain PyTorch loop (no Lightning) with AdamW, cosine schedule + 1-epoch linear warmup, mixed precision, gradient clipping 1.0, early stopping on val AUROC with ties broken by val loss (patience 7, max 30 epochs; ADR-012), best-checkpoint restore, resume support, per-epoch logging to `train_log.csv`. Separate learning rate for backbone vs head (`train.backbone_lr_mult`).
- [x] **2.2 Evaluation.** (979a09d, 832107f) `evaluation/metrics.py` (all metrics in CLAUDE.md §5), `evaluation/stats.py` (bootstrap CI, fast DeLong, McNemar exact, Holm), `evaluation/aggregate.py` (combines runs → per-config mean ± std and pooled OOF metrics with CIs; writes CSV + Markdown).
    - AC: `test_metrics.py` and `test_stats.py` pass.
- [x] **2.3 Model registry.** (d385ffb, 1197e58; LoRA added with E2) `models/registry.py`, `models/backbones.py` (timm/open_clip wrappers returning `{"logits","embedding"}`), `models/heads.py`, `models/cnn_small.py` (the original 3-conv CNN, rebuilt), `models/handcrafted.py` (color histogram 3×32 bins + HOG + uniform LBP histogram → standardized → logistic regression / XGBoost; scalers fitted on train only).
    - AC: `test_models.py` passes for all registered models.
- [x] **2.4 Figure utilities.** (3dd1d4a) `scripts/make_figures.py` reads only `results/tables/`; consistent style module; saves PNG (300 dpi) + PDF.

### E1 · Baseline ladder

- [x] **2.5** (02e1cae) Write `docs/experiments/E01.md`.
- [x] **2.6** (0a7cdbb) Run tiers under the full protocol at 224 px with the default recipe:
    1. handcrafted + logistic regression; handcrafted + XGBoost
    2. original CNN (trained from scratch)
    3. ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny (full fine-tuning, backbone LR × 0.1)
    4. ViT-B/16, Swin-Tiny (same)
    5. DINOv2 ViT-S/14, DINOv2 ViT-B/14, CLIP ViT-B/16 (linear probe AND full fine-tuning)
    6. zero-shot CLIP ViT-B/16 with prompt ensemble (`"a photo of authentic medicine packaging"` vs `"a photo of counterfeit medicine packaging"`, plus 4 paraphrases each, averaged); evaluated on the same held-out folds (no training)
    7. ~~one vision–language model, zero-shot~~ — skipped (Sophie, 2026-10-10; ADR-008).
- [x] **2.7** (0a7cdbb; ADR-016) Optuna (≤ 30 trials, fold 0 inner val) for LR, weight decay, and backbone LR multiplier for the best model of each tier; freeze settings.
- [x] **2.8** (1714be0; ADR-014) Aggregate → `results/tables/e01_summary.{csv,md}`; figure `e01_auroc_vs_params` (log x-axis); paired DeLong of every model vs the best, Holm-corrected → `e01_pairwise.md`.
    - AC: all tiers complete; EXPERIMENT_LOG entry; ADR naming the two best backbones.

### E2 · Adaptation strategy

- [x] **2.9** (053e606) Write `docs/experiments/E02.md`.
- [x] **2.10** (0a7cdbb) For the best two E1 backbones: linear probe; last-2-blocks fine-tuning; full fine-tuning (backbone LR × 0.1); LoRA rank 8 and 16 on attention `qkv` projections via `peft` (verify target module names on the actual timm model).
- [x] **2.11** (1714be0; ADR-015) Aggregate → `e02_summary`; figure `e02_auroc_by_strategy`; table includes trainable params and wall-clock per run.
    - AC: ADR choosing the global-branch backbone + strategy for E4.

### E3 · Label audit

- [x] **2.12** (4a682bf) Run `cleanlab.filter.find_label_issues` on pooled OOF probabilities of the E1-best model (average over seeds). Export flagged images to a review sheet in the annotation app.
- [x] **2.13** (both removed, 2026-10-13) Sophie reviews; decisions saved to `data/annotations/label_audit.csv` (`image_id, original_label, decision {keep|relabel|remove}, new_label, reason`).
- [x] **2.14** (data v2 built 2026-10-13; retrain = stage 0 of the Phase 3 kit) If any relabel/remove: create `manifest_v2.csv` and `splits/v2.json`; retrain E1-best on v2; table before vs after. Locked-test membership is copied unchanged from v1 (ADR-002); only the development pool is re-folded (same seeds and procedure). Test images are not audited.
    - AC: ADR recording the data version used from now on.

**CHECKPOINT 2.** Tag `v0.1-baselines`.

---

## Phase 3 · Proposed model (E4–E5)

E5 runs first on the global-only model (cheap); its winning recipe is then used for every E4 arm.

### E5 · Training recipe

- [ ] **3.1** Write `docs/experiments/E05.md`.
- [ ] **3.2** On the E2-chosen global-only model, one factor at a time vs default: + horizontal flip; + strong color jitter incl. hue 0.1; + RandAugment (n=2, m=9); + MixUp (α=0.2); + CutMix (α=1.0); focal loss (γ=2, class-balanced α); label smoothing 0.1; SupCon pretraining (20 epochs, temperature 0.1, projection head 128-d) then linear head.
- [ ] **3.3** Aggregate ΔAUROC and ΔECE with CIs vs default → `e05_summary`.
    - AC: ADR fixing the recipe (combine only components that individually improved; verify the combination with one extra full-protocol run).

### E4 · Multi-branch model

- [ ] **3.4** Write `docs/experiments/E04.md`.
- [ ] **3.5 Components** (each with unit tests):
    - `models/mil.py`: gated-attention MIL pooling (Ilse et al., 2018), input `(B, N, D)` → bag embedding `(B, D')` + attention `(B, N)` summing to 1.
    - `models/frequency.py`: grayscale model input (448 px, ADR-017; originally a 512 px center crop) → `log(1 + |FFT|)` centered magnitude + 8×8 block-DCT coefficient-energy map (resized to the same grid), stacked as 2 channels → 4-block CNN (32-64-128-256, BN, ReLU, max-pool) → global average pool → 256-d.
    - `models/router.py`: linear layer on the global embedding → 4-way packaging-type logits over `blister | packaging | both | other`; `not_a_drug` images get no router loss (ADR-009); auxiliary cross-entropy weight 0.3.
    - `models/fusion.py`: gated fusion — project each branch to 256-d, gate = softmax(MLP([e_g; e_l; e_f])) over 3 branches, fused = Σ gate_i · e_i; also `concat` mode for the ablation.
    - `models/multibranch.py`: input 448 px (ADR-017; originally 896 px); global branch = 224 px resize → backbone; local branch = 2×2 grid of 224 px tiles (originally 4×4) → the same backbone (shared weights, configurable) → MIL; expert heads = 4 linear heads (one per router class), output logits = Σ_k router_prob_k · head_k(fused) (soft) or argmax head (hard). Returns `logits, embedding, gate_weights, router_probs, mil_attention`.
    - AC: forward pass on GPU with batch 8 fits in the GPU PC's 8 GB (RTX 5060, ADR-010) with mixed precision (report peak memory); if not, enable gradient checkpointing on the backbone and/or gradient accumulation, and report.
- [ ] **3.6 Arms** (full protocol, E5 recipe): full; − local; − frequency; − router (single head); concat fusion; separate local backbone; 4×4 tiles at 896 px (ablation; ADR-017 swapped it with the main 2×2/448 design); hard routing; global-only (reference = E2 winner).
- [ ] **3.7** Aggregate → `e04_ablation` (ΔAUROC vs full, CI, Holm-corrected DeLong); figure `e04_gate_weights_by_packaging` (mean gate weight per branch per packaging type from OOF outputs).
    - AC: ADR defining the final model; components that did not improve are dropped and reported.

**CHECKPOINT 3.** Tag `v0.2-model`.

---

## Phase 4 · Generalization (E6–E9)

All experiments use the five comparison models.

### E6 · Data efficiency

- [ ] **4.1** `docs/experiments/E06.md`. Subsample each outer training portion to 10/25/50/100% (grouped, stratified; 3 draws per fraction, seed = draw index); val/cal/held-out unchanged.
- [ ] **4.2** Aggregate; figure `e06_learning_curves` (log-x, mean line + 95% CI band per model).

### E7 · Cross-dataset transfer

- [ ] **4.3** `docs/experiments/E07.md`. Train on all development images of source A (inner train/val/cal split by group), test on all development images of source B; then reverse. 3 seeds. Reference = within-source 5-fold results.
- [ ] **4.4** Table `e07_transfer`: in-source AUROC, cross-source AUROC, drop, CI.

### E8 · Open-set (leave-one-packaging-type-out)

- [ ] **4.5** `docs/experiments/E08.md`. For each packaging type with ≥ 30 images in both classes (report which qualify), train on the other types and test on the held-out type. 3 seeds.
- [ ] **4.6** Methods: five comparison models; PatchCore via `anomalib` (backbone `wide_resnet50_2`, fitted on authentic training images only; verify API for the installed version); prototype verifier (`models/prototype.py`: L2-normalized DINOv2 embeddings, score = mean cosine distance to k=5 nearest authentic training embeddings).
- [ ] **4.7** Table `e08_open_set` (type × method); figure `e08_seen_vs_unseen`.

### E9 · Robustness

- [ ] **4.8** `data/corruptions.py`: Gaussian blur, motion blur, Gaussian noise, JPEG compression, brightness, contrast, rotation, perspective warp, synthetic glare (additive elliptical highlight); 5 severity levels each, parameters listed in `configs/eval/corruptions.yaml`. Unit test: severity 0 = identity; severity monotonic in a simple image statistic.
- [ ] **4.9** Evaluate the trained fold models (no retraining) on corrupted held-out folds; seeds fixed for corruption randomness.
- [ ] **4.10** Figure `e09_severity_curves` (one panel per corruption); table `e09_mean_corruption_auroc`; table `e09_quality_limits` = highest severity per corruption at which the final model's AUROC ≥ 0.85 (threshold configurable; confirm with Sophie).

**CHECKPOINT 4.**

---

## Phase 5 · Reliability (E10–E11)

### E10 · Calibration, selective, conformal

- [ ] **5.1** `docs/experiments/E10.md`. `evaluation/calibration.py`: temperature scaling fitted on `cal` by minimizing NLL (LBFGS); ECE, Brier, reliability-diagram data.
- [ ] **5.2** Uncertainty sources: softmax confidence; 3-seed deep ensemble; MC dropout (20 passes; dropout enabled only in heads). `evaluation/selective.py`: risk–coverage curves and AURC.
- [ ] **5.3** `evaluation/conformal.py`: split conformal with LAC and APS scores, marginal and class-conditional (Mondrian), α ∈ {0.05, 0.10}; quantile fitted on `cal`; evaluated on held-out folds. A set of size 2 = abstain.
    - AC: `test_conformal.py` passes.
- [ ] **5.4** Tables `e10_calibration` (ECE/Brier before/after per model), `e10_conformal` (coverage overall + per class, mean set size, singleton rate, abstention rate); figures `e10_reliability_diagrams`, `e10_risk_coverage`.
    - AC: chosen conformal method's per-class coverage within 2 pp of target; ADR choosing the method.

### E11 · OOD guard

- [ ] **5.5** `evaluation/ood.py`: MSP, energy score (T=1), Mahalanobis (class-conditional means, shared covariance with shrinkage, fitted on train embeddings), kNN (k=50, cosine, normalized penultimate features), ensemble disagreement. Threshold = 95% acceptance of in-distribution `val`.
- [ ] **5.6** Table `e11_ood` (AUROC, FPR@95TPR; near vs far).
- [ ] **5.7** End-to-end: `evaluation/pipeline.py` chains OOD guard → model → temperature → conformal on a mixed stream (held-out folds + all OOD images). Table `e11_end_to_end`: rejection rate, abstention rate, error rate on issued predictions; full system vs bare classifier.

**CHECKPOINT 5.** Tag `v0.3-reliable`. (Minimum viable path continues at Phase 8 if Sophie decides so.)

---

## Phase 6 · Interpretability (E12–E13)

### E12 · Faithfulness

- [ ] **6.1** `interpret/cams.py` (Grad-CAM, Grad-CAM++ via `grad-cam`), integrated gradients (Captum), `interpret/attention_rollout.py`, MIL tile-attention upsampled to image size, random-map baseline.
- [ ] **6.2** `evaluation/faithfulness.py`: deletion and insertion AUC (5% steps, blurred-image baseline); model-randomization sanity check (cascading re-initialization from the top layer; Spearman correlation of maps vs original).
- [ ] **6.3** Localization on E14 synthetic counterfeits (run after 7.4): pointing-game accuracy and IoU of the top-15% map region vs edit mask.
- [ ] **6.4** Table `e12_faithfulness`; figure `e12_examples` (4 images × 6 methods).

### E13 · Concept bottleneck + TCAV

- [ ] **6.5** `models/concept_bottleneck.py`: backbone → 7 sigmoid concept outputs → logistic-regression label head on concepts only; independent, sequential and joint training (Koh et al., 2020); concept loss masked where labels are missing or `unsure`.
- [ ] **6.6** Intervention: replace the k most uncertain predicted concepts with annotated values, k = 0..7; task AUROC per k.
- [ ] **6.7** `interpret/tcav.py`: concept activation vectors from annotated positives vs random negatives at 2–3 layers of the final model; 20 random-concept repeats; two-sided t-test.
- [ ] **6.8** Tables `e13_concepts`, `e13_cbm_vs_blackbox`; figures `e13_intervention`, `e13_tcav`.

**CHECKPOINT 6.**

---

## Phase 7 · Synthesis and adversary (E14–E15)

### E14 · Generative counterfeit synthesis

- [ ] **7.1** `synthesis/ocr.py` (EasyOCR text boxes, cached per image).
- [ ] **7.2** `synthesis/procedural.py`: edit types `font_swap, char_substitution, dose_digit_change, local_hue_shift (10–30°), local_blur_or_rehalftone, logo_warp, local_jpeg`; each returns edited image, binary mask, edit type, strength. Only authentic images from training partitions; output inherits `group_id` and fold; saved under `data/processed/synthetic/` with `synthetic_manifest.csv`.
    - AC: `test_leakage.py` covers synthetic inheritance.
- [ ] **7.3** `synthesis/inpaint.py`: Stable Diffusion inpainting (verify model ID and licence; ask Sophie before downloading weights) on masks derived from OCR/logo regions; prompts from a fixed list in config.
- [ ] **7.4** Quality check: random 100 synthetic images per generator into the annotation app; Sophie labels plausible/implausible → `data/annotations/synthetic_quality.csv`.
- [ ] **7.5** Arms (final model, full protocol): synthetic:real counterfeit ratio 0, 0.5, 1, 2, 4 per generator; synthetic-only (no real counterfeits); E8 open-set re-run with the best ratio. Evaluation always on real held-out images only.
- [ ] **7.6** Figure `e14_ratio_curve`; tables `e14_synthetic_only`, `e14_open_set_gain`.

### E15 · Adaptive adversary

- [ ] **7.7** `adversarial/attacks.py`: FGSM, PGD-20 (step ε/8) at L∞ ε ∈ {1,2,4,8}/255 via `torchattacks`, targeted toward "authentic", on counterfeit held-out images.
- [ ] **7.8** `adversarial/physical.py`: optimize global brightness/contrast/hue/saturation parameters only (bounded), and a square patch ≤ 5% of image area at a fixed location, by gradient ascent.
- [ ] **7.9** `adversarial/blackbox.py`: random search over procedural-edit parameters, 200 score-only queries per image.
- [ ] **7.10** Defences: PGD adversarial training (ε=4/255, 7 steps, 50% adversarial per batch); synthetic-edit training (E14 best); both.
- [ ] **7.11** Arms race: 5 rounds; each round the black-box attacker searches evasive edits on training-fold counterfeits, detector retrains including them, fresh search on held-out folds measures evasion rate.
- [ ] **7.12** Figures `e15_success_vs_eps`, `e15_arms_race`; table `e15_defences` (clean AUROC cost vs robust gain).

**CHECKPOINT 7.**

---

## Phase 8 · Deployment (E16 + app)

- [ ] **8.1** `training/distill.py`: students MobileNetV3-Large and EfficientNet-Lite0; loss = 0.5·CE + 0.5·T²·KL(teacher‖student), T ∈ {2,4}; teacher = final model (or 3-seed ensemble, per ADR). Full protocol.
- [ ] **8.2** `deploy/export_onnx.py` (opset documented; numerical check vs PyTorch outputs, max abs diff ≤ 1e-4 for FP32), `deploy/quantize.py` (ONNX Runtime static INT8, 200 training images for calibration).
- [ ] **8.3** Re-fit temperature and conformal thresholds for every exported model; re-run E10 metrics.
- [ ] **8.4** `deploy/latency.py`: median of 100 runs after 10 warm-ups, batch 1, laptop CPU; phone-browser latency via a minimal ONNX Runtime Web page (record device model and browser).
- [ ] **8.5** Figure `e16_pareto`; table `e16_deployment` (AUROC, ECE, coverage, size MB, latency ms).
- [ ] **8.6 App** (`app/`): photo capture/upload → quality checks from `e09_quality_limits` → OOD guard → model → calibrated probability + conformal set → label `Likely authentic | Likely counterfeit | Uncertain — verify manually` → best E12 heatmap (+ E13 concepts if included) → footer with intended use and model-card link.
    - AC: outputs on 20 held-out images identical to the offline pipeline (automated test); runs in a phone browser.

**CHECKPOINT 8.** Tag `v0.4-deploy`.

---

## Phase 9 · Optional extensions (only if Sophie approves)

- [ ] **9.1 X1 Federated.** `federated/fedavg.py`: sites = source datasets; 4-site non-IID variant split by packaging type; centralized vs local-only vs FedAvg vs FedProx (μ ∈ {0.01, 0.1}); 50 rounds × 1 local epoch. Figure `x1_rounds`; table `x1_summary`.
- [ ] **9.2 X2 Human–AI study.** Build the study app only after Sophie confirms consent forms and review-board approval are in place. Within-subject, 3 counterbalanced blocks × 20 balanced held-out images; store anonymous participant codes only. Analysis script: accuracy, sensitivity, specificity, over-reliance rate, mixed-effects logistic regression (participant random intercept).

---

## Phase 10 · Final evaluation, release, write-up

- [ ] **10.1 Freeze.** ADR listing the final models (E1 best, E4 final, E16 student). Tag `v1.0-rc`.
- [ ] **10.2 Locked test evaluation.** `scripts/final_eval.py --i-understand-this-is-final`; writes `results/tables/final_test.{csv,md}` and `results/LOCK`. Report results exactly as produced.
- [ ] **10.3 Benchmark release.** Frozen splits, evaluation harness, baseline table, leaderboard Markdown, weights (release assets), Dockerfile, `make reproduce` that regenerates every table and figure from run folders.
    - AC: a clean clone + Docker reproduces all tables bit-for-bit from committed run folders.
- [ ] **10.4 Model card** `docs/MODEL_CARD.md`: intended use, out-of-scope uses, data, metrics with CIs, per-packaging-type performance, calibration, known failure modes (E7–E9, E15).
- [ ] **10.5 Paper assets.** `results/paper/` with final versions of Fig 1–8 and Tab 1–4 (see the plan doc's figure list); a `numbers.json` holding every number cited in the text, each with its source file.
- [ ] **10.6** Update README with the full reproduction guide. Tag `v1.0`.

**FINAL CHECKPOINT.**
