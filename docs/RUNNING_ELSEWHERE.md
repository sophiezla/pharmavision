# Running experiments on the GPU PC

The development laptop has no CUDA GPU (see `results/tables/bench_laptop-cpu.md`), so full
experiment runs happen on a separate GPU PC. The run folders come back to the main repository as
a verified zip. This page is the complete procedure; every experiment page
(`docs/experiments/EXX.md`) adds only its job list.

## Rules that still apply

- Run only from a **clean checkout of a pushed commit**. A dirty tree is refused for any run
  other than `fold=0 seed=0` (CLAUDE.md section 7), and `env.json` records the commit.
- Never touch the locked test set: runs use `splits/*.json` through the code, which only loads
  development partitions.
- Weights stay on the GPU PC (`checkpoints/`); bundles never contain them.
- Split E images never leave Sophie's machines and are never committed. Copy them by USB only,
  and only if an experiment needs them.

## One-time setup

1. Install Python 3.13, Git and a current NVIDIA driver. Run `nvidia-smi` and note the CUDA version
   it reports.
2. Clone and check out the branch or tag named in the experiment page:
   ```bash
   git clone https://github.com/sophiezla/pharmavision.git
   cd pharmavision
   git checkout <branch-or-tag>
   ```
3. Install with a CUDA build of the pinned torch version. Pick the CUDA wheel index that
   pytorch.org lists for this torch version and that your driver supports, for example:
   ```bash
   python scripts/tasks.py setup --no-hooks --torch-index https://download.pytorch.org/whl/cu128
   ```
4. Check the GPU is visible:
   ```bash
   .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   On Windows use `.venv\Scripts\python.exe`.
5. Measure this machine (used for every GPU-hour estimate):
   ```bash
   .venv/bin/python scripts/bench_machine.py --name gpu-pc
   ```
   This writes `results/tables/bench_gpu-pc.md`; it travels back in the first results bundle.
6. Run the test suite once: `python scripts/tasks.py test`.

## Data

`data/` is not in git. After Phase 1, the main repository produces a data bundle (processed PNGs
plus manifest; the script is added in Phase 1). Copy it to the GPU PC and unpack it into `data/`
at the repository root. To keep data or weights on another disk, set
`PHARMACHECKED_DATA_DIR` and/or `PHARMACHECKED_CHECKPOINTS_DIR` and pass `paths=remote`.

## Running a batch

1. `git pull` and check out the commit named in the experiment page.
2. Run the job list from the experiment page. Every command ends in `paths=remote`, for example:
   ```bash
   .venv/bin/python -m pharmachecked.run experiment=e01_baselines model=resnet50 paths=remote
   ```
   An interrupted run resumes with `train.resume=true` (available from Phase 2).
3. Note anything unusual (crash, out of memory, restart) in that run's `notes.md`.

## Bringing results back

1. On the GPU PC, pack the run folders (weights are left out automatically):
   ```bash
   .venv/bin/python scripts/pack_results.py --glob "e01_*"
   ```
   This writes `bundles/results_<host>_<timestamp>.zip`.
2. Copy the zip to the laptop.
3. On the laptop, verify and install it:
   ```bash
   python scripts/unpack_results.py path/to/results_<host>_<timestamp>.zip
   ```
   Every file's SHA-256 is checked first. Nothing is written if any file fails, or if a file
   would replace different existing content.
4. Commit the installed run folders (CLAUDE.md section 12).
