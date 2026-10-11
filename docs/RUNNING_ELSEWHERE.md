# Running experiments on the GPU PC (the GPU kit)

The development laptop has no CUDA GPU, so full runs happen on Sophie's GPU PC (RTX 5060, 8 GB,
Windows; ADR-010). That PC needs **no commands, no git, no Python install and no AI**: everything
travels as one zip (ADR-011).

## On the GPU PC (Sophie)

1. Unzip `PRIVATE_gpu_kit_<name>_<commit>.zip` anywhere with enough free space (the README in the
   kit gives the number).
2. Double-click **`1_SETUP.bat`**. This runs only once per kit: it downloads Python 3.13 and the
   pinned packages (PyTorch CUDA 13.0 build), then checks the graphics card. It needs internet and
   takes 10–30 minutes. If it says the graphics card is not usable, install the newest NVIDIA
   driver and double-click it again.
3. Double-click **`2_RUN.bat`**. It benchmarks the GPU once, runs the kit's job list (finished
   runs are skipped and interrupted runs resume, so it can simply be double-clicked again), keeps
   Windows awake while working, and writes **`RESULTS_<name>_<timestamp>.zip`** next to itself.
4. Copy the RESULTS zip back to the laptop.

The kit is PRIVATE: it contains Split E images, which never leave Sophie's machines and are never
uploaded or committed. Model weights stay in the kit folder on the GPU PC.

## On the laptop

- **Build a kit:** commit everything, then
  `python scripts/make_gpu_kit.py --jobs jobs/<file>.txt --name <name>`
  (`--selftest` builds a CPU/toy-data kit for a dry run on the laptop).
- **Install results:** `python scripts/unpack_results.py <RESULTS zip>`. It verifies every
  file's SHA-256 before writing, never overwrites different content, and installs runs, the GPU
  benchmark table and job logs. Then commit the run folders (CLAUDE.md §12).

## What the kit contains

- `pharmavision/`: a snapshot of the committed source at one commit (`git archive`), plus
  `SNAPSHOT.json` (commit + SHA-256 of every source file). Runs record that commit, and a
  changed source file marks them dirty.
- `pharmavision/data/processed/images/`: development images (the locked test set excluded) and
  external images.
- `tools/uv.exe`: Astral's uv, checked against its published SHA-256. It installs Python and
  the packages.
- `KIT.json`: the job list to run. `1_SETUP.bat`, `2_RUN.bat`, `README_FIRST.txt`.
