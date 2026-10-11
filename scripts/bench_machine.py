"""Measure training and inference throughput of every configured backbone on this machine.

Used to turn "single-run time x number of runs" into GPU-hour estimates (CLAUDE.md section 12)
for whichever machine runs an experiment. Weights are random (speed does not depend on them),
so nothing is downloaded. Writes ``results/tables/bench_<name>.md``.

Usage:
    python scripts/bench_machine.py --name gpu-pc [--batch 32] [--size 224] [--device cuda]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from importlib import metadata

import timm
import torch
from omegaconf import DictConfig

from pharmachecked.config import CONFIG_DIR, REPO_ROOT
from pharmachecked.utils.io import atomic_write_text, read_yaml
from pharmachecked.utils.logging import configure_logging, get_logger
from pharmachecked.utils.run_dir import slug

log = get_logger("bench_machine")
# open_clip image towers are benchmarked through their timm equivalent (same architecture).
OPEN_CLIP_TO_TIMM = {("ViT-B-16", "openai"): "vit_base_patch16_clip_224"}


def timm_arch(cfg: DictConfig) -> str:
    """Return the timm architecture (no pretrained tag) to benchmark for a model config."""
    if cfg.source == "timm":
        return str(cfg.timm_name).split(".")[0]
    return OPEN_CLIP_TO_TIMM[(cfg.open_clip_arch, cfg.open_clip_pretrained)]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    arch: str, device: torch.device, batch: int, size: int, warmup: int, iters: int
) -> tuple[float, float, float | None]:
    """Return (train img/s, eval img/s, peak CUDA memory in GB or None) for one architecture.

    Training is a full forward/backward/AdamW step on a 1-logit head, under bf16 autocast on
    CUDA (fp32 on CPU), matching full fine-tuning in E1.
    """
    kwargs = {"img_size": size} if "dinov2" in arch or "clip" in arch else {}
    model = timm.create_model(arch, pretrained=False, num_classes=1, **kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, size, size, device=device)
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for i in range(warmup + iters):
        if i == warmup:
            _sync(device)
            start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp:
            loss = model(x).float().mean()
        loss.backward()
        opt.step()
    _sync(device)
    train_ips = iters * batch / (time.perf_counter() - start)

    model.eval()
    with torch.inference_mode(), amp:
        for i in range(warmup + iters):
            if i == warmup:
                _sync(device)
                start = time.perf_counter()
            model(x)
    _sync(device)
    eval_ips = iters * batch / (time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return train_ips, eval_ips, peak


def main() -> int:
    """Benchmark every model config and write the table; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=slug(platform.node() or "host"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=None, help="default 32 on CUDA, 16 on CPU")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--iters", type=int, default=None, help="default 20 on CUDA, 3 on CPU")
    args = parser.parse_args()
    configure_logging()
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"
    batch = args.batch or (32 if on_gpu else 16)
    iters = args.iters or (20 if on_gpu else 3)
    warmup = 3 if on_gpu else 1
    configs = sorted(
        (
            p
            for p in (CONFIG_DIR / "model").glob("*.yaml")
            if read_yaml(p).get("source") in ("timm", "open_clip")  # pretrained backbones only
        ),
        key=lambda p: p.stem,
    )
    if on_gpu:
        torch.backends.cudnn.benchmark = True
        hardware = torch.cuda.get_device_name(device)
    else:
        hardware = (
            f"{platform.processor() or platform.machine()}, {torch.get_num_threads()} threads"
        )
    rows = []
    for path in configs:
        cfg = read_yaml(path)
        arch = timm_arch(cfg)
        train_ips, eval_ips, peak = measure(arch, device, batch, args.size, warmup, iters)
        log.info("%s (%s): train %.2f img/s, eval %.2f img/s", cfg.name, arch, train_ips, eval_ips)
        peak_txt = f"{peak:.2f}" if peak is not None else "n/a"
        rows.append(f"| {cfg.name} | `{arch}` | {train_ips:.2f} | {eval_ips:.2f} | {peak_txt} |")
    lines = [
        f"# Throughput benchmark: {args.name}",
        "",
        f"Generated by `scripts/bench_machine.py`. Device: {device.type} ({hardware}); "
        f"torch {metadata.version('torch')}, timm {metadata.version('timm')}, "
        f"Python {platform.python_version()}, {platform.system()}.",
        f"Batch {batch}, {args.size}x{args.size} px, {warmup} warm-up + {iters} timed "
        f"iterations, random weights, {'bf16 autocast' if on_gpu else 'fp32'}. Training = "
        "forward + backward + AdamW step (full fine-tuning).",
        "",
        "| Model | Architecture | Train img/s | Eval img/s | Peak GPU memory (GB) |",
        "|---|---|---|---|---|",
        *rows,
    ]
    out = REPO_ROOT / "results" / "tables" / f"bench_{slug(args.name)}.md"
    atomic_write_text(out, "\n".join(lines) + "\n")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
