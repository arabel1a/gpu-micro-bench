# GPU Memory Subsystem Microbenchmark

Pointer-chase latency and sequential bandwidth as a function of working set size, with explicit PTX cache hints (`ca`, `cg`, `cs`) and a forced-DRAM mode (`cn` = L2 flushed before measurement).

## Requirements

- NVIDIA GPU(s) with CUDA toolkit (nvcc, sm_75+ supported)
- Python 3.12+ managed by [uv](https://github.com/astral-sh/uv)
- OpenMP (for parallel chain building on host)

## Build

```bash
cd micro-benchmarking
make
```

This compiles `memtest.cu` into `./memtest` for sm_75 and sm_86.

## Run benchmarks

```bash
# Full run (all sizes, all GPUs from config)
uv run python run.py

# Smoke test (4 sizes, fast)
uv run python run.py --config-name=smoke

# Override from CLI
uv run python run.py gpu_ids=[0] overwrite=true
```

Configuration lives in `conf/config.yaml` (all defaults) and `conf/smoke.yaml` (minimal delta for quick tests).

## Generate plots and tables

```bash
uv run python plot.py

# Or for smoke results
uv run python plot.py --config-name=smoke
```

## Outputs

All outputs go to `{output_dir}/` (default: `results/memtest/`):

```
results/memtest/
  NVIDIA_CMP_90HX.json          # raw data per GPU
  NVIDIA_GeForce_RTX_2080_SUPER.json
  results.md                     # markdown tables
  plots/
    lat_single_*.png             # per-GPU latency (1 thread)
    lat_multi_*.png              # per-GPU latency (all SMs)
    bw_*.png                     # per-GPU bandwidth
    lat_single_cross.png         # cross-GPU latency comparison
    lat_multi_cross.png
    bw_cross_gpu.png
    lat_dram_hint_bar.png        # DRAM latency bar chart by hint
```

## Cache hints

| Hint | PTX instruction | Caches used |
|------|----------------|-------------|
| `ca` | `ld.global.ca` | L1 + L2 (default) |
| `cg` | `ld.global.cg` | L2 only (bypass L1) |
| `cs` | `ld.global.cs` | streaming (bypass L1, evict-first in L2) |
| `cn` | `ld.global.ca` + flush | L2 flushed before measurement (guaranteed DRAM) |
