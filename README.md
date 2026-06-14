# GPU Microbenchmark Suite

Memory subsystem characterization and kernel isolation benchmarks for NVIDIA GPUs.

## Reproduce

```bash
git clone https://github.com/arabel1a/gpu-micro-bench
cd gpu-micro-bench
./reproduce.sh            # add a GPU index if you have several cards: ./reproduce.sh 1
```

## Benchmarks

### memtest — Memory subsystem characterization

Pointer-chase latency and sequential bandwidth as a function of working set size, with explicit PTX cache hints (`ca`, `cg`, `cs`) and a forced-DRAM mode (`cn` = L2 flushed before measurement).

Also includes:
- **BLOCK_COST**: Per-block dispatch overhead — 128-thread blocks (4 warps × 32), `__launch_bounds__(128,1)`, reading L1-cached 4 KB buffer with data-dependent PTX loads. Varies blocks from 1×n_sms to 64×n_sms. Isolates SM block scheduling cost from cache/DRAM effects.
- **CONC_WARP**: Single-SM warp concurrency — pointer chase with 1–32 warps per block, L2 flushed, 64 MB buffer. Tests whether the warp scheduler overlaps DRAM latency across warps.
- **CONC_BLOCK**: GPU-wide block concurrency — pointer chase with 1 to 16×n_sms blocks. Tests block-level occupancy scaling.
- **L2_PERSIST**: Inter-kernel L2 persistence — flushes L2, then launches 10 sequential-read kernels back-to-back. Compares cold (launch 1) vs warm (launches 2–10) bandwidth to measure whether L2 retains data across kernel launches.

### arithmtest — Instruction-level throughput/latency

Inline PTX assembly microbenchmark for all arithmetic instruction types: FP32 (FADD, FMUL, FFMA), FP16x2 (HADD2, HMUL2, HFMA2), INT32 (IADD, IMAD), INT8 dot (DP4A), bitwise (SHL, SHR, AND, OR, XOR, LOP3, PRMT), and tensor core MMA (HMMA F16/BF16/TF32, IMMA S8). Two modes per op: throughput (8 independent chains) and latency (1 dependent chain). Architecture guards for sm_75 vs sm_80+.

### arithm2 / ctrl / tensor — Auto-generated extension benches

Three Python generators sit alongside `gen_arithm.py` and produce additional `.cu` benchmarks via the same `bench_common.py` pipeline (compile across sm_75/80/86, SASS-verify, JSON output, markdown tables).

- **`gen_arithm2.py`** — formats `gen_arithm.py` doesn't cover: FP64 scalar (DADD/DMUL/DFMA + div/rcp/sqrt emulation), TF32 scalar `cvt.rna.tf32.f32` (with a small drift step to defeat ptxas const-folding), predicate ops (`setp.lt + selp`, `slct.s32`), and packed video SIMD (`vadd2`/`vsub2`/`vadd4`/`vsub4`/`vabsdiff2`/`vabsdiff4`/`vmin.s32`/`vmax.s32`).
- **`gen_ctrl.py`** — control flow & warp collectives: `shfl.sync.{bfly,idx,down}`, `vote.sync.{ballot,all,any}`, `redux.sync.{add,min,or}` (sm_80+), `match.sync.any`, `bar.sync 0`, `bar.warp.sync`, `bar.arrive`, predicated and divergent branches.
- **`gen_tensor.py`** — Ampere-supported `mma.sync` shapes: `m16n8k16` (f16/bf16 → f32), `m16n8k8` (tf32 → f32), `m16n8k32` (s8/u8 → s32), `m8n8k4` (f64 → f64). Single-chain (`_lat`) and 4-way round-robin (`_tput`) variants. Single SM × 4 warps so the per-SM mma issue rate is what's measured.

### mmvq_bench — Isolated MMVQ kernel benchmark

Extracts the exact `mul_mat_vec_q` kernel and `vec_dot_q1_0_g128_q8_1` from llama.cpp. Runs on random data with no framework overhead. Configurable layer shapes (default: Qwen3-8B). Reports per-layer and aggregate TPS with effective bandwidth.

### SASS verification

Each `gen_*.py` runs `verify_sass()` after compile. It uses `cuobjdump` to count
SASS instructions per generated kernel and warns when a kernel has fewer than
~50 instructions — a strong signal that ptxas constant-folded the inline asm
chain to a no-op (e.g. `r += 1` ×8 collapses to `IADD3 R, R, 8`). Warnings are
informational; they don't fail the build. The threshold respects each Insn's
`min_sm` so `__CUDA_ARCH__`-guarded kernels don't false-positive on lower archs.
Full SASS for any kernel: `cuobjdump --dump-sass bin/<bench>_bench`.

## Key Results

https://github.com/arabel1a/ml-on-cmp
