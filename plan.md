# GPU Micro-Bench Plan

## Done
- Diagnosed cn == ca bug: all three cn measurements used `measure_*<CA>` template with multi-pass n_hops. The L2 flush before measurement was ineffective because CA loads re-cached data during the timed kernel, and high n_hops amortized the initial cold misses.
- First fix attempt (ld.global.cv): cv behaves same as cg on Ampere — L2 is in the data path for all global loads, no PTX hint bypasses it.
- Final fix: dedicated cold-measurement functions (`measure_latency_cold`, `measure_latency_cold_multi`, `measure_seq_bw_cold`) that flush L2 before each single-pass rep (5 reps averaged). Single pass = n_hops = n_elems, so each element visited exactly once = guaranteed L2 miss. Uses CG loads to also bypass L1.
- Verified: cn now shows ~177ns (DRAM) vs cg ~145ns (L2) for small buffers; BW 125 GB/s (DRAM) vs 330 GB/s (cached).
- But cn still ≈ cg on 3090 (not CMP 90HX). Root cause: 4x L2 flush buffer insufficient for Ampere's 16-32 way set-associative L2. Each L2 set only saw 4 conflicting lines, chain data survived in remaining ways.
- Fix: increased flush buffer from 4x to 32x L2 size. Guarantees full eviction even at 32-way associativity. ~192MB for 3090's 6MB L2.
- Updated cross-GPU plots to show ca/cg/cn (dropped cs for clarity).
- Added all plots to README (cross-GPU + per-GPU in collapsible sections).
- CMP 90HX analysis: latencies identical to 3090 (same sm_86), L2 step position matches smaller L2 size. But DRAM bandwidth is ~30-40% lower (582 vs 821 GB/s) — real memory controller/channel difference, not just cache.

## Arithm coverage extension (2026-04-25)
- Step 0: Added FP64 helpers `_db`/`_du`/`_dt` to `bench_common.py` and `'d'`→`double` out_type plumbing through the runner template.
- Step 1: Created `gen_arithm2.py` covering FP64 (add/sub/mul/fma/min/max/div_rn/rcp_rn/sqrt_rn), TF32 cvt (rna+add to break const-fold), predicate (`setp+selp.lt.s32/.f32`, `slct.s32`), and packed-video SIMD (`vadd2/vsub2/vadd4/vsub4/vabsdiff2/vabsdiff4/vmin/vmax`).
- Bug fix in `bench_common.verify_sass`: regex `kern_(\w+)` was over-capturing into the C++ mangled param suffix (e.g. `f64_abs_latPVdi` instead of `f64_abs_lat`), so name comparison never matched and warnings never fired. Replaced with `kern_([a-z0-9_]+)` and added `min_sm` skip so `__CUDA_ARCH__`-guarded kernels don't false-positive.
- Findings from re-running verify on existing `arithm_bench`:
  - Many `*_neg`/`*_abs`/`*_not`/`*_cnot` kernels are operand-modifiers on Ampere — no isolated SASS insn — and ptxas folds chains. Their numbers in existing results.md are loop-overhead, not the op.
  - Several latency kernels (`s32_add_lat`, `s32_sub_lat`, `s32_mul_lo_lat`, `b32_and/or/xor/shl/shr_lat`, etc.) get const-folded: ptxas unrolls and reduces e.g. `r += 1` ×8×N to `IADD3 R, R, 0x20`. Throughput kernels (8 independent chains) survive intact.
  - In `gen_arithm2.py`, dropped `f64_neg`/`f64_abs` (Ampere has no isolated FP64 neg/abs) and `setp_selp_eq_s32` (degenerate when r0==k0), and broke the `tf32_cvt_rna` self-fixed-point with a small `add.rn.f32` drift step.

- Step 3: Created `gen_ctrl.py` covering warp shfl (bfly/idx/down), vote (ballot/all/any), redux (add/min/or — sm_80+), match.any, barriers (bar.sync, bar.warp.sync, bar.arrive), and predicated/divergent branches. All custom_tput/custom_lat kernels.
  - PTX gotchas hit: (1) loop-unrolling of `for(i<n)` made ptxas duplicate `.reg .pred` declarations and labels — fix is `{ ... }` PTX-scope wrappers around every asm body **plus** `#pragma unroll 1` on the for-loop; (2) `shfl.sync` takes 5 operands (d, a, b, c, membermask), not 4.
  - SASS spot-check (sm_86) confirms expected mnemonics: SHFL.BFLY/IDX/DOWN, VOTE.ANY/ALL, REDUX.SUM.S32/MIN.U32/OR, MATCH.ANY, BAR.SYNC.DEFER_BLOCKING, BAR.ARV, all at ~40 occurrences per kernel.
  - Real findings (worth keeping, not bugs): `bar.warp.sync` compiles to NOPs on Ampere (no separate sync insn — handled by scheduler dependency tracking); predicated branches with target == next-insn collapse to NOPs at ptxas level; `shfl.sync.idx` with constant lane folds in latency variant.

- Step 2: Created `gen_tensor.py` covering Ampere-supported mma.sync shapes — m16n8k16 f16/bf16/f32, m16n8k8 tf32/f32, m16n8k32 s8/u8/s32, m8n8k4 f64/f64. All custom kernels, 1 chain × 8 mma/iter for `_lat`, 4 round-robin chains × 8 mma/iter for `_tput`. Each kernel uses `__launch_bounds__(128,1)` and `<<<1,(32,4)>>>` so we get 4 warps on 1 SM (per-SM issue rate).
  - SASS spot-check (sm_86) confirms expected mnemonics: HMMA.16816.F32 (f16), HMMA.16816.F32.BF16, HMMA.1688.F32.TF32, IMMA.16832.S8.S8 / U8.U8, DMMA.884 — exactly 8 per kernel as designed.
  - Smoke-run results (n_iters=100k, 5 warmup, 20 reps; 1 SM × 4 warps; ns_per_op = wallclock / (iters × ops_per_iter)):
    - **CMP 170HX (sm_80, GA100 die)**: f16/bf16/tf32/s8/u8 mma all ≈ 181.7 ns/op; **f64 mma = 726 ns/op (4× slower than f16)**. Matches GA100's published FP64-tensor:FP16-tensor ratio.
    - **CMP 90HX (sm_86, GA102 die)**: f16/bf16/tf32/s8/u8 mma all ≈ 284.5 ns/op (1.57× slower than 170HX per-SM, despite 90HX having fewer SMs total); **f64 mma = 331 ns/op (only 1.16× slower than f16)** — unusual, suggests GA102's DMMA path isn't disproportionately slower per-issue on this chip.
    - tput vs lat reports identical numbers — even with 4 round-robin D accumulators, ptxas/scheduler can't extract additional ILP across the 8 mmas/iter. The mma issue port is the bottleneck.

## Verification of "compiler didn't optimize out the instructions"
- Fixed `verify_sass` (was silently never matching due to over-greedy regex — see Step 1 notes). Now actually runs.
- All 3 new benches (arithm2, ctrl, tensor) verified end-to-end: every kernel inspected for expected SASS mnemonics, kernels with degenerate fold-down behavior either dropped (`f64_neg`, `f64_abs`, `setp_selp_eq_s32`) or fixed with a non-foldable drift step (`tf32_cvt_rna` + add).
- Pre-existing `arithm_bench` re-audited: chain-foldable integer ops (`s32_add_lat`, `b32_and_lat` etc.) collapse to a single IADD3-by-constant + loop overhead. Throughput kernels (8 independent chains) survive intact. Existing results.md numbers for those *_lat columns reflect loop overhead, not the op's true latency.
