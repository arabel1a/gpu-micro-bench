#!/usr/bin/env bash
#
# reproduce.sh — one-shot GPU microbenchmark runner.
#
# Detects your GPU's compute capability, builds the benchmarks for that exact
# architecture, runs them, and prints all numbers to stdout between clearly
# marked BEGIN/END blocks.
#
# HOW TO USE:
#   1. git clone https://github.com/arabel1a/gpu-micro-bench
#   2. cd gpu-micro-bench
#   3. ./reproduce.sh            # uses GPU 0
#      ./reproduce.sh 1          # uses GPU 1 (if you have several cards)
#   4. Copy EVERYTHING this script prints (from the first "====" line to the
#      last) and paste it back to whoever asked you to run it.
#
# Requirements: an NVIDIA GPU, the CUDA toolkit (nvcc + make), and OpenMP
# (libgomp — already present with any normal gcc). No Python needed.
#
set -uo pipefail

GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----- benchmark parameters (kept identical to conf/*.yaml so numbers are
# ----- directly comparable to the published CMP 90HX / 170HX results) --------
MEMTEST_SIZES="2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728"
MEMTEST_HOPS="50000"
ARITHM_ITERS="100000"; ARITHM_WARMUP="10"; ARITHM_REPS="50"
MMVQ_WARMUP="20"; MMVQ_REPS="100"
# Qwen3-8B Q1_0_g128 layer shapes  (name:nrows:ncols:count_per_token)
MMVQ_LAYERS="q_proj:4096:4096:32,k_proj:1024:4096:32,v_proj:1024:4096:32,o_proj:4096:4096:32,gate_proj:12288:4096:32,up_proj:12288:4096:32,down_proj:4096:12288:32,lm_head:151669:4096:1"

say() { echo "$@" >&2; }   # progress goes to stderr so it never pollutes the paste

# ----------------------------------------------------------------------------
# 0. sanity checks — find nvcc even if it isn't on a non-login PATH (common on
#    headless mining rigs where CUDA is installed under /opt or /usr/local)
# ----------------------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
    for d in /opt/cuda/bin /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [ -x "$d/nvcc" ]; then export PATH="$d:$PATH"; break; fi
    done
fi
command -v nvcc >/dev/null 2>&1 || { say "ERROR: nvcc not found. Install the CUDA toolkit, or add its bin dir to PATH (e.g. export PATH=/usr/local/cuda/bin:\$PATH)."; exit 1; }
command -v make >/dev/null 2>&1 || { say "ERROR: make not found."; exit 1; }
say "==> Using $(command -v nvcc)"

# ----------------------------------------------------------------------------
# 1. detect compute capability of the chosen GPU (compile a tiny probe — works
#    on ANY card and any driver version, unlike parsing nvidia-smi)
# ----------------------------------------------------------------------------
say "==> Detecting GPU $GPU_ID ..."
PROBE="$(mktemp -d)/probe"
cat > "${PROBE}.cu" <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    int id = argc > 1 ? atoi(argv[1]) : 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, id) != cudaSuccess) { fprintf(stderr, "no such GPU\n"); return 1; }
    // arch<TAB>name<TAB>n_sms<TAB>l2_kb
    printf("%d%d\t%s\t%d\t%d\n", p.major, p.minor, p.name,
           p.multiProcessorCount, (int)(p.l2CacheSize / 1024));
    return 0;
}
EOF
nvcc -o "$PROBE" "${PROBE}.cu" 2>/dev/null || { say "ERROR: failed to compile the probe — is the CUDA toolkit healthy?"; exit 1; }
PROBE_OUT="$("$PROBE" "$GPU_ID")" || { say "ERROR: GPU $GPU_ID not available."; exit 1; }
ARCH="$(printf '%s' "$PROBE_OUT" | cut -f1)"
GPU_NAME="$(printf '%s' "$PROBE_OUT" | cut -f2)"
N_SMS="$(printf '%s' "$PROBE_OUT" | cut -f3)"
L2_KB="$(printf '%s' "$PROBE_OUT" | cut -f4)"
say "    GPU: ${GPU_NAME}  (sm_${ARCH}, ${N_SMS} SMs, ${L2_KB} KB L2)"

# ----------------------------------------------------------------------------
# 2. build the four binaries for THIS architecture
# ----------------------------------------------------------------------------
say "==> Building benchmarks for sm_${ARCH} (this can take a minute) ..."
make clean >/dev/null 2>&1 || true
BUILD_TARGETS="bin/memtest bin/arithmtest_gen bin/mmvq_bench bin/mmvq_bench_dp2a_prmt"
if ! make ARCHS="$ARCH" $BUILD_TARGETS >/tmp/cmp_build.log 2>&1; then
    say "ERROR: build failed. Last 30 lines:"
    tail -n 30 /tmp/cmp_build.log >&2
    exit 1
fi
say "    build OK"

# ----------------------------------------------------------------------------
# 3. run everything; raw CSV output is wrapped in BEGIN/END markers
# ----------------------------------------------------------------------------
run_bench() {
    local tag="$1"; shift
    echo "===== BEGIN ${tag} ====="
    if ! "$@" 2>/tmp/cmp_run.log; then
        echo "(FAILED — stderr below)"
        tail -n 20 /tmp/cmp_run.log
    fi
    echo "===== END ${tag} ====="
    echo
}

echo "========================================================================"
echo " gpu-micro-bench results"
echo " gpu_name=${GPU_NAME}  arch=sm_${ARCH}  n_sms=${N_SMS}  l2_kb=${L2_KB}"
echo " gpu_id=${GPU_ID}"
echo "========================================================================"
echo

say "==> [1/4] memtest (memory subsystem) ..."
run_bench "MEMTEST" ./bin/memtest "$GPU_ID" "$MEMTEST_SIZES" "$MEMTEST_HOPS"

say "==> [2/4] arithmtest (instruction throughput/latency) ..."
run_bench "ARITHM" ./bin/arithmtest_gen "$GPU_ID" "$ARITHM_ITERS" "$ARITHM_WARMUP" "$ARITHM_REPS"

say "==> [3/4] mmvq (Qwen3-8B Q1_0, baseline DP4A) ..."
run_bench "MMVQ_BASE" ./bin/mmvq_bench "$GPU_ID" "$MMVQ_WARMUP" "$MMVQ_REPS" "$MMVQ_LAYERS"

say "==> [4/4] mmvq (Qwen3-8B Q1_0, DP2A+PRMT replacement) ..."
run_bench "MMVQ_DP2A_PRMT" ./bin/mmvq_bench_dp2a_prmt "$GPU_ID" "$MMVQ_WARMUP" "$MMVQ_REPS" "$MMVQ_LAYERS"

echo "========================================================================"
echo " DONE — copy everything from the first '====' line above to here."
echo "========================================================================"
say "==> Done. Copy the stdout (the ==== blocks) and paste it back."
