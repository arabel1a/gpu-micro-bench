"""
Shared infrastructure for PTX instruction benchmark generators.

Each generator defines instructions and calls gen_and_build() to produce
a .cu file, compile it, and verify SASS. The generated binary uses the
same ARITH CSV output format parsed by run.py.

The make_table() function reads JSON results and produces markdown tables
with automatic throttle classification and SASS instruction count markers.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
BIN = ROOT / "bin"

N_CHAINS = 8
OPS_PER_ITER = 8


# ============================================================================
# Instruction specification
# ============================================================================

@dataclass
class Insn:
    name: str           # unique benchmark name (used in output)
    ptx: str            # PTX mnemonic (for docs/reference)
    asm_op: str         # asm volatile template.  %0=accumulator, %1..%N=constants
    ctype: str          # C type: "float", "unsigned", "int", "unsigned short"
    constraint: str     # asm constraint: "f", "r", "h"
    init: str           # accumulator init expression
    const_decls: list   # [(ctype, name, value), ...] for constants
    min_sm: int = 0     # minimum SM version (75, 80, 86, ...) or 0 = any
    cat: str = ""       # category label
    # For multi-register ops (mma, wmma), use custom_tput/custom_lat instead
    custom_tput: str = ""  # full kernel body (replaces auto-generation)
    custom_lat: str = ""


# --- Helpers for common patterns ---

def _fb(name, ptx, asm, init, k, min_sm=0, cat="fp32"):
    """FP32 binary: op %0, %0, %1"""
    return Insn(name, ptx, asm, "float", "f", init,
                [("float", "k0", k)], min_sm, cat)

def _fu(name, ptx, asm, init, min_sm=0, cat="fp32"):
    """FP32 unary: op %0, %0"""
    return Insn(name, ptx, asm, "float", "f", init, [], min_sm, cat)

def _ft(name, ptx, asm, init, k, min_sm=0, cat="fp32"):
    """FP32 ternary (fma-style): op %0, %0, %1, %0"""
    return Insn(name, ptx, asm, "float", "f", init,
                [("float", "k0", k)], min_sm, cat)

def _ib(name, ptx, asm, init, k, min_sm=0, cat="int32"):
    """INT binary: op %0, %0, %1"""
    return Insn(name, ptx, asm, "int", "r", init,
                [("int", "k0", k)], min_sm, cat)

def _ub(name, ptx, asm, init, k, min_sm=0, cat="int32"):
    """UINT binary: op %0, %0, %1"""
    return Insn(name, ptx, asm, "unsigned", "r", init,
                [("unsigned", "k0", k)], min_sm, cat)

def _iu(name, ptx, asm, init, min_sm=0, cat="int32"):
    """INT unary: op %0, %0"""
    return Insn(name, ptx, asm, "int", "r", init, [], min_sm, cat)

def _uu(name, ptx, asm, init, min_sm=0, cat="bits"):
    """UINT unary"""
    return Insn(name, ptx, asm, "unsigned", "r", init, [], min_sm, cat)

def _hb(name, ptx, asm, init, k, min_sm=0, cat="fp16x2"):
    """FP16x2 (packed u32) binary"""
    return Insn(name, ptx, asm, "unsigned", "r", init,
                [("unsigned", "k0", k)], min_sm, cat)

def _hu(name, ptx, asm, init, min_sm=0, cat="fp16x2"):
    """FP16x2 (packed u32) unary"""
    return Insn(name, ptx, asm, "unsigned", "r", init, [], min_sm, cat)


# ============================================================================
# Code generation
# ============================================================================

def gen_kernel_tput(insn: Insn) -> str:
    """Generate throughput kernel (8 independent chains in one asm block)."""
    if insn.custom_tput:
        return insn.custom_tput

    c = insn.constraint
    lines = []
    lines.append(f"__global__ void __launch_bounds__(128,1)")
    lines.append(f"kern_{insn.name}_tput(volatile {insn.ctype}* __restrict__ out, int n) {{")

    if insn.min_sm > 0:
        lines.append(f"#if __CUDA_ARCH__ >= {insn.min_sm * 10}")

    init = insn.init
    chain_inits = ", ".join(f"r{i}={init}" for i in range(N_CHAINS))
    lines.append(f"    {insn.ctype} {chain_inits};")

    for ctype, name, val in insn.const_decls:
        lines.append(f"    {ctype} {name} = {val};")

    n_const = len(insn.const_decls)
    out_constraints = ",".join(f'"+{c}"(r{i})' for i in range(N_CHAINS))
    in_constraints = ",".join(f'"{c}"({d[1]})' for d in insn.const_decls)

    asm_lines = []
    for i in range(N_CHAINS):
        op = insn.asm_op
        op = op.replace("%0", f"%{i}")
        for k in range(n_const):
            op = op.replace(f"%{k+1}", f"%{N_CHAINS + k}")
        asm_lines.append(f'        "{op}\\n\\t"')

    lines.append(f"    for(int i=0; i<n; i++) asm volatile(")
    lines.append("\n".join(asm_lines))
    if in_constraints:
        lines.append(f"        :{out_constraints}:{in_constraints});")
    else:
        lines.append(f"        :{out_constraints});")

    lines.append(f"    out[threadIdx.y*32+threadIdx.x] = r0;")

    if insn.min_sm > 0:
        lines.append(f"#endif")
    lines.append("}")
    return "\n".join(lines)


def gen_kernel_lat(insn: Insn) -> str:
    """Generate latency kernel (1 dependent chain, 8 ops per iter)."""
    if insn.custom_lat:
        return insn.custom_lat

    c = insn.constraint
    lines = []
    lines.append(f"__global__ void __launch_bounds__(128,1)")
    lines.append(f"kern_{insn.name}_lat(volatile {insn.ctype}* __restrict__ out, int n) {{")

    if insn.min_sm > 0:
        lines.append(f"#if __CUDA_ARCH__ >= {insn.min_sm * 10}")

    lines.append(f"    {insn.ctype} r0 = {insn.init};")
    for ctype, name, val in insn.const_decls:
        lines.append(f"    {ctype} {name} = {val};")

    n_const = len(insn.const_decls)
    in_constraints = ",".join(f'"{c}"({d[1]})' for d in insn.const_decls)

    asm_lines = []
    for _ in range(OPS_PER_ITER):
        asm_lines.append(f'        "{insn.asm_op}\\n\\t"')

    lines.append(f"    for(int i=0; i<n; i++) asm volatile(")
    lines.append("\n".join(asm_lines))
    if in_constraints:
        lines.append(f'        :"+{c}"(r0):{in_constraints});')
    else:
        lines.append(f'        :"+{c}"(r0));')

    lines.append(f"    out[threadIdx.y*32+threadIdx.x] = r0;")

    if insn.min_sm > 0:
        lines.append(f"#endif")
    lines.append("}")
    return "\n".join(lines)


def gen_cu_file(instructions: list[Insn], group_name: str, extra_headers: str = "") -> str:
    """Generate the full .cu source file."""
    parts = []

    parts.append(f"""\
// AUTO-GENERATED by gen_{group_name}.py — do not edit manually.
// PTX {group_name} instruction throughput/latency benchmark.
//
// Usage: ./{group_name}_bench <gpu_id> <n_iters> [n_warmup] [n_reps]
//
// Output format (compatible with run.py parse_arithm_output):
//   GPU,<name>,<sm>,<n_sms>,<l2_kb>
//   ARITH,<op>,<mode>,<n_iters>,<total_ns>,<ns_per_op>,<ops_per_ns>,<wall_ms>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
{extra_headers}
#define CHECK(call) do {{ \\
    cudaError_t e = (call); \\
    if (e != cudaSuccess) {{ \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, \\
                cudaGetErrorString(e)); exit(1); }} \\
}} while(0)
""")

    cats_seen = []
    for insn in instructions:
        if insn.cat not in cats_seen:
            cats_seen.append(insn.cat)
            parts.append(f"\n// {'='*70}")
            parts.append(f"// {insn.cat.upper()}")
            parts.append(f"// {'='*70}\n")
        parts.append(gen_kernel_tput(insn))
        parts.append("")
        parts.append(gen_kernel_lat(insn))
        parts.append("")

    # Runner template
    parts.append("""
// ============================================================================
// Runner
// ============================================================================

struct Test {
    const char* name;
    void* tput;
    void* lat;
    int min_sm;
    char out_type;   // 'f'=float, 'i'=int, 'u'=unsigned, 's'=unsigned short
};

template<typename T>
float bench_kernel(void* fn_ptr, T* d_out, int n_iters, int n_warmup, int n_reps) {
    typedef void (*KFn)(volatile T*, int);
    KFn fn = (KFn)fn_ptr;
    dim3 block(32, 4);
    for (int w = 0; w < n_warmup; w++) fn<<<1, block>>>(d_out, n_iters);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    float total_ms = 0;
    for (int r = 0; r < n_reps; r++) {
        CHECK(cudaEventRecord(t0));
        fn<<<1, block>>>(d_out, n_iters);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_ms += ms;
    }
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return total_ms / n_reps;
}
""")

    # Test table
    parts.append("Test tests[] = {")
    for insn in instructions:
        otype = {'float': 'f', 'int': 'i', 'unsigned': 'u',
                 'unsigned short': 's'}[insn.ctype]
        sm = insn.min_sm if insn.min_sm > 0 else 75
        parts.append(f'    {{"{insn.name}", (void*)kern_{insn.name}_tput, '
                     f'(void*)kern_{insn.name}_lat, {sm}, \'{otype}\'}},')
    parts.append("};")
    parts.append(f"const int n_tests = sizeof(tests)/sizeof(tests[0]);")

    # Main
    parts.append("""
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <gpu_id> <n_iters> [n_warmup] [n_reps]\\n", argv[0]);
        return 1;
    }

    int gpu_id = atoi(argv[1]);
    int n_iters = atoi(argv[2]);
    int n_warmup = argc > 3 ? atoi(argv[3]) : 10;
    int n_reps   = argc > 4 ? atoi(argv[4]) : 50;

    CHECK(cudaSetDevice(gpu_id));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    int l2_kb = prop.l2CacheSize / 1024;
    int gpu_sm = prop.major * 10 + prop.minor;

    printf("GPU,%s,sm_%d%d,%d,%d\\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, l2_kb);
    fprintf(stderr, "GPU: %s (sm_%d), SMs: %d\\n",
            prop.name, gpu_sm, prop.multiProcessorCount);
    fprintf(stderr, "n_iters=%d, n_warmup=%d, n_reps=%d, ops_per_iter=8\\n",
            n_iters, n_warmup, n_reps);

    void* d_buf;
    CHECK(cudaMalloc(&d_buf, 128 * 8));

    for (int t = 0; t < n_tests; t++) {
        if (gpu_sm < tests[t].min_sm) {
            fprintf(stderr, "  %s: skipped (needs sm_%d, have sm_%d)\\n",
                    tests[t].name, tests[t].min_sm, gpu_sm);
            continue;
        }
        fprintf(stderr, "  %s ...\\n", tests[t].name);

        for (int mode = 0; mode < 2; mode++) {
            void* fn = mode == 0 ? tests[t].tput : tests[t].lat;
            const char* mode_str = mode == 0 ? "tput" : "lat";
            int divisor = n_iters * 8;

            float avg_ms = 0;
            switch (tests[t].out_type) {
                case 'f': avg_ms = bench_kernel<float>( fn, (float*)d_buf, n_iters, n_warmup, n_reps); break;
                case 'i': avg_ms = bench_kernel<int>(   fn, (int*)d_buf,   n_iters, n_warmup, n_reps); break;
                case 'u': avg_ms = bench_kernel<unsigned>(fn, (unsigned*)d_buf, n_iters, n_warmup, n_reps); break;
                case 's': avg_ms = bench_kernel<unsigned short>(fn, (unsigned short*)d_buf, n_iters, n_warmup, n_reps); break;
            }

            double total_ns = avg_ms * 1e6;
            double ns_per_op = total_ns / divisor;
            double ops_per_ns = (total_ns > 0) ? divisor / total_ns : 0;
            printf("ARITH,%s,%s,%d,%.1f,%.3f,%.4f,%.1f\\n",
                   tests[t].name, mode_str, n_iters, total_ns, ns_per_op,
                   ops_per_ns, avg_ms * n_reps);
        }
    }

    CHECK(cudaFree(d_buf));
    return 0;
}
""")

    return "\n".join(parts)


# ============================================================================
# SASS verification
# ============================================================================

def get_sass_counts(binary: Path) -> dict:
    """Get SASS instruction counts per (arch, kernel_short_name).

    Returns: {(sm_ver, kernel_short_name): count}
    """
    result = subprocess.run(
        ["cuobjdump", "--dump-sass", str(binary)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: cuobjdump failed: {result.stderr}", file=sys.stderr)
        return {}

    kernel_pat = re.compile(r"Function\s*:\s*(\S+)")
    arch_pat = re.compile(r"code for sm_(\d+)")
    insn_pat = re.compile(r"/\*[0-9a-f]+\*/\s+(.+);")

    cur_arch = None
    cur_kern = None
    counts = {}

    for line in result.stdout.split("\n"):
        m = arch_pat.search(line)
        if m:
            cur_arch = int(m.group(1))
            continue
        m = kernel_pat.search(line)
        if m:
            cur_kern = m.group(1)
            km = re.search(r"kern_(\w+)", cur_kern)
            if km and cur_arch:
                counts[(cur_arch, km.group(1))] = 0
            continue
        if cur_kern and cur_arch and insn_pat.search(line):
            km = re.search(r"kern_(\w+)", cur_kern)
            if km:
                counts[(cur_arch, km.group(1))] += 1

    return counts


def verify_sass(binary: Path, instructions: list[Insn]) -> bool:
    """Check SASS to verify instructions weren't optimized away."""
    counts = get_sass_counts(binary)
    if not counts:
        return False

    ok = True
    min_expected = 50
    for insn in instructions:
        for suffix in ["tput", "lat"]:
            found = False
            for (arch, kname), count in counts.items():
                if kname == f"{insn.name}_{suffix}":
                    found = True
                    if count < min_expected:
                        print(f"WARNING: kern_{insn.name}_{suffix} (sm_{arch}) has only {count} "
                              f"SASS instructions (expected >>{min_expected}).",
                              file=sys.stderr)
                        ok = False
                    break
            # Not found is OK — might be __CUDA_ARCH__ guarded
    return ok


def sass_differs_across_archs(binary: Path) -> set:
    """Return set of op names where SASS count differs unusually across archs.

    For tput kernels, we compare arch pairs. If the difference between
    two archs for a given op deviates from the median difference (constant
    overhead per arch), we flag it.
    """
    counts = get_sass_counts(binary)
    if not counts:
        return set()

    # Group by op, mode, arch
    ops = {}
    for (arch, kname), count in counts.items():
        # kname is like "f32_add_tput" or "f32_add_lat"
        if kname.endswith("_tput"):
            op = kname[:-5]
            ops.setdefault(op, {})[arch] = count

    archs = sorted(set(a for (a, _) in counts.keys()))
    if len(archs) < 2:
        return set()

    # Use highest two archs for comparison
    a1, a2 = archs[-2], archs[-1]
    deltas = []
    op_deltas = {}
    for op, arch_counts in ops.items():
        if a1 in arch_counts and a2 in arch_counts:
            d = arch_counts[a2] - arch_counts[a1]
            deltas.append(d)
            op_deltas[op] = d

    if not deltas:
        return set()

    # Median delta
    deltas.sort()
    median = deltas[len(deltas) // 2]

    # Flag ops that deviate from median by more than 16 instructions
    flagged = set()
    for op, d in op_deltas.items():
        if abs(d - median) > 16:
            flagged.add(op)

    return flagged


# ============================================================================
# Table generation
# ============================================================================

def load_results(path: Path) -> tuple:
    """Load results JSON. Returns (gpu_name, {op: {mode: ns_per_op}})."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for t in data["tests"]:
        result.setdefault(t["op"], {})[t["mode"]] = t["ns_per_op"]
    return data["gpu"]["name"], result


def throttle_label(ratio):
    if ratio is None:
        return "—"
    if ratio <= 1.5:
        return "OK"
    if ratio <= 3.0:
        return f"~{ratio:.1f}×"
    return f"**{ratio:.1f}×**"


def make_table(results_dir: Path, categories: dict, sass_flagged: set = None,
               o0_check: str = "") -> str:
    """Generate markdown table from JSON results in results_dir.

    Args:
        results_dir: directory containing *.json result files
        categories: {cat_name: [op_names]} for ordering
        sass_flagged: set of op names where SASS differs across archs
        o0_check: brief note about -O0 check result

    Returns: markdown string
    """
    if sass_flagged is None:
        sass_flagged = set()

    files = sorted(results_dir.glob("*.json"))
    if not files:
        return "No result files found."

    gpus = []
    all_data = []
    for f in files:
        name, data = load_results(f)
        gpus.append(name)
        all_data.append(data)

    # Find baseline (prefer 2080S, then 3090)
    baseline_idx = None
    for i, name in enumerate(gpus):
        if "2080" in name:
            baseline_idx = i
            break
    if baseline_idx is None:
        for i, name in enumerate(gpus):
            if "3090" in name:
                baseline_idx = i
                break
    if baseline_idx is None:
        baseline_idx = 0

    # Get all ops
    all_ops = set()
    for d in all_data:
        all_ops.update(d.keys())

    cat_order = list(categories.keys())

    def classify(op):
        for cat, ops in categories.items():
            if op in ops:
                return cat
        return "Other"

    def sort_key(op):
        cat = classify(op)
        ci = cat_order.index(cat) if cat in cat_order else 999
        return (ci, op)

    all_ops = sorted(all_ops, key=sort_key)

    lines = []
    lines.append(f"# Results\n")
    if o0_check:
        lines.append(f"**-O0 check**: {o0_check}\n")
    lines.append(f"**Baseline**: {gpus[baseline_idx]}\n")
    if sass_flagged:
        lines.append(f"**\\*** = SASS instruction count differs unusually across architectures for this op\n")
    lines.append("")

    # Header
    hdr = "| Category | Instruction |"
    sep = "|----------|-------------|"
    for i, name in enumerate(gpus):
        short = name.replace("NVIDIA ", "").replace("GeForce ", "")
        hdr += f" {short} tput | {short} lat |"
        sep += "-------:|-------:|"
    for i, name in enumerate(gpus):
        if i == baseline_idx:
            continue
        short = name.replace("NVIDIA ", "").replace("GeForce ", "")
        hdr += f" {short} ratio | Throttle |"
        sep += "-------:|----------|"
    lines.append(hdr)
    lines.append(sep)

    prev_cat = None
    for op in all_ops:
        cat = classify(op)
        cat_display = cat if cat != prev_cat else ""
        prev_cat = cat

        sass_mark = " \\*" if op in sass_flagged else ""
        row = f"| {cat_display} | `{op}`{sass_mark} |"

        for i, d in enumerate(all_data):
            tput = d.get(op, {}).get("tput")
            lat = d.get(op, {}).get("lat")
            tput_s = f"{tput:.3f}" if tput else "—"
            lat_s = f"{lat:.3f}" if lat else "—"
            row += f" {tput_s} | {lat_s} |"

        baseline_d = all_data[baseline_idx]
        for i, d in enumerate(all_data):
            if i == baseline_idx:
                continue
            b_tput = baseline_d.get(op, {}).get("tput")
            c_tput = d.get(op, {}).get("tput")
            if b_tput and c_tput and b_tput > 0:
                ratio = c_tput / b_tput
                row += f" {ratio:.2f} | {throttle_label(ratio)} |"
            else:
                row += " — | — |"

        lines.append(row)

    # Category summaries
    lines.append("")
    lines.append("## Throttling Summary by Category\n")

    for cat in cat_order + ["Other"]:
        ops_in_cat = [op for op in all_ops if classify(op) == cat]
        if not ops_in_cat:
            continue

        lines.append(f"### {cat}\n")

        for i, d in enumerate(all_data):
            if i == baseline_idx:
                continue
            short = gpus[i].replace("NVIDIA ", "").replace("GeForce ", "")
            baseline_d = all_data[baseline_idx]

            ratios = []
            for op in ops_in_cat:
                b_tput = baseline_d.get(op, {}).get("tput")
                c_tput = d.get(op, {}).get("tput")
                if b_tput and c_tput and b_tput > 0:
                    ratios.append((op, c_tput / b_tput))

            if not ratios:
                lines.append(f"- **{short}**: no data\n")
                continue

            r_vals = [r[1] for r in ratios]
            r_min, r_max = min(r_vals), max(r_vals)
            r_avg = sum(r_vals) / len(r_vals)

            if r_max <= 1.5:
                verdict = "UNTHROTTLED"
            elif r_min <= 1.5 and r_max > 3.0:
                verdict = "MIXED"
            elif r_min > 3.0:
                verdict = "THROTTLED"
            else:
                verdict = "MILDLY THROTTLED"

            lines.append(f"**{short}**: {verdict} (range {r_min:.2f}×–{r_max:.2f}×, avg {r_avg:.2f}×)")

            if r_max / max(r_min, 0.01) > 2.0:
                fast = [(op, r) for op, r in ratios if r <= 1.5]
                mid = [(op, r) for op, r in ratios if 1.5 < r <= 3.0]
                slow = [(op, r) for op, r in ratios if r > 3.0]
                if fast:
                    lines.append(f"- Unthrottled: {', '.join(f'`{o}` ({r:.2f}×)' for o,r in fast)}")
                if mid:
                    lines.append(f"- Mildly: {', '.join(f'`{o}` ({r:.2f}×)' for o,r in mid)}")
                if slow:
                    lines.append(f"- Throttled: {', '.join(f'`{o}` ({r:.2f}×)' for o,r in slow)}")
            else:
                detail = ", ".join(f"`{op}` ({r:.2f}×)" for op, r in ratios)
                lines.append(f"- {detail}")
            lines.append("")

    return "\n".join(lines)


# ============================================================================
# CLI helpers
# ============================================================================

def standard_argparser(group_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Generate PTX {group_name} benchmarks")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only generate .cu file, don't compile or verify")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing binary's SASS")
    parser.add_argument("--table-only", action="store_true",
                        help="Only regenerate results table from existing JSONs")
    parser.add_argument("--archs", default="75,80,86",
                        help="Comma-separated SM versions (default: 75,80,86)")
    parser.add_argument("--output", default=None,
                        help="Output .cu path")
    return parser


def gen_and_build(group_name: str, instructions: list[Insn], categories: dict,
                  args=None, extra_headers: str = ""):
    """Full pipeline: generate .cu, compile, verify SASS, generate table."""
    if args is None:
        parser = standard_argparser(group_name)
        args = parser.parse_args()

    cu_path = Path(args.output) if args.output else SRC / f"{group_name}_bench.cu"
    bin_path = BIN / f"{group_name}_bench"
    results_dir = ROOT / "results" / group_name

    if args.table_only:
        sass_flagged = set()
        if bin_path.exists():
            sass_flagged = sass_differs_across_archs(bin_path)
        table = make_table(results_dir, categories, sass_flagged,
                           o0_check="No effect — asm volatile prevents all optimization")
        out = results_dir / "results.md"
        out.write_text(table)
        print(f"Table: {out}")
        return

    if not args.verify_only:
        print(f"Generating {len(instructions)} instruction benchmarks "
              f"({len(instructions)*2} kernels)...")
        cu_code = gen_cu_file(instructions, group_name, extra_headers)
        cu_path.parent.mkdir(parents=True, exist_ok=True)
        cu_path.write_text(cu_code)
        print(f"Written: {cu_path} ({len(cu_code)} bytes)")

        cats = {}
        for insn in instructions:
            cats.setdefault(insn.cat, []).append(insn.name)
        for cat, names in cats.items():
            print(f"  {cat}: {len(names)} ops — {', '.join(names)}")

    if args.gen_only:
        return

    # Compile
    archs = args.archs.split(",")
    gencode = " ".join(
        f"-gencode arch=compute_{a},code=sm_{a}" for a in archs
    )
    cmd = f"nvcc -O3 {gencode} -Xcompiler -fopenmp -lgomp -o {bin_path} {cu_path}"
    print(f"\nCompiling: {cmd}")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMPILE FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Binary: {bin_path}")

    # Verify SASS
    print("\nVerifying SASS...")
    if verify_sass(bin_path, instructions):
        print("SASS verification: OK")
    else:
        print("SASS verification: WARNINGS found", file=sys.stderr)
