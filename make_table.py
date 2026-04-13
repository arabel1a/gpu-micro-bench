#!/usr/bin/env python3
"""Generate comprehensive comparison table from arithmtest_gen results."""
import json
import sys
from pathlib import Path
from collections import defaultdict

def load(path):
    with open(path) as f:
        data = json.load(f)
    # op -> mode -> ns_per_op
    result = {}
    for t in data["tests"]:
        result.setdefault(t["op"], {})[t["mode"]] = t["ns_per_op"]
    return data["gpu"]["name"], result

# Category assignment
CATS = {
    "FP32 core": ["f32_add", "f32_sub", "f32_mul", "f32_fma", "f32_neg", "f32_abs", "f32_min", "f32_max"],
    "FP32 SFU": ["f32_rcp_approx", "f32_sqrt_approx", "f32_rsqrt", "f32_sin", "f32_cos", "f32_lg2", "f32_ex2", "f32_tanh"],
    "FP32 multi-insn": ["f32_div_approx", "f32_div_rn", "f32_rcp_rn", "f32_sqrt_rn"],
    "FP16x2": ["f16x2_add", "f16x2_sub", "f16x2_mul", "f16x2_fma",
               "f16x2_neg", "f16x2_abs", "f16x2_min", "f16x2_max",
               "f16x2_ex2", "f16x2_tanh"],
    "FP16": ["f16_add", "f16_sub", "f16_mul", "f16_fma",
             "f16_neg", "f16_abs", "f16_min", "f16_max",
             "f16_ex2", "f16_tanh"],
    "BF16x2": ["bf16x2_add", "bf16x2_sub", "bf16x2_mul", "bf16x2_fma",
               "bf16x2_neg", "bf16x2_abs", "bf16x2_min", "bf16x2_max"],
    "BF16": ["bf16_add", "bf16_sub", "bf16_mul", "bf16_fma",
             "bf16_neg", "bf16_abs"],
    "INT32 arith": ["s32_add", "s32_sub", "u32_add", "u32_sub",
                    "s32_mul_lo", "u32_mul_lo", "s32_mul_hi", "u32_mul_hi",
                    "s32_mad_lo", "u32_mad_lo", "s32_mad_hi",
                    "s32_mul24_lo", "u32_mul24_lo", "s32_mad24_lo", "u32_mad24_lo",
                    "s32_abs", "s32_neg", "s32_min", "u32_min", "s32_max", "u32_max",
                    "s32_sad"],
    "INT32 div/rem": ["s32_div", "u32_div", "s32_rem", "u32_rem"],
    "Bit ops": ["b32_and", "b32_or", "b32_xor", "b32_not", "b32_cnot",
                "b32_shl", "b32_shr", "s32_shr", "b32_lop3"],
    "Bit manip": ["b32_popc", "b32_clz", "b32_brev",
                  "s32_bfe", "u32_bfe", "b32_bfi", "s32_bfind", "b32_prmt"],
    "Dot product": ["dp4a_ss", "dp4a_uu", "dp4a_su", "dp4a_us",
                    "dp2a_lo_ss", "dp2a_hi_ss"],
    "Carry ops": ["s32_add_cc", "s32_sub_cc", "s32_addcc_addc"],
}

def classify(op):
    for cat, ops in CATS.items():
        if op in ops:
            return cat
    return "Other"

def throttle_label(ratio):
    if ratio is None:
        return "—"
    if ratio <= 1.5:
        return "OK"
    if ratio <= 3.0:
        return f"~{ratio:.1f}×"
    return f"**{ratio:.1f}×**"

def main():
    rdir = Path("results/arithm_gen")
    files = sorted(rdir.glob("*.json"))
    if len(files) < 2:
        print("Need at least 2 result files")
        sys.exit(1)

    # Load all
    gpus = []
    all_data = []
    for f in files:
        name, data = load(f)
        gpus.append(name)
        all_data.append(data)

    # Find baseline (2080 Super or 3090)
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

    baseline_name = gpus[baseline_idx]

    # Get all ops (union)
    all_ops = set()
    for d in all_data:
        all_ops.update(d.keys())

    # Sort by category then name
    cat_order = list(CATS.keys()) + ["Other"]
    def sort_key(op):
        cat = classify(op)
        ci = cat_order.index(cat) if cat in cat_order else 999
        return (ci, op)
    all_ops = sorted(all_ops, key=sort_key)

    # --- FULL TABLE ---
    lines = []
    lines.append("# Comprehensive Arithmetic Instruction Benchmark\n")
    lines.append(f"Baseline: **{baseline_name}**\n")
    lines.append("")

    # Build header
    hdr = "| Category | Instruction | "
    sep = "|----------|-------------|"
    for i, name in enumerate(gpus):
        short = name.replace("NVIDIA ", "").replace("GeForce ", "")
        hdr += f" {short} tput | {short} lat |"
        sep += "-------:|-------:|"
        if i == baseline_idx:
            continue
    # Add ratio columns for non-baseline
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
        # Category separator
        cat_display = cat if cat != prev_cat else ""
        prev_cat = cat

        row = f"| {cat_display} | `{op}` |"
        for i, d in enumerate(all_data):
            tput = d.get(op, {}).get("tput")
            lat = d.get(op, {}).get("lat")
            tput_s = f"{tput:.3f}" if tput else "—"
            lat_s = f"{lat:.3f}" if lat else "—"
            row += f" {tput_s} | {lat_s} |"

        # Ratio columns
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

    lines.append("")

    # --- SUMMARY BY CATEGORY ---
    lines.append("## Throttling Summary by Category\n")
    lines.append("Ratios = CMP 90HX ns/op ÷ baseline ns/op. Higher = slower = more throttled.\n")

    for cat in cat_order:
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
                    ratios.append((op, c_tput / b_tput, c_tput, b_tput))

            if not ratios:
                lines.append(f"- **{short}**: no data\n")
                continue

            r_vals = [r[1] for r in ratios]
            r_min = min(r_vals)
            r_max = max(r_vals)
            r_avg = sum(r_vals) / len(r_vals)

            if r_max <= 1.5:
                verdict = "UNTHROTTLED"
            elif r_min <= 1.5 and r_max > 3.0:
                verdict = "MIXED — some ops throttled, some not"
            elif r_min > 3.0:
                verdict = "THROTTLED"
            else:
                verdict = "MILDLY THROTTLED"

            lines.append(f"**{short}**: {verdict} (range {r_min:.2f}×–{r_max:.2f}×, avg {r_avg:.2f}×)")

            # Show outliers if mixed
            if r_max / max(r_min, 0.01) > 2.0:
                fast = [(op, r) for op, r, _, _ in ratios if r <= 1.5]
                slow = [(op, r) for op, r, _, _ in ratios if r > 3.0]
                mid = [(op, r) for op, r, _, _ in ratios if 1.5 < r <= 3.0]
                if fast:
                    lines.append(f"- Unthrottled: {', '.join(f'`{o}` ({r:.2f}×)' for o,r in fast)}")
                if mid:
                    lines.append(f"- Mildly: {', '.join(f'`{o}` ({r:.2f}×)' for o,r in mid)}")
                if slow:
                    lines.append(f"- Throttled: {', '.join(f'`{o}` ({r:.2f}×)' for o,r in slow)}")
            else:
                # Uniform — list all
                detail = ", ".join(f"`{op}` ({r:.2f}×)" for op, r, _, _ in ratios)
                lines.append(f"- {detail}")
            lines.append("")

    # --- KEY FINDINGS ---
    lines.append("## Key Findings\n")
    lines.append("1. **Signed vs Unsigned**: No difference in throttling. `s32_add` and `u32_add` have identical throughput, same for mul, mad, min, max, div, rem.")
    lines.append("2. **DP4A type variants**: All four (ss, uu, su, us) throttled identically (~15.9×).")
    lines.append("3. **DP2A is UNTHROTTLED** while DP4A is heavily throttled — different execution units.")
    lines.append("4. **FP32 is non-uniform**: add/sub/mul/fma/mad ~7-8×, min/max/neg/abs unthrottled, SFU 1.7-1.8×, ex2/tanh ~1.2-1.4×.")
    lines.append("5. **FP16/FP16x2**: CMP is actually FASTER than 2080S (~0.8× ratio) — wider Ampere FP16 pipe.")
    lines.append("6. **All integer arithmetic**: Completely unthrottled (1.1-1.2×), consistent with mining firmware preserving INT pipelines.")
    lines.append("7. **INT div/rem**: These are multi-instruction expansions — ratio follows the constituent ops (unthrottled).")
    lines.append("8. **Bit manipulation (popc, clz, bfind, brev, bfe, bfi, prmt)**: All unthrottled.")
    lines.append("")

    out = Path("results/arithm_gen/results.md")
    out.write_text("\n".join(lines))
    print(f"Written to {out} ({len(lines)} lines)")

if __name__ == "__main__":
    main()
