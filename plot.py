"""Generate plots and markdown tables from benchmark JSON results.

Usage:
  uv run python plot.py                          # memtest (default)
  uv run python plot.py bench=mmvq               # mmvq tables only
  uv run python plot.py --config-name=smoke       # uses smoke output dir
"""

import json
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

HINTS = ["ca", "cg", "cs", "cn"]
HINT_LABELS = {
    "ca": "ca (L1+L2)",
    "cg": "cg (L2 only)",
    "cs": "cs (stream)",
    "cn": "cn (L2 flushed)",
}
HINT_COLORS = {"ca": "#2196F3", "cg": "#FF9800", "cs": "#9C27B0", "cn": "#F44336"}
HINT_MARKERS = {"ca": "o", "cg": "s", "cs": "^", "cn": "x"}

GPU_COLORS = plt.cm.tab10.colors


def fmt_size(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b // (1024 * 1024)} MB"
    return f"{b // 1024} KB"


def short_name(name: str) -> str:
    return (name.replace("NVIDIA ", "").replace("GeForce ", "")
            .replace("RTX ", "").replace("SUPER", "S").replace(" ", ""))


def load_results(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


# ============================================================
# Memtest plot functions — latency / bandwidth (existing)
# ============================================================

def _plot_hint_lines(ax, rows, sizes, key_suffix, hint_list=HINTS):
    for h in hint_list:
        vals = [r[f"{h}_{key_suffix}"] for r in rows]
        if any(v is None for v in vals):
            valid = [(s, v) for s, v in zip(sizes, vals) if v is not None]
            if not valid:
                continue
            sx, sy = zip(*valid)
            ax.semilogx(sx, sy, label=HINT_LABELS[h], color=HINT_COLORS[h],
                         marker=HINT_MARKERS[h], linestyle="--" if h == "cn" else "-")
        else:
            ax.semilogx(sizes, vals, label=HINT_LABELS[h], color=HINT_COLORS[h],
                         marker=HINT_MARKERS[h], linestyle="--" if h == "cn" else "-")


def plot_latency_single(gpu, plot_dir):
    rows = gpu["tests"].get("LAT_WARM", [])
    if not rows:
        return
    sizes = np.array([r["size_bytes"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_hint_lines(ax, rows, sizes, "ns")
    ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color="gray", ls=":", alpha=0.7,
               label=f"L2 = {gpu['gpu']['l2_kb']} KB")
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Latency (ns)")
    ax.set_title(f"Pointer-Chase Latency (1 thread) — {gpu['gpu']['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes[::2])
    ax.set_xticklabels([fmt_size(s) for s in sizes[::2]], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / f"lat_single_{short_name(gpu['gpu']['name'])}.png", dpi=150)
    plt.close(fig)


def plot_latency_multi(gpu, plot_dir):
    rows = gpu["tests"].get("LAT_MULTI", [])
    if not rows:
        return
    sizes = np.array([r["size_bytes"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_hint_lines(ax, rows, sizes, "ns")
    ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color="gray", ls=":", alpha=0.7,
               label=f"L2 = {gpu['gpu']['l2_kb']} KB")
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Latency (ns)")
    ax.set_title(f"Pointer-Chase Latency ({gpu['gpu']['n_sms']} SMs) — {gpu['gpu']['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes[::2])
    ax.set_xticklabels([fmt_size(s) for s in sizes[::2]], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / f"lat_multi_{short_name(gpu['gpu']['name'])}.png", dpi=150)
    plt.close(fig)


def plot_bandwidth(gpu, plot_dir):
    rows = gpu["tests"].get("BW_WARM", [])
    if not rows:
        return
    sizes = np.array([r["size_bytes"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_hint_lines(ax, rows, sizes, "gbps")
    ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color="gray", ls=":", alpha=0.7,
               label=f"L2 = {gpu['gpu']['l2_kb']} KB")
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title(f"Sequential Bandwidth (all SMs) — {gpu['gpu']['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes[::2])
    ax.set_xticklabels([fmt_size(s) for s in sizes[::2]], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / f"bw_{short_name(gpu['gpu']['name'])}.png", dpi=150)
    plt.close(fig)


def plot_cross_gpu_latency(gpus, test_key, title, fname, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    cross_hints = [("ca", "o", "-", 2), ("cg", "s", "-.", 1.5), ("cn", "x", "--", 1.5)]
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get(test_key, [])
        if not rows:
            continue
        label = short_name(gpu["gpu"]["name"])
        for h, marker, ls, lw in cross_hints:
            vals = [(r["size_bytes"], r[f"{h}_ns"]) for r in rows if r[f"{h}_ns"] is not None]
            if not vals:
                continue
            sx, sy = zip(*vals)
            ax.semilogx(sx, sy, label=f"{label} {h}", color=GPU_COLORS[i],
                         marker=marker, ls=ls, lw=lw)
        ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color=GPU_COLORS[i], ls=":", alpha=0.4)
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Latency (ns)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    sizes0 = np.array([r["size_bytes"] for r in gpus[0]["tests"].get(test_key, [])])
    if len(sizes0):
        ax.set_xticks(sizes0[::2])
        ax.set_xticklabels([fmt_size(s) for s in sizes0[::2]], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{fname}.png", dpi=150)
    plt.close(fig)


def plot_cross_gpu_bandwidth(gpus, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    cross_hints = [("ca", "o", "-", 2), ("cg", "s", "-.", 1.5), ("cn", "x", "--", 1.5)]
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get("BW_WARM", [])
        if not rows:
            continue
        sizes = np.array([r["size_bytes"] for r in rows])
        label = short_name(gpu["gpu"]["name"])
        for h, marker, ls, lw in cross_hints:
            ax.semilogx(sizes, [r[f"{h}_gbps"] for r in rows],
                         label=f"{label} {h}", color=GPU_COLORS[i], marker=marker, ls=ls, lw=lw)
        ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color=GPU_COLORS[i], ls=":", alpha=0.4)
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Sequential Bandwidth — Cross-GPU (ca / cg / cn)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    sizes0 = np.array([r["size_bytes"] for r in gpus[0]["tests"].get("BW_WARM", [])])
    if len(sizes0):
        ax.set_xticks(sizes0[::2])
        ax.set_xticklabels([fmt_size(s) for s in sizes0[::2]], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "bw_cross_gpu.png", dpi=150)
    plt.close(fig)


def plot_dram_hint_bars(gpus, plot_dir):
    """Bar chart: DRAM latency by hint, per GPU."""
    names = []
    data = {h: [] for h in HINTS}
    for gpu in gpus:
        rows = gpu["tests"].get("LAT_WARM", [])
        if not rows:
            continue
        names.append(short_name(gpu["gpu"]["name"]))
        last = rows[-1]
        for h in HINTS:
            val = last[f"{h}_ns"]
            data[h].append(val if val is not None else 0)

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, h in enumerate(HINTS):
        ax.bar(x + i * width, data[h], width, label=HINT_LABELS[h], color=HINT_COLORS[h])
    ax.set_ylabel("Latency (ns)")
    ax.set_title(f"DRAM Latency by Cache Hint ({fmt_size(rows[-1]['size_bytes'])})")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "lat_dram_hint_bar.png", dpi=150)
    plt.close(fig)


# ============================================================
# New plots — concurrency, block cost, L2 persistence
# ============================================================

def plot_conc_warp(gpus, plot_dir):
    """Warp concurrency: ns/hop and throughput (hops/us) vs n_warps, per GPU."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get("CONC_WARP", [])
        if not rows:
            continue
        label = short_name(gpu["gpu"]["name"])
        nw = [r["n_warps"] for r in rows]
        ax1.plot(nw, [r["ns_per_hop"] for r in rows], marker="o",
                 color=GPU_COLORS[i], label=label)
        ax2.plot(nw, [r["hops_per_us"] for r in rows], marker="s",
                 color=GPU_COLORS[i], label=label)
        # ideal linear scaling from single-warp throughput
        base = rows[0]["hops_per_us"]
        ax2.plot(nw, [base * w for w in nw], ls="--", alpha=0.4,
                 color=GPU_COLORS[i], label=f"{label} ideal")
    ax1.set_xlabel("Warps per block")
    ax1.set_ylabel("Latency (ns/hop)")
    ax1.set_title("Warp Concurrency — Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Warps per block")
    ax2.set_ylabel("Throughput (hops/us)")
    ax2.set_title("Warp Concurrency — Throughput")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "conc_warp.png", dpi=150)
    plt.close(fig)


def plot_conc_block(gpus, plot_dir):
    """Block concurrency: ns/hop and throughput (hops/us) vs n_blocks, per GPU."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get("CONC_BLOCK", [])
        if not rows:
            continue
        label = short_name(gpu["gpu"]["name"])
        nb = [r["n_blocks"] for r in rows]
        ax1.semilogx(nb, [r["ns_per_hop"] for r in rows], marker="o",
                      color=GPU_COLORS[i], label=label)
        ax2.loglog(nb, [r["hops_per_us"] for r in rows], marker="s",
                   color=GPU_COLORS[i], label=label)
        # ideal linear scaling
        base = rows[0]["hops_per_us"]
        ax2.loglog(nb, [base * b for b in nb], ls="--", alpha=0.4,
                   color=GPU_COLORS[i], label=f"{label} ideal")
    ax1.set_xlabel("Number of blocks")
    ax1.set_ylabel("Latency (ns/hop)")
    ax1.set_title("Block Concurrency — Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Number of blocks")
    ax2.set_ylabel("Throughput (hops/us)")
    ax2.set_title("Block Concurrency — Throughput")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "conc_block.png", dpi=150)
    plt.close(fig)


def plot_block_cost(gpus, plot_dir):
    """Block dispatch overhead: total time and per-block cost vs blocks/SM."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get("BLOCK_COST", [])
        if not rows:
            continue
        label = short_name(gpu["gpu"]["name"])
        bpsm = [r["blocks_per_sm"] for r in rows]
        ax1.plot(bpsm, [r["total_us"] for r in rows], marker="o",
                 color=GPU_COLORS[i], label=label)
        ax2.plot(bpsm, [r["us_per_block"] for r in rows], marker="s",
                 color=GPU_COLORS[i], label=label)
    ax1.set_xlabel("Blocks per SM")
    ax1.set_ylabel("Total time (us)")
    ax1.set_title("Block Dispatch — Total Execution Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Blocks per SM")
    ax2.set_ylabel("Cost per block (us)")
    ax2.set_title("Block Dispatch — Per-Block Overhead")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "block_cost.png", dpi=150)
    plt.close(fig)


def plot_l2_persist(gpus, plot_dir):
    """L2 persistence: cold vs warm bandwidth by working set size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = [("-", "o"), ("--", "x")]  # cold=solid, warm=dashed
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get("L2_PERSIST", [])
        if not rows:
            continue
        label = short_name(gpu["gpu"]["name"])
        sizes = [r["size_bytes"] for r in rows]
        ax.semilogx(sizes, [r["cold_bw_gbps"] for r in rows],
                     ls="-", marker="o", color=GPU_COLORS[i], label=f"{label} cold")
        ax.semilogx(sizes, [r["warm_bw_gbps"] for r in rows],
                     ls="--", marker="x", color=GPU_COLORS[i], label=f"{label} warm")
        ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color=GPU_COLORS[i], ls=":", alpha=0.4)
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("L2 Persistence — Cold (1st launch) vs Warm (2nd-10th)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # x-tick labels from first GPU with data
    for gpu in gpus:
        rows = gpu["tests"].get("L2_PERSIST", [])
        if rows:
            sizes = [r["size_bytes"] for r in rows]
            ax.set_xticks(sizes[::2])
            ax.set_xticklabels([fmt_size(s) for s in sizes[::2]], rotation=45, ha="right")
            break
    fig.tight_layout()
    fig.savefig(plot_dir / "l2_persist.png", dpi=150)
    plt.close(fig)


# ============================================================
# Markdown tables — memtest
# ============================================================

def write_explanation_table(f):
    f.write("\n## Cache Hint Explanation\n\n")
    f.write("| | Small (fits L1) | Medium (fits L2) | Large (DRAM) |\n")
    f.write("|---|---|---|---|\n")
    f.write("| **ca** (L1+L2) | L1 latency | L2 latency | DRAM latency |\n")
    f.write("| **cg** (L2 only) | L2 latency | L2 latency | DRAM latency |\n")
    f.write("| **cs** (stream) | ~DRAM latency | ~DRAM latency | DRAM latency |\n")
    f.write("| **cn** (flushed) | DRAM latency | DRAM latency | DRAM latency |\n")
    f.write("\n")
    f.write("- `ca`: `ld.global.ca` — cache at all levels (L1 + L2). Default.\n")
    f.write("- `cg`: `ld.global.cg` — bypass L1, cache in L2 only.\n")
    f.write("- `cs`: `ld.global.cs` — streaming hint, bypass L1, evict-first in L2.\n")
    f.write("- `cn`: `ld.global.cg` + L2 flushed before each rep. Reported for sizes >= 64 KB.\n")


def write_latency_md(f, gpu, test_key, title):
    rows = gpu["tests"].get(test_key, [])
    if not rows:
        return
    f.write(f"\n### {title} — {gpu['gpu']['name']} (L2={gpu['gpu']['l2_kb']} KB)\n\n")
    f.write("| Size | ca (L1+L2) | cg (L2 only) | cs (stream) | cn (flushed) | wall ms |\n")
    f.write("|------|-----------|-------------|------------|-------------|--------|\n")
    for r in rows:
        cn_str = f"{r['cn_ns']:>10.1f} ns" if r['cn_ns'] is not None else "          —"
        f.write(f"| {fmt_size(r['size_bytes']):>6} "
                f"| {r['ca_ns']:>8.1f} ns "
                f"| {r['cg_ns']:>10.1f} ns "
                f"| {r['cs_ns']:>9.1f} ns "
                f"| {cn_str} "
                f"| {r['wall_ms']:>7.0f} |\n")


def write_bw_md(f, gpu, test_key, title):
    rows = gpu["tests"].get(test_key, [])
    if not rows:
        return
    f.write(f"\n### {title} — {gpu['gpu']['name']} (L2={gpu['gpu']['l2_kb']} KB)\n\n")
    f.write("| Size | ca (L1+L2) | cg (L2 only) | cs (stream) | cn (flushed) | wall ms |\n")
    f.write("|------|-----------|-------------|------------|-------------|--------|\n")
    for r in rows:
        f.write(f"| {fmt_size(r['size_bytes']):>6} "
                f"| {r['ca_gbps']:>7.1f} GB/s "
                f"| {r['cg_gbps']:>9.1f} GB/s "
                f"| {r['cs_gbps']:>8.1f} GB/s "
                f"| {r['cn_gbps']:>9.1f} GB/s "
                f"| {r['wall_ms']:>7.0f} |\n")


def write_block_cost_md(f, gpu):
    rows = gpu["tests"].get("BLOCK_COST", [])
    if not rows:
        return
    f.write(f"\n### Block Dispatch Overhead — {gpu['gpu']['name']}\n\n")
    f.write("| blocks/SM | total us | us/block | wall ms |\n")
    f.write("|-----------|---------|---------|--------|\n")
    for r in rows:
        f.write(f"| {r['blocks_per_sm']:>9} "
                f"| {r['total_us']:>7.3f} "
                f"| {r['us_per_block']:>7.3f} "
                f"| {r['wall_ms']:>7.1f} |\n")


def write_conc_warp_md(f, gpu):
    rows = gpu["tests"].get("CONC_WARP", [])
    if not rows:
        return
    f.write(f"\n### Warp Concurrency (1 SM, DRAM) — {gpu['gpu']['name']}\n\n")
    f.write("| n_warps | ns/hop | hops/us | wall ms |\n")
    f.write("|---------|--------|---------|--------|\n")
    for r in rows:
        f.write(f"| {r['n_warps']:>7} "
                f"| {r['ns_per_hop']:>6.1f} "
                f"| {r['hops_per_us']:>7.2f} "
                f"| {r['wall_ms']:>7.1f} |\n")


def write_conc_block_md(f, gpu):
    rows = gpu["tests"].get("CONC_BLOCK", [])
    if not rows:
        return
    f.write(f"\n### Block Concurrency (GPU-wide, DRAM) — {gpu['gpu']['name']}\n\n")
    f.write("| n_blocks | ns/hop | hops/us | wall ms |\n")
    f.write("|----------|--------|---------|--------|\n")
    for r in rows:
        f.write(f"| {r['n_blocks']:>8} "
                f"| {r['ns_per_hop']:>6.1f} "
                f"| {r['hops_per_us']:>7.2f} "
                f"| {r['wall_ms']:>7.1f} |\n")


def write_l2_persist_md(f, gpu):
    rows = gpu["tests"].get("L2_PERSIST", [])
    if not rows:
        return
    f.write(f"\n### L2 Persistence — {gpu['gpu']['name']}\n\n")
    f.write("| Size | cold GB/s | warm GB/s | warm/cold | wall ms |\n")
    f.write("|------|----------|----------|-----------|--------|\n")
    for r in rows:
        ratio = r["warm_bw_gbps"] / r["cold_bw_gbps"] if r["cold_bw_gbps"] > 0 else 0
        f.write(f"| {fmt_size(r['size_bytes']):>6} "
                f"| {r['cold_bw_gbps']:>8.1f} "
                f"| {r['warm_bw_gbps']:>8.1f} "
                f"| {ratio:>8.2f}x "
                f"| {r['wall_ms']:>7.1f} |\n")


# ============================================================
# Markdown tables — mmvq
# ============================================================

def write_mmvq_md(f, gpus):
    f.write("# MMVQ Kernel Isolation Benchmark Results\n\n")

    # Per-GPU layer tables
    for gpu in gpus:
        name = gpu["gpu"]["name"]
        f.write(f"\n## {name} ({gpu['gpu']['sm']}, {gpu['gpu']['n_sms']} SMs)\n\n")

        f.write("| Layer | nrows | ncols | count | us/token | us/call | weight MB | eff BW GB/s |\n")
        f.write("|-------|-------|-------|-------|----------|---------|-----------|------------|\n")
        for layer in gpu["layers"]:
            f.write(f"| {layer['name']:>9} "
                    f"| {layer['nrows']:>5} "
                    f"| {layer['ncols']:>5} "
                    f"| {layer['count']:>5} "
                    f"| {layer['us_per_token']:>8.1f} "
                    f"| {layer['us_per_call']:>7.2f} "
                    f"| {layer['weight_mb']:>9.2f} "
                    f"| {layer['eff_bw_gbps']:>10.2f} |\n")

        s = gpu["summary"]
        f.write(f"\n**Summary**: {s['us_per_token']:.1f} us/token = "
                f"**{s['equiv_tps']:.1f} TPS**, "
                f"total weights {s['total_weight_mb']:.1f} MB\n")

    # Cross-GPU comparison table (if multiple)
    if len(gpus) >= 2:
        f.write("\n---\n\n## Cross-GPU Comparison\n\n")
        headers = ["Metric"] + [short_name(g["gpu"]["name"]) for g in gpus]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

        # Summary row
        vals = [f"{g['summary']['equiv_tps']:.1f}" for g in gpus]
        f.write("| TPS | " + " | ".join(vals) + " |\n")
        vals = [f"{g['summary']['us_per_token']:.0f}" for g in gpus]
        f.write("| us/token | " + " | ".join(vals) + " |\n")
        vals = [f"{g['summary']['total_weight_mb']:.0f}" for g in gpus]
        f.write("| total weight MB | " + " | ".join(vals) + " |\n")

        # Per-layer BW comparison
        f.write("\n### Effective Bandwidth by Layer (GB/s)\n\n")
        headers = ["Layer"] + [short_name(g["gpu"]["name"]) for g in gpus]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

        # Use first GPU's layer list as reference
        for j, layer in enumerate(gpus[0]["layers"]):
            vals = []
            for gpu in gpus:
                if j < len(gpu["layers"]):
                    vals.append(f"{gpu['layers'][j]['eff_bw_gbps']:.1f}")
                else:
                    vals.append("—")
            f.write(f"| {layer['name']} | " + " | ".join(vals) + " |\n")


# ============================================================
# Markdown tables + plots — arithm
# ============================================================

def write_arithm_md(f, gpus):
    f.write("# Arithmetic Instruction Throughput Benchmark Results\n\n")
    f.write("1 block, 128 threads (4 warps), 8 independent chains (tput) or 1 chain (lat).\n\n")

    # Per-GPU tables
    for gpu in gpus:
        name = gpu["gpu"]["name"]
        f.write(f"\n## {name} ({gpu['gpu']['sm']}, {gpu['gpu']['n_sms']} SMs)\n\n")
        f.write("| Op | Mode | ns/op | ops/ns | wall ms |\n")
        f.write("|----|------|-------|--------|--------|\n")
        for t in gpu["tests"]:
            f.write(f"| {t['op']:>4} | {t['mode']:>4} "
                    f"| {t['ns_per_op']:>7.3f} "
                    f"| {t['ops_per_ns']:>6.4f} "
                    f"| {t['wall_ms']:>7.1f} |\n")

    # Cross-GPU comparison (throughput mode only)
    if len(gpus) >= 2:
        f.write("\n---\n\n## Cross-GPU Comparison (throughput mode)\n\n")
        names = [short_name(g["gpu"]["name"]) for g in gpus]
        ops = sorted(set(t["op"] for t in gpus[0]["tests"]))

        headers = ["Op"] + [f"{n} ns/op" for n in names]
        if len(gpus) == 2:
            headers.append("ratio")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

        for op in ops:
            vals = []
            for gpu in gpus:
                tput = [t for t in gpu["tests"] if t["op"] == op and t["mode"] == "tput"]
                vals.append(tput[0]["ns_per_op"] if tput else None)
            row = [op]
            for v in vals:
                row.append(f"{v:.3f}" if v else "—")
            if len(gpus) == 2 and all(v is not None for v in vals):
                row.append(f"{vals[0]/vals[1]:.1f}x")
            f.write("| " + " | ".join(row) + " |\n")


def plot_arithm_bars(gpus, plot_dir):
    """Bar chart: ns/op per instruction, grouped by GPU."""
    ops = []
    seen = set()
    for t in gpus[0]["tests"]:
        if t["mode"] == "tput" and t["op"] not in seen:
            ops.append(t["op"])
            seen.add(t["op"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(ops))
    width = 0.8 / len(gpus)

    for i, gpu in enumerate(gpus):
        label = short_name(gpu["gpu"]["name"])
        tput_vals = []
        lat_vals = []
        for op in ops:
            tput = [t for t in gpu["tests"] if t["op"] == op and t["mode"] == "tput"]
            lat = [t for t in gpu["tests"] if t["op"] == op and t["mode"] == "lat"]
            tput_vals.append(tput[0]["ns_per_op"] if tput else 0)
            lat_vals.append(lat[0]["ns_per_op"] if lat else 0)
        ax1.bar(x + i * width, tput_vals, width, label=label, color=GPU_COLORS[i])
        ax2.bar(x + i * width, lat_vals, width, label=label, color=GPU_COLORS[i])

    for ax, title in [(ax1, "Reciprocal Throughput"), (ax2, "Latency")]:
        ax.set_ylabel("ns / op")
        ax.set_title(f"Instruction {title}")
        ax.set_xticks(x + width * (len(gpus) - 1) / 2)
        ax.set_xticklabels(ops)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(plot_dir / "arithm_bars.png", dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def run_memtest_plots(cfg: DictConfig):
    results_dir = Path(cfg.output_dir)
    gpus = load_results(results_dir)
    if not gpus:
        log.error("No .json files in %s", results_dir)
        return

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Per-GPU plots
    for gpu in gpus:
        plot_latency_single(gpu, plot_dir)
        plot_latency_multi(gpu, plot_dir)
        plot_bandwidth(gpu, plot_dir)

    # Cross-GPU plots
    if len(gpus) >= 2:
        plot_cross_gpu_latency(gpus, "LAT_WARM",
                               "Pointer-Chase Latency (1 thread) — Cross-GPU",
                               "lat_single_cross", plot_dir)
        plot_cross_gpu_latency(gpus, "LAT_MULTI",
                               "Pointer-Chase Latency (all SMs) — Cross-GPU",
                               "lat_multi_cross", plot_dir)
        plot_cross_gpu_bandwidth(gpus, plot_dir)
        plot_dram_hint_bars(gpus, plot_dir)
        plot_conc_warp(gpus, plot_dir)
        plot_conc_block(gpus, plot_dir)
        plot_block_cost(gpus, plot_dir)
        plot_l2_persist(gpus, plot_dir)

    log.info("Plots saved to %s", plot_dir)

    # Markdown tables
    md_path = results_dir / "results.md"
    with open(md_path, "w") as f:
        f.write("# GPU Memory Subsystem Benchmark Results\n")
        write_explanation_table(f)
        for gpu in gpus:
            f.write(f"\n---\n\n## {gpu['gpu']['name']} ({gpu['gpu']['sm']}, "
                    f"{gpu['gpu']['n_sms']} SMs, L2={gpu['gpu']['l2_kb']} KB)\n")
            write_latency_md(f, gpu, "LAT_WARM", "Pointer-Chase Latency (1 thread, warm)")
            write_latency_md(f, gpu, "LAT_MULTI", "Pointer-Chase Latency (all SMs, warm)")
            write_bw_md(f, gpu, "BW_WARM", "Sequential Bandwidth (all SMs, warm)")
            write_block_cost_md(f, gpu)
            write_conc_warp_md(f, gpu)
            write_conc_block_md(f, gpu)
            write_l2_persist_md(f, gpu)
    log.info("Markdown saved to %s", md_path)


def run_mmvq_plots(cfg: DictConfig):
    results_dir = Path(cfg.output_dir)
    gpus = load_results(results_dir)
    if not gpus:
        log.error("No .json files in %s", results_dir)
        return

    md_path = results_dir / "results.md"
    with open(md_path, "w") as f:
        write_mmvq_md(f, gpus)
    log.info("Markdown saved to %s", md_path)


def run_arithm_plots(cfg: DictConfig):
    results_dir = Path(cfg.output_dir)
    gpus = load_results(results_dir)
    if not gpus:
        log.error("No .json files in %s", results_dir)
        return

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    if len(gpus) >= 2:
        plot_arithm_bars(gpus, plot_dir)
    log.info("Plots saved to %s", plot_dir)

    md_path = results_dir / "results.md"
    with open(md_path, "w") as f:
        write_arithm_md(f, gpus)
    log.info("Markdown saved to %s", md_path)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.bench.name == "mmvq":
        run_mmvq_plots(cfg)
    elif cfg.bench.name == "arithm":
        run_arithm_plots(cfg)
    elif cfg.bench.name == "arithm_gen":
        run_arithm_plots(cfg)
 
    else:
        run_memtest_plots(cfg)


if __name__ == "__main__":
    main()
