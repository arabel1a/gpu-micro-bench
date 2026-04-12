"""Generate plots and markdown tables from memtest JSON results.

Usage:
  uv run python plot.py                          # uses cfg.output_dir from config
  uv run python plot.py --config-name=smoke       # uses smoke output dir
  uv run python plot.py output_dir=results/custom  # override
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
# Plot functions — one per visualization
# ============================================================

def plot_latency_single(gpu, plot_dir):
    rows = gpu["tests"].get("LAT_WARM", [])
    if not rows:
        return
    sizes = np.array([r["size_bytes"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 6))
    for h in HINTS:
        ax.semilogx(sizes, [r[f"{h}_ns"] for r in rows],
                     label=HINT_LABELS[h], color=HINT_COLORS[h], marker=HINT_MARKERS[h],
                     linestyle="--" if h == "cn" else "-")
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
    for h in HINTS:
        ax.semilogx(sizes, [r[f"{h}_ns"] for r in rows],
                     label=HINT_LABELS[h], color=HINT_COLORS[h], marker=HINT_MARKERS[h],
                     linestyle="--" if h == "cn" else "-")
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
    for h in HINTS:
        ax.semilogx(sizes, [r[f"{h}_gbps"] for r in rows],
                     label=HINT_LABELS[h], color=HINT_COLORS[h], marker=HINT_MARKERS[h],
                     linestyle="--" if h == "cn" else "-")
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
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get(test_key, [])
        if not rows:
            continue
        sizes = np.array([r["size_bytes"] for r in rows])
        label = short_name(gpu["gpu"]["name"])
        ax.semilogx(sizes, [r["ca_ns"] for r in rows],
                     label=f"{label} ca", color=colors[i], marker="o", lw=2)
        ax.semilogx(sizes, [r["cn_ns"] for r in rows],
                     label=f"{label} cn", color=colors[i], marker="x", ls="--", lw=1.5)
        ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color=colors[i], ls=":", alpha=0.4)
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
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, gpu in enumerate(gpus):
        rows = gpu["tests"].get("BW_WARM", [])
        if not rows:
            continue
        sizes = np.array([r["size_bytes"] for r in rows])
        label = short_name(gpu["gpu"]["name"])
        ax.semilogx(sizes, [r["ca_gbps"] for r in rows],
                     label=f"{label} ca", color=colors[i], marker="o", lw=2)
        ax.semilogx(sizes, [r["cn_gbps"] for r in rows],
                     label=f"{label} cn", color=colors[i], marker="x", ls="--", lw=1.5)
        ax.axvline(gpu["gpu"]["l2_kb"] * 1024, color=colors[i], ls=":", alpha=0.4)
    ax.set_xlabel("Working Set Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Sequential Bandwidth — Cross-GPU (ca vs cn)")
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
            data[h].append(last[f"{h}_ns"])

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
# Markdown tables
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
    f.write("- `cn`: `ld.global.ca` + L2 flushed (read 4x L2 trash). Guaranteed DRAM.\n")


def write_latency_md(f, gpu, test_key, title):
    rows = gpu["tests"].get(test_key, [])
    if not rows:
        return
    f.write(f"\n### {title} — {gpu['gpu']['name']} (L2={gpu['gpu']['l2_kb']} KB)\n\n")
    f.write("| Size | ca (L1+L2) | cg (L2 only) | cs (stream) | cn (flushed) | wall ms |\n")
    f.write("|------|-----------|-------------|------------|-------------|--------|\n")
    for r in rows:
        f.write(f"| {fmt_size(r['size_bytes']):>6} "
                f"| {r['ca_ns']:>8.1f} ns "
                f"| {r['cg_ns']:>10.1f} ns "
                f"| {r['cs_ns']:>9.1f} ns "
                f"| {r['cn_ns']:>10.1f} ns "
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


# ============================================================
# Main
# ============================================================

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    results_dir = Path(cfg.output_dir)
    gpus = load_results(results_dir)
    if not gpus:
        log.error("No .json files in %s", results_dir)
        return

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for gpu in gpus:
        plot_latency_single(gpu, plot_dir)
        plot_latency_multi(gpu, plot_dir)
        plot_bandwidth(gpu, plot_dir)

    if len(gpus) >= 2:
        plot_cross_gpu_latency(gpus, "LAT_WARM",
                               "Pointer-Chase Latency (1 thread) — Cross-GPU",
                               "lat_single_cross", plot_dir)
        plot_cross_gpu_latency(gpus, "LAT_MULTI",
                               "Pointer-Chase Latency (all SMs) — Cross-GPU",
                               "lat_multi_cross", plot_dir)
        plot_cross_gpu_bandwidth(gpus, plot_dir)
        plot_dram_hint_bars(gpus, plot_dir)

    log.info("Plots saved to %s", plot_dir)

    # Markdown tables to file
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
    log.info("Markdown saved to %s", md_path)


if __name__ == "__main__":
    main()
