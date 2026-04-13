"""Run GPU microbenchmarks and save results as JSON.

Usage:
  uv run python run.py                              # memtest (default), both GPUs
  uv run python run.py bench=mmvq                   # MMVQ kernel isolation bench
  uv run python run.py --config-name=smoke           # quick memtest smoke test
  uv run python run.py bench=mmvq bench.n_reps=50    # MMVQ with fewer reps
"""

import json
import logging
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ============================================================================
# memtest
# ============================================================================

def parse_memtest_output(stdout: str) -> dict:
    result = {"gpu": {}, "tests": {}}
    for line in stdout.strip().split("\n"):
        parts = line.strip().split(",")
        tag = parts[0]
        if tag == "GPU":
            result["gpu"] = {
                "name": parts[1],
                "sm": parts[2],
                "n_sms": int(parts[3]),
                "l2_kb": int(parts[4]),
            }
        elif tag in ("LAT_WARM", "LAT_MULTI"):
            result["tests"].setdefault(tag, []).append({
                "size_bytes": int(parts[1]),
                "ca_ns": float(parts[2]),
                "cg_ns": float(parts[3]),
                "cs_ns": float(parts[4]),
                "cn_ns": float(parts[5]) if parts[5] else None,
                "wall_ms": float(parts[6]),
            })
        elif tag == "BW_WARM":
            result["tests"].setdefault(tag, []).append({
                "size_bytes": int(parts[1]),
                "ca_gbps": float(parts[2]),
                "cg_gbps": float(parts[3]),
                "cs_gbps": float(parts[4]),
                "cn_gbps": float(parts[5]),
                "wall_ms": float(parts[6]),
            })
        elif tag == "BLOCK_COST":
            result["tests"].setdefault(tag, []).append({
                "n_blocks": int(parts[1]),
                "blocks_per_sm": int(parts[2]),
                "total_us": float(parts[3]),
                "us_per_block": float(parts[4]),
                "wall_ms": float(parts[5]),
            })
        elif tag == "CONC_WARP":
            result["tests"].setdefault(tag, []).append({
                "n_warps": int(parts[1]),
                "ns_per_hop": float(parts[2]),
                "hops_per_us": float(parts[3]),
                "wall_ms": float(parts[4]),
            })
        elif tag == "CONC_BLOCK":
            result["tests"].setdefault(tag, []).append({
                "n_blocks": int(parts[1]),
                "ns_per_hop": float(parts[2]),
                "hops_per_us": float(parts[3]),
                "wall_ms": float(parts[4]),
            })
        elif tag == "L2_PERSIST":
            result["tests"].setdefault(tag, []).append({
                "size_bytes": int(parts[1]),
                "cold_bw_gbps": float(parts[2]),
                "warm_bw_gbps": float(parts[3]),
                "wall_ms": float(parts[4]),
            })
    return result


def run_memtest(cfg: DictConfig):
    out_dir = Path(cfg.output_dir)
    if out_dir.exists() and any(out_dir.glob("*.json")):
        if not cfg.overwrite:
            raise FileExistsError(f"Results exist in {out_dir} and overwrite=false")
        log.info("Overwriting existing results in %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes_csv = ",".join(str(s) for s in cfg.bench.sizes)

    for gpu_id in cfg.gpu_ids:
        cmd = [cfg.bench.binary, str(gpu_id), sizes_csv, str(cfg.bench.n_hops_min)]
        log.info("Running: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("GPU %d failed:\n%s", gpu_id, proc.stderr)
            continue

        log.info(proc.stderr.strip())
        result = parse_memtest_output(proc.stdout)
        gpu_name = result["gpu"]["name"].strip().replace(" ", "_")
        out_path = out_dir / f"{gpu_name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        log.info("Saved: %s", out_path)


# ============================================================================
# mmvq
# ============================================================================

def parse_mmvq_output(stdout: str) -> dict:
    result = {"gpu": {}, "layers": [], "summary": {}}
    for line in stdout.strip().split("\n"):
        parts = line.strip().split(",")
        tag = parts[0]
        if tag == "GPU":
            result["gpu"] = {
                "name": parts[1],
                "sm": parts[2],
                "n_sms": int(parts[3]),
                "l2_kb": int(parts[4]),
            }
        elif tag == "LAYER":
            result["layers"].append({
                "name": parts[1],
                "nrows": int(parts[2]),
                "ncols": int(parts[3]),
                "count": int(parts[4]),
                "us_per_token": float(parts[5]),
                "us_per_call": float(parts[6]),
                "weight_mb": float(parts[7]),
                "eff_bw_gbps": float(parts[8]),
            })
        elif tag == "SUMMARY":
            result["summary"] = {
                "us_per_token": float(parts[1]),
                "equiv_tps": float(parts[2]),
                "total_weight_mb": float(parts[3]),
            }
    return result


def run_mmvq(cfg: DictConfig):
    out_dir = Path(cfg.output_dir)
    if out_dir.exists() and any(out_dir.glob("*.json")):
        if not cfg.overwrite:
            raise FileExistsError(f"Results exist in {out_dir} and overwrite=false")
        log.info("Overwriting existing results in %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build layers CSV from config: "name:nrows:ncols:count,..."
    layer_parts = []
    for name, spec in cfg.bench.layers.items():
        layer_parts.append(f"{name}:{spec.nrows}:{spec.ncols}:{spec.count}")
    layers_csv = ",".join(layer_parts)

    for gpu_id in cfg.gpu_ids:
        cmd = [
            cfg.bench.binary, str(gpu_id),
            str(cfg.bench.n_warmup), str(cfg.bench.n_reps),
            layers_csv,
        ]
        log.info("Running: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("GPU %d failed:\n%s", gpu_id, proc.stderr)
            continue

        log.info(proc.stderr.strip())
        result = parse_mmvq_output(proc.stdout)
        gpu_name = result["gpu"]["name"].strip().replace(" ", "_")
        out_path = out_dir / f"{gpu_name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        log.info("Saved: %s", out_path)


# ============================================================================
# arithm
# ============================================================================

def parse_arithm_output(stdout: str) -> dict:
    result = {"gpu": {}, "tests": []}
    for line in stdout.strip().split("\n"):
        parts = line.strip().split(",")
        tag = parts[0]
        if tag == "GPU":
            result["gpu"] = {
                "name": parts[1],
                "sm": parts[2],
                "n_sms": int(parts[3]),
                "l2_kb": int(parts[4]),
            }
        elif tag == "ARITH":
            result["tests"].append({
                "op": parts[1],
                "mode": parts[2],
                "n_iters": int(parts[3]),
                "total_ns": float(parts[4]),
                "ns_per_op": float(parts[5]),
                "ops_per_ns": float(parts[6]),
                "wall_ms": float(parts[7]),
            })
    return result


def run_arithm(cfg: DictConfig):
    out_dir = Path(cfg.output_dir)
    if out_dir.exists() and any(out_dir.glob("*.json")):
        if not cfg.overwrite:
            raise FileExistsError(f"Results exist in {out_dir} and overwrite=false")
        log.info("Overwriting existing results in %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gpu_id in cfg.gpu_ids:
        cmd = [
            cfg.bench.binary, str(gpu_id),
            str(cfg.bench.n_iters), str(cfg.bench.n_warmup), str(cfg.bench.n_reps),
        ]
        log.info("Running: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("GPU %d failed:\n%s", gpu_id, proc.stderr)
            continue

        log.info(proc.stderr.strip())
        result = parse_arithm_output(proc.stdout)
        gpu_name = result["gpu"]["name"].strip().replace(" ", "_")
        out_path = out_dir / f"{gpu_name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        log.info("Saved: %s", out_path)


# ============================================================================
# dispatch
# ============================================================================

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    fn = hydra.utils.get_method(cfg.bench._target_)
    fn(cfg)


if __name__ == "__main__":
    main()
