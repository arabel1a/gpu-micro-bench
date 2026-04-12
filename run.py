"""Run GPU memory microbenchmark and save results as JSON.

Usage:
  uv run python run.py                        # full run, both GPUs
  uv run python run.py --config-name=smoke     # quick smoke test
"""

import json
import logging
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def parse_output(stdout: str) -> dict:
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
                "cn_ns": float(parts[5]),
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

    return result


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = Path(cfg.output_dir)
    if out_dir.exists() and any(out_dir.glob("*.json")):
        if not cfg.overwrite:
            raise FileExistsError(f"Results exist in {out_dir} and overwrite=false")
        log.info("Overwriting existing results in %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes_csv = ",".join(str(s) for s in cfg.sizes)

    for gpu_id in cfg.gpu_ids:
        cmd = [cfg.binary, str(gpu_id), sizes_csv, str(cfg.n_hops_min)]
        log.info("Running: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("GPU %d failed:\n%s", gpu_id, proc.stderr)
            continue

        log.info(proc.stderr.strip())
        result = parse_output(proc.stdout)
        gpu_name = result["gpu"]["name"].strip().replace(" ", "_")
        out_path = out_dir / f"{gpu_name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        log.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
