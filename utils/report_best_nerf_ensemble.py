#!/usr/bin/env python3
"""Report the best run in a Nerfstudio ensemble by combined metric."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _combo(psnr: float, ssim: float, lpips: float) -> float:
    return (
        (10.0 ** (-float(psnr) / 10.0))
        * math.sqrt(max(0.0, 1.0 - float(ssim)))
        * float(lpips)
    ) ** (1.0 / 3.0)


def _find_eval_file(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("outputs/*/*/*/eval_all_images.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No eval_all_images.jsonl found under {run_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple eval_all_images.jsonl files found under {run_dir}: "
            f"{[str(m) for m in matches]}"
        )
    return matches[0]


def _best_row(eval_file: Path, max_step: int = -1) -> Tuple[Dict, float]:
    best_row: Optional[Dict] = None
    best_combo: Optional[float] = None

    with eval_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not all(k in row for k in ("step", "psnr", "ssim", "lpips")):
                continue
            step = int(row["step"])
            if max_step >= 0 and step > max_step:
                continue
            combo = _combo(float(row["psnr"]), float(row["ssim"]), float(row["lpips"]))
            if best_combo is None or combo < best_combo:
                best_combo = combo
                best_row = row

    if best_row is None or best_combo is None:
        raise RuntimeError(f"No valid psnr/ssim/lpips rows found in {eval_file}")
    return best_row, best_combo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Given a models_<scene> directory, scan nerf_ensemble_* runs, find the best "
            "checkpoint in each run by combined metric, then report the best run overall."
        )
    )
    parser.add_argument(
        "--model-root",
        required=True,
        type=Path,
        help="Scene-level root such as /cluster/.../models_flowers",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=-1,
        help="Optional maximum step to consider. -1 means all.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional JSON file to write all per-run summaries.",
    )
    args = parser.parse_args()

    model_root = args.model_root.resolve()
    run_dirs = sorted(p for p in model_root.glob("nerf_ensemble_*") if p.is_dir())
    if not run_dirs:
        raise RuntimeError(f"No nerf_ensemble_* directories found under {model_root}")

    summaries: List[Dict] = []
    for run_dir in run_dirs:
        eval_file = _find_eval_file(run_dir)
        best_row, best_combo = _best_row(eval_file, max_step=args.max_step)
        summaries.append(
            {
                "run_dir": str(run_dir),
                "eval_file": str(eval_file),
                "step": int(best_row["step"]),
                "avg_metric": best_combo,
                "psnr": float(best_row["psnr"]),
                "ssim": float(best_row["ssim"]),
                "lpips": float(best_row["lpips"]),
            }
        )

    summaries.sort(key=lambda r: r["avg_metric"])
    best = summaries[0]

    print(f"Best ensemble run under: {model_root}")
    print(f"run_dir: {best['run_dir']}")
    print(f"step: {best['step']}")
    print(f"avg metric: {best['avg_metric']:.6f}")
    print(f"psnr: {best['psnr']:.6f}")
    print(f"ssim: {best['ssim']:.6f}")
    print(f"lpips: {best['lpips']:.6f}")

    print("\nAll runs:")
    for i, row in enumerate(summaries, start=1):
        print(
            f"{i:2d}. {Path(row['run_dir']).name}: "
            f"avg={row['avg_metric']:.6f} | step={row['step']} | "
            f"psnr={row['psnr']:.6f} | ssim={row['ssim']:.6f} | lpips={row['lpips']:.6f}"
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
