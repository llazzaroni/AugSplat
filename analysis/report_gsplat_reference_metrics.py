#!/usr/bin/env python3
"""Report best combined metric per run and metrics at a reference run step."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STEP_RE = re.compile(r"^(?P<stage>[a-zA-Z0-9_]+)_step(?P<step>\d+)\.json$")


def _load_from_stats_dir(run_dir: Path, stage: str, max_step: int = -1) -> List[Tuple[int, Dict]]:
    stats_dir = run_dir / "stats"
    if not stats_dir.is_dir():
        return []

    rows: List[Tuple[int, Dict]] = []
    for p in sorted(stats_dir.glob("*.json")):
        m = STEP_RE.match(p.name)
        if not m or m.group("stage") != stage:
            continue
        step = int(m.group("step"))
        if max_step >= 0 and step > max_step:
            continue
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append((step, payload))
    rows.sort(key=lambda x: x[0])
    return rows


def _load_from_aggregate(run_dir: Path, step_interval: int, max_step: int = -1) -> List[Tuple[int, Dict]]:
    agg = run_dir / "gsplat_stats.json"
    if not agg.is_file():
        return []
    with agg.open("r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        return []

    rows: List[Tuple[int, Dict]] = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        step = i * step_interval
        if max_step >= 0 and step > max_step:
            continue
        rows.append((step, item))
    return rows


def _load_run(
    run_dir: Path, source: str, stage: str, step_interval: int, max_step: int = -1
) -> Tuple[List[Tuple[int, Dict]], str]:
    rows: List[Tuple[int, Dict]] = []
    used_source = source
    if source in ("auto", "stats_dir"):
        rows = _load_from_stats_dir(run_dir, stage=stage, max_step=max_step)
        if rows:
            used_source = "stats_dir"
    if not rows and source in ("auto", "aggregate"):
        rows = _load_from_aggregate(run_dir, step_interval=step_interval, max_step=max_step)
        if rows:
            used_source = "aggregate"
    return rows, used_source


def _combo(psnr: float, ssim: float, lpips: float) -> float:
    return (
        (10.0 ** (-float(psnr) / 10.0))
        * math.sqrt(max(0.0, 1.0 - float(ssim)))
        * float(lpips)
    ) ** (1.0 / 3.0)


def _metrics_at_step(rows: List[Tuple[int, Dict]], target_step: int) -> Tuple[Optional[Dict], Optional[int], bool]:
    exact = {step: payload for step, payload in rows}.get(target_step)
    if exact is not None:
        return exact, target_step, True
    if not rows:
        return None, None, False
    nearest_step, nearest_payload = min(rows, key=lambda x: abs(x[0] - target_step))
    return nearest_payload, nearest_step, False


def _extract_record(step: int, payload: Dict) -> Optional[Dict]:
    psnr = payload.get("psnr")
    ssim = payload.get("ssim")
    lpips = payload.get("lpips")
    if not isinstance(psnr, (int, float)):
        return None
    if not isinstance(ssim, (int, float)):
        return None
    if not isinstance(lpips, (int, float)):
        return None
    return {
        "step": int(step),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": float(lpips),
        "combo": _combo(float(psnr), float(ssim), float(lpips)),
    }


def _find_best(rows: List[Tuple[int, Dict]]) -> Optional[Dict]:
    best: Optional[Dict] = None
    for step, payload in rows:
        rec = _extract_record(step, payload)
        if rec is None:
            continue
        if best is None or rec["combo"] < best["combo"]:
            best = rec
    return best


def _format_record(label: str, rec: Dict) -> str:
    return (
        f"{label}: step={rec['step']} | combo={rec['combo']:.6f} | "
        f"psnr={rec['psnr']:.6f} | ssim={rec['ssim']:.6f} | lpips={rec['lpips']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For each run, report the best combined PSNR/SSIM/LPIPS score and metrics at that step. "
            "Also report metrics for all runs at the reference run's best step."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        action="append",
        type=Path,
        help="Run directory (repeat --run-dir for multiple runs).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label for each --run-dir (same order). Repeat as needed.",
    )
    parser.add_argument(
        "--reference-run-dir",
        required=True,
        type=Path,
        help="Special run directory whose best step is used as cross-run reference.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "stats_dir", "aggregate"],
        default="auto",
        help="Where to read stats from.",
    )
    parser.add_argument(
        "--stage",
        default="val",
        help="Stage prefix for stats_dir files (default: val).",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=100,
        help="Step interval used to infer steps from gsplat_stats.json.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=-1,
        help="Optional max step to include. -1 means all.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional JSON output file.",
    )
    args = parser.parse_args()

    run_dirs = [p.resolve() for p in args.run_dir]
    ref_run_dir = args.reference_run_dir.resolve()
    labels = args.label if args.label is not None else []
    if labels and len(labels) != len(run_dirs):
        raise ValueError(
            f"Number of --label ({len(labels)}) must match number of --run-dir ({len(run_dirs)})."
        )
    if not labels:
        labels = [p.name for p in run_dirs]

    runs = []
    for run_dir, label in zip(run_dirs, labels):
        rows, used_source = _load_run(
            run_dir,
            source=args.source,
            stage=args.stage,
            step_interval=args.step_interval,
            max_step=args.max_step,
        )
        if not rows:
            print(f"[WARN] {label}: no stats found in {run_dir}")
            continue
        best = _find_best(rows)
        if best is None:
            print(f"[WARN] {label}: no valid psnr/ssim/lpips rows in {run_dir}")
            continue
        runs.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "used_source": used_source,
                "rows": rows,
                "best": best,
            }
        )

    if not runs:
        raise RuntimeError("No valid runs found.")

    ref_run = None
    for run in runs:
        if Path(run["run_dir"]) == ref_run_dir:
            ref_run = run
            break
    if ref_run is None:
        raise RuntimeError(
            f"Reference run {ref_run_dir} is not among the valid --run-dir inputs."
        )

    print("Best point per run")
    for run in runs:
        print(_format_record(run["label"], run["best"]))

    ref_best = ref_run["best"]
    ref_step = int(ref_best["step"])
    print("")
    print(
        f"Reference step from {ref_run['label']}: "
        f"step={ref_step} (best combo for reference run)"
    )
    print(_format_record(ref_run["label"], ref_best))

    at_reference = []
    for run in runs:
        payload, chosen_step, exact = _metrics_at_step(run["rows"], ref_step)
        if payload is None or chosen_step is None:
            continue
        rec = _extract_record(chosen_step, payload)
        if rec is None:
            continue
        rec["exact_step_match"] = exact
        at_reference.append(
            {
                "label": run["label"],
                "run_dir": run["run_dir"],
                "used_source": run["used_source"],
                **rec,
            }
        )

    print("")
    print("All runs at reference step")
    for rec in at_reference:
        suffix = "" if rec["exact_step_match"] else " [nearest available step]"
        print(_format_record(rec["label"], rec) + suffix)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "reference_run_dir": str(ref_run_dir),
            "reference_label": ref_run["label"],
            "reference_best": ref_best,
            "best_per_run": [
                {
                    "label": run["label"],
                    "run_dir": run["run_dir"],
                    "used_source": run["used_source"],
                    **run["best"],
                }
                for run in runs
            ],
            "at_reference_step": at_reference,
        }
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
