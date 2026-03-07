"""Run a comparable schedule experiment suite that always includes baseline.

This script launches a sequence of `train.py` runs with matched settings:
1) baseline Muon (no power schedules)
2) fixed alternating p in {0,1}
3) entropy alternating p in {0,1}

Use this to keep "schedule vs baseline" comparisons consistent.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(shlex.quote(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def build_common_args(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        "train.py",
        "config/train_shakespeare_char.py",
        "--dataset=shakespeare_char",
        "--use_muon=True",
        f"--max_iters={args.max_iters}",
        f"--eval_interval={args.eval_interval}",
        f"--eval_iters={args.eval_iters}",
        f"--batch_size={args.batch_size}",
        f"--block_size={args.block_size}",
        f"--n_layer={args.n_layer}",
        f"--n_head={args.n_head}",
        f"--n_embd={args.n_embd}",
        "--compile=False",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        "--gradient_accumulation_steps=1",
        f"--wandb_log={str(args.wandb_log)}",
        f"--wandb_project={args.wandb_project}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduled-Muon suite with baseline.")
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--wandb-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="muon-schedule-lab")
    parser.add_argument("--name-prefix", type=str, default="suite")
    args = parser.parse_args()

    common = build_common_args(args)

    runs = [
        (
            f"{args.name_prefix}_baseline_muon",
            [
                "--enable_power_schedules=False",
                f"--out_dir=out_{args.name_prefix}_baseline_muon",
            ],
        ),
        (
            f"{args.name_prefix}_fixed_alt_p01_period20",
            [
                "--enable_power_schedules=True",
                "--power_backend=poly",
                "--power_schedule_type=fixed_alternating",
                "--power_p_low=0.0",
                "--power_p_high=1.0",
                "--power_alternation_period=20",
                "--power_log_interval=25",
                f"--out_dir=out_{args.name_prefix}_fixed_alt_p01_period20",
            ],
        ),
        (
            f"{args.name_prefix}_fixed_alt_p01_period60",
            [
                "--enable_power_schedules=True",
                "--power_backend=poly",
                "--power_schedule_type=fixed_alternating",
                "--power_p_low=0.0",
                "--power_p_high=1.0",
                "--power_alternation_period=60",
                "--power_log_interval=25",
                f"--out_dir=out_{args.name_prefix}_fixed_alt_p01_period60",
            ],
        ),
        (
            f"{args.name_prefix}_entropy_alt_p01",
            [
                "--enable_power_schedules=True",
                "--power_backend=poly",
                "--power_schedule_type=entropy_alternating",
                "--power_p_low=0.0",
                "--power_p_high=1.0",
                "--power_entropy_low=0.64",
                "--power_entropy_high=0.66",
                "--power_entropy_initial_mode=low",
                "--power_log_interval=25",
                f"--out_dir=out_{args.name_prefix}_entropy_alt_p01",
            ],
        ),
    ]

    for run_name, extra in runs:
        cmd = common + [f"--wandb_run_name={run_name}"] + extra
        run_cmd(cmd)

    print("Suite completed.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
