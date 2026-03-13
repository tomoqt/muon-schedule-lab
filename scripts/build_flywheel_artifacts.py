from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def load_ckpt(run_dir: str) -> dict:
    path = REPO_ROOT / run_dir / "ckpt.pt"
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    return {
        "run_dir": run_dir,
        "path": path,
        "iter_num": int(ckpt["iter_num"]),
        # In this training code, when always_save_checkpoint=True the checkpoint
        # field named "best_val_loss" is overwritten at every eval. Treat it as
        # the final recorded validation loss for the run horizon.
        "final_val": float(ckpt["best_val_loss"]),
        "config": cfg,
        "run_name": cfg.get("wandb_run_name", run_dir),
    }


def save_fig(fig: plt.Figure, rel_path: str) -> None:
    out = REPO_ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt_scale(scale: float) -> str:
    return f"{scale:.1f}".replace(".", "p")


def build_char_fixed_alpha_plot() -> None:
    run_dirs = [
        "out_long2000_fixed_alpha_m0p5_p_2p0",
        "out_long2000_fixed_alpha_0p0_p_1p0",
        "out_long2000_fixed_alpha_0p25_p_0p5",
        "out_long2000_fixed_alpha_0p5_p_0p0",
        "out_long2000_fixed_alpha_0p75_p_m0p5",
        "out_long2000_fixed_alpha_1p0_p_m1p0",
        "out_long2000_fixed_alpha_refine_0p49_p_0p02",
        "out_long2000_fixed_alpha_refine_0p48_p_0p04",
        "out_long2000_fixed_alpha_refine_0p45_p_0p1",
        "out_long2000_fixed_alpha_refine_0p4_p_0p2",
        "out_long2000_fixed_alpha_refine_0p35_p_0p3",
        "out_long2000_fixed_alpha_refine_0p3_p_0p4",
    ]
    runs = []
    for run_dir in run_dirs:
        row = load_ckpt(run_dir)
        cfg = row["config"]
        p = float(cfg["power_p_low"])
        alpha = 0.5 * (1.0 - p)
        row["alpha"] = alpha
        row["p"] = p
        runs.append(row)
    runs.sort(key=lambda r: r["alpha"])

    baseline = load_ckpt("out_long2000_baseline_muon")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [r["alpha"] for r in runs],
        [r["final_val"] for r in runs],
        marker="o",
        linewidth=1.8,
        color="#0f766e",
        label="fixed alpha",
    )
    ax.axhline(
        baseline["final_val"],
        linestyle="--",
        linewidth=1.4,
        color="#7c2d12",
        label=f"baseline Muon ({baseline['final_val']:.4f})",
    )
    best = min(runs, key=lambda r: r["final_val"])
    ax.scatter([best["alpha"]], [best["final_val"]], s=90, color="#be123c", zorder=5)
    ax.annotate(
        f"best alpha={best['alpha']:.2f}\nfinal val={best['final_val']:.4f}",
        (best["alpha"], best["final_val"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("alpha")
    ax.set_ylabel("final val loss")
    ax.set_title("Shakespeare-char fixed alpha sweep (2000 steps)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    save_fig(fig, "assets/char_fixed_alpha_full.png")


def build_char_alternating_plot() -> None:
    records = [
        ("fixed_alt_period20", 300, load_ckpt("out_exp_fixed_alt_period20_long300")),
        ("fixed_alt_period60", 300, load_ckpt("out_exp_fixed_alt_period60_long300")),
        ("entropy_alt", 300, load_ckpt("out_exp_entropy_alt_long300")),
        ("baseline_muon", 500, load_ckpt("out_exp_baseline_muon_long500")),
        ("fixed_alt_period20", 500, load_ckpt("out_exp_fixed_alt_period20_long500")),
        ("fixed_alt_period60", 500, load_ckpt("out_exp_fixed_alt_period60_long500")),
        ("entropy_alt", 500, load_ckpt("out_exp_entropy_alt_long500")),
        ("baseline_muon", 2000, load_ckpt("out_long2000_baseline_muon")),
        ("fixed_alpha05", 2000, load_ckpt("out_long2000_alpha05_p0_constant")),
        ("fixed_alt_period20", 2000, load_ckpt("out_long2000_fixed_alt_p01_period20")),
        ("fixed_alt_period60", 2000, load_ckpt("out_long2000_fixed_alt_p01_period60")),
        ("entropy_alt", 2000, load_ckpt("out_long2000_entropy_alt_p01")),
    ]
    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for method, steps, row in records:
        by_method[method].append((steps, row["final_val"]))

    colors = {
        "baseline_muon": "#111827",
        "fixed_alpha05": "#0f766e",
        "fixed_alt_period20": "#2563eb",
        "fixed_alt_period60": "#9333ea",
        "entropy_alt": "#dc2626",
    }
    labels = {
        "baseline_muon": "baseline Muon",
        "fixed_alpha05": "fixed alpha=0.5",
        "fixed_alt_period20": "fixed alternating period=20",
        "fixed_alt_period60": "fixed alternating period=60",
        "entropy_alt": "entropy alternating",
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method, points in by_method.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[method], label=labels[method])
    ax.set_xlabel("training steps")
    ax.set_ylabel("best val loss")
    ax.set_title("Shakespeare-char alternating schedule checks across horizons")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, "assets/char_alternating_horizons.png")


def _load_fineweb_v2_means(methods: list[str]) -> dict[str, list[tuple[float, float, float]]]:
    seeds = [1337, 1338, 1339]
    scales = [0.5, 1.0, 1.5, 2.0]
    result: dict[str, list[tuple[float, float, float]]] = {}
    for method in methods:
        rows = []
        for scale in scales:
            values = []
            for seed in seeds:
                run_dir = f"out_fineweb_small_lr_sweep_poly_v2_{method}_s{fmt_scale(scale)}_seed{seed}"
                values.append(load_ckpt(run_dir)["final_val"])
            mu, sigma = mean_std(values)
            rows.append((scale, mu, sigma))
        result[method] = rows
    return result


def build_fineweb_baseline_fixed_plot() -> None:
    data = _load_fineweb_v2_means(["baseline_muon", "fixed_alpha05"])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = {"baseline_muon": "#111827", "fixed_alpha05": "#0f766e"}
    labels = {"baseline_muon": "baseline Muon", "fixed_alpha05": "fixed alpha=0.5"}
    for method in ["baseline_muon", "fixed_alpha05"]:
        xs = [x for x, _, _ in data[method]]
        ys = [y for _, y, _ in data[method]]
        es = [e for _, _, e in data[method]]
        ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1.8, capsize=3, color=colors[method], label=labels[method])
    ax.set_xlabel("LR scale")
    ax.set_ylabel("final val loss (mean ± std)")
    ax.set_title("FineWeb sanity check: baseline vs fixed alpha=0.5")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    save_fig(fig, "assets/fineweb_baseline_fixed_alpha_meanstd.png")


def build_fineweb_entropy_plot() -> None:
    data = _load_fineweb_v2_means(["baseline_muon", "law_powerg2", "law_powerg2_osc"])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = {
        "baseline_muon": "#111827",
        "law_powerg2": "#b45309",
        "law_powerg2_osc": "#7c3aed",
    }
    labels = {
        "baseline_muon": "baseline Muon",
        "law_powerg2": "entropy law power(g=2)",
        "law_powerg2_osc": "entropy law power(g=2) + oscillation",
    }
    for method in ["baseline_muon", "law_powerg2", "law_powerg2_osc"]:
        xs = [x for x, _, _ in data[method]]
        ys = [y for _, y, _ in data[method]]
        es = [e for _, _, e in data[method]]
        ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1.8, capsize=3, color=colors[method], label=labels[method])
    ax.set_xlabel("LR scale")
    ax.set_ylabel("final val loss (mean ± std)")
    ax.set_title("FineWeb entropy-law sweeps under polynomial backend")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    save_fig(fig, "assets/fineweb_entropy_law_meanstd.png")


def build_fineweb_long5000_plot() -> None:
    rows = []
    csv_path = OUTPUTS_DIR / "flywheel_long5000_lr_micro_20260308" / "long5000_lr_micro_results.csv"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "method": row["method"],
                    "scale": float(row["scale"]),
                    "best_val": float(row["best_val"]),
                    "final_val": float(row["final_val"]),
                }
            )

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    for vals in by_method.values():
        vals.sort(key=lambda r: r["scale"])

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = {"baseline_muon": "#111827", "law_powerg2_osc": "#7c3aed"}
    labels = {"baseline_muon": "baseline Muon", "law_powerg2_osc": "law power(g=2) + oscillation"}
    for method, vals in by_method.items():
        xs = [r["scale"] for r in vals]
        ys = [r["best_val"] for r in vals]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[method], label=labels[method])
    ax.set_xlabel("LR scale")
    ax.set_ylabel("best val loss")
    ax.set_title("FineWeb 5000-step matched-LR benchmark")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    save_fig(fig, "assets/fineweb_long5000_lr_micro_val_vs_scale.png")


def build_fineweb_gating_plot() -> None:
    run_map = {
        ("baseline_muon", 500): "out_flywheel_fineweb_gate500_baseline_muon_s2p0_seed1337",
        ("law_powerg2_osc", 500): "out_flywheel_fineweb_gate500_law_powerg2_osc_s2p0_seed1337",
        ("baseline_muon", 2000): "out_flywheel_fineweb_gate2000_baseline_muon_s2p0_seed1337",
        ("law_powerg2_osc", 2000): "out_flywheel_fineweb_gate2000_law_powerg2_osc_s2p0_seed1337",
    }
    rows = []
    missing = []
    for key, run_dir in run_map.items():
        path = REPO_ROOT / run_dir / "ckpt.pt"
        if not path.exists():
            missing.append(run_dir)
            continue
        method, steps = key
        row = load_ckpt(run_dir)
        rows.append({"method": method, "steps": steps, "final_val": row["final_val"]})
    if missing:
        print(f"skipping fineweb gating plot; missing checkpoints: {', '.join(missing)}")
        return

    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append((row["steps"], row["final_val"]))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {"baseline_muon": "#111827", "law_powerg2_osc": "#7c3aed"}
    labels = {"baseline_muon": "baseline Muon", "law_powerg2_osc": "law power(g=2) + oscillation"}
    for method, points in by_method.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[method], label=labels[method])
    ax.set_xlabel("training steps")
    ax.set_ylabel("final val loss")
    ax.set_title("FineWeb gating rerun at LR scale 2.0")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    save_fig(fig, "assets/fineweb_gating_500_2000.png")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    build_char_fixed_alpha_plot()
    build_char_alternating_plot()
    build_fineweb_baseline_fixed_plot()
    build_fineweb_entropy_plot()
    build_fineweb_long5000_plot()
    build_fineweb_gating_plot()


if __name__ == "__main__":
    main()
