#!/usr/bin/env python3
"""Plot per-carry-length test accuracy and loss curves.

Usage:
    python scripts/plot_curves.py --results-dir experiments/<run_name>
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_run(results_dir: Path):
    with open(results_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)
    return cfg, metrics


def geometric_prob(k: int, base: int) -> float:
    return (1.0 / base) ** k * (1.0 - 1.0 / base)


def extract_curves(metrics, carry_keys, split="iid", metric="acc"):
    steps = []
    curves = {k: [] for k in carry_keys}
    agg = []

    for entry in metrics:
        if split not in entry:
            continue
        steps.append(entry["step"])
        agg.append(entry[split][f"agg_{metric}"])
        pc = entry[split]["per_carry"]
        for k in carry_keys:
            sk = str(k)
            if sk in pc:
                curves[k].append(pc[sk][metric])
            else:
                curves[k].append(float("nan"))

    return np.array(steps), curves, np.array(agg)


MAX_PLOT_K = 200


def make_plot(
    steps, curves, agg, carry_keys, base, results_dir,
    split="iid", metric="acc", log_x=False, log_y=False,
):
    plot_keys = [k for k in carry_keys if k <= MAX_PLOT_K]
    if not plot_keys:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    probs = {k: geometric_prob(k, base) for k in plot_keys}
    valid_probs = [p for p in probs.values() if p > 0]
    if valid_probs:
        cmap = mpl.colormaps["viridis"].reversed()
        norm = mpl.colors.LogNorm(vmin=min(valid_probs), vmax=max(valid_probs))
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        sm = None

    for k in plot_keys:
        vals = curves[k]
        if all(math.isnan(v) for v in vals):
            continue
        p = probs.get(k, 0.001)
        color = sm.to_rgba(p) if sm else None
        ax.plot(steps, vals, color=color, alpha=0.7, linewidth=0.8)

    ax.plot(steps, agg, color="red", linewidth=3, label="Aggregate")
    ax.legend(loc="best", fontsize=9)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ylabel = "Sequence Accuracy" if metric == "acc" else "CE Loss"
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(f"{split.upper()} {ylabel}")
    ax.set_title(f"{split.upper()} {ylabel} by Carry Length (base {base})")

    if sm and valid_probs:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("P(k) under uniform sampling")

        k_ticks = [k for k in plot_keys if k > 0]
        if len(k_ticks) > 12:
            step = max(1, len(k_ticks) // 10)
            k_ticks = k_ticks[::step]
            if plot_keys[-1] not in k_ticks:
                k_ticks.append(plot_keys[-1])
        p_ticks = [geometric_prob(k, base) for k in k_ticks]

        cbar_ax2 = cbar.ax.twinx()
        cbar_ax2.set_ylim(cbar.ax.get_ylim())
        cbar_ax2.set_yscale("log")
        cbar_ax2.set_yticks(p_ticks)
        cbar_ax2.set_yticklabels([f"k={k}" for k in k_ticks], fontsize=7)
        cbar_ax2.tick_params(length=3, pad=2)

    fig.tight_layout()

    parts = []
    if log_x:
        parts.append("logx")
    if log_y:
        parts.append("logy")
    suffix = ("_" + "_".join(parts)) if parts else ""
    out = results_dir / "plots" / f"{split}_{metric}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cfg, metrics = load_run(results_dir)
    base = cfg.get("base", 2)

    if not metrics:
        return

    iid_keys = set()
    ood_keys = set()
    for entry in metrics:
        if "iid" in entry:
            iid_keys.update(int(k) for k in entry["iid"]["per_carry"].keys())
        if "ood" in entry:
            ood_keys.update(int(k) for k in entry["ood"]["per_carry"].keys())

    iid_keys = sorted(iid_keys)
    ood_keys = sorted(ood_keys)

    for metric in ("acc", "loss"):
        for split, keys in [("iid", iid_keys), ("ood", ood_keys)]:
            if not keys:
                continue
            steps, curves, agg = extract_curves(metrics, keys, split, metric)
            for log_x in (False, True):
                make_plot(steps, curves, agg, keys, base, results_dir, split, metric, log_x=log_x, log_y=False)
                if metric == "loss":
                    make_plot(steps, curves, agg, keys, base, results_dir, split, metric, log_x=log_x, log_y=True)


if __name__ == "__main__":
    main()
