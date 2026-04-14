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


def make_plot(
    steps, curves, agg, carry_keys, base, results_dir,
    split="iid", metric="acc", log_y=False,
):
    fig, ax = plt.subplots(figsize=(12, 7))

    probs = {k: geometric_prob(k, base) for k in carry_keys}
    valid_probs = [p for p in probs.values() if p > 0]
    if valid_probs:
        cmap = mpl.colormaps["viridis"].reversed()
        norm = mpl.colors.LogNorm(vmin=min(valid_probs), vmax=max(valid_probs))
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        sm = None

    for k in carry_keys:
        vals = curves[k]
        if all(math.isnan(v) for v in vals):
            continue
        p = probs.get(k, 0.001)
        color = sm.to_rgba(p) if sm else None
        ax.plot(steps, vals, color=color, alpha=0.7, linewidth=0.8, label=f"k={k}")

    ax.plot(steps, agg, color="red", linewidth=3, label=f"Aggregate")

    if log_y:
        ax.set_yscale("log")

    ylabel = "Sequence Accuracy" if metric == "acc" else "CE Loss"
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(f"{split.upper()} {ylabel}")
    ax.set_title(f"{split.upper()} {ylabel} by Carry Length")

    if sm and valid_probs:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Carry length (colored by P(k) under uniform)")

    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()

    suffix = "_logy" if log_y else ""
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
        if iid_keys:
            steps, curves, agg = extract_curves(metrics, iid_keys, "iid", metric)
            make_plot(steps, curves, agg, iid_keys, base, results_dir, "iid", metric, log_y=False)
            if metric == "loss":
                make_plot(steps, curves, agg, iid_keys, base, results_dir, "iid", metric, log_y=True)

        if ood_keys:
            steps, curves, agg = extract_curves(metrics, ood_keys, "ood", metric)
            make_plot(steps, curves, agg, ood_keys, base, results_dir, "ood", metric, log_y=False)
            if metric == "loss":
                make_plot(steps, curves, agg, ood_keys, base, results_dir, "ood", metric, log_y=True)


if __name__ == "__main__":
    main()
