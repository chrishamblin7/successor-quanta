#!/usr/bin/env python3
"""Plot the analytical and empirical carry-length distribution for uniform sampling.

Usage:
    python scripts/plot_carry_dist.py [--n 1000] [--samples 100000] [--output carry_distribution.png]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def analytical_carry_prob(k: int, n: int, base: int) -> float:
    if k == n:
        return (1.0 / base) ** n
    return (1.0 / base) ** k * (1.0 - 1.0 / base)


def empirical_carry_dist(n: int, base: int, num_samples: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    digits = rng.integers(0, base, size=(num_samples, n))
    is_max = digits == (base - 1)

    counts = np.zeros(num_samples, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        still_trailing = counts == (n - 1 - i)
        counts += (is_max[:, i] & still_trailing).astype(np.int64)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--output", type=str, default="carry_distribution.png")
    args = parser.parse_args()

    bases = [2, 3, 10]
    fig, axes = plt.subplots(1, len(bases), figsize=(5 * len(bases), 5))
    if len(bases) == 1:
        axes = [axes]

    for ax, base in zip(axes, bases):
        max_k_plot = min(args.n, 40 if base == 2 else 15 if base == 3 else 8)
        ks = np.arange(0, max_k_plot + 1)
        analytical = np.array([analytical_carry_prob(k, args.n, base) for k in ks])

        empirical = empirical_carry_dist(args.n, base, args.samples)
        emp_counts = np.bincount(empirical, minlength=max_k_plot + 1)[:max_k_plot + 1]
        emp_probs = emp_counts / args.samples

        ax.semilogy(ks, analytical, "b-o", markersize=4, label="Analytical: $(1/b)^k (1-1/b)$")
        ax.semilogy(ks, emp_probs, "rx", markersize=6, label=f"Empirical (n={args.samples:,})")

        ax.set_xlabel("Carry length k")
        ax.set_ylabel("P(carry = k)")
        ax.set_title(f"Base {base}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Carry-length distribution under uniform sampling (n={args.n} digits)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
