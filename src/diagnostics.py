"""OOD diagnostic sequence visualizations.

Generates per-sequence images showing input digits, carry structure,
prediction errors, and per-token cross-entropy for incorrect predictions.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

mpl.use("Agg")

ROW_HEIGHT = 0.35
DIAG_K_STEP = 20
N_SAMPLES_PER_K = 3


def render_sequence_diagnostic(
    input_digits: np.ndarray,
    target_digits: np.ndarray,
    pred_digits: np.ndarray,
    token_ce: np.ndarray,
    carry_k: int,
    base: int,
) -> plt.Figure:
    """Render a 4-row diagnostic strip for one sequence.

    Row 1: input digits (viridis, normalized to [0, base-1])
    Row 2: carry structure (white=unchanged, red=trigger, black=carry chain)
    Row 3: prediction error (white=correct, red=incorrect)
    Row 4: per-token cross-entropy (viridis)
    """
    n = len(input_digits)

    row_input = input_digits.astype(np.float64) / max(base - 1, 1)

    structure = np.ones((n, 3), dtype=np.float64)  # white
    if carry_k > 0:
        structure[-carry_k:] = [0, 0, 0]  # black for carry chain
    if carry_k < n:
        structure[n - carry_k - 1] = [1, 0, 0]  # red for trigger

    errors = pred_digits != target_digits
    error_rgb = np.ones((n, 3), dtype=np.float64)
    error_rgb[errors] = [1, 0, 0]

    cmap = mpl.colormaps["viridis"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 4 * ROW_HEIGHT + 1.2),
                             gridspec_kw={"hspace": 0.5})

    row_labels = ["Input", "Structure", "Error", "CE Loss"]

    axes[0].imshow(row_input[np.newaxis, :], aspect="auto", cmap="viridis",
                   vmin=0, vmax=1, interpolation="nearest")

    axes[1].imshow(structure[np.newaxis, :, :], aspect="auto", interpolation="nearest")

    axes[2].imshow(error_rgb[np.newaxis, :, :], aspect="auto", interpolation="nearest")

    ce_max = max(token_ce.max(), 1e-6)
    axes[3].imshow(token_ce[np.newaxis, :], aspect="auto", cmap="viridis",
                   vmin=0, vmax=ce_max, interpolation="nearest")

    for ax, label in zip(axes, row_labels):
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)

    n_err = int(errors.sum())
    fig.suptitle(f"k={carry_k}  |  {n_err}/{n} tokens wrong  |  "
                 f"mean CE={token_ce.mean():.4f}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def generate_ood_diagnostics(
    model,
    data,
    device: str,
    run_dir: Path,
    step: int,
    base: int,
    n_samples: int = N_SAMPLES_PER_K,
    k_step: int = DIAG_K_STEP,
):
    """Generate diagnostic plots for a sample of OOD k values with errors."""
    diag_dir = run_dir / "plots" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    all_ks = sorted(data.ood_inputs.keys())
    if not all_ks:
        return

    sampled_ks = all_ks[::k_step]
    if all_ks[-1] not in sampled_ks:
        sampled_ks.append(all_ks[-1])

    model.eval()
    with torch.no_grad():
        for k in sampled_ks:
            inp = data.ood_inputs[k]
            tgt = data.ood_targets[k]
            N = len(inp)

            all_preds = []
            all_ce = []
            for start in range(0, N, 256):
                end = min(start + 256, N)
                xb = inp[start:end].to(device)
                tb = tgt[start:end].to(device)
                logits = model(xb)
                V = logits.shape[-1]

                ce = F.cross_entropy(
                    logits.reshape(-1, V), tb.reshape(-1), reduction="none"
                ).view(end - start, -1)
                all_preds.append(logits.argmax(dim=-1).cpu())
                all_ce.append(ce.cpu())

            preds = torch.cat(all_preds).numpy()
            ce_all = torch.cat(all_ce).numpy()
            inp_np = inp.numpy()
            tgt_np = tgt.numpy()

            seq_correct = (preds == tgt_np).all(axis=1)
            wrong_idxs = np.where(~seq_correct)[0]

            if len(wrong_idxs) == 0:
                continue

            chosen = wrong_idxs[:n_samples]

            figs = []
            for idx in chosen:
                fig = render_sequence_diagnostic(
                    inp_np[idx], tgt_np[idx], preds[idx],
                    ce_all[idx], k, base,
                )
                figs.append(fig)

            if len(figs) == 1:
                figs[0].savefig(diag_dir / f"step{step}_k{k}.png", dpi=150)
                plt.close(figs[0])
            else:
                combined_h = sum(f.get_size_inches()[1] for f in figs) + 0.3 * (len(figs) - 1)
                combined_w = figs[0].get_size_inches()[0]
                combined, combined_axes = plt.subplots(
                    len(figs), 1, figsize=(combined_w, combined_h),
                    gridspec_kw={"hspace": 0.6},
                )
                for i, fig_single in enumerate(figs):
                    fig_single.savefig(
                        diag_dir / f"step{step}_k{k}_sample{i}.png", dpi=150
                    )
                    plt.close(fig_single)

    model.train()
