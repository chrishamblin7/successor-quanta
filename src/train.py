import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from .config import ExperimentConfig
from .data import SuccessorData
from .model import SuccessorTransformer
from .utils import set_seed, sync_to_gcs


EVAL_BATCH_SIZE = 256


def evaluate_set(
    model, inputs: torch.Tensor, targets: torch.Tensor,
    carries: np.ndarray, device: str,
) -> tuple[dict, float, float]:
    """Evaluate on a test set, returning per-carry-length and aggregate metrics.

    Accuracy = fraction of samples where ALL output positions are correct.
    """
    model.eval()
    all_correct = []
    all_losses = []
    N = len(inputs)

    with torch.no_grad():
        for start in range(0, N, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, N)
            xb = inputs[start:end].to(device)
            tb = targets[start:end].to(device)

            logits = model(xb)  # (B, n, V)
            B_chunk, T, V = logits.shape

            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), tb.reshape(-1), reduction="none"
            ).view(B_chunk, T)
            mean_loss_per_sample = loss_per_token.mean(dim=1)

            preds = logits.argmax(dim=-1)
            correct_per_sample = (preds == tb).all(dim=1)

            all_losses.append(mean_loss_per_sample.cpu())
            all_correct.append(correct_per_sample.cpu())

    model.train()

    all_losses = torch.cat(all_losses).numpy()
    all_correct = torch.cat(all_correct).numpy()

    unique_k = sorted(set(carries.tolist()))
    per_carry = {}
    for k in unique_k:
        mask = carries == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        per_carry[k] = {
            "acc": float(all_correct[mask].mean()),
            "loss": float(all_losses[mask].mean()),
            "n": int(n_k),
        }

    agg_acc = float(all_correct.mean())
    agg_loss = float(all_losses.mean())
    return per_carry, agg_acc, agg_loss


def train(cfg: ExperimentConfig):
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    run_name = cfg.auto_run_name()
    run_dir = Path("experiments") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    cfg.save(str(run_dir / "config.yaml"))

    script_src = Path(__file__).resolve().parent.parent / "experiments" / "train_successor.py"
    if script_src.exists():
        shutil.copy2(script_src, run_dir / "train_script.py")

    notes = (
        f"# {run_name}\n\n"
        f"Command: python experiments/train_successor.py --config <config>\n\n"
        f"Config: {json.dumps({k: v for k, v in vars(cfg).items()}, default=str, indent=2)}\n"
    )
    (run_dir / "notes.md").write_text(notes)

    wandb_kwargs = dict(project=cfg.wandb_project, name=run_name, config={
        k: v for k, v in vars(cfg).items() if not k.startswith("_")
    })
    if cfg.resume:
        wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)

    data = SuccessorData(cfg)

    model = SuccessorTransformer(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pos_emb_type=cfg.pos_emb_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {run_name} ===")
    print(f"Params  : {n_params:,}")
    print(f"Seq len : {cfg.seq_len}")
    print(f"Base    : {cfg.base}")
    print(f"Sampler : {cfg.sampler_type}")
    print(f"Pos emb : {cfg.pos_emb_type}")
    print(f"Device  : {device}")
    wandb.log({"n_params": n_params}, step=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    rng = np.random.default_rng(cfg.seed + 1)
    metrics_log: list[dict] = []
    start_step = 1

    if cfg.resume:
        start_step, metrics_log = _try_resume(model, optimizer, run_dir)

    early_eval_steps = {0, 1, 10, 50, 100}

    def _run_eval(step_num, train_loss_val):
        iid_per_carry, iid_acc, iid_loss = evaluate_set(
            model, data.iid_inputs, data.iid_targets,
            data.iid_carries, device,
        )

        ood_per_carry = {}
        ood_total_correct = 0
        ood_total_loss = 0.0
        ood_total_n = 0
        for k, inp in data.ood_inputs.items():
            pk, _, _ = evaluate_set(
                model, inp, data.ood_targets[k],
                data.ood_carries[k], device,
            )
            if k in pk:
                ood_per_carry[k] = pk[k]
                ood_total_correct += pk[k]["acc"] * pk[k]["n"]
                ood_total_loss += pk[k]["loss"] * pk[k]["n"]
                ood_total_n += pk[k]["n"]

        ood_acc = ood_total_correct / max(ood_total_n, 1)
        ood_loss = ood_total_loss / max(ood_total_n, 1)

        log_dict = {
            "iid/acc": iid_acc, "iid/loss": iid_loss,
            "ood/acc": ood_acc, "ood/loss": ood_loss,
        }
        for k, v in iid_per_carry.items():
            log_dict[f"iid_acc/carry_{k}"] = v["acc"]
            log_dict[f"iid_loss/carry_{k}"] = v["loss"]
        for k, v in ood_per_carry.items():
            log_dict[f"ood_acc/carry_{k}"] = v["acc"]
            log_dict[f"ood_loss/carry_{k}"] = v["loss"]
        wandb.log(log_dict, step=step_num)

        entry = {
            "step": step_num,
            "train_loss": train_loss_val,
            "iid": {
                "agg_acc": iid_acc, "agg_loss": iid_loss,
                "per_carry": {str(k): v for k, v in iid_per_carry.items()},
            },
            "ood": {
                "agg_acc": ood_acc, "agg_loss": ood_loss,
                "per_carry": {str(k): v for k, v in ood_per_carry.items()},
            },
        }
        metrics_log.append(entry)

        elapsed = time.time() - t0
        print(
            f"[{elapsed:7.1f}s] step {step_num:>6d}  "
            f"train_loss={train_loss_val:.4f}  "
            f"iid_acc={iid_acc:.4f}  ood_acc={ood_acc:.4f}"
        )

        _save_metrics(metrics_log, run_dir)
        _regenerate_plots(run_dir)

    t0 = time.time()

    if start_step <= 1 and 0 in early_eval_steps:
        _run_eval(0, float("nan"))

    try:
        for step in range(start_step, cfg.num_steps + 1):
            inputs, targets, _ = data.sample_batch(rng, cfg.batch_size, device)

            logits = model(inputs)  # (B, n, V)
            V = logits.shape[-1]
            loss = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                wandb.log({"train/loss": loss.item()}, step=step)

            should_eval = (
                step % cfg.eval_every == 0
                or step in early_eval_steps
            )
            if should_eval:
                _run_eval(step, loss.item())

            if step % cfg.checkpoint_every == 0:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    run_dir / "checkpoints" / f"step_{step}.pt",
                )
                _save_metrics(metrics_log, run_dir)
                sync_to_gcs(str(run_dir), cfg.gcs_bucket + run_name)

    except KeyboardInterrupt:
        print("\nInterrupted -- saving checkpoint...")

    _save_metrics(metrics_log, run_dir)
    torch.save(model.state_dict(), run_dir / "checkpoints" / "model_final.pt")
    sync_to_gcs(str(run_dir), cfg.gcs_bucket + run_name)
    wandb.finish()
    print(f"Done. Results in {run_dir}")


def _try_resume(model, optimizer, run_dir):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        print("[resume] No checkpoints directory found, starting from scratch")
        return 1, []

    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        print("[resume] No checkpoint files found, starting from scratch")
        return 1, []

    latest = ckpts[-1]
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    print(f"[resume] Restored from {latest.name} (step {step})")

    metrics_log = []
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_log = json.load(f)
        metrics_log = [e for e in metrics_log if e["step"] <= step]
        print(f"[resume] Loaded {len(metrics_log)} metric entries up to step {step}")

    return step + 1, metrics_log


def _save_metrics(metrics_log, run_dir):
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)


def _regenerate_plots(run_dir: Path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "plot_curves.py"
    if not script.exists():
        return
    try:
        subprocess.Popen(
            [sys.executable, str(script), "--results-dir", str(run_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
