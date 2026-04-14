# Successor Quanta

Experiments testing whether transformers learn the successor function as a compact algorithm or as a bag of carry-length-specific heuristics (quanta).

## Task

Given an n-digit base-b number (0-padded, MSB first), predict its successor (mod b^n).

The carry length k — the number of trailing (b-1) digits — determines the difficulty.
Under uniform sampling, P(carry = k) = (1/b)^k * (1 - 1/b), a geometric distribution
that makes long carries exponentially rare in training data.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Run

```bash
python experiments/train_successor.py --config experiments/configs/L2_D128_base2_rope.yaml
```

Override any config field:

```bash
python experiments/train_successor.py --config experiments/configs/L2_D128_base2_rope.yaml seed=123 lr=0.001
```

## Experiments

Results are saved to `experiments/<run_name>/` with:
- `config.yaml` — resolved config
- `notes.md` — command and metadata
- `checkpoints/` — model checkpoints
- `plots/` — per-carry-length accuracy and loss curves
- `metrics.json` — full metrics history

## Evaluation

Two test sets:
- **IID**: uniformly sampled strings (same distribution as training)
- **OOD**: strings with forced long carry chains (k = 5, 10, 15, 20, 25, 30)

Per-carry-length accuracy and CE loss are tracked for both.
