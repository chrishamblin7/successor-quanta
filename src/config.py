from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    # --- Task ---
    n_positions: int = 1000
    base: int = 2
    sampler_type: str = "uniform"  # "uniform" or "power_law"
    carry_beta: float = 1.5       # exponent for power_law sampler: P(k) ∝ k^{-β}

    # --- Model ---
    n_layers: int = 2
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    pos_emb_type: str = "rope"  # "rope", "learned", or "sinusoidal"

    # --- Training ---
    lr: float = 3e-4
    weight_decay: float = 0.0
    num_steps: int = 100_000
    batch_size: int = 64
    eval_every: int = 1000
    checkpoint_every: int = 10_000

    # --- Eval ---
    iid_test_size: int = 4096
    ood_test_carries: list = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    ood_samples_per_carry: int = 512

    # --- Infrastructure ---
    seed: int = 42
    run_name: str = ""
    gcs_bucket: str = "gs://cloud/misc/chris/successor-quanta/"
    device: str = "cuda"
    wandb_project: str = "successor-quanta"
    resume: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @property
    def vocab_size(self) -> int:
        return self.base + 1  # digits 0..base-1, plus SEP

    @property
    def seq_len(self) -> int:
        return 2 * self.n_positions + 1  # input + SEP + output

    @property
    def sep_token(self) -> int:
        return self.base  # last token id

    def auto_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        parts = [
            f"L{self.n_layers}",
            f"D{self.d_model}",
            f"base{self.base}",
            self.pos_emb_type,
        ]
        if self.sampler_type != "uniform":
            parts.append(self.sampler_type)
        if self.weight_decay > 0:
            parts.append(f"wd{self.weight_decay}")
        parts.append(f"s{self.seed}")
        return "_".join(parts)
