import numpy as np
import torch


def compute_successor(digits: np.ndarray, base: int) -> tuple[np.ndarray, int]:
    """Compute successor of an n-digit base-b number (MSB first).

    Returns (successor_digits, carry_length).
    carry_length = number of trailing (base-1) digits that get reset to 0.
    """
    n = len(digits)
    out = digits.copy()
    carry = 0
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            val = out[i] + 1
        else:
            val = out[i] + carry
        if val >= base:
            out[i] = 0
            carry = 1
        else:
            out[i] = val
            carry = 0
            break
    carry_length = 0
    for i in range(n - 1, -1, -1):
        if digits[i] == base - 1:
            carry_length += 1
        else:
            break
    return out, carry_length


def compute_successor_batch(digits: np.ndarray, base: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized successor for a batch of n-digit numbers. Shape: (B, n)."""
    B, n = digits.shape
    out = digits.copy()

    is_max = digits == (base - 1)
    trailing_max = np.zeros(B, dtype=np.int64)
    carry_active = np.ones(B, dtype=bool)

    for i in range(n - 1, -1, -1):
        if i == n - 1:
            new_val = out[:, i] + 1
        else:
            new_val = out[:, i] + carry_active.astype(np.int64)

        overflow = new_val >= base
        out[:, i] = np.where(overflow, 0, np.where(carry_active, new_val, out[:, i]))
        carry_active = carry_active & overflow

    for i in range(n - 1, -1, -1):
        mask = is_max[:, i]
        trailing_max += mask.astype(np.int64) * (trailing_max == (n - 1 - i))

    return out, trailing_max


def _count_trailing_max(digits: np.ndarray, base: int) -> np.ndarray:
    """Count trailing (base-1) digits for a batch. Shape: (B, n) -> (B,)."""
    B, n = digits.shape
    is_max = (digits == base - 1)
    counts = np.zeros(B, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        still_trailing = (counts == (n - 1 - i))
        counts += (is_max[:, i] & still_trailing).astype(np.int64)
    return counts


def sample_uniform_batch(
    rng: np.random.Generator, batch_size: int, n_positions: int, base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample uniform random n-digit base-b strings and compute successors.

    Returns (input_digits, output_digits, carry_lengths), all shape (B, n) or (B,).
    """
    digits = rng.integers(0, base, size=(batch_size, n_positions))
    out = digits.copy()
    carry_active = np.ones(batch_size, dtype=bool)

    for i in range(n_positions - 1, -1, -1):
        increment = carry_active.astype(np.int64)
        new_val = out[:, i] + increment
        overflow = new_val >= base
        out[:, i] = np.where(carry_active, new_val % base, out[:, i])
        carry_active = carry_active & overflow

    carry_lengths = _count_trailing_max(digits, base)
    return digits, out, carry_lengths


def sample_powerlaw_batch(
    rng: np.random.Generator, batch_size: int, n_positions: int, base: int, beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample strings with power-law distributed carry lengths.

    Draw k from P(k) ∝ k^{-β} for k=1..n_positions, plus k=0 gets weight 1.
    Then construct a string with exactly k trailing (base-1) digits.
    """
    max_k = n_positions
    ks = np.arange(0, max_k + 1)
    weights = np.ones(max_k + 1, dtype=np.float64)
    weights[1:] = ks[1:].astype(np.float64) ** (-beta)
    weights /= weights.sum()

    sampled_k = rng.choice(ks, size=batch_size, p=weights)

    digits = rng.integers(0, base, size=(batch_size, n_positions))

    for i in range(batch_size):
        k = sampled_k[i]
        if k > 0:
            digits[i, -k:] = base - 1
            if k < n_positions:
                if digits[i, n_positions - k - 1] == base - 1:
                    digits[i, n_positions - k - 1] = rng.integers(0, base - 1)

    out = digits.copy()
    carry_active = np.ones(batch_size, dtype=bool)
    for i in range(n_positions - 1, -1, -1):
        increment = carry_active.astype(np.int64)
        new_val = out[:, i] + increment
        overflow = new_val >= base
        out[:, i] = np.where(carry_active, new_val % base, out[:, i])
        carry_active = carry_active & overflow

    return digits, out, sampled_k


def encode_batch(
    input_digits: np.ndarray, output_digits: np.ndarray, sep_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode (input, output) pairs into token sequences and targets.

    Sequence: [x_1, ..., x_n, SEP, y_1, ..., y_n]
    Targets:  loss on positions n+1 .. 2n (the output digits), ignore rest.
    Returns (tokens, targets) where targets[i] = -100 for ignored positions.
    """
    B, n = input_digits.shape
    seq_len = 2 * n + 1

    tokens = np.zeros((B, seq_len), dtype=np.int64)
    tokens[:, :n] = input_digits
    tokens[:, n] = sep_token
    tokens[:, n + 1:] = output_digits

    targets = np.full((B, seq_len), -100, dtype=np.int64)
    targets[:, n + 1:] = output_digits

    return torch.from_numpy(tokens), torch.from_numpy(targets)


class SuccessorData:
    """Online data generator for the successor task with IID and OOD test sets."""

    def __init__(self, cfg):
        self.n_positions = cfg.n_positions
        self.base = cfg.base
        self.sep_token = cfg.sep_token
        self.sampler_type = cfg.sampler_type
        self.carry_beta = cfg.carry_beta

        test_rng = np.random.default_rng(cfg.seed + 9999)
        self._build_iid_test(test_rng, cfg.iid_test_size)
        self._build_ood_test(test_rng, cfg.ood_test_carries, cfg.ood_samples_per_carry)

    def _build_iid_test(self, rng: np.random.Generator, n: int):
        inp, out, carries = sample_uniform_batch(rng, n, self.n_positions, self.base)
        tokens, targets = encode_batch(inp, out, self.sep_token)
        self.iid_tokens = tokens
        self.iid_targets = targets
        self.iid_carries = carries

    def _build_ood_test(self, rng: np.random.Generator, carry_ks: list[int], samples_per: int):
        self.ood_tokens = {}
        self.ood_targets = {}
        self.ood_carries = {}

        for k in carry_ks:
            if k > self.n_positions:
                continue
            digits = rng.integers(0, self.base, size=(samples_per, self.n_positions))
            digits[:, -k:] = self.base - 1
            if k < self.n_positions:
                for i in range(samples_per):
                    if digits[i, self.n_positions - k - 1] == self.base - 1:
                        digits[i, self.n_positions - k - 1] = rng.integers(0, self.base - 1)

            out = digits.copy()
            carry_active = np.ones(samples_per, dtype=bool)
            for j in range(self.n_positions - 1, -1, -1):
                increment = carry_active.astype(np.int64)
                new_val = out[:, j] + increment
                overflow = new_val >= self.base
                out[:, j] = np.where(carry_active, new_val % self.base, out[:, j])
                carry_active = carry_active & overflow

            tokens, targets = encode_batch(digits, out, self.sep_token)
            self.ood_tokens[k] = tokens
            self.ood_targets[k] = targets
            self.ood_carries[k] = np.full(samples_per, k, dtype=np.int64)

    def sample_batch(
        self, rng: np.random.Generator, batch_size: int, device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        if self.sampler_type == "power_law":
            inp, out, carries = sample_powerlaw_batch(
                rng, batch_size, self.n_positions, self.base, self.carry_beta,
            )
        else:
            inp, out, carries = sample_uniform_batch(
                rng, batch_size, self.n_positions, self.base,
            )
        tokens, targets = encode_batch(inp, out, self.sep_token)
        return tokens.to(device), targets.to(device), carries
