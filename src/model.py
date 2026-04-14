import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, n_heads, T, head_dim)."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0,
                 pos_emb_type: str = "rope"):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pos_emb_type = pos_emb_type

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor,
                rope_cos: torch.Tensor | None = None,
                rope_sin: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_emb_type == "rope" and rope_cos is not None:
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.o_proj(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SuccessorTransformer(nn.Module):
    """Decoder-only transformer for the successor task.

    Input: sequence [x_1, ..., x_n, SEP, y_1, ..., y_n]
    Output: per-position logits over vocab (predictions used at output positions).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
        pos_emb_type: str = "rope",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pos_emb_type = pos_emb_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        if pos_emb_type == "learned":
            self.pos_emb = nn.Embedding(seq_len, d_model)
        elif pos_emb_type == "sinusoidal":
            self.register_buffer("pos_emb", _sinusoidal_embeddings(seq_len, d_model))
        # rope: no embedding table, applied inside attention

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, pos_emb_type)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.tok_emb(x)

        if self.pos_emb_type == "learned":
            pos = torch.arange(T, device=x.device)
            h = h + self.pos_emb(pos)
        elif self.pos_emb_type == "sinusoidal":
            h = h + self.pos_emb[:T]

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )

        rope_cos, rope_sin = None, None
        if self.pos_emb_type == "rope":
            rope_cos, rope_sin = _build_rope_cache(T, self.head_dim, x.device)

        for block in self.blocks:
            h = block(h, causal_mask, rope_cos, rope_sin)

        h = self.ln_f(h)
        return self.head(h)
