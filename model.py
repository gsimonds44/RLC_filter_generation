"""
A minimal ten-layer Transformer (each layer = self-attention + MLP block),
implemented from scratch in PyTorch.

Run this file directly to see a forward pass on random token ids.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ----------------------------------------------------------------------
# Multi-Head Self-Attention
# ----------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: optional (seq_len, seq_len) boolean/float mask, True/1 = masked out
        """
        B, T, C = x.shape

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split into heads: (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.bool(), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads

        return self.out_proj(out)


# ----------------------------------------------------------------------
# Position-wise Feed-Forward (MLP) block
# ----------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------
# A single Transformer block: Attention + MLP, each with residual + LayerNorm
# (pre-norm formulation, as used in GPT-style models)
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)   # residual around attention
        x = x + self.mlp(self.ln2(x))          # residual around MLP
        return x


# ----------------------------------------------------------------------
# Full ten-layer Transformer model
# ----------------------------------------------------------------------
class TenLayerTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 224,
        n_heads: int = 8,
        d_ff: int = 896,
        max_seq_len: int = 384,
        dropout: float = 0.0,
        n_layers: int = 10,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # n_layers transformer layers (attention + MLP each), 10 by default
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.max_seq_len = max_seq_len

        # Gradient checkpointing: recompute each layer's activations during
        # backward instead of keeping them all resident at once, trading
        # ~20-30% more compute for a large cut in activation memory. Only
        # applies during training -- inference has no backward pass.
        self.grad_checkpoint = grad_checkpoint

    @staticmethod
    def causal_mask(seq_len: int, device=None) -> torch.Tensor:
        """Upper-triangular mask (True = masked/forbidden position)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)

    def forward(self, idx: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        """
        idx: (batch, seq_len) integer token ids
        returns logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, "sequence length exceeds max_seq_len"

        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(idx) + self.pos_emb(positions)            # (B, T, d_model)
        x = self.dropout(x)

        mask = self.causal_mask(T, device=idx.device) if use_causal_mask else None

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 1000
    batch_size = 2
    seq_len = 16

    model = TenLayerTransformer(vocab_size=vocab_size)  # current defaults: 224/8/896/10

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}  (batch, seq_len, vocab_size)")