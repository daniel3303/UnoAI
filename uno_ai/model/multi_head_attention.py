from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from uno_ai.model.rope_embedding import RoPEEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads: int = n_heads
        self.head_dim: int = dim // n_heads
        self.scale: float = self.head_dim ** -0.5

        # Standard attention projections
        self.q_proj: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.k_proj: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.v_proj: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.o_proj: nn.Linear = nn.Linear(dim, dim, bias=False)

        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.rope: RoPEEmbedding = RoPEEmbedding(self.head_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size: int
        seq_len: int
        dim: int
        batch_size, seq_len, dim = x.shape

        # Compute Q, K, V
        q: Tensor = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k: Tensor = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v: Tensor = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores: Tensor = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        attn_weights: Tensor = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out: Tensor = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.o_proj(out)