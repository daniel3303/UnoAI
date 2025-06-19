# ./uno_ai/model/rope_embedding.py
import torch
from torch import Tensor, nn


class RoPEEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.dim: int = dim

        # Precompute frequency matrix
        inv_freq: Tensor = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        # x: [batch, seq_len, heads, head_dim]
        device = x.device
        dtype = x.dtype

        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim//2]

        # Create cos and sin tensors
        cos = freqs.cos()  # [seq_len, head_dim//2]
        sin = freqs.sin()  # [seq_len, head_dim//2]

        return self.apply_rotary(x, cos, sin)

    def apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # x: [batch, seq_len, heads, head_dim]
        # cos, sin: [seq_len, head_dim//2]

        batch_size, seq_len, n_heads, head_dim = x.shape

        # Reshape x to separate even/odd dimensions
        x = x.view(batch_size, seq_len, n_heads, head_dim // 2, 2)
        x1 = x[..., 0]  # [batch, seq_len, heads, head_dim//2] - even indices
        x2 = x[..., 1]  # [batch, seq_len, heads, head_dim//2] - odd indices

        # Expand cos and sin to match x1, x2 dimensions
        # cos, sin: [seq_len, head_dim//2] -> [1, seq_len, 1, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Combine back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # [batch, seq_len, heads, head_dim//2, 2]
        rotated = rotated.view(batch_size, seq_len, n_heads, head_dim)  # [batch, seq_len, heads, head_dim]

        return rotated