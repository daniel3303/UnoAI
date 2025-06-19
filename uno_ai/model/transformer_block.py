from typing import Optional

from torch import Tensor, nn

from uno_ai.model.multi_head_attention import MultiHeadAttention
from uno_ai.model.swi_glu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention: MultiHeadAttention = MultiHeadAttention(dim, n_heads, dropout)
        self.feed_forward: SwiGLU = SwiGLU(dim, int(dim * mlp_ratio), dropout)

        # Using LayerNorm
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(dim)
        self.ffn_norm: nn.LayerNorm = nn.LayerNorm(dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Pre-norm architecture with residual connections
        attn_out: Tensor = self.attention(self.attention_norm(x), mask)
        x = x + self.dropout(attn_out)

        ffn_out: Tensor = self.feed_forward(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)

        return x