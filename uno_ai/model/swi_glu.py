import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1: nn.Linear = nn.Linear(dim, hidden_dim, bias=False)
        self.w2: nn.Linear = nn.Linear(hidden_dim, dim, bias=False)
        self.w3: nn.Linear = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gated: Tensor = F.silu(self.w1(x)) * self.w3(x)
        dropped: Tensor = self.dropout(gated)
        return self.w2(dropped)