from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from uno_ai.model.transformer_block import TransformerBlock

"""
UNO Token Vocabulary (85 tokens total):

CARD TOKENS (0-81):
RED cards (0-19):
  0-9: Red number cards (0-9)
  10: Red Skip
  11: Red Reverse  
  12: Red Draw Two

GREEN cards (20-39):
  20-29: Green number cards (0-9)
  30: Green Skip
  31: Green Reverse
  32: Green Draw Two

BLUE cards (40-59):
  40-49: Blue number cards (0-9)
  50: Blue Skip
  51: Blue Reverse
  52: Blue Draw Two

YELLOW cards (60-79):
  60-69: Yellow number cards (0-9)
  70: Yellow Skip
  71: Yellow Reverse
  72: Yellow Draw Two

WILD cards (80-81):
  80: Wild
  81: Wild Draw Four

ACTION TOKENS (82-83):
  82: Draw card action
  83: Pass turn action

SPECIAL TOKENS (84):
  84: [PAD] - Padding token
"""

class UNOTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int = 85,  # Total UNO vocabulary size
            dim: int = 512,
            n_layers: int = 6,
            n_heads: int = 8,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.vocab_size = vocab_size

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.layers: nn.ModuleList = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm: nn.LayerNorm = nn.LayerNorm(dim)
        self.lm_head: nn.Linear = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, tokens: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size: int
        seq_len: int
        batch_size, seq_len = tokens.shape

        # Token embeddings
        x: Tensor = self.embedding(tokens)
        x = self.dropout(x)

        # Create causal mask for autoregressive generation
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Pass through transformer layers
        layer: TransformerBlock
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final normalization and projection
        x = self.norm(x)
        logits: Tensor = self.lm_head(x)

        return logits

# Token mapping utilities
class UNOTokens:
    """Utility class for UNO token mappings"""

    # Card token ranges
    RED_START = 0
    GREEN_START = 20
    BLUE_START = 40
    YELLOW_START = 60
    WILD_START = 80

    # Action tokens
    DRAW_ACTION = 82
    PASS_ACTION = 83

    # Special tokens
    PAD = 84

    VOCAB_SIZE = 85

    @staticmethod
    def card_to_token(card) -> int:
        """Convert UNO card to token ID"""
        from uno_ai.environment.uno_game import CardColor, CardType

        # Color base offsets
        color_offset = {
            CardColor.RED: UNOTokens.RED_START,
            CardColor.GREEN: UNOTokens.GREEN_START,
            CardColor.BLUE: UNOTokens.BLUE_START,
            CardColor.YELLOW: UNOTokens.YELLOW_START,
        }

        if card.color == CardColor.WILD:
            if card.type == CardType.WILD:
                return 80
            elif card.type == CardType.WILD_DRAW_FOUR:
                return 81
        else:
            base = color_offset[card.color]
            if card.type == CardType.NUMBER:
                return base + card.number
            elif card.type == CardType.SKIP:
                return base + 10
            elif card.type == CardType.REVERSE:
                return base + 11
            elif card.type == CardType.DRAW_TWO:
                return base + 12

        return UNOTokens.PAD  # Fallback to PAD for unknown cards

    @staticmethod
    def token_to_card_info(token: int) -> dict:
        """Convert token ID back to card information"""
        if token < UNOTokens.GREEN_START:  # Red cards
            return UNOTokens._decode_colored_card(token, "RED", UNOTokens.RED_START)
        elif token < UNOTokens.BLUE_START:  # Green cards
            return UNOTokens._decode_colored_card(token, "GREEN", UNOTokens.GREEN_START)
        elif token < UNOTokens.YELLOW_START:  # Blue cards
            return UNOTokens._decode_colored_card(token, "BLUE", UNOTokens.BLUE_START)
        elif token < UNOTokens.WILD_START:  # Yellow cards
            return UNOTokens._decode_colored_card(token, "YELLOW", UNOTokens.YELLOW_START)
        elif token == 80:
            return {"color": "WILD", "type": "WILD", "number": None}
        elif token == 81:
            return {"color": "WILD", "type": "WILD_DRAW_FOUR", "number": None}
        elif token == UNOTokens.DRAW_ACTION:
            return {"action": "DRAW"}
        elif token == UNOTokens.PASS_ACTION:
            return {"action": "PASS"}
        elif token == UNOTokens.PAD:
            return {"special": "PAD"}
        else:
            return {"special": f"UNKNOWN_TOKEN_{token}"}

    @staticmethod
    def _decode_colored_card(token: int, color: str, base: int) -> dict:
        """Helper to decode colored card tokens"""
        offset = token - base
        if 0 <= offset <= 9:
            return {"color": color, "type": "NUMBER", "number": offset}
        elif offset == 10:
            return {"color": color, "type": "SKIP", "number": None}
        elif offset == 11:
            return {"color": color, "type": "REVERSE", "number": None}
        elif offset == 12:
            return {"color": color, "type": "DRAW_TWO", "number": None}
        else:
            return {"color": color, "type": "UNKNOWN", "number": None}