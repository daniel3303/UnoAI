from typing import Optional

import torch
from torch import nn

from uno_ai.environment.uno_vocabulary import UNOVocabulary
from uno_ai.layers.transformer_block import TransformerBlock


class PPOAgent(nn.Module):
    def __init__(self, vocab_size: int = UNOVocabulary.VOCAB_SIZE, dim: int = 512, n_layers: int = 16, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=UNOVocabulary.PAD)
        self.dropout = nn.Dropout(0.1)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout=0.1)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)

        # Policy head - outputs probability over entire vocabulary
        self.policy_head = nn.Linear(dim, vocab_size)

        # Value head (state value estimation)
        self.value_head = nn.Linear(dim, 1)

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
    
    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len = tokens.shape
    
        # Create attention mask for PAD tokens if not provided
        if attention_mask is None:
            # Mask out PAD tokens - 1 for valid tokens, 0 for PAD
            pad_mask = (tokens != UNOVocabulary.PAD)
            attention_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    
        # Token embeddings (PAD tokens will have zero embeddings due to padding_idx)
        x = self.token_embedding(tokens)
        x = self.dropout(x)
    
        # Pass through transformer layers with attention masking
        for layer in self.layers:
            x = layer(x, attention_mask)  # PAD positions won't be attended to

        # Final normalization
        x = self.norm(x)

        # Use the last token's representation for policy and value
        last_hidden = x[:, -1, :]  # [batch_size, dim]

        # Get action logits over entire vocabulary and state value
        action_logits = self.policy_head(last_hidden)
        state_value = self.value_head(last_hidden).squeeze(-1)

        return action_logits, state_value

    def get_action_and_value(self, tokens: torch.Tensor, action_mask: torch.Tensor, action: Optional[torch.Tensor] = None):
        action_logits, value = self.forward(tokens)

        # Apply action mask (set invalid actions to very negative values)
        masked_logits = action_logits.clone()
        masked_logits[~action_mask] = -1e8

        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=masked_logits)

        if action is None:
            action = action_dist.sample()

        return action, action_dist.log_prob(action), action_dist.entropy(), value