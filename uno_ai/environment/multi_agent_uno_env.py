# ./uno_ai/environment/multi_agent_env.py
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from uno_ai.environment.uno_env import UNOEnv
from uno_ai.model.uno_transformer import UNOTokens


@dataclass
class OpponentConfig:
    agent_players: List[int]  # Which players are trained agents
    random_players: List[int]  # Which players are random
    env_players: List[int]    # Which players use environment logic

class MultiAgentUNOEnv(UNOEnv):
    def __init__(self, num_players: int = 4, render_mode: Optional[str] = None):
        super().__init__(num_players, render_mode)
        self.opponent_config: Optional[OpponentConfig] = None
        self.trained_agents: Dict[int, Any] = {}  # Store multiple agent instances
        self.vocab_size = UNOTokens.VOCAB_SIZE

    def set_opponent_config(self, config: OpponentConfig):
        """Set the current opponent configuration"""
        self.opponent_config = config

    def add_trained_agent(self, player_id: int, agent):
        """Add a trained agent for a specific player"""
        self.trained_agents[player_id] = agent

    def get_action_for_player(self, player_id: int, obs) -> int:
        """Get action for a specific player based on their type"""
        if not self.opponent_config:
            return self.action_space.sample()  # Default random

        if player_id in self.opponent_config.agent_players:
            # Use trained agent
            if player_id in self.trained_agents:
                return self._get_agent_action(player_id, obs)
            else:
                return self._get_random_valid_action(player_id)

        elif player_id in self.opponent_config.random_players:
            return self._get_random_valid_action(player_id)

        elif player_id in self.opponent_config.env_players:
            return self._get_env_action(player_id)

        return self._get_random_valid_action(player_id)

    def _get_agent_action(self, player_id: int, obs) -> int:
        """Get action from trained agent"""
        agent = self.trained_agents[player_id]

        try:
            with torch.no_grad():
                device = next(agent.parameters()).device
                obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(device)
                action_mask, token_to_hand_index = self._create_action_mask_for_player(player_id)
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)
                action_token, _, _, _ = agent.get_action_and_value(obs_tensor, action_mask_tensor)

                # Convert to environment action
                return self._convert_token_to_env_action(action_token.item(), token_to_hand_index)
        except Exception as e:
            print(f"Error getting agent action for player {player_id}: {e}")
            return self._get_random_valid_action(player_id)

    def _get_random_valid_action(self, player_id: int) -> int:
        """Get random valid action"""
        if not self.game:
            return 7

        valid_actions = self.game.get_valid_actions(player_id)
        if valid_actions:
            return random.choice(valid_actions)
        return 7  # Draw card

    def _get_env_action(self, player_id: int) -> int:
        """Get action using simple heuristics"""
        if not self.game:
            return 7

        valid_actions = self.game.get_valid_actions(player_id)
        if not valid_actions:
            return 7  # Draw card

        # Simple heuristic: prefer special cards, then high numbers
        hand = self.game.players_hands[player_id]
        best_action = valid_actions[0]
        best_score = -1

        for action in valid_actions:
            if action < len(hand):
                card = hand[action]
                score = self._calculate_card_score(card)
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

    def _calculate_card_score(self, card) -> int:
        """Simple card scoring for environment players"""
        from uno_ai.environment.uno_game import CardType

        if card.type in [CardType.WILD, CardType.WILD_DRAW_FOUR]:
            return 100
        elif card.type in [CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO]:
            return 50
        else:
            return card.number if card.number else 0

    def _create_action_mask_for_player(self, player_id: int):
        """Create action mask for specific player"""
        mask = np.zeros(self.vocab_size, dtype=bool)
        token_to_hand_index = {}

        if not self.game:
            # Fallback - only allow draw action
            mask[UNOTokens.DRAW_ACTION] = True
            token_to_hand_index[UNOTokens.DRAW_ACTION] = -1
            return mask, token_to_hand_index

        hand = self.game.players_hands[player_id]
        valid_card_indices = self.game.get_valid_actions(player_id)

        for hand_index in valid_card_indices:
            if hand_index < len(hand):
                card = hand[hand_index]
                card_token = UNOTokens.card_to_token(card)
                mask[card_token] = True
                token_to_hand_index[card_token] = hand_index

        # Always enable draw action
        mask[UNOTokens.DRAW_ACTION] = True
        token_to_hand_index[UNOTokens.DRAW_ACTION] = -1

        return mask, token_to_hand_index

    def _convert_token_to_env_action(self, action_token: int, token_to_hand_index: Dict[int, int]) -> int:
        """Convert action token to environment action"""
        if action_token == UNOTokens.DRAW_ACTION:
            return 7
        elif action_token in token_to_hand_index:
            return token_to_hand_index[action_token]
        else:
            print(f"Warning: Invalid action token {action_token}, falling back to draw")
            return 7  # Fallback to draw