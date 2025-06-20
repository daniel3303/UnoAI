class RewardCalculator:
    """Handles reward calculation for UNO training"""

    def __init__(self):
        # Reward weights - easy to tune
        self.rewards = {
            'win': 50.0,
            'card_played': 1,
            'card_drawn': -1,
            'uno_achieved': 5.0,  # 1 card left
            'close_to_uno': 1.0,   # 2 cards left
            'special_card_bonus': 0,
            'wild_card_bonus': 0,
            'turn_penalty': -0.1,
            'invalid_action': -1,
        }

    def calculate_reward(self, info: dict, env_reward: float) -> float:
        """Calculate comprehensive reward based on game state"""
        reward = 0.0

        # 1. Win condition (use environment reward)
        if env_reward > 0:
            return self.rewards['win']

        # 2. Hand size changes
        prev_size = info.get('prev_hand_size', 0)
        current_size = info.get('current_hand_size', 0)

        if current_size < prev_size:
            # Played cards - positive reward
            cards_played = prev_size - current_size
            reward += cards_played * self.rewards['card_played']

        elif current_size > prev_size:
            # Drew cards - small penalty
            cards_drawn = current_size - prev_size
            reward += cards_drawn * self.rewards['card_drawn']

        # 3. UNO situation bonuses
        if current_size == 1:
            reward += self.rewards['uno_achieved']
        elif current_size == 2:
            reward += self.rewards['close_to_uno']

        # 4. Turn penalty (encourage efficiency)
        reward += self.rewards['turn_penalty']

        # 5. Invalid action penalty
        if not info.get('valid', True):
            reward += self.rewards['invalid_action']

        return reward

    def calculate_card_type_bonus(self, card_info: dict) -> float:
        """Calculate bonus based on card type played"""
        if not card_info:
            return 0.0

        card_type = card_info.get('type', '')

        if card_type in ['wild', 'wild_draw_four']:
            return self.rewards['wild_card_bonus']
        elif card_type in ['skip', 'reverse', 'draw_two']:
            return self.rewards['special_card_bonus']
        else:
            return 0.0