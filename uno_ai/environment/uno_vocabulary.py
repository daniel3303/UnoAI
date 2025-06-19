# uno_vocabulary.py
"""
UNO Token Vocabulary (100 tokens total):

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

GAME MODE TOKENS (84-85):
 84: Normal rules mode
 85: Street rules mode

CARD_BACK TOKEN (86):
 86: Card back (hidden card)

OPPONENT TOKEN (87):
 87: Opponent identifier

NUMBER TOKENS (88-97):
 88-97: Number digits 0-9

SPECIAL TOKENS (98-99):
 98: [PAD] - Padding token
 99: [SEP] - Separator token
"""
from typing import List

from uno_ai.environment.uno_game import CardColor, CardType, GameMode


# Token mapping utilities
class UNOVocabulary:
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

    # Game mode tokens
    NORMAL_MODE = 84
    STREET_MODE = 85

    # Card back token
    CARD_BACK = 86

    # Opponent identifier
    OPPONENT = 87

    # Number tokens (0-9)
    NUMBER_START = 88

    # Special tokens
    SEP = 98
    PAD = 99

    VOCAB_SIZE = 100

    @staticmethod
    def card_to_token(card) -> int:
        """Convert UNO card to token ID"""
        # Color base offsets
        color_offset = {
            CardColor.RED: UNOVocabulary.RED_START,
            CardColor.GREEN: UNOVocabulary.GREEN_START,
            CardColor.BLUE: UNOVocabulary.BLUE_START,
            CardColor.YELLOW: UNOVocabulary.YELLOW_START,
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

        return UNOVocabulary.PAD  # Fallback to PAD for unknown cards

    @staticmethod
    def game_mode_to_token(game_mode) -> int:
        """Convert game mode to token ID"""
        if game_mode == GameMode.NORMAL:
            return UNOVocabulary.NORMAL_MODE
        elif game_mode == GameMode.STREET:
            return UNOVocabulary.STREET_MODE

        return UNOVocabulary.PAD  # Fallback for unknown modes

    @staticmethod
    def number_to_tokens(number: int) -> List[int]:
        """Convert number to sequence of digit tokens"""
        if number == 0:
            return [UNOVocabulary.NUMBER_START]

        digits = []
        while number > 0:
            digit = number % 10
            digits.append(UNOVocabulary.NUMBER_START + digit)
            number //= 10

        return digits[::-1]  # Reverse to get correct order

    @staticmethod
    def tokens_to_number(tokens: List[int]) -> int:
        """Convert sequence of digit tokens back to number"""
        number = 0
        for token in tokens:
            if UNOVocabulary.NUMBER_START <= token <= UNOVocabulary.NUMBER_START + 9:
                digit = token - UNOVocabulary.NUMBER_START
                number = number * 10 + digit
            else:
                break  # Stop at first non-digit token
        return number

    @staticmethod
    def token_to_card_info(token: int) -> dict:
        """Convert token ID back to card information"""
        if token < UNOVocabulary.GREEN_START:  # Red cards
            return UNOVocabulary._decode_colored_card(token, "RED", UNOVocabulary.RED_START)
        elif token < UNOVocabulary.BLUE_START:  # Green cards
            return UNOVocabulary._decode_colored_card(token, "GREEN", UNOVocabulary.GREEN_START)
        elif token < UNOVocabulary.YELLOW_START:  # Blue cards
            return UNOVocabulary._decode_colored_card(token, "BLUE", UNOVocabulary.BLUE_START)
        elif token < UNOVocabulary.WILD_START:  # Yellow cards
            return UNOVocabulary._decode_colored_card(token, "YELLOW", UNOVocabulary.YELLOW_START)
        elif token == 80:
            return {"color": "WILD", "type": "WILD", "number": None}
        elif token == 81:
            return {"color": "WILD", "type": "WILD_DRAW_FOUR", "number": None}
        elif token == UNOVocabulary.DRAW_ACTION:
            return {"action": "DRAW"}
        elif token == UNOVocabulary.PASS_ACTION:
            return {"action": "PASS"}
        elif token == UNOVocabulary.NORMAL_MODE:
            return {"game_mode": "NORMAL"}
        elif token == UNOVocabulary.STREET_MODE:
            return {"game_mode": "STREET"}
        elif token == UNOVocabulary.CARD_BACK:
            return {"card": "CARD_BACK"}
        elif token == UNOVocabulary.OPPONENT:
            return {"identifier": "OPPONENT"}
        elif UNOVocabulary.NUMBER_START <= token <= UNOVocabulary.NUMBER_START + 9:
            digit = token - UNOVocabulary.NUMBER_START
            return {"number": digit}
        elif token == UNOVocabulary.PAD:
            return {"special": "PAD"}
        elif token == UNOVocabulary.SEP:
            return {"special": "SEP"}
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