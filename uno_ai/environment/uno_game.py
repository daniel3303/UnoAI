# uno_game.py
import random
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple


class CardColor(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    WILD = "wild"


class CardType(Enum):
    NUMBER = "number"
    SKIP = "skip"
    REVERSE = "reverse"
    DRAW_TWO = "draw_two"
    WILD = "wild"
    WILD_DRAW_FOUR = "wild_draw_four"


class GameMode(Enum):
    NORMAL = "normal" # Normal UNO rules
    STREET = "street" # Street rules with stacking and drawing until playable


class Card:
    def __init__(self, color: CardColor, card_type: CardType, number: Optional[int] = None):
        self.color = color
        self.type = card_type
        self.number = number

    def __str__(self) -> str:
        if self.type == CardType.NUMBER:
            return f"{self.color.value}_{self.number}"
        return f"{self.color.value}_{self.type.value}"

    def __repr__(self) -> str:
        return self.__str__()

    def can_play_on(self, other: 'Card') -> bool:
        """Check if this card can be played on top of another card"""
        # Wild cards can always be played
        if self.type in [CardType.WILD, CardType.WILD_DRAW_FOUR]:
            return True

        # Same color always works
        if self.color == other.color:
            return True

        # Same type (both action cards of same type)
        if self.type == other.type and self.type != CardType.NUMBER:
            return True

        # Same number (both number cards with same number)
        if (self.type == CardType.NUMBER and
                other.type == CardType.NUMBER and
                self.number == other.number):
            return True

        return False

    def is_draw_card(self) -> bool:
        """Check if this is a draw card (+2 or +4)"""
        return self.type in [CardType.DRAW_TWO, CardType.WILD_DRAW_FOUR]

    def get_draw_amount(self) -> int:
        """Get the number of cards this card forces to draw"""
        if self.type == CardType.DRAW_TWO:
            return 2
        elif self.type == CardType.WILD_DRAW_FOUR:
            return 4
        return 0


class UNOGame:
    def __init__(self, num_players: int = 4, game_mode: GameMode = GameMode.NORMAL):
        self.num_players = num_players
        self.game_mode = game_mode
        self.players_hands: List[List[Card]] = [[] for _ in range(num_players)]
        self.deck: List[Card] = []
        self.discard_pile: List[Card] = []
        self.discard_pile_with_players: List[Tuple[int, Card]] = []  # Track (player_id, card)
        self.current_player = 0
        self.direction = 1  # 1 for clockwise, -1 for counter-clockwise
        self.game_over = False
        self.winner: Optional[int] = None
        self.pending_draw_stack = 0  # For street rules stacking
        self._create_deck()
        self._deal_cards()

    def _create_deck(self) -> None:
        """Create a standard UNO deck"""
        self.deck = []

        # Regular colored cards
        for color in [CardColor.RED, CardColor.GREEN, CardColor.BLUE, CardColor.YELLOW]:
            # Number cards (0-9, with 0 appearing once and 1-9 appearing twice)
            self.deck.append(Card(color, CardType.NUMBER, 0))
            for number in range(1, 10):
                self.deck.append(Card(color, CardType.NUMBER, number))
                self.deck.append(Card(color, CardType.NUMBER, number))

            # Action cards (2 of each per color)
            for _ in range(2):
                self.deck.append(Card(color, CardType.SKIP))
                self.deck.append(Card(color, CardType.REVERSE))
                self.deck.append(Card(color, CardType.DRAW_TWO))

        # Wild cards
        for _ in range(4):
            self.deck.append(Card(CardColor.WILD, CardType.WILD))
            self.deck.append(Card(CardColor.WILD, CardType.WILD_DRAW_FOUR))

        random.shuffle(self.deck)

    def _deal_cards(self) -> None:
        """Deal 7 cards to each player and place first card on discard pile"""
        for _ in range(7):
            for player in range(self.num_players):
                self.players_hands[player].append(self.deck.pop())
    
        # Place first card (ensure it's not a wild card)
        while True:
            first_card = self.deck.pop()
            if first_card.type not in [CardType.WILD, CardType.WILD_DRAW_FOUR]:
                self.discard_pile.append(first_card)
                # Use -1 to indicate system/initial card (not played by any player)
                self.discard_pile_with_players.append((-1, first_card))
                break

    def get_valid_actions(self, player: int) -> List[int]:
        """Get list of valid card indices that can be played"""
        if self.game_over:
            return []

        hand = self.players_hands[player]
        top_card = self.discard_pile[-1]
        valid_indices = []

        # In street rules with pending draw stack, only draw cards can be played
        if self.game_mode == GameMode.STREET and self.pending_draw_stack > 0:
            for i, card in enumerate(hand):
                if card.is_draw_card() and card.can_play_on(top_card):
                    valid_indices.append(i)
        else:
            for i, card in enumerate(hand):
                if card.can_play_on(top_card):
                    valid_indices.append(i)

        return valid_indices

    def play_card(self, player: int, card_index: int, chosen_color: Optional[CardColor] = None) -> Dict[str, Any]:
        """Play a card from player's hand"""
        if self.game_over:
            return {"valid": False, "reason": "Game is over"}
    
        if player != self.current_player:
            return {"valid": False, "reason": "Not player's turn"}
    
        hand = self.players_hands[player]
        if card_index >= len(hand) or card_index < 0:
            return {"valid": False, "reason": "Invalid card index"}
    
        card = hand[card_index]
        top_card = self.discard_pile[-1] if self.discard_pile else None
    
        if not top_card:
            return {"valid": False, "reason": "No card on discard pile"}
    
        # Street rules: if there's a pending draw stack, only allow draw cards
        if self.game_mode == GameMode.STREET and self.pending_draw_stack > 0:
            if not card.is_draw_card() or not card.can_play_on(top_card):
                return {"valid": False, "reason": "Must play a draw card to stack or draw cards"}
        elif not card.can_play_on(top_card):
            return {"valid": False, "reason": "Card cannot be played"}
    
        # Remove card from hand and add to discard pile
        played_card = hand.pop(card_index)
    
        # Handle wild cards
        if played_card.type in [CardType.WILD, CardType.WILD_DRAW_FOUR]:
            if chosen_color and chosen_color != CardColor.WILD:
                played_card = Card(chosen_color, played_card.type)
            else:
                played_card = Card(CardColor.RED, played_card.type)
    
        self.discard_pile.append(played_card)
        self.discard_pile_with_players.append((player, played_card))  # Track who played it
    
        # Check for win
        if len(hand) == 0:
            self.game_over = True
            self.winner = player
            return {"valid": True, "winner": player, "game_over": True}
    
        # Handle special card effects
        result = self._handle_card_effects(played_card)
    
        # Move to next player
        if not result.get("skip_next", False):
            self._next_player()
    
        return {"valid": True, **result}

    def draw_card(self, player: int) -> Dict[str, Any]:
        """Player draws a card from deck"""
        if self.game_over:
            return {"valid": False, "reason": "Game is over"}

        if player != self.current_player:
            return {"valid": False, "reason": "Not player's turn"}

        # Street rules: if there's a pending draw stack, must draw all stacked cards
        if self.game_mode == GameMode.STREET and self.pending_draw_stack > 0:
            self._force_draw(player, self.pending_draw_stack)
            self.pending_draw_stack = 0
            self._next_player()
            return {"valid": True, "drew_cards": self.pending_draw_stack, "turn_ended": True}

        # Street rules: keep drawing until a playable card is found
        if self.game_mode == GameMode.STREET:
            return self._street_draw_until_playable(player)

        # Normal rules: draw one card
        return self._normal_draw_card(player)

    def _street_draw_until_playable(self, player: int) -> Dict[str, Any]:
        """Street rules: draw cards until finding a playable one"""
        cards_drawn = 0
        top_card = self.discard_pile[-1] if self.discard_pile else None

        while True:
            if len(self.deck) == 0:
                self._reshuffle_deck()

            if len(self.deck) == 0:
                self._next_player()
                return {"valid": True, "cards_drawn": cards_drawn, "turn_ended": True, "reason": "No cards available"}

            card = self.deck.pop()
            self.players_hands[player].append(card)
            cards_drawn += 1

            if top_card and card.can_play_on(top_card):
                return {"valid": True, "cards_drawn": cards_drawn, "must_play": True, "drawn_card_index": len(self.players_hands[player]) - 1}

    def _normal_draw_card(self, player: int) -> Dict[str, Any]:
        """Normal rules: draw one card"""
        if len(self.deck) == 0:
            self._reshuffle_deck()

        if len(self.deck) == 0:
            self._next_player()
            return {"valid": True, "can_play_drawn": False, "turn_ended": True, "reason": "No cards available"}

        card = self.deck.pop()
        self.players_hands[player].append(card)
        drawn_card_index = len(self.players_hands[player]) - 1

        top_card = self.discard_pile[-1] if self.discard_pile else None
        if top_card and card.can_play_on(top_card):
            return {
                "valid": True,
                "can_play_drawn": True,
                "drawn_card_index": drawn_card_index,
                "must_play_or_pass": True
            }
        else:
            self._next_player()
            return {"valid": True, "can_play_drawn": False, "turn_ended": True}

    def _handle_card_effects(self, card: Card) -> Dict[str, Any]:
        """Handle special effects of played cards"""
        result = {}

        if card.type == CardType.SKIP:
            self._next_player()
            result["skip_next"] = True

        elif card.type == CardType.REVERSE:
            self.direction *= -1
            if self.num_players == 2:
                self._next_player()
                result["skip_next"] = True

        elif card.type in [CardType.DRAW_TWO, CardType.WILD_DRAW_FOUR]:
            draw_amount = card.get_draw_amount()

            if self.game_mode == GameMode.STREET:
                # Street rules: add to pending stack
                self.pending_draw_stack += draw_amount
                result["draw_stack"] = self.pending_draw_stack
            else:
                # Normal rules: force next player to draw immediately
                next_player = (self.current_player + self.direction) % self.num_players
                self._force_draw(next_player, draw_amount)
                self._next_player()
                result["skip_next"] = True

        return result

    def _force_draw(self, player: int, num_cards: int) -> None:
        """Force a player to draw cards"""
        for _ in range(num_cards):
            if len(self.deck) == 0:
                self._reshuffle_deck()
            if len(self.deck) > 0:
                self.players_hands[player].append(self.deck.pop())

    def _next_player(self) -> None:
        """Move to next player"""
        self.current_player = (self.current_player + self.direction) % self.num_players

    def _reshuffle_deck(self) -> None:
        """Reshuffle discard pile back into deck"""
        if len(self.discard_pile) > 1:
            top_card = self.discard_pile.pop()
            top_player_card = self.discard_pile_with_players.pop()
    
            self.deck = self.discard_pile[:]
            self.discard_pile = [top_card]
            self.discard_pile_with_players = [top_player_card]
            random.shuffle(self.deck)

    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state"""
        return {
            "current_player": self.current_player,
            "hands_sizes": [len(hand) for hand in self.players_hands],
            "top_card": self.discard_pile[-1] if self.discard_pile else None,
            "direction": self.direction,
            "deck_size": len(self.deck),
            "game_over": self.game_over,
            "winner": self.winner,
            "game_mode": self.game_mode.value,
            "pending_draw_stack": self.pending_draw_stack
        }