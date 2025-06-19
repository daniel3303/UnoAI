import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
import math
import time

from uno_ai.environment.uno_game import UNOGame, Card, CardColor, CardType
from uno_ai.model.uno_transformer import UNOTokens
from uno_ai.utils.asset_manager import AssetManager

class UNOEnv(gym.Env):
    def __init__(self, num_players: int = 4, render_mode: Optional[str] = None):
        super().__init__()

        self.num_players = num_players
        self.render_mode = render_mode
        self.game: Optional[UNOGame] = None

        # Action space: 0-6 play card, 7 draw card
        self.action_space = spaces.Discrete(8)

        # Observation space: sequence of tokens
        max_seq_len = 1000
        vocab_size = 200
        self.observation_space = spaces.Box(
            low=0, high=vocab_size-1, shape=(max_seq_len,), dtype=np.int32
        )

        # Initialize asset manager
        self.asset_manager = AssetManager()

        # Initialize pygame if rendering
        if render_mode == "human":
            self._init_pygame()


    def _init_pygame(self) -> None:
        """Initialize pygame with proper scaling support"""
        pygame.init()
    
        # Set DPI scale factor (simplified approach)
        self.dpi_scale = 1.0  # Default scale factor
    
        # Try to detect high DPI on different platforms
        try:
            if sys.platform == "darwin":  # macOS
                # On macOS, pygame handles retina automatically in most cases
                self.dpi_scale = 1.0
            elif sys.platform == "win32":  # Windows
                try:
                    import ctypes
                    # Get DPI awareness
                    user32 = ctypes.windll.user32
                    user32.SetProcessDPIAware()
                    dc = user32.GetDC(0)
                    dpi_x = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)
                    self.dpi_scale = max(1.0, dpi_x / 96.0)
                    user32.ReleaseDC(0, dc)
                except:
                    self.dpi_scale = 1.0
            else:  # Linux and others
                self.dpi_scale = 1.0
        except:
            self.dpi_scale = 1.0
    
        # Base window size
        base_width = 1600
        base_height = 1200
    
        # Apply scaling
        self.screen_width = int(base_width * self.dpi_scale)
        self.screen_height = int(base_height * self.dpi_scale)
    
        print(f"DPI Scale Factor: {self.dpi_scale}")
        print(f"Screen Resolution: {self.screen_width}x{self.screen_height}")
    
        # Create display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("UNO AI Training")
        self.clock = pygame.time.Clock()
    
        # Scale fonts
        base_font_small = 28
        base_font_medium = 36
        base_font_large = 48
        base_font_xl = 64
    
        self.font_small = pygame.font.Font(None, int(base_font_small * self.dpi_scale))
        self.font_medium = pygame.font.Font(None, int(base_font_medium * self.dpi_scale))
        self.font_large = pygame.font.Font(None, int(base_font_large * self.dpi_scale))
        self.font_xl = pygame.font.Font(None, int(base_font_xl * self.dpi_scale))
    
        # Scale card dimensions
        base_card_width = 80
        base_card_height = 120
        base_card_radius = 12
    
        self.card_width = int(base_card_width * self.dpi_scale)
        self.card_height = int(base_card_height * self.dpi_scale)
        self.card_radius = int(base_card_radius * self.dpi_scale)
    
        # Colors
        self.bg_color = (25, 35, 45)
        self.surface_color = (35, 45, 55)
        self.accent_color = (64, 224, 208)
        self.text_color = (255, 255, 255)
        self.text_secondary = (180, 180, 180)
    
        # Load assets with scaled dimensions
        self.asset_manager.load_assets(self.card_width, self.card_height)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.game = UNOGame(self.num_players)
        obs = self._get_observation()
        info = self.game.get_game_state()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.game is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        if self.game.game_over:
            obs = self._get_observation()
            info = self.game.get_game_state()
            return obs, 0.0, True, False, info
        
        current_player = self.game.current_player
        
        # Store game state before action
        prev_hand_size = len(self.game.players_hands[current_player])
                
        try:
            if action < 7:  # Play card from hand
                valid_actions = self.game.get_valid_actions(current_player)
                if action < len(self.game.players_hands[current_player]) and action in valid_actions:
                    card = self.game.players_hands[current_player][action]
                    if card.type in [CardType.WILD, CardType.WILD_DRAW_FOUR]:
                        chosen_color = self._choose_wild_color(current_player)
                        result = self.game.play_card(current_player, action, chosen_color)
                    else:
                        result = self.game.play_card(current_player, action)
                else:
                    result = {"valid": False, "reason": "Invalid card selection"}
        
            elif action == 7:  # Draw card
                result = self.game.draw_card(current_player)
        
            else:
                result = {"valid": False, "reason": "Invalid action number"}
        
        except Exception as e:
            print(f"Error in step: {e}")
            result = {"valid": False, "reason": f"Exception: {str(e)}"}
        
        terminated = self.game.game_over
        truncated = False
        
        obs = self._get_observation()
        info = self.game.get_game_state()
        if result:
            info.update(result)
        
        # Add additional info for reward calculation in trainer
        info.update({
            'prev_hand_size': prev_hand_size,
            'current_hand_size': len(self.game.players_hands[current_player]),
            'action_taken': action,
            'current_player': current_player
        })
        
        # Only return 1 if this player wins, 0 otherwise
        reward = 1.0 if (result.get("winner") == current_player) else 0.0
        
        return obs, reward, terminated, truncated, info


    def _choose_wild_color(self, player: int) -> CardColor:
        """Choose color for wild card based on most common color in hand"""
        hand = self.game.players_hands[player]
        color_counts = {
            CardColor.RED: 0,
            CardColor.GREEN: 0,
            CardColor.BLUE: 0,
            CardColor.YELLOW: 0
        }
    
        for card in hand:
            if card.color in color_counts:
                color_counts[card.color] += 1
    
        # Return the most common color, default to red if tie
        max_color = max(color_counts, key=color_counts.get)
        return max_color if color_counts[max_color] > 0 else CardColor.RED
    
    def _get_observation(self) -> np.ndarray:
        """Convert game state to observation tokens"""
        tokens = []
    
        if self.game:
            # Add game history (last few cards)
            history = self.game.discard_pile[-10:]
            for card in history:
                tokens.append(self._card_to_token(card))
    
            # Add current player's hand directly - no separators or complex encoding
            current_player = self.game.current_player
            for card in self.game.players_hands[current_player]:
                tokens.append(self._card_to_token(card))
    
            # Add other players' hand sizes as simple counts using card tokens
            # We'll represent hand size by repeating PAD tokens (simple but valid)
            for i in range(self.num_players):
                if i != current_player:
                    hand_size = min(len(self.game.players_hands[i]), 10)  # Cap at 10 for efficiency
                    for _ in range(hand_size):
                        tokens.append(UNOTokens.PAD)
    
        # Pad to fixed length
        max_len = self.observation_space.shape[0]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([UNOTokens.PAD] * (max_len - len(tokens)))
    
        return np.array(tokens, dtype=np.int32)
    
    def _card_to_token(self, card: Card) -> int:
        """Convert card to token ID using the standardized UNO token system"""
        return UNOTokens.card_to_token(card)

    def _number_to_tokens(self, number: int) -> List[int]:
        """Convert number to digit tokens"""
        if number == 0:
            return [110]

        digits = []
        while number > 0:
            digits.append(110 + (number % 10))
            number //= 10

        return digits[::-1]

    def render(self) -> None:
        if self.render_mode == "human" and self.game:
            self._render_pygame()

    def _render_pygame(self) -> None:
        """Render the game interface"""
        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear with background
        self.screen.fill(self.bg_color)
        self._draw_background_pattern()

        center_x, center_y = self.screen_width // 2, self.screen_height // 2

        # Draw center game area
        self._draw_center_area(center_x, center_y)

        # Draw players
        self._draw_players()

        # Draw UI panels
        self._draw_ui_panels()

        pygame.display.flip()
        self.clock.tick(30)

    def _scale(self, value: float) -> int:
        """Scale a value by DPI factor"""
        if hasattr(self, 'dpi_scale'):
            return int(value * self.dpi_scale)
        else:
            return int(value)  # Fallback if dpi_scale not set
    
    def _scale_rect(self, x: float, y: float, width: float, height: float) -> pygame.Rect:
        """Create a DPI-scaled rectangle"""
        return pygame.Rect(
            self._scale(x),
            self._scale(y),
            self._scale(width),
            self._scale(height)
        )
    
    def _scale_point(self, x: float, y: float) -> Tuple[int, int]:
        """Scale a point by DPI factor"""
        return (self._scale(x), self._scale(y))

    def _draw_background_pattern(self) -> None:
        """Draw subtle background pattern"""
        grid_size = 50
        grid_color = (30, 40, 50)

        for x in range(0, self.screen_width, grid_size):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.screen_height), 1)
        for y in range(0, self.screen_height, grid_size):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.screen_width, y), 1)

    def _draw_center_area(self, center_x: int, center_y: int) -> None:
        """Draw center area with discard pile and deck"""
        # Draw center platform (larger)
        platform_radius = self._scale(220)  # Increased from 180
        pygame.draw.circle(self.screen, self.surface_color, (center_x, center_y), platform_radius)
        pygame.draw.circle(self.screen, self.accent_color, (center_x, center_y), platform_radius, self._scale(3))
    
        # Draw discard pile with larger cards and better positioning
        if self.game and self.game.discard_pile:
            top_card = self.game.discard_pile[-1]
            discard_x = center_x - self._scale(80)  # More spacing
            discard_y = center_y - self._scale(20)  # Moved up slightly
    
            # Draw multiple cards for pile effect
            for i in range(3):
                offset = self._scale(i * 3)  # Slightly more offset
                self._draw_card(top_card, discard_x + offset, discard_y + offset, glow=(i == 2))
    
            # Add label (positioned better)
            label = self.font_small.render("DISCARD PILE", True, self.text_secondary)
            label_rect = label.get_rect(center=(discard_x, discard_y - self._scale(90)))
            self.screen.blit(label, label_rect)
    
        # Draw deck with larger cards and better positioning
        if self.game and self.game.deck:
            deck_x = center_x + self._scale(80)  # More spacing
            deck_y = center_y - self._scale(20)  # Moved up slightly
    
            # Draw stack effect
            for i in range(4):
                offset = self._scale(i * 3)  # Slightly more offset
                self._draw_card_back(deck_x - offset, deck_y - offset)
    
            # Deck count badge (better positioned)
            count_text = str(len(self.game.deck))
            self._draw_badge(deck_x + self._scale(50), deck_y - self._scale(30), count_text, (255, 140, 0))
    
            # Add label (positioned better)
            label = self.font_small.render("DECK", True, self.text_secondary)
            label_rect = label.get_rect(center=(deck_x, deck_y - self._scale(90)))
            self.screen.blit(label, label_rect)
    
        # Draw direction indicator (better positioned)
        if self.game:
            direction_text = "CLOCKWISE" if self.game.direction == 1 else "COUNTER-CLOCKWISE"
            direction_surface = self.font_medium.render(direction_text, True, self.accent_color)
            direction_rect = direction_surface.get_rect(center=(center_x, center_y + self._scale(120)))
            self.screen.blit(direction_surface, direction_rect)

    def _draw_players(self) -> None:
        """Draw players in circular layout"""
        for i in range(self.num_players):
            angle = (2 * math.pi * i / self.num_players) - math.pi/2
    
            is_current = self.game and i == self.game.current_player
            hand = self.game.players_hands[i] if self.game else []
    
            self._draw_player_area(i, hand, is_current, angle)

    def _draw_player_area(self, player_id: int, hand: List[Card],
                          is_current: bool, angle: float) -> None:
        """Draw individual player area positioned radially from center"""
        # Panel dimensions
        panel_width = self._scale(400)
        panel_height = self._scale(240)
    
        # Calculate the distance from center to the panel center
        center_radius = self._scale(350)
    
        # Calculate the perpendicular distance from panel center to its closest edge
        # This is the projection of the panel's half-dimensions onto the radial direction
        half_width = panel_width / 2
        half_height = panel_height / 2
    
        # The closest edge distance is determined by projecting the panel dimensions
        # onto the direction vector from center to panel
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
    
        # Calculate which edge is closest by finding minimum distance
        # Check all four edges and find the minimum perpendicular distance
        edge_distances = [
            abs(half_width * sin_angle),   # top/bottom edges
            abs(half_height * cos_angle)   # left/right edges
        ]
    
        panel_edge_distance = min(edge_distances)
        panel_center_distance = center_radius + panel_edge_distance + self._scale(10)  # Small buffer
    
        # Calculate panel center position based on angle
        game_center_x = self.screen_width // 2
        game_center_y = self.screen_height // 2
    
        panel_x = game_center_x + panel_center_distance * cos_angle
        panel_y = game_center_y + panel_center_distance * sin_angle
    
        # Create panel rectangle centered at calculated position
        panel_rect = pygame.Rect(
            int(panel_x - panel_width // 2),
            int(panel_y - panel_height // 2),
            panel_width,
            panel_height
        )
    
        # Panel colors
        panel_color = (45, 55, 65) if is_current else self.surface_color
        border_color = self.accent_color if is_current else (60, 70, 80)
        border_width = self._scale(4) if is_current else self._scale(2)
    
        # Draw panel
        pygame.draw.rect(self.screen, panel_color, panel_rect, border_radius=self._scale(15))
        pygame.draw.rect(self.screen, border_color, panel_rect, width=border_width, border_radius=self._scale(15))
    
        # Player info header
        header_y = panel_y - panel_height // 2 + self._scale(20)
    
        # Player name
        player_name = f"PLAYER {player_id + 1}"
        if is_current:
            player_name += " <--"
    
        name_color = self.accent_color if is_current else self.text_color
        name_surface = self.font_medium.render(player_name, True, name_color)
        name_rect = name_surface.get_rect(center=(int(panel_x), int(header_y)))
        self.screen.blit(name_surface, name_rect)
    
        # Hand count badge
        hand_count = len(hand)
        badge_color = (255, 69, 58) if hand_count <= 2 else (255, 140, 0) if hand_count <= 5 else (52, 199, 89)
        self._draw_badge(panel_x + self._scale(80), header_y, str(hand_count), badge_color)
    
        # Draw cards in fan layout
        cards_y = panel_y + self._scale(20)
        max_cards_shown = min(8, len(hand))
    
        if max_cards_shown > 0:
            card_spacing = min(self._scale(60), (panel_width - self._scale(40)) // max_cards_shown)
            start_x = panel_x - (max_cards_shown - 1) * card_spacing // 2
    
            valid_actions = self.game.get_valid_actions(player_id) if self.game and is_current else []
    
            for j in range(max_cards_shown):
                card_x = start_x + j * card_spacing
                card = hand[j] if j < len(hand) else None
    
                if card:
                    # Highlight playable cards
                    is_playable = is_current and j in valid_actions
                    self._draw_card(card, card_x, cards_y,
                                    highlight=is_playable, small=True)
    
        # Show "..." if more cards
        if len(hand) > max_cards_shown:
            dots_surface = self.font_medium.render("...", True, self.text_secondary)
            dots_rect = dots_surface.get_rect(center=(int(panel_x), int(cards_y + self._scale(80))))
            self.screen.blit(dots_surface, dots_rect)
            
        
    def _draw_card(self, card: Card, x: float, y: float, highlight: bool = False,
                   small: bool = False, glow: bool = False) -> None:
        """Draw a card using high-resolution images with dynamic scaling"""
        # Calculate target dimensions
        if small:
            target_width = int(self.card_width * 0.8)
            target_height = int(self.card_height * 0.8)
        else:
            target_width = self.card_width
            target_height = self.card_height
    
        # Try to get card image at the exact size we need
        card_image = self.asset_manager.get_card_image(card, target_width, target_height)
    
        if card_image:
            # Calculate position
            card_rect = card_image.get_rect(center=(int(x), int(y)))
    
            # Draw glow effect if needed
            if glow or highlight:
                glow_color = (255, 215, 0) if highlight else self.accent_color
                for i in range(3):
                    glow_rect = card_rect.inflate(self._scale(4 + i*2), self._scale(4 + i*2))
                    glow_alpha = 120 - i*40
                    glow_surface = pygame.Surface((glow_rect.width, glow_rect.height))
                    glow_surface.set_alpha(glow_alpha)
                    glow_surface.fill(glow_color)
                    self.screen.blit(glow_surface, glow_rect)
    
            # Draw the high-quality card image
            self.screen.blit(card_image, card_rect)
    
            # Add a subtle border for better definition
            pygame.draw.rect(self.screen, (255, 255, 255, 100), card_rect,
                             width=1, border_radius=self._scale(8))
    
        else:
            # Fallback to programmatic rendering
            self._draw_card_programmatic(card, x, y, highlight, small, glow)

    def _draw_card_back(self, x: float, y: float, small: bool = False) -> None:
        """Draw card back using high-resolution image"""
    
        # Calculate target dimensions
        if small:
            target_width = int(self.card_width * 0.8)
            target_height = int(self.card_height * 0.8)
        else:
            target_width = self.card_width
            target_height = self.card_height
    
        card_back_image = self.asset_manager.get_card_back_image(target_width, target_height)
    
        if card_back_image:
            card_rect = card_back_image.get_rect(center=(int(x), int(y)))
            self.screen.blit(card_back_image, card_rect)
    
            # Add border
            pygame.draw.rect(self.screen, (255, 255, 255, 100), card_rect,
                             width=1, border_radius=self._scale(8))
        else:
            # Fallback to programmatic rendering
            self._draw_card_back_programmatic(x, y)

    def _draw_card_programmatic(self, card: Card, x: float, y: float, highlight: bool = False,
                                small: bool = False, glow: bool = False) -> None:
        """Fallback programmatic card rendering"""
        # Card dimensions
        width = int(self.card_width * 0.7) if small else self.card_width
        height = int(self.card_height * 0.7) if small else self.card_height

        card_rect = pygame.Rect(int(x - width//2), int(y - height//2), width, height)

        # Glow effect
        if glow or highlight:
            glow_color = (255, 215, 0) if highlight else self.accent_color
            glow_rect = card_rect.inflate(8, 8)
            pygame.draw.rect(self.screen, glow_color, glow_rect, border_radius=self.card_radius + 4)

        # Card colors
        color_schemes = {
            CardColor.RED: [(220, 20, 60), (178, 34, 34)],
            CardColor.GREEN: [(34, 139, 34), (0, 100, 0)],
            CardColor.BLUE: [(30, 144, 255), (0, 0, 139)],
            CardColor.YELLOW: [(255, 215, 0), (255, 140, 0)],
            CardColor.WILD: [(138, 43, 226), (75, 0, 130)]
        }

        primary_color, secondary_color = color_schemes.get(card.color, [(128, 128, 128), (64, 64, 64)])

        # Draw card with gradient effect
        self._draw_gradient_rect(card_rect, primary_color, secondary_color)
        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, width=2, border_radius=self.card_radius)

        # Card content
        self._draw_card_content(card, card_rect, small)

    def _draw_card_back_programmatic(self, x: float, y: float) -> None:
        """Fallback programmatic card back rendering"""
        card_rect = pygame.Rect(int(x - self.card_width//2), int(y - self.card_height//2),
                                self.card_width, self.card_height)

        # Gradient background
        self._draw_gradient_rect(card_rect, (25, 25, 112), (72, 61, 139))
        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, width=2, border_radius=self.card_radius)

        # UNO logo
        logo_surface = self.font_large.render("UNO", True, (255, 255, 255))
        logo_rect = logo_surface.get_rect(center=card_rect.center)
        self.screen.blit(logo_surface, logo_rect)

    def _draw_card_content(self, card: Card, rect: pygame.Rect, small: bool = False) -> None:
        """Draw card content"""
        font = self.font_small if small else self.font_large

        # Determine card text and symbols
        if card.type == CardType.NUMBER:
            main_text = str(card.number)
            symbol = main_text
        elif card.type == CardType.SKIP:
            main_text = "SKIP"
            symbol = "SKIP"
        elif card.type == CardType.REVERSE:
            main_text = "REV"
            symbol = "REV"
        elif card.type == CardType.DRAW_TWO:
            main_text = "+2"
            symbol = "+2"
        elif card.type == CardType.WILD:
            main_text = "WILD"
            symbol = "WILD"
        elif card.type == CardType.WILD_DRAW_FOUR:
            main_text = "+4"
            symbol = "+4"
        else:
            main_text = "?"
            symbol = "?"

        # Text color
        text_color = (0, 0, 0) if card.color == CardColor.YELLOW else (255, 255, 255)

        # Draw main symbol in center
        symbol_surface = font.render(symbol, True, text_color)
        symbol_rect = symbol_surface.get_rect(center=rect.center)
        self.screen.blit(symbol_surface, symbol_rect)

        # Draw corner indicators
        if not small:
            corner_font = self.font_small
            corner_text = main_text if card.type == CardType.NUMBER else symbol[:2]

            # Top-left
            tl_surface = corner_font.render(corner_text, True, text_color)
            self.screen.blit(tl_surface, (rect.left + 8, rect.top + 8))

            # Bottom-right (rotated)
            br_surface = pygame.transform.rotate(tl_surface, 180)
            br_rect = br_surface.get_rect()
            br_rect.right = rect.right - 8
            br_rect.bottom = rect.bottom - 8
            self.screen.blit(br_surface, br_rect)

    def _draw_ui_panels(self) -> None:
        """Draw UI information panels"""
        if not self.game:
            return

        # Game status text
        status_texts = []

        if self.game.game_over:
            status_texts.append(f"Winner: Player {self.game.winner + 1}!")
            
            
        if not status_texts:
            return 

        # Top panel - Game status
        top_panel = pygame.Rect(50, 30, self.screen_width - 100, 80)
        pygame.draw.rect(self.screen, self.surface_color, top_panel, border_radius=10)
        pygame.draw.rect(self.screen, self.accent_color, top_panel, width=2, border_radius=10)

        for i, text in enumerate(status_texts):
            color = self.accent_color if "Winner" in text else self.text_color
            text_surface = self.font_medium.render(text, True, color)
            self.screen.blit(text_surface, (top_panel.left + 30, top_panel.top + 20 + i * 25))

        # Bottom panel - Instructions
        if not self.game.game_over:
            instruction_text = "AI is playing automatically - Watch the game progress"
            instruction_surface = self.font_small.render(instruction_text, True, self.text_secondary)
            instruction_rect = instruction_surface.get_rect(center=(self.screen_width//2, self.screen_height - 30))
            self.screen.blit(instruction_surface, instruction_rect)

    def _draw_badge(self, x: float, y: float, text: str, color: Tuple[int, int, int]) -> None:
        """Draw a badge/pill"""
        badge_surface = self.font_small.render(text, True, (255, 255, 255))
        badge_rect = badge_surface.get_rect()
        badge_rect.inflate_ip(16, 8)
        badge_rect.center = (int(x), int(y))

        # Draw badge background
        pygame.draw.rect(self.screen, color, badge_rect, border_radius=badge_rect.height // 2)

        # Draw badge text
        text_rect = badge_surface.get_rect(center=badge_rect.center)
        self.screen.blit(badge_surface, text_rect)

    def _draw_gradient_rect(self, rect: pygame.Rect, color1: Tuple[int, int, int],
                            color2: Tuple[int, int, int]) -> None:
        """Draw a vertical gradient rectangle"""
        for y in range(rect.height):
            blend = y / rect.height if rect.height > 0 else 0
            r = int(color1[0] * (1 - blend) + color2[0] * blend)
            g = int(color1[1] * (1 - blend) + color2[1] * blend)
            b = int(color1[2] * (1 - blend) + color2[2] * blend)
        pygame.draw.line(self.screen, (r, g, b),
                         (rect.left, rect.top + y), (rect.right, rect.top + y))
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions for current player"""
        if not self.game or self.game.game_over:
            return []
    
        current_player = self.game.current_player
        valid_card_actions = self.game.get_valid_actions(current_player)
        valid_actions = valid_card_actions + [7]  # Always can draw
        return valid_actions
    
    def close(self) -> None:
        """Close the environment"""
        if hasattr(self, 'screen'):
            pygame.quit()