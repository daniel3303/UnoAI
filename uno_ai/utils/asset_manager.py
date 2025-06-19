# uno_ai/utils/asset_manager.py
import os
import pygame
from typing import Dict, Optional, Tuple
from uno_ai.environment.uno_game import Card, CardColor, CardType

class AssetManager:
    def __init__(self):
        self.card_images: Dict[str, pygame.Surface] = {}
        self.original_card_images: Dict[str, pygame.Surface] = {}  # Store originals
        self.card_back_image: Optional[pygame.Surface] = None
        self.original_card_back: Optional[pygame.Surface] = None
        self.assets_loaded = False
        self.current_scale = (80, 120)  # Track current scale

        # Get the path to assets directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.assets_path = os.path.join(project_root, "assets", "images", "cards")

    def load_assets(self, card_width: int = 80, card_height: int = 120) -> None:
        """Load all card images in full resolution"""
        # Only reload if scale changed or not loaded yet
        if self.assets_loaded and self.current_scale == (card_width, card_height):
            return

        self.current_scale = (card_width, card_height)

        try:
            # Load colored cards
            colors = ["red", "green", "blue", "yellow"]

            for color in colors:
                color_path = os.path.join(self.assets_path, color)

                # Load number cards (0-9)
                for number in range(10):
                    key = f"{color}_{number}"
                    self._load_card_image(color_path, f"{color}_{number}", key, card_width, card_height)

                # Load action cards
                actions = ["skip", "reverse", "draw_two"]
                for action in actions:
                    key = f"{color}_{action}"
                    self._load_card_image(color_path, f"{color}_{action}", key, card_width, card_height)

            # Load wild cards
            wild_path = os.path.join(self.assets_path, "wild")
            self._load_card_image(wild_path, "wild", "wild", card_width, card_height)
            self._load_card_image(wild_path, "wild_draw_four", "wild_draw_four", card_width, card_height)

            # Load card back
            self._load_card_back(card_width, card_height)

            self.assets_loaded = True

        except Exception as e:
            print(f"Error loading assets: {e}")
            print("Falling back to programmatic card rendering")
            self.assets_loaded = False

    def _load_card_image(self, folder_path: str, filename: str, key: str,
                         target_width: int, target_height: int) -> None:
        """Load a card image in full resolution, trying multiple formats"""
        # Try different image formats
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        for ext in extensions:
            image_path = os.path.join(folder_path, filename + ext)
            if os.path.exists(image_path):
                try:
                    # Load original image at full resolution
                    original_image = pygame.image.load(image_path)
                    original_image = original_image.convert_alpha()

                    # Store original for future scaling
                    self.original_card_images[key] = original_image

                    # Create scaled version for current use
                    scaled_image = self._scale_image_high_quality(
                        original_image, target_width, target_height
                    )
                    self.card_images[key] = scaled_image

                    return

                except Exception as e:
                    print(f"Failed to load {image_path}: {e}")
                    continue

        print(f"Could not find image for {filename} in {folder_path}")

    def _load_card_back(self, target_width: int, target_height: int) -> None:
        """Load card back image"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        for ext in extensions:
            back_path = os.path.join(self.assets_path, "card_back" + ext)
            if os.path.exists(back_path):
                try:
                    # Load original
                    original_back = pygame.image.load(back_path)
                    original_back = original_back.convert_alpha()
                    self.original_card_back = original_back

                    # Create scaled version
                    self.card_back_image = self._scale_image_high_quality(
                        original_back, target_width, target_height
                    )

                    return

                except Exception as e:
                    print(f"Failed to load {back_path}: {e}")
                    continue

    def _scale_image_high_quality(self, image: pygame.Surface,
                                  target_width: int, target_height: int) -> pygame.Surface:
        """Scale image with highest quality using multiple algorithms"""
        original_width, original_height = image.get_size()

        # If image is already the target size, return copy
        if original_width == target_width and original_height == target_height:
            return image.copy()

        # Calculate scaling ratios
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height

        # Choose best scaling method based on scaling ratio
        if width_ratio > 2.0 or height_ratio > 2.0:
            # Upscaling by more than 2x - use smoothscale
            scaled = pygame.transform.smoothscale(image, (target_width, target_height))
        elif width_ratio < 0.5 or height_ratio < 0.5:
            # Downscaling by more than 2x - use multi-step scaling for better quality
            scaled = self._multi_step_scale(image, target_width, target_height)
        else:
            # Normal scaling - use smoothscale
            scaled = pygame.transform.smoothscale(image, (target_width, target_height))

        return scaled

    def _multi_step_scale(self, image: pygame.Surface,
                          target_width: int, target_height: int) -> pygame.Surface:
        """Multi-step scaling for better quality when downscaling significantly"""
        current_image = image
        current_width, current_height = image.get_size()

        # Scale down in steps of no more than 50% at a time
        while current_width > target_width * 2 or current_height > target_height * 2:
            new_width = max(target_width, current_width // 2)
            new_height = max(target_height, current_height // 2)

            current_image = pygame.transform.smoothscale(current_image, (new_width, new_height))
            current_width, current_height = new_width, new_height

        # Final scaling to exact target size
        if current_width != target_width or current_height != target_height:
            current_image = pygame.transform.smoothscale(current_image, (target_width, target_height))

        return current_image

    def rescale_images(self, new_width: int, new_height: int) -> None:
        """Rescale all images to new dimensions using original high-res sources"""
        if not self.assets_loaded or not self.original_card_images:
            return

        print(f"Rescaling images to {new_width}x{new_height}")

        # Rescale card images
        for key, original_image in self.original_card_images.items():
            self.card_images[key] = self._scale_image_high_quality(
                original_image, new_width, new_height
            )

        # Rescale card back
        if self.original_card_back:
            self.card_back_image = self._scale_image_high_quality(
                self.original_card_back, new_width, new_height
            )

        self.current_scale = (new_width, new_height)

    def get_card_image(self, card: Card, width: Optional[int] = None,
                       height: Optional[int] = None) -> Optional[pygame.Surface]:
        """Get the image for a specific card, optionally at a custom size"""
        if not self.assets_loaded:
            return None

        key = self._card_to_key(card)

        # If custom size requested and we have the original
        if width and height and key in self.original_card_images:
            if (width, height) != self.current_scale:
                # Create custom-sized version from original
                original = self.original_card_images[key]
                return self._scale_image_high_quality(original, width, height)

        return self.card_images.get(key)

    def get_card_back_image(self, width: Optional[int] = None,
                            height: Optional[int] = None) -> Optional[pygame.Surface]:
        """Get the card back image, optionally at a custom size"""
        # If custom size requested and we have the original
        if width and height and self.original_card_back:
            if (width, height) != self.current_scale:
                return self._scale_image_high_quality(self.original_card_back, width, height)

        return self.card_back_image

    def get_image_info(self) -> Dict[str, any]:
        """Get information about loaded images"""
        info = {
            'loaded': self.assets_loaded,
            'current_scale': self.current_scale,
            'card_count': len(self.card_images),
            'original_count': len(self.original_card_images)
        }

        if self.original_card_images:
            # Get some sample original sizes
            sample_key = next(iter(self.original_card_images))
            original_size = self.original_card_images[sample_key].get_size()
            info['sample_original_size'] = original_size

        return info

    def _card_to_key(self, card: Card) -> str:
        """Convert card to image key"""
        if card.type == CardType.NUMBER:
            return f"{card.color.value}_{card.number}"
        elif card.type == CardType.SKIP:
            return f"{card.color.value}_skip"
        elif card.type == CardType.REVERSE:
            return f"{card.color.value}_reverse"
        elif card.type == CardType.DRAW_TWO:
            return f"{card.color.value}_draw_two"
        elif card.type == CardType.WILD:
            return "wild"
        elif card.type == CardType.WILD_DRAW_FOUR:
            return "wild_draw_four"

        return "unknown"