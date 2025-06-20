import os
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from uno_ai.agents.ppo_agent import PPOAgent
from uno_ai.environment.uno_env import UNOEnv
from uno_ai.environment.uno_game import GameMode
from uno_ai.environment.uno_vocabulary import UNOVocabulary
from uno_ai.training.ppo_buffer import PPOBuffer
from uno_ai.training.ppo_config import PPOConfig
from uno_ai.training.reward_calculator import RewardCalculator



class PPOTrainer:
    def __init__(self, config: PPOConfig = PPOConfig(), env: Optional[UNOEnv] = None):
        self.config = config
        self.device = self._get_best_device()
        self.vocab_size = UNOVocabulary.VOCAB_SIZE

        # Initialize environment with random game modes
        self.env = UNOEnv(num_players=4, game_mode=GameMode.NORMAL, render_mode=None) if env is None else env

        # Initialize agent
        self.agent = PPOAgent(vocab_size=self.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate)

        # Initialize buffer
        obs_dim = self.env.observation_space.shape[0]
        
        self.buffer = PPOBuffer(config.buffer_size, obs_dim, self.vocab_size)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()

        # Training metrics
        self.episode_rewards = deque(maxlen=10)
        self.episode_lengths = deque(maxlen=10)
        self.episode_wins = deque(maxlen=10)

        # Add agent position tracking
        self.agent_player_id = 0  # Which player the agent is controlling

    def _get_best_device(self):
        """Select the best available device with MPS support"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

        return device

    def count_parameters(self):
        """Count total trainable parameters"""
        total_params = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        return total_params
    
    def collect_rollouts(self):
        """Collect rollouts for training with custom reward calculation and random agent positions"""
        import random
        from uno_ai.environment.uno_game import GameMode
    
        # Randomly assign agent to different positions and game modes
        self.agent_player_id = random.randint(0, 3)
        game_mode = random.choice([GameMode.NORMAL, GameMode.STREET])
    
        # Reset environment with new configuration
        self.env.game_mode = game_mode
        obs, _ = self.env.reset()
    
        episode_reward = 0
        episode_length = 0
        agent_episode_reward = 0
    
        for _ in range(self.config.buffer_size):
            obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(self.device)
    
            current_player = self.env.game.current_player
            episode_length += 1
    
            if current_player == self.agent_player_id:  # Agent's turn
                # Use environment's action mask method
                action_mask, token_to_hand_index = self.env.create_action_mask(current_player)
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
    
                with torch.no_grad():
                    action_token, log_prob, _, value = self.agent.get_action_and_value(obs_tensor, action_mask_tensor)
    
                action_token_item = action_token.item()
    
                # Take step
                next_obs, env_reward, terminated, truncated, info = self.env.step(action_token_item)
    
                if info.get('valid', True) is False:
                    env_reward += self.reward_calculator.rewards['invalid_action']
    
                # Calculate custom reward based on game state
                reward = self.reward_calculator.calculate_reward(info, env_reward)
    
                # Store experience
                self.buffer.store(
                    obs, action_mask, action_token_item, reward,
                    value.item(), log_prob.item(), terminated or truncated
                )
    
                agent_episode_reward += reward
    
            else:
                # Other players - use random/heuristic actions
                valid_actions = self.env.get_valid_actions()
                if valid_actions:
                    action_token = random.choice(valid_actions)
                else:
                    action_token = UNOVocabulary.DRAW_ACTION
    
                next_obs, env_reward, terminated, truncated, info = self.env.step(action_token)
    
            # Always track total episode reward (for all players)
            episode_reward += env_reward
            done = terminated or truncated
            obs = next_obs
    
            if done:
                # Track wins for the agent
                winner = info.get('winner')
                is_win = winner == self.agent_player_id if winner is not None else False
                self.episode_wins.append(1.0 if is_win else 0.0)
    
                self.buffer.finish_path(0)
                self.episode_rewards.append(agent_episode_reward)
                self.episode_lengths.append(episode_length)
    
                # Start new episode with new random position and game mode
                self.agent_player_id = random.randint(0, 3)
                game_mode = random.choice([GameMode.NORMAL, GameMode.STREET])
                self.env.game_mode = game_mode
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                agent_episode_reward = 0
    
            if self.buffer.ptr >= self.config.buffer_size:
                if not done and current_player == self.agent_player_id:
                    obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(self.device)
                    action_mask, _ = self.env.create_action_mask(current_player)
                    action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
    
                    with torch.no_grad():
                        _, _, _, last_value = self.agent.get_action_and_value(obs_tensor, action_mask_tensor)
                    self.buffer.finish_path(last_value.item())
                break

    def update_policy(self, data: Dict[str, torch.Tensor]):
        """Update policy using PPO"""
        observations = data['observations'].to(self.device)
        action_masks = data['action_masks'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        returns = data['returns'].to(self.device)
        advantages = data['advantages'].to(self.device)
        old_values = data['values'].to(self.device)

        # Convert to batches
        batch_size = self.config.batch_size
        indices = torch.randperm(len(observations))

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.config.ppo_epochs):
            for start in range(0, len(observations), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_obs = observations[batch_indices]
                batch_action_masks = action_masks[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]

                # Get new policy values
                _, new_log_probs, entropy, new_values = self.agent.get_action_and_value(
                    batch_obs, batch_action_masks, batch_actions
                )

                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate policy loss
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Calculate value loss
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # Calculate entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                total_loss = (
                        policy_loss +
                        self.config.value_loss_coef * value_loss -
                        self.config.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

        return {
            'policy_loss': total_policy_loss,
            'value_loss': total_value_loss,
            'entropy_loss': total_entropy_loss
        }

    def train(self, total_timesteps: int):
        """Main training loop with enhanced logging"""
        timesteps_collected = 0
        update_count = 0

        # Show model info before training
        total_params = self.count_parameters()
        print(f"Starting PPO training for {total_timesteps:,} timesteps")
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Buffer size: {self.config.buffer_size}")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Device: {self.device}")
        print("-" * 80)

        while timesteps_collected < total_timesteps:
            self.collect_rollouts()
            timesteps_collected += self.config.buffer_size

            data = self.buffer.get()
            losses = self.update_policy(data)
            update_count += 1

            # Log training progress
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            win_rate = np.mean(self.episode_wins) if self.episode_wins else 0

            print(f"Update {update_count:4d} | Steps: {timesteps_collected:8,}/{total_timesteps:,} | "
                  f"Reward: {avg_reward:6.2f} | Length: {avg_length:5.1f} | WinRate: {win_rate:.3f} | "
                  f"PolicyLoss: {losses['policy_loss']:6.4f} | ValueLoss: {losses['value_loss']:6.4f} | "
                  f"Entropy: {losses['entropy_loss']:6.4f}")

            if update_count % 10 == 0:
                self.save_model(f"checkpoints/uno_ppo_model_update_{update_count}.pt")

    def save_model(self, filepath: str):
        """Save model checkpoint"""

        # Create the parent directory if it doesn't exist
        if os.path.dirname(filepath) and not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False) # Allow more than weights because config claasses are also saved in the checkpoint
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab_size = checkpoint.get('vocab_size', UNOVocabulary.VOCAB_SIZE)
        print(f"Model loaded from {filepath}")