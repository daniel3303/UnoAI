import os
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from uno_ai.environment.uno_env import UNOEnv
from uno_ai.training.ppo_config import PPOConfig
from uno_ai.training.reward_calculator import RewardCalculator


class PPOBuffer:
    def __init__(self, size: int, obs_dim: int, vocab_size: int):
        self.size = size
        self.obs_dim = obs_dim
        self.vocab_size = vocab_size
        self.reset()
    
    def reset(self):
        self.observations = np.zeros((self.size, self.obs_dim), dtype=np.int32)
        self.action_masks = np.zeros((self.size, self.vocab_size), dtype=np.bool_)
        self.actions = np.zeros(self.size, dtype=np.int32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.bool_)
        self.ptr = 0
        self.path_start_idx = 0
    
    def store(self, obs, action_mask, action, reward, value, log_prob, done):
        self.observations[self.ptr] = obs
        self.action_masks[self.ptr] = action_mask
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        return True  # Successfully stored
    
    def finish_path(self, last_value=0):
        """Calculate advantages and returns for the current trajectory"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
    
        # Calculate GAE advantages
        deltas = rewards[:-1] + 0.99 * values[1:] - values[:-1]
        advantages = self._discount_cumsum(deltas, 0.99 * 0.95)
    
        # Calculate returns
        returns = advantages + self.values[path_slice]
    
        # Store advantages and returns
        if not hasattr(self, 'advantages'):
            self.advantages = np.zeros(self.size, dtype=np.float32)
            self.returns = np.zeros(self.size, dtype=np.float32)
    
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
    
        self.path_start_idx = self.ptr
    
    def get(self):
        """Get all data and normalize advantages - handle partial fills"""
        # Use actual collected experiences, not the full buffer size
        actual_size = self.ptr
    
        if actual_size == 0:
            raise ValueError("Buffer is empty, cannot get data")
    
        # Only process the experiences we actually collected
        observations = self.observations[:actual_size]
        action_masks = self.action_masks[:actual_size]
        actions = self.actions[:actual_size]
        rewards = self.rewards[:actual_size]
        values = self.values[:actual_size]
        log_probs = self.log_probs[:actual_size]
        dones = self.dones[:actual_size]
    
        # Check if we have advantages computed
        if not hasattr(self, 'advantages') or self.advantages is None:
            print(f"Warning: No advantages computed, creating dummy advantages for {actual_size} experiences")
            self.advantages = np.zeros(self.size, dtype=np.float32)
            self.returns = self.values.copy()  # Simple fallback
    
        advantages = self.advantages[:actual_size]
        returns = self.returns[:actual_size] if hasattr(self, 'returns') and self.returns is not None else values
    
        # Normalize advantages
        if len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = np.zeros_like(advantages)
    
        data = dict(
            observations=torch.tensor(observations, dtype=torch.long),
            action_masks=torch.tensor(action_masks, dtype=torch.bool),
            actions=torch.tensor(actions, dtype=torch.long),
            returns=torch.tensor(returns, dtype=torch.float32),
            advantages=torch.tensor(advantages, dtype=torch.float32),
            log_probs=torch.tensor(log_probs, dtype=torch.float32),
            values=torch.tensor(values, dtype=torch.float32)
        )
    
        # Reset buffer
        self.reset()
        return data
    
    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sum"""
        return np.array([np.sum(discount**np.arange(len(x)-i) * x[i:]) for i in range(len(x))])


class PPOTrainer:
    def __init__(self, config: PPOConfig = PPOConfig()):
        self.config = config
        self.device = self._get_best_device()
        self.vocab_size = UNOTokens.VOCAB_SIZE

        # Initialize environment
        self.env = UNOEnv(num_players=4, render_mode=None)

        # Initialize agent
        self.agent = PPOAgent(vocab_size=self.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate)

        # Initialize buffer
        obs_dim = self.env.observation_space.shape[0]
        self.buffer = PPOBuffer(config.buffer_size, obs_dim, self.vocab_size)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()

        # Training metrics
        self.episode_rewards = deque(maxlen=1)
        self.episode_lengths = deque(maxlen=1)
        self.episode_wins = deque(maxlen=1)

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
        """Collect rollouts for training with custom reward calculation"""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        agent_episode_reward = 0  # Track agent-specific reward separately
    
        for _ in range(self.config.buffer_size):
            obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(self.device)
    
            current_player = self.env.game.current_player
    
            # Always track episode length
            episode_length += 1
    
            if current_player == 0:  # Agent's turn
                # Use environment's action mask method
                action_mask, token_to_hand_index = self.env.create_action_mask(current_player)
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
    
                with torch.no_grad():
                    action_token, log_prob, _, value = self.agent.get_action_and_value(obs_tensor, action_mask_tensor)
    
                action_token_item = action_token.item()
    
                # Take step
                next_obs, env_reward, terminated, truncated, info = self.env.step(action_token_item)
    
                if info['valid'] is False:
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
                    import random
                    action_token = random.choice(valid_actions)
                else:
                    action_token = UNOTokens.DRAW_ACTION
    
                next_obs, env_reward, terminated, truncated, info = self.env.step(action_token)
    
            # Always track total episode reward (for all players)
            episode_reward += env_reward
            done = terminated or truncated
            obs = next_obs
    
            if done:
                # Track wins
                winner = info.get('winner')
                is_win = winner == 0 if winner is not None else False  # Agent is player 0
                self.episode_wins.append(1.0 if is_win else 0.0)
    
                self.buffer.finish_path(0)
                self.episode_rewards.append(agent_episode_reward)  # Use agent-specific reward
                self.episode_lengths.append(episode_length)
    
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                agent_episode_reward = 0
    
            if self.buffer.ptr >= self.config.buffer_size:
                if not done and current_player == 0:
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
        self.vocab_size = checkpoint.get('vocab_size', UNOTokens.VOCAB_SIZE)
        print(f"Model loaded from {filepath}")