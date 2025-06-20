import numpy as np
import torch


class PPOBuffer:
    def __init__(self, size: int, obs_dim: int, vocab_size: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.size = size
        self.obs_dim = obs_dim
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
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

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)

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
