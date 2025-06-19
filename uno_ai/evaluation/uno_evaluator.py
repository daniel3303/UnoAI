import time
from typing import List, Optional

import torch
import numpy as np

from uno_ai.agents.ppo_agent import PPOAgent
from uno_ai.environment.multi_agent_uno_env import MultiAgentUNOEnv, OpponentConfig
from uno_ai.environment.uno_game import GameMode


class UNOEvaluator:
    def __init__(self, num_players: int = 4, game_mode: GameMode = GameMode.NORMAL, model_paths: Optional[List[str]] = None):
        self.game_mode = game_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_players = num_players
        self.agents = {}

        # Load models for each player if provided
        if model_paths:
            for player_id, model_path in enumerate(model_paths):
                if model_path and player_id < num_players:
                    self.load_agent_for_player(player_id, model_path)

    def load_agent_for_player(self, player_id: int, model_path: str):
        """Load trained model for a specific player"""
        agent = PPOAgent().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        self.agents[player_id] = agent
        print(f"Model loaded for player {player_id} from {model_path}")

    def evaluate(self, num_episodes: int = 100, render: bool = False, delay: float = 1):
        """Evaluate the configured players"""
        # Use multi-agent environment
        env = MultiAgentUNOEnv(num_players=self.num_players, game_mode=self.game_mode, render_mode="human" if render else None)

        # Configure which players use trained agents vs environment players
        agent_players = list(self.agents.keys())
        env_players = [i for i in range(self.num_players) if i not in agent_players]

        opponent_config = OpponentConfig(
            agent_players=agent_players,
            env_players=env_players,
            random_players=[]
        )
        env.set_opponent_config(opponent_config)

        # Add trained agents to environment
        for player_id, agent in self.agents.items():
            env.add_trained_agent(player_id, agent)

        total_rewards = []
        win_rates = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                current_player = env.game.current_player
                action_token = env.get_action_for_player(current_player, obs)

                obs, reward, terminated, truncated, info = env.step(action_token)

                if current_player == 0:  # Track player 0
                    episode_reward += reward

                episode_length += 1

                if render:
                    env.render()
                    time.sleep(delay)

                if terminated or truncated:
                    total_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    win_rates.append(1 if info.get('winner') == 0 else 0)
                    break

            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")

        # Print results
        avg_reward = np.mean(total_rewards)
        win_rate = np.mean(win_rates) * 100
        avg_length = np.mean(episode_lengths)

        print(f"\nEvaluation Results over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Episode Length: {avg_length:.1f}")

        env.close()
        return avg_reward, win_rate, avg_length