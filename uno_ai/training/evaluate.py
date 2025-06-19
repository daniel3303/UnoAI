# ./uno_ai/training/evaluate.py
import torch
import numpy as np
from uno_ai.training.ppo_trainer import PPOAgent
from uno_ai.environment.uno_env import UNOEnv

class UNOEvaluator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = PPOAgent().to(self.device)
        self.load_model(model_path)
        self.agent.eval()

    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")

    def evaluate(self, num_episodes: int = 100, render: bool = False):
        """Evaluate the trained agent"""
        env = UNOEnv(num_players=4, render_mode="human" if render else None)

        total_rewards = []
        win_rates = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                # Get action from trained agent using token-based system
                obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(self.device)

                current_player = env.game.current_player
                action_mask, _ = env.create_action_mask(current_player)
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_token, _, _, _ = self.agent.get_action_and_value(obs_tensor, action_mask_tensor)

                action_token_item = action_token.item()

                # Environment now handles token conversion
                obs, reward, terminated, truncated, info = env.step(action_token_item)
                episode_reward += reward
                episode_length += 1

                if render:
                    env.render()

                if terminated or truncated:
                    total_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                    # Check if agent won (assuming player 0 is the agent)
                    if info.get('winner') == 0:
                        win_rates.append(1)
                    else:
                        win_rates.append(0)
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

def main():
    evaluator = UNOEvaluator("uno_ppo_final_model.pt")
    evaluator.evaluate(num_episodes=100, render=False)

if __name__ == "__main__":
    main()