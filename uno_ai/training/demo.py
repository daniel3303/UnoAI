# uno_ai/training/demo.py
import argparse
from uno_ai.training.evaluate import UNOEvaluator
from uno_ai.environment.uno_env import UNOEnv

def main():
    parser = argparse.ArgumentParser(description="Demo UNO AI")
    parser.add_argument("--model-path", type=str, default="uno_ppo_final_model.pt",
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--render", action="store_true",
                        help="Render the game visually")

    args = parser.parse_args()

    if args.model_path and args.model_path != "none":
        # Use trained model
        evaluator = UNOEvaluator(args.model_path)
        evaluator.evaluate(num_episodes=args.episodes, render=args.render)
    else:
        # Use random agent
        env = UNOEnv(num_players=4, render_mode="human" if args.render else None)

        for episode in range(args.episodes):
            obs, _ = env.reset()
            episode_reward = 0

            while True:
                # Random valid action
                valid_actions = env.get_valid_actions()
                if valid_actions:
                    import random
                    action = random.choice(valid_actions)
                else:
                    action = 7  # Draw card

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if args.render:
                    env.render()

                if terminated or truncated:
                    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
                    break

        env.close()

if __name__ == "__main__":
    main()