# test_uno_game.py
import time

from uno_ai.environment.uno_env import UNOEnv


def test_game():
    env = UNOEnv(num_players=3, render_mode="human")
    obs, _ = env.reset()

    print("Starting UNO game!")

    for step in range(1000):  # Max steps
        # Simple random policy for testing
        action = env.action_space.sample()

        # Prefer playing cards over drawing if possible
        if env.game:
            valid_actions = env.game.get_valid_actions(env.game.current_player)
            if valid_actions:
                action = valid_actions[0]  # Play first valid card
            else:
                action = 7  # Draw card

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        time.sleep(4)  # Slow down for visualization

        if terminated or truncated:
            print(f"Game over! Winner: {info.get('winner', 'None')}")
            break

    env.close()

if __name__ == "__main__":
    test_game()