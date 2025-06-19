import argparse
import random
import time

from uno_ai.environment.uno_vocabulary import UNOVocabulary
from uno_ai.evaluation.uno_evaluator import UNOEvaluator
from uno_ai.environment.uno_env import UNOEnv


def main():
    parser = argparse.ArgumentParser(description="Demo UNO AI")
    parser.add_argument("--num-players", type=int, default=4,
                        help="Number of players (2-4)")
    parser.add_argument("--model-paths", type=str, nargs='+', default=None,
                        help="Paths to trained models for each player (use 'none' for environment players)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--render", action="store_true",
                        help="Render the game visually")
    parser.add_argument("--delay", type=float, default=1,
                        help="Delay in seconds between moves (default: 1)")

    args = parser.parse_args()

    # Process model paths - convert 'none'/'environment' to None
    model_paths = []
    for path in args.model_paths:
        if path.lower() in ['none', 'environment']:
            model_paths.append(None)
        else:
            model_paths.append(path)

    # Pad or truncate to match num_players
    while len(model_paths) < args.num_players:
        model_paths.append(None)
    model_paths = model_paths[:args.num_players]

    # Use evaluator with configured players
    evaluator = UNOEvaluator(num_players=args.num_players, model_paths=model_paths)
    evaluator.evaluate(num_episodes=args.episodes, render=args.render, delay=args.delay)


if __name__ == "__main__":
    main()