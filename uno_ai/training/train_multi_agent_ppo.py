import argparse
import logging
from uno_ai.training.multi_agent_ppo_trainer import MultiAgentPPOTrainer
from uno_ai.training.ppo_config import PPOConfig

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Agent UNO PPO")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Buffer size for rollouts")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Number of PPO epochs per update")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save-path", type=str, default="models/uno_multi_agent_final.pt", help="Path to save the final model")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs
    )

    trainer = MultiAgentPPOTrainer(config)

    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_model(args.load_checkpoint)

    trainer.train(total_timesteps=args.timesteps)
    trainer.save_model(args.save_path)

if __name__ == "__main__":
    main()