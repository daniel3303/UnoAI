from uno_ai.training.parallel_multi_agent_ppo_trainer import ParallelMultiAgentTrainer
from uno_ai.training.ppo_trainer import PPOConfig

def main():
    """Entry point for parallel training with mode selection"""
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Train UNO AI with parallel environments")
    parser.add_argument("--multiprocess", action="store_true",
                        help="Use true multiprocessing (default: sequential)")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--buffer-size", type=int, default=2048,
                        help="Buffer size")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")

    args = parser.parse_args()

    # Check if MPS is available and warn about multiprocessing
    if args.multiprocess and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("⚠️  MPS (Apple Silicon GPU) detected with multiprocessing enabled.")
        print("   Training will use CPU in worker processes and GPU in main process.")
        print("   This is normal and expected behavior.")

    print(f"Mode: {'Multiprocessing' if args.multiprocess else 'Sequential'}")
    print(f"Environments: {args.num_envs}")

    config = PPOConfig(
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ppo_epochs=4
    )

    # Create trainer with mode selection
    trainer = ParallelMultiAgentTrainer(
        config,
        num_parallel_envs=args.num_envs,
        use_multiprocessing=args.multiprocess
    )

    trainer.train(total_timesteps=args.timesteps)

    model_name = f"uno_{'multiprocess' if args.multiprocess else 'sequential'}_final.pt"
    trainer.save_model(model_name)
    print(f"Training completed! Model saved as: {model_name}")

if __name__ == "__main__":
    main()