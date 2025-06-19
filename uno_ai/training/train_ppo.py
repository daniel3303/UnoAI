# uno_ai/training/train_ppo.py
from uno_ai.training.ppo_config import PPOConfig
from uno_ai.training.ppo_trainer import PPOTrainer

def main():
    # Configure training
    config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64,
        buffer_size=1024
    )

    # Create trainer
    trainer = PPOTrainer(config)

    # Train the model
    total_timesteps = 1_000_000  # Adjust based on your needs
    trainer.train(total_timesteps)

    # Save final model
    trainer.save_model("models/uno_ppo_final_model.pt")

if __name__ == "__main__":
    main()