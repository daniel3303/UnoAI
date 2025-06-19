from uno_ai.training.multi_agent_ppo_trainer import MultiAgentPPOTrainer
from uno_ai.training.ppo_trainer import PPOConfig

def main():
    config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=2048,  # Larger buffer for multi-agent
        batch_size=64,
        ppo_epochs=4
    )

    trainer = MultiAgentPPOTrainer(config)
    trainer.train(total_timesteps=4_000_000)
    trainer.save_model("uno_multi_agent_final.pt")

if __name__ == "__main__":
    main()