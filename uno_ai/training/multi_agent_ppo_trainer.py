import logging
import random
from collections import deque

import numpy as np
import torch

from uno_ai.agents.ppo_agent import PPOAgent
from uno_ai.environment.multi_agent_uno_env import MultiAgentUNOEnv, OpponentConfig
from uno_ai.environment.uno_game import GameMode
from uno_ai.environment.uno_vocabulary import UNOVocabulary
from uno_ai.training.multi_agent_config import MultiAgentTrainingConfig, TrainingScenario
from uno_ai.training.ppo_config import PPOConfig
from uno_ai.training.ppo_trainer import PPOTrainer, PPOBuffer, RewardCalculator

logger = logging.getLogger(__name__)



class MultiAgentPPOTrainer(PPOTrainer):
    def __init__(self, config: PPOConfig = PPOConfig()):
        # Initialize base trainer but replace environment
        super().__init__(config)
        self.vocab_size = UNOVocabulary.VOCAB_SIZE

        # Use multi-agent environment
        self.current_num_players = 4  # Default
        self.env = MultiAgentUNOEnv(num_players=self.current_num_players, render_mode=None)

        # Training configuration
        self.training_config = MultiAgentTrainingConfig()

        # Multiple agent instances for self-play
        self.agents = {}
        self.optimizers = {}

        # Create primary agent (the one being trained)
        self.primary_agent_id = 0
        self._create_agent(self.primary_agent_id, is_primary=True)

        # Initialize buffer and other components
        obs_dim = self.env.observation_space.shape[0]
        self.buffer = PPOBuffer(config.buffer_size, obs_dim, self.vocab_size)
        self.reward_calculator = RewardCalculator()

        # Training metrics
        self.episode_rewards = deque(maxlen=10)
        self.episode_lengths = deque(maxlen=10)
        self.episode_wins = deque(maxlen=10)
        self.scenario_stats = {}  # This will track {scenario_name: {'episodes': [], 'wins': []}}


    def _create_agent(self, agent_id: int, is_primary: bool = False):
        """Create an agent instance"""
        agent = PPOAgent(vocab_size=self.vocab_size).to(self.device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=self.config.learning_rate)

        self.agents[agent_id] = agent
        self.optimizers[agent_id] = optimizer

        if is_primary:
            self.agent = agent  # Keep reference for compatibility
            self.optimizer = optimizer

    def _setup_scenario(self, scenario: TrainingScenario):
        """Setup environment for a specific training scenario"""
        # Recreate environment if player count changed
        if scenario.num_players != self.current_num_players:
            self.current_num_players = scenario.num_players
            self.env = MultiAgentUNOEnv(num_players=self.current_num_players, render_mode=None)

        # Validate scenario configuration
        all_players = set(range(scenario.num_players))
        scenario_players = set(scenario.agent_players + scenario.random_players + scenario.env_players)

        if scenario_players != all_players:
            raise ValueError(f"Scenario {scenario.name} doesn't assign all players. "
                             f"Expected: {all_players}, Got: {scenario_players}")

        # Create opponent config
        opponent_config = OpponentConfig(
            agent_players=scenario.agent_players,
            random_players=scenario.random_players,
            env_players=scenario.env_players
        )

        self.env.set_opponent_config(opponent_config)

        # Add required agent instances
        for player_id in scenario.agent_players:
            if player_id not in self.agents:
                self._create_agent(player_id)
            self.env.add_trained_agent(player_id, self.agents[player_id])


    def collect_rollouts(self):
        """Enhanced rollout collection with scenario sampling and random agent positions"""
        try:
            # Sample training scenario
            scenario = self.training_config.sample_scenario()
            self._setup_scenario(scenario)
    
            # Randomly assign primary agent to different positions
            self.primary_agent_id = random.choice(scenario.agent_players) if scenario.agent_players else random.randint(0, scenario.num_players - 1)
    
            # Randomly choose game mode
            game_mode = random.choice([GameMode.NORMAL, GameMode.STREET])
            self.env.game_mode = game_mode
    
            # Initialize scenario tracking if needed
            if scenario.name not in self.scenario_stats:
                self.scenario_stats[scenario.name] = {'episodes': [], 'wins': []}
    
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            consecutive_same_player = 0
            last_current_player = -1
    
            for step in range(self.config.buffer_size):
                if not self.env.game or self.env.game.game_over:
                    logger.debug("Game over or no game, resetting...")
                    obs, _ = self.env.reset()
                    continue
    
                current_player = self.env.game.current_player
    
                # Safety check for infinite loops
                if current_player == last_current_player:
                    consecutive_same_player += 1
                    if consecutive_same_player > 20:
                        logger.debug(f"Stuck on player {current_player} for {consecutive_same_player} steps! Resetting game.")
                        obs, _ = self.env.reset()
                        episode_reward = 0
                        episode_length = 0
                        consecutive_same_player = 0
                        continue
                else:
                    consecutive_same_player = 0
    
                last_current_player = current_player
    
                # Only collect experience for primary agent
                if current_player == self.primary_agent_id:
                    try:
                        obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(self.device)
                        action_mask, token_to_hand_index = self.env.create_action_mask(current_player)
                        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
    
                        with torch.no_grad():
                            action_token, log_prob, _, value = self.agents[self.primary_agent_id].get_action_and_value(
                                obs_tensor, action_mask_tensor
                            )
    
                        # Use the token directly as environment action
                        env_action = action_token.item()
                        buffer_action = action_token.item()
                    except Exception as e:
                        print(f"Error getting primary agent action: {e}")
                        env_action = UNOVocabulary.DRAW_ACTION
                        buffer_action = UNOVocabulary.DRAW_ACTION
    
                else:
                    # Get action from the appropriate opponent
                    try:
                        env_action = self.env.get_action_for_player(current_player, obs)
                    except Exception as e:
                        logger.debug(f"Error getting opponent action: {e}")
                        env_action = UNOVocabulary.DRAW_ACTION
    
                    # Initialize variables to avoid reference errors
                    action_mask = None
                    buffer_action = None
                    value = None
                    log_prob = None
    
                # Take step
                try:
                    next_obs, env_reward, terminated, truncated, info = self.env.step(env_action)
                    done = terminated or truncated
                except Exception as e:
                    logger.debug(f"Error taking step: {e}")
                    # Reset and continue
                    obs, _ = self.env.reset()
                    continue
    
                # Only store experience for primary agent
                if current_player == self.primary_agent_id and action_mask is not None:
                    reward = self.reward_calculator.calculate_reward(info, env_reward)
    
                    self.buffer.store(
                        obs, action_mask, buffer_action, reward,
                        value.item(), log_prob.item(), done
                    )
    
                    logger.debug(f"Agent {current_player} took action {UNOVocabulary.token_to_card_info(action_token)}")
                    logger.debug(f"Stored reply for player {current_player} with reward {reward}. Valid: {info.get('valid', True)} | Hand size: {info.get('current_hand_size', 0)} | Previous hand size: {info.get('prev_hand_size', 0)}")
                    episode_reward += reward
    
                episode_length += 1
                obs = next_obs
    
                if done:
                    # Track wins for primary agent
                    winner = info.get('winner')
                    logger.debug(f"Winner is: {winner} (Primary agent: {self.primary_agent_id})")
                    is_win = winner == self.primary_agent_id if winner is not None else False
    
                    # Record this episode for the current scenario
                    self.scenario_stats[scenario.name]['episodes'].append(1)
                    self.scenario_stats[scenario.name]['wins'].append(1 if is_win else 0)
    
                    # Also track in the general deques
                    self.episode_wins.append(1.0 if is_win else 0.0)
    
                    self.buffer.finish_path(0)
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
    
                    # Start new episode with potentially different scenario and agent position
                    scenario = self.training_config.sample_scenario()
                    self._setup_scenario(scenario)
    
                    # Randomly assign primary agent to different positions
                    self.primary_agent_id = random.choice(scenario.agent_players) if scenario.agent_players else random.randint(0, scenario.num_players - 1)
    
                    # Randomly choose game mode
                    self.env.game_mode = game_mode
    
                    # Initialize new scenario if needed
                    if scenario.name not in self.scenario_stats:
                        self.scenario_stats[scenario.name] = {'episodes': [], 'wins': []}
    
                    game_mode = random.choice([GameMode.NORMAL, GameMode.STREET])
                    obs, _ = self.env.reset(game_mode=game_mode)
                    episode_reward = 0
                    episode_length = 0
    
                if self.buffer.ptr >= self.config.buffer_size:
                    # Store current episode metrics even if not done
                    if episode_reward != 0 or episode_length > 0:
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
    
                    if not done and current_player == self.primary_agent_id:
                        try:
                            obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(self.device)
                            action_mask, _ = self.env.create_action_mask(current_player)
                            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
    
                            with torch.no_grad():
                                _, _, _, last_value = self.agents[self.primary_agent_id].get_action_and_value(
                                    obs_tensor, action_mask_tensor
                                )
                            self.buffer.finish_path(last_value.item())
                        except Exception as e:
                            print(f"Error getting last value: {e}")
                            self.buffer.finish_path(0)
                    break
    
        except Exception as e:
            print(f"Critical error in collect_rollouts: {e}")
            # Emergency reset
            self.env.reset()
            raise

    def train(self, total_timesteps: int):
        """Enhanced training with scenario tracking"""
        timesteps_collected = 0
        update_count = 0

        # Print scenario distribution
        self.training_config.print_scenario_distribution()

        print(f"\nStarting Multi-Agent PPO training for {total_timesteps:,} timesteps")
        print(f"Total scenarios: {len(self.training_config.scenarios)}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.agent.parameters() if p.requires_grad):,}")
        print("-" * 80)

        while timesteps_collected < total_timesteps:
            self.collect_rollouts()
            timesteps_collected += self.config.buffer_size

            data = self.buffer.get()
            losses = self.update_policy(data)
            update_count += 1
            
            # Log training progress with scenario stats
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            win_rate = np.mean(self.episode_wins) if self.episode_wins else 0
            
            print(f"Update {update_count:4d} | Steps: {timesteps_collected:8,}/{total_timesteps:,} | "
                  f"Players: {self.current_num_players} | "
                  f"Reward: {avg_reward:6.2f} | Game Length: {avg_length:5.1f} | Win Rate: {win_rate:.3f}")
            
            # Print scenario statistics every 40 updates
            if update_count % 40 == 0:
                self._print_scenario_stats()
            
            if update_count % 40 == 0:
                self.save_model(f"checkpoints/uno_multi_agent_model_update_{update_count}.pt")
    
    def _print_scenario_stats(self):
        """Print statistics for each training scenario"""
        print("\nScenario Statistics:")
        print("-" * 60)

        scenarios_player_counts = {s.num_players for s in self.training_config.scenarios}
        
        # Group by player count
        for player_count in scenarios_player_counts:
            scenarios = self.training_config.get_scenarios_by_player_count(player_count)
            scenario_names = [s.name for s in scenarios]

            if any(name in self.scenario_stats for name in scenario_names):
                print(f"\n{player_count}-Player Games:")
                for scenario_name in scenario_names:
                    if scenario_name in self.scenario_stats:
                        stats = self.scenario_stats[scenario_name]
                        total_episodes = len(stats['episodes'])
                        total_wins = sum(stats['wins'])

                        if total_episodes > 0:
                            win_rate = (total_wins / total_episodes) * 100
                            print(f"  {scenario_name:25} | {total_episodes:3d} episodes | {win_rate:5.1f}% wins")
        print()
