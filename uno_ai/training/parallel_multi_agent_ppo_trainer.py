# ./uno_ai/training/parallel_multi_agent_trainer.py
import torch
import numpy as np
import multiprocessing as mp
from collections import deque
from typing import Dict, List, Tuple, Any
import pickle
import time
from queue import Empty

from uno_ai.environment.multi_agent_uno_env import MultiAgentUNOEnv, OpponentConfig
from uno_ai.training.multi_agent_ppo_trainer import MultiAgentPPOTrainer
from uno_ai.training.ppo_trainer import PPOConfig, PPOBuffer, RewardCalculator, PPOAgent
from uno_ai.training.multi_agent_config import MultiAgentTrainingConfig
from uno_ai.model.uno_transformer import UNOTokens

class Experience:
    """Serializable experience for multiprocessing"""
    def __init__(self, obs, action_mask, action, reward, value, log_prob, done):
        self.obs = obs
        self.action_mask = action_mask
        self.action = action
        self.reward = reward
        self.value = value
        self.log_prob = log_prob
        self.done = done

class WorkerRequest:
    """Request sent to worker process"""
    def __init__(self, request_type: str, data: Any = None):
        self.type = request_type  # 'step', 'reset', 'get_action', 'close'
        self.data = data

class WorkerResponse:
    """Response from worker process"""
    def __init__(self, response_type: str, data: Any = None, success: bool = True, error: str = None):
        self.type = response_type
        self.data = data
        self.success = success
        self.error = error

def worker_process(worker_id: int, request_queue: mp.Queue, response_queue: mp.Queue,
                   agent_state_dict_cpu: Dict, config_dict: Dict):
    """Worker process that runs a single environment"""
    try:
        # Set up environment in worker process
        env = MultiAgentUNOEnv(num_players=4, render_mode=None)
        training_config = MultiAgentTrainingConfig()
        reward_calculator = RewardCalculator()

        # Create local agent for action generation (ALWAYS use CPU in workers)
        device = torch.device("cpu")  # Force CPU in workers
        vocab_size = UNOTokens.VOCAB_SIZE
        agent = PPOAgent(vocab_size=vocab_size).to(device)

        # Load state dict (already on CPU)
        agent.load_state_dict(agent_state_dict_cpu)
        agent.eval()

        # Worker state
        current_obs = None
        episode_reward = 0.0
        episode_length = 0
        current_scenario = None

        print(f"Worker {worker_id} started on CPU")

        while True:
            try:
                # Get request from main process
                request = request_queue.get(timeout=1.0)

                if request.type == 'close':
                    break

                elif request.type == 'reset':
                    # Reset environment with new scenario
                    scenario = training_config.sample_scenario()
                    current_scenario = scenario

                    # Setup environment
                    if scenario.num_players != env.num_players:
                        env = MultiAgentUNOEnv(num_players=scenario.num_players, render_mode=None)

                    opponent_config = OpponentConfig(
                        agent_players=scenario.agent_players,
                        random_players=scenario.random_players,
                        env_players=scenario.env_players
                    )
                    env.set_opponent_config(opponent_config)

                    # Add agent for self-play scenarios
                    for player_id in scenario.agent_players:
                        env.add_trained_agent(player_id, agent)

                    obs, info = env.reset()
                    current_obs = obs
                    episode_reward = 0.0
                    episode_length = 0

                    response = WorkerResponse('reset', {
                        'obs': obs,
                        'scenario': scenario.name,
                        'num_players': scenario.num_players
                    })
                    response_queue.put(response)

                elif request.type == 'step':
                    if env.game and not env.game.game_over:
                        current_player = env.game.current_player
                        primary_agent_id = 0  # Assuming primary agent is always player 0

                        if current_player == primary_agent_id:
                            # Get action from agent
                            obs_tensor = torch.tensor(current_obs, dtype=torch.long).unsqueeze(0).to(device)
                            action_mask, token_mapping = env._create_action_mask_for_player(current_player)
                            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)

                            with torch.no_grad():
                                action_token, log_prob, _, value = agent.get_action_and_value(
                                    obs_tensor, action_mask_tensor
                                )

                            # Convert to environment action
                            if action_token.item() == UNOTokens.DRAW_ACTION:
                                env_action = 7
                            elif action_token.item() in token_mapping:
                                env_action = token_mapping[action_token.item()]
                            else:
                                env_action = 7

                            # Take step
                            next_obs, env_reward, terminated, truncated, info = env.step(env_action)
                            done = terminated or truncated

                            # Calculate reward
                            reward = reward_calculator.calculate_reward(info, env_reward)
                            episode_reward += reward
                            episode_length += 1

                            # Create experience
                            experience = Experience(
                                obs=current_obs,
                                action_mask=action_mask,
                                action=action_token.item(),
                                reward=reward,
                                value=value.item(),
                                log_prob=log_prob.item(),
                                done=done
                            )

                            current_obs = next_obs

                            response_data = {
                                'experience': experience,
                                'done': done,
                                'episode_reward': episode_reward if done else None,
                                'episode_length': episode_length if done else None,
                                'winner': info.get('winner') if done else None,
                                'is_primary_turn': True,
                                'scenario_name': current_scenario.name if current_scenario else None
                            }

                        else:
                            # Opponent turn - just step environment
                            env_action = env.get_action_for_player(current_player, current_obs)
                            next_obs, _, terminated, truncated, info = env.step(env_action)
                            done = terminated or truncated
                            episode_length += 1

                            current_obs = next_obs

                            response_data = {
                                'experience': None,
                                'done': done,
                                'episode_reward': episode_reward if done else None,
                                'episode_length': episode_length if done else None,
                                'winner': info.get('winner') if done else None,
                                'is_primary_turn': False,
                                'scenario_name': current_scenario.name if current_scenario else None
                            }

                        response = WorkerResponse('step', response_data)
                    else:
                        # Game over, need reset
                        response = WorkerResponse('step', {'done': True, 'needs_reset': True})

                    response_queue.put(response)

                elif request.type == 'update_agent':
                    # Update agent weights (state dict should be on CPU)
                    agent_state_dict_cpu = request.data
                    agent.load_state_dict(agent_state_dict_cpu)
                    response = WorkerResponse('update_agent', {'success': True})
                    response_queue.put(response)

            except Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                response = WorkerResponse('error', error=str(e), success=False)
                response_queue.put(response)

    except Exception as e:
        print(f"Worker {worker_id} failed to initialize: {e}")
    finally:
        print(f"Worker {worker_id} terminated")

class ParallelMultiAgentTrainer(MultiAgentPPOTrainer):
    def __init__(self, config: PPOConfig = PPOConfig(), num_parallel_envs: int = 8, use_multiprocessing: bool = True):
        # Store multiprocessing flag
        self.use_multiprocessing = use_multiprocessing
        self.num_parallel_envs = num_parallel_envs

        if use_multiprocessing:
            # Initialize for multiprocessing mode
            self.config = config
            self.device = self._get_best_device()
            self.vocab_size = UNOTokens.VOCAB_SIZE

            # Training configuration
            self.training_config = MultiAgentTrainingConfig()

            # Create primary agent (the one being trained)
            self.primary_agent_id = 0
            self.agents = {}
            self.optimizers = {}
            self._create_agent(self.primary_agent_id, is_primary=True)

            # Initialize buffer and other components
            obs_dim = 1000  # Fixed size for UNO observations
            self.buffer = PPOBuffer(config.buffer_size, obs_dim, self.vocab_size)
            self.reward_calculator = RewardCalculator()

            # Training metrics
            self.episode_rewards = deque(maxlen=100)
            self.episode_lengths = deque(maxlen=100)
            self.episode_wins = deque(maxlen=100)
            self.scenario_stats = {}

            # Multiprocessing setup
            self.workers = []
            self.request_queues = []
            self.response_queues = []

            # Set multiprocessing start method to avoid MPS issues
            self._setup_multiprocessing()
            self._start_workers()
        else:
            # Use original sequential implementation
            super().__init__(config)

            # Override with sequential environments
            self.envs = [
                MultiAgentUNOEnv(num_players=4, render_mode=None)
                for _ in range(num_parallel_envs)
            ]
            self.env = self.envs[0]  # Keep reference for compatibility

            # Environment states for sequential processing
            self.env_states = [None] * num_parallel_envs
            self.env_episode_rewards = [0.0] * num_parallel_envs
            self.env_episode_lengths = [0] * num_parallel_envs
            self.env_current_scenarios = [None] * num_parallel_envs

        print(f"Initialized {'multiprocess' if use_multiprocessing else 'sequential'} trainer with {num_parallel_envs} environments")

    def _setup_multiprocessing(self):
        """Setup multiprocessing to work with MPS/GPU"""
        try:
            # Set start method to 'spawn' to avoid sharing GPU tensors
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'")
        except RuntimeError:
            # Start method already set
            print(f"Multiprocessing start method: {mp.get_start_method()}")


    def _get_cpu_state_dict(self, agent):
        """Get agent state dict moved to CPU for serialization"""
        state_dict = agent.state_dict()
        cpu_state_dict = {}
        for key, tensor in state_dict.items():
            cpu_state_dict[key] = tensor.cpu()
        return cpu_state_dict


    def _start_workers(self):
        """Start worker processes (multiprocessing mode only)"""
        if not self.use_multiprocessing:
            return

        print(f"Starting {self.num_parallel_envs} worker processes...")

        for i in range(self.num_parallel_envs):
            request_queue = mp.Queue()
            response_queue = mp.Queue()

            # Get agent state dict on CPU for serialization
            agent_state_dict_cpu = self._get_cpu_state_dict(self.agents[self.primary_agent_id])
            config_dict = {
                'buffer_size': self.config.buffer_size,
                'vocab_size': self.vocab_size
            }

            worker = mp.Process(
                target=worker_process,
                args=(i, request_queue, response_queue, agent_state_dict_cpu, config_dict)
            )
            worker.start()

            self.workers.append(worker)
            self.request_queues.append(request_queue)
            self.response_queues.append(response_queue)

        # Reset all workers
        self._reset_all_workers()
        print(f"All {self.num_parallel_envs} workers ready")


    def _reset_all_workers(self):
        """Reset all worker environments (multiprocessing mode only)"""
        if not self.use_multiprocessing:
            return

        # Send reset requests
        for request_queue in self.request_queues:
            request_queue.put(WorkerRequest('reset'))

        # Wait for reset responses
        for i, response_queue in enumerate(self.response_queues):
            try:
                response = response_queue.get(timeout=5.0)
                if response.success:
                    print(f"Worker {i} reset with scenario: {response.data['scenario']}")
                else:
                    print(f"Worker {i} reset failed: {response.error}")
            except Empty:
                print(f"Worker {i} reset timeout")

    def _update_worker_agents(self):
        """Update agent weights in all workers (multiprocessing mode only)"""
        if not self.use_multiprocessing:
            return

        # Get CPU version of state dict for workers
        agent_state_dict_cpu = self._get_cpu_state_dict(self.agents[self.primary_agent_id])

        for request_queue in self.request_queues:
            request_queue.put(WorkerRequest('update_agent', agent_state_dict_cpu))


    def collect_rollouts(self):
        """Collect rollouts using either multiprocessing or sequential mode"""
        if self.use_multiprocessing:
            self._collect_rollouts_multiprocess()
        else:
            self._collect_rollouts_sequential()

    def _collect_rollouts_multiprocess(self):
        """Collect experiences from multiple worker processes"""
        experiences_collected = 0
        max_steps = self.config.buffer_size * 5  # Increase safety limit
        step_count = 0
    
        print(f"Starting collection, target: {self.config.buffer_size} experiences")
    
        for step in range(max_steps):
            step_count += 1
    
            if self.buffer.is_full():
                print(f"Buffer full at step {step}, collected {experiences_collected} experiences")
                break
    
            # Send step requests to all workers
            active_workers = 0
            for i, request_queue in enumerate(self.request_queues):
                try:
                    request_queue.put(WorkerRequest('step'))
                    active_workers += 1
                except Exception as e:
                    print(f"Failed to send request to worker {i}: {e}")
    
            if active_workers == 0:
                print("No active workers, stopping collection")
                break
    
            # Collect responses from all workers
            responses_received = 0
            for worker_id, response_queue in enumerate(self.response_queues):
                try:
                    response = response_queue.get(timeout=5.0)  # Increased timeout
                    responses_received += 1
    
                    if response.success and response.data:
                        data = response.data
    
                        # Store experience if it's from primary agent
                        if data.get('is_primary_turn') and data.get('experience') and not self.buffer.is_full():
                            exp = data['experience']
                            success = self.buffer.store(
                                exp.obs, exp.action_mask, exp.action,
                                exp.reward, exp.value, exp.log_prob, exp.done
                            )
                            if success:
                                experiences_collected += 1
    
                                # Finish trajectory on episode end
                                if exp.done:
                                    self.buffer.finish_path(0)
    
                        # Handle episode completion
                        if data.get('done'):
                            if data.get('episode_reward') is not None:
                                self.episode_rewards.append(data['episode_reward'])
                                self.episode_lengths.append(data['episode_length'])
    
                                winner = data.get('winner')
                                is_win = winner == self.primary_agent_id if winner is not None else False
                                self.episode_wins.append(1.0 if is_win else 0.0)
    
                                # Track scenario stats
                                scenario_name = data.get('scenario_name')
                                if scenario_name:
                                    if scenario_name not in self.scenario_stats:
                                        self.scenario_stats[scenario_name] = {'count': 0, 'wins': 0}
                                    self.scenario_stats[scenario_name]['count'] += 1
                                    if is_win:
                                        self.scenario_stats[scenario_name]['wins'] += 1
    
                            # Reset this worker
                            try:
                                self.request_queues[worker_id].put(WorkerRequest('reset'))
                            except Exception as e:
                                print(f"Failed to reset worker {worker_id}: {e}")
                    else:
                        if not response.success:
                            print(f"Worker {worker_id} error: {response.error}")
    
                except Empty:
                    print(f"Worker {worker_id} timeout on step {step}")
                    continue
                except Exception as e:
                    print(f"Error collecting from worker {worker_id}: {e}")
                    continue
    
            # Progress logging
            if step % 50 == 0 or experiences_collected >= self.config.buffer_size:
                progress = (experiences_collected / self.config.buffer_size) * 100
                print(f"Step {step}: {experiences_collected}/{self.config.buffer_size} experiences ({progress:.1f}%)")
    
            # Check if we have enough experiences
            if experiences_collected >= self.config.buffer_size:
                print(f"Reached target experiences at step {step}")
                break
    
            # Safety check: if we're not making progress, break
            if step > 100 and experiences_collected == 0:
                print("No experiences collected after 100 steps, stopping")
                break
    
        print(f"Collection completed: {experiences_collected} experiences in {step_count} steps")
    
        # Ensure we have at least one complete trajectory
        if experiences_collected > 0 and not hasattr(self.buffer, 'advantages'):
            print("Finalizing incomplete trajectories...")
            self.buffer.finish_path(0)
    
        return experiences_collected

    def _collect_rollouts_sequential(self):
        """Original sequential collection method (from previous implementation)"""
        # Reset all environments with random scenarios
        observations = []

        for env_idx in range(self.num_parallel_envs):
            try:
                scenario = self._setup_env_scenario(env_idx)
                obs, _ = self.envs[env_idx].reset()
                observations.append(obs)
                self.env_episode_rewards[env_idx] = 0.0
                self.env_episode_lengths[env_idx] = 0

                # Ensure scenario is properly stored
                if self.env_current_scenarios[env_idx] is None:
                    print(f"Warning: Scenario not set for env {env_idx}, retrying...")
                    self.env_current_scenarios[env_idx] = scenario

            except Exception as e:
                print(f"Error setting up environment {env_idx}: {e}")
                # Fallback: create default scenario
                from uno_ai.training.multi_agent_config import TrainingScenario
                default_scenario = TrainingScenario("vs_random", 4, [0], [1, 2, 3], [], 1.0)
                self.env_current_scenarios[env_idx] = default_scenario
                obs, _ = self.envs[env_idx].reset()
                observations.append(obs)
                self.env_episode_rewards[env_idx] = 0.0
                self.env_episode_lengths[env_idx] = 0

        experiences_collected = 0
        max_steps_per_collection = self.config.buffer_size * 2  # Safety limit

        for step in range(max_steps_per_collection):
            # Check if buffer is full
            if self.buffer.is_full():
                print(f"Buffer full after {experiences_collected} experiences, stopping collection")
                break

            # Prepare batch for environments where primary agent should act
            batch_data = []

            for env_idx in range(self.num_parallel_envs):
                env = self.envs[env_idx]
                if env.game and not env.game.game_over:
                    current_player = env.game.current_player

                    if current_player == self.primary_agent_id:
                        obs = observations[env_idx]
                        action_mask, token_mapping = env._create_action_mask_for_player(current_player)
                        batch_data.append((env_idx, obs, action_mask, token_mapping))

            # Get batched actions for primary agent
            batch_results = self._collect_batch_actions(batch_data)

            # Step all environments
            new_observations = []

            for env_idx in range(self.num_parallel_envs):
                env = self.envs[env_idx]

                if env.game and not env.game.game_over:
                    current_player = env.game.current_player

                    if current_player == self.primary_agent_id and env_idx in batch_results:
                        # Use batched result for primary agent
                        result = batch_results[env_idx]

                        # Store experience before taking step
                        obs = observations[env_idx]

                        # Take step
                        next_obs, env_reward, terminated, truncated, info = env.step(result['env_action'])
                        done = terminated or truncated

                        # Calculate reward
                        reward = self.reward_calculator.calculate_reward(info, env_reward)

                        # Check buffer capacity before storing
                        if not self.buffer.is_full():
                            success = self.buffer.store(
                                obs, result['action_mask'], result['action_token'], reward,
                                result['value'], result['log_prob'], done
                            )

                            if success:
                                experiences_collected += 1
                                self.env_episode_rewards[env_idx] += reward
                            else:
                                print(f"Warning: Failed to store experience in buffer")
                        else:
                            print(f"Buffer full, skipping experience storage")

                        new_observations.append(next_obs)
                        self.env_episode_lengths[env_idx] += 1

                        # Handle episode completion
                        if done:
                            self._handle_episode_completion(env_idx, info)

                            # Reset environment
                            try:
                                new_scenario = self._setup_env_scenario(env_idx)
                                next_obs, _ = env.reset()
                                self.env_episode_rewards[env_idx] = 0.0
                                self.env_episode_lengths[env_idx] = 0
                                new_observations[env_idx] = next_obs

                                # Verify scenario was set
                                if self.env_current_scenarios[env_idx] is None:
                                    self.env_current_scenarios[env_idx] = new_scenario

                            except Exception as e:
                                print(f"Error resetting environment {env_idx}: {e}")
                                # Keep old observation as fallback
                                new_observations[env_idx] = observations[env_idx]

                    else:
                        # Get action from opponent
                        env_action = env.get_action_for_player(current_player, observations[env_idx])
                        next_obs, _, terminated, truncated, info = env.step(env_action)
                        done = terminated or truncated

                        new_observations.append(next_obs)
                        self.env_episode_lengths[env_idx] += 1

                        # Handle episode completion
                        if done:
                            self._handle_episode_completion(env_idx, info)

                            # Reset environment
                            try:
                                new_scenario = self._setup_env_scenario(env_idx)
                                next_obs, _ = env.reset()
                                self.env_episode_rewards[env_idx] = 0.0
                                self.env_episode_lengths[env_idx] = 0
                                new_observations[env_idx] = next_obs
                            except Exception as e:
                                print(f"Error resetting environment {env_idx}: {e}")
                                new_observations.append(observations[env_idx])
                        else:
                            new_observations[env_idx] = next_obs
                else:
                    # Environment is done, keep current observation
                    new_observations.append(observations[env_idx])

            observations = new_observations

            # Break if we've collected enough experiences
            if experiences_collected >= self.config.buffer_size:
                print(f"Collected target {self.config.buffer_size} experiences, stopping")
                break

            # Safety check: if no environments are active, break
            active_envs = sum(1 for env in self.envs if env.game and not env.game.game_over)
            if active_envs == 0:
                print("No active environments, breaking collection loop")
                break

        print(f"Collection completed: {experiences_collected} experiences in {step + 1} steps")

        # Finalize any incomplete trajectories
        self._finalize_trajectories(observations)

    def update_policy(self, data: Dict[str, torch.Tensor]):
        """Update policy and sync with workers if using multiprocessing"""
        losses = super().update_policy(data)

        if self.use_multiprocessing:
            # Update agent weights in all workers
            self._update_worker_agents()
        else:
            # Use parent's sequential update logic
            for env_idx in range(self.num_parallel_envs):
                env = self.envs[env_idx]
                scenario = self.env_current_scenarios[env_idx]

                # Safety check for scenario
                if scenario is not None:
                    for player_id in scenario.agent_players:
                        if player_id != self.primary_agent_id and player_id in self.agents:
                            try:
                                # Sync opponent agents with latest trained weights
                                self.agents[player_id].load_state_dict(
                                    self.agents[self.primary_agent_id].state_dict()
                                )
                                env.add_trained_agent(player_id, self.agents[player_id])
                            except Exception as e:
                                print(f"Error updating agent {player_id} in env {env_idx}: {e}")
                else:
                    print(f"Warning: No scenario set for environment {env_idx}")

        return losses

    def close(self):
        """Clean shutdown of worker processes (multiprocessing mode only)"""
        if not self.use_multiprocessing:
            return

        print("Shutting down worker processes...")

        # Send close requests
        for request_queue in self.request_queues:
            request_queue.put(WorkerRequest('close'))

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()

        print("All workers shut down")


    def train(self, total_timesteps: int):
        """Training loop with either multiprocessing or sequential mode"""
        try:
            timesteps_collected = 0
            update_count = 0
    
            # Print initialization info
            self.training_config.print_scenario_distribution()
    
            mode = "Multiprocess" if self.use_multiprocessing else "Sequential"
            print(f"\nStarting {mode} Multi-Agent PPO training")
            print(f"Environments: {self.num_parallel_envs}")
            print(f"Total timesteps: {total_timesteps:,}")
            print(f"Buffer size: {self.config.buffer_size}")
            if self.use_multiprocessing:
                print(f"Expected speedup: ~{self.num_parallel_envs}x")
            print("-" * 80)
    
            consecutive_failures = 0
            max_failures = 3
    
            while timesteps_collected < total_timesteps:
                # Use appropriate collection method
                if self.use_multiprocessing:
                    experiences_collected = self._collect_rollouts_multiprocess()
                else:
                    self.collect_rollouts()
                    experiences_collected = self.config.buffer_size  # Assume full collection in sequential mode
    
                # Check if collection was successful
                if experiences_collected < self.config.buffer_size * 0.1:  # Less than 10% of target
                    consecutive_failures += 1
                    print(f"Collection failure {consecutive_failures}/{max_failures}: only {experiences_collected} experiences")
    
                    if consecutive_failures >= max_failures:
                        print("Too many consecutive collection failures, stopping training")
                        break
    
                    # Try to restart workers
                    if self.use_multiprocessing:
                        print("Attempting to restart workers...")
                        self.close()
                        self._start_workers()
    
                    continue
                else:
                    consecutive_failures = 0  # Reset failure counter
    
                timesteps_collected += experiences_collected
    
                # Only update policy if we have enough data
                try:
                    data = self.buffer.get()
    
                    # Check if we have valid data
                    if len(data['observations']) == 0:
                        print("No valid data in buffer, skipping update")
                        continue
    
                    losses = self.update_policy(data)
                    update_count += 1
    
                    # Enhanced logging
                    avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                    avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                    win_rate = np.mean(self.episode_wins) if self.episode_wins else 0
    
                    print(f"Update {update_count:4d} | Steps: {timesteps_collected:8,}/{total_timesteps:,} | "
                          f"Mode: {mode} | Experiences: {experiences_collected} | "
                          f"Reward: {avg_reward:6.2f} | WinRate: {win_rate:.3f}")
    
                    # Use parent's scenario stats printing
                    if update_count % 20 == 0:
                        self._print_scenario_stats()
    
                    if update_count % 50 == 0:
                        model_name = f"uno_{'multiprocess' if self.use_multiprocessing else 'sequential'}_update_{update_count}.pt"
                        self.save_model(model_name)
    
                except Exception as e:
                    print(f"Error during policy update: {e}")
                    import traceback
                    traceback.print_exc()
                    consecutive_failures += 1
    
                    if consecutive_failures >= max_failures:
                        print("Too many update failures, stopping training")
                        break
    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.use_multiprocessing:
                self.close()


    def _setup_env_scenario(self, env_idx: int):
        """Setup a random scenario for a specific environment (sequential mode only)"""
        if self.use_multiprocessing:
            # This method is only used in sequential mode
            return None

        scenario = self.training_config.sample_scenario()
        env = self.envs[env_idx]

        # Track scenario (reuse parent method logic)
        if scenario.name not in self.scenario_stats:
            self.scenario_stats[scenario.name] = {'count': 0, 'wins': 0}
        self.scenario_stats[scenario.name]['count'] += 1

        # Handle player count changes
        if scenario.num_players != env.num_players:
            self.envs[env_idx] = MultiAgentUNOEnv(
                num_players=scenario.num_players,
                render_mode=None
            )
            env = self.envs[env_idx]

        # Configure opponents
        opponent_config = OpponentConfig(
            agent_players=scenario.agent_players,
            random_players=scenario.random_players,
            env_players=scenario.env_players
        )
        env.set_opponent_config(opponent_config)

        # Add agent instances
        for player_id in scenario.agent_players:
            if player_id not in self.agents:
                self._create_agent(player_id)
            env.add_trained_agent(player_id, self.agents[player_id])

        self.env_current_scenarios[env_idx] = scenario
        return scenario

    def _collect_batch_actions(self, batch_data: List[Tuple[int, np.ndarray, np.ndarray, Dict]]):
        """
        Collect actions for primary agent across multiple environments in batch (sequential mode only)
        
        Args:
            batch_data: List of (env_idx, obs, action_mask, token_mapping) tuples
        
        Returns:
            Dictionary mapping env_idx to (action_token, value, log_prob, env_action)
        """
        if self.use_multiprocessing or not batch_data:
            return {}

        # Separate batch components
        env_indices = [item[0] for item in batch_data]
        batch_obs = np.array([item[1] for item in batch_data])
        batch_action_masks = np.array([item[2] for item in batch_data])
        token_mappings = [item[3] for item in batch_data]

        # Process batch through agent
        batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.long).to(self.device)
        batch_mask_tensor = torch.tensor(batch_action_masks, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            action_tokens, log_probs, _, values = self.agents[self.primary_agent_id].get_action_and_value(
                batch_obs_tensor, batch_mask_tensor
            )

        # Convert to numpy
        action_tokens = action_tokens.cpu().numpy()
        values = values.cpu().numpy()
        log_probs = log_probs.cpu().numpy()

        # Prepare results
        results = {}
        for i, env_idx in enumerate(env_indices):
            action_token = action_tokens[i]
            env_action = self._convert_token_to_env_action(action_token, token_mappings[i])

            results[env_idx] = {
                'action_token': action_token,
                'value': values[i],
                'log_prob': log_probs[i],
                'env_action': env_action,
                'action_mask': batch_action_masks[i],
                'token_mapping': token_mappings[i]
            }

        return results

    def _convert_token_to_env_action(self, action_token: int, token_mapping: Dict[int, int]) -> int:
        """Convert action token to environment action"""
        if action_token == UNOTokens.DRAW_ACTION:
            return 7
        elif action_token in token_mapping:
            return token_mapping[action_token]
        else:
            print(f"Warning: Invalid action token {action_token}, falling back to draw")
            return 7  # Fallback to draw

    def _handle_episode_completion(self, env_idx: int, info: Dict):
        """Handle episode completion for a specific environment (sequential mode only)"""
        if self.use_multiprocessing:
            return  # Handled in worker processes

        winner = info.get('winner')
        is_win = winner == self.primary_agent_id if winner is not None else False
        self.episode_wins.append(1.0 if is_win else 0.0)

        # Track scenario wins - with null check
        scenario = self.env_current_scenarios[env_idx]
        if scenario is not None and is_win:
            # Additional safety check
            if scenario.name in self.scenario_stats:
                self.scenario_stats[scenario.name]['wins'] += 1
            else:
                # Initialize if somehow missing
                self.scenario_stats[scenario.name] = {'count': 1, 'wins': 1}

        # Store episode metrics
        self.episode_rewards.append(self.env_episode_rewards[env_idx])
        self.episode_lengths.append(self.env_episode_lengths[env_idx])

    def _finalize_trajectories(self, observations: List[np.ndarray]):
        """Finalize any incomplete trajectories in the buffer (sequential mode only)"""
        if self.use_multiprocessing:
            return  # Not needed in multiprocessing mode

        # Check if we need to compute final values for incomplete episodes
        for env_idx in range(self.num_parallel_envs):
            env = self.envs[env_idx]
            if env.game and not env.game.game_over:
                current_player = env.game.current_player
                if current_player == self.primary_agent_id:
                    # Compute final value
                    obs = observations[env_idx]
                    obs_tensor = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(self.device)
                    action_mask, _ = env._create_action_mask_for_player(current_player)
                    action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        _, _, _, last_value = self.agents[self.primary_agent_id].get_action_and_value(
                            obs_tensor, action_mask_tensor
                        )

                    self.buffer.finish_path(last_value.item())
                    break  # Only need to finish one path

    def _print_scenario_stats(self):
        """Print statistics for each training scenario"""
        mode = "Multiprocess" if self.use_multiprocessing else "Sequential"
        print(f"\n{mode} Training Scenario Statistics:")
        print("-" * 60)

        if not self.scenario_stats:
            print("  No scenarios completed yet.")
            return

        # Group by player count (reuse parent logic but with safety)
        for player_count in [3, 4]:
            scenarios = self.training_config.get_scenarios_by_player_count(player_count)
            scenario_names = [s.name for s in scenarios]

            found_scenarios = False
            for scenario_name in scenario_names:
                if scenario_name in self.scenario_stats:
                    if not found_scenarios:
                        print(f"\n{player_count}-Player Games:")
                        found_scenarios = True

                    stats = self.scenario_stats[scenario_name]
                    win_rate = stats['wins'] / max(stats['count'], 1) * 100
                    print(f"  {scenario_name:25} | {stats['count']:3d} episodes | {win_rate:5.1f}% wins")

        if self.use_multiprocessing:
            print(f"\nActive Workers: {len(self.workers)} processes")
        else:
            # Show active environments info for sequential mode
            active_scenarios = [s.name if s else "None" for s in self.env_current_scenarios]
            scenario_counts = {}
            for scenario_name in active_scenarios:
                scenario_counts[scenario_name] = scenario_counts.get(scenario_name, 0) + 1

            print(f"\nActive Environment Scenarios:")
            for scenario_name, count in scenario_counts.items():
                print(f"  {scenario_name}: {count} environments")
        print()

    def create_action_mask(self, current_player: int):
        """Create action mask for current player - compatibility method"""
        if self.use_multiprocessing:
            # This shouldn't be called in multiprocessing mode
            raise NotImplementedError("create_action_mask not used in multiprocessing mode")
        else:
            # Use the first environment as reference
            return self.envs[0]._create_action_mask_for_player(current_player)

    def count_parameters(self):
        """Count total trainable parameters"""
        total_params = sum(p.numel() for p in self.agents[self.primary_agent_id].parameters() if p.requires_grad)
        return total_params

    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.agents[self.primary_agent_id].state_dict(),
            'optimizer_state_dict': self.optimizers[self.primary_agent_id].state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size,
            'use_multiprocessing': self.use_multiprocessing,
            'num_parallel_envs': self.num_parallel_envs
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.agents[self.primary_agent_id].load_state_dict(checkpoint['model_state_dict'])
        self.optimizers[self.primary_agent_id].load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab_size = checkpoint.get('vocab_size', UNOTokens.VOCAB_SIZE)

        # Update workers if in multiprocessing mode
        if self.use_multiprocessing:
            self._update_worker_agents()

        print(f"Model loaded from {filepath}")