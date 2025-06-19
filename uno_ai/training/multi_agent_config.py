from dataclasses import dataclass
from typing import List, Optional
import random

@dataclass
class TrainingScenario:
    name: str
    num_players: int  # Add this to specify player count
    agent_players: List[int]
    random_players: List[int]
    env_players: List[int]
    weight: float = 1.0  # Probability weight for this scenario

class MultiAgentTrainingConfig:
    def __init__(self):
        self.scenarios = [
            # 4-player scenarios
            # Self-play scenarios
            TrainingScenario("self_play_4", 4, [0, 1, 2, 3], [], [], 0.05),
            TrainingScenario("self_play_2_vs_random", 4, [0, 1], [2, 3], [], 0.05),

            # Mixed 4-player scenarios
            TrainingScenario("vs_random_4", 4, [0], [1, 2, 3], [], 0.05),
            TrainingScenario("vs_env_4", 4, [0], [], [1, 2, 3], 0.05),
            TrainingScenario("mixed_4", 4, [0, 1], [2], [3], 0.05),

            # 3-player scenarios
            # Self-play scenarios
            TrainingScenario("self_play_3", 3, [0, 1, 2], [], [], 0.05),
            TrainingScenario("self_play_2_vs_random_3", 3, [0, 1], [2], [], 0.05),

            # Mixed 3-player scenarios  
            TrainingScenario("vs_random_3", 3, [0], [1, 2], [], 0.05),
            TrainingScenario("vs_env_3", 3, [0], [], [1, 2], 0.05),
            TrainingScenario("agent_vs_random_vs_env", 3, [0], [1], [2], 0.05),
            
            # 2-player scenarios
            TrainingScenario("self_play_2_player", 2, [0, 1], [], [], 0.20),
            TrainingScenario("vs_random_2_player", 2, [0], [1], [], 0.10),
            TrainingScenario("vs_env_2_player", 2, [0], [], [1], 0.20),
        ]

    def sample_scenario(self) -> TrainingScenario:
        """Sample a training scenario based on weights"""
        weights = [s.weight for s in self.scenarios]
        return random.choices(self.scenarios, weights=weights)[0]

    def get_scenario_by_name(self, name: str) -> Optional[TrainingScenario]:
        """Get specific scenario by name"""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None

    def get_scenarios_by_player_count(self, num_players: int) -> List[TrainingScenario]:
        """Get all scenarios for a specific player count"""
        return [s for s in self.scenarios if s.num_players == num_players]

    def print_scenario_distribution(self):
        """Print the distribution of scenarios"""
        total_weight = sum(s.weight for s in self.scenarios)

        print("Training Scenario Distribution:")
        print("-" * 50)

        scenarios_player_counts = {s.num_players for s in self.scenarios}

        # Group by player count
        for player_count in scenarios_player_counts:
            scenarios = self.get_scenarios_by_player_count(player_count)
            if scenarios:
                print(f"\n{player_count}-Player Scenarios:")
                for scenario in scenarios:
                    percentage = (scenario.weight / total_weight) * 100
                    agents = len(scenario.agent_players)
                    randoms = len(scenario.random_players)
                    envs = len(scenario.env_players)
                    print(f"  {scenario.name:25} | {percentage:5.1f}% | "
                          f"Agents: {agents}, Random: {randoms}, Env: {envs}")