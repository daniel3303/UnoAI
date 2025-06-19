# UNO AI - Reinforcement Learning Training Environment

A comprehensive UNO card game implementation with AI agents trained using Proximal Policy Optimization (PPO). This project provides a complete training environment for developing intelligent UNO players with support for multi-agent scenarios, self-play training, and real-time visualization.

## 🎮 Features

### Core Components
- **Custom Gymnasium Environment**: Full UNO game implementation following OpenAI Gym standards
- **Pygame Visualization**: Real-time rendering of game states with high-quality card graphics and DPI scaling
- **Multi-Agent Training**: Support for various training scenarios including self-play and mixed opponents
- **Transformer-based Architecture**: Custom UNO transformer model with specialized token vocabulary
- **PPO Implementation**: Proximal Policy Optimization with custom reward shaping for UNO gameplay

### Training Capabilities
- **Self-Play Training**: Agents learn by playing against copies of themselves
- **Mixed Opponent Training**: Train against combinations of AI agents, random players, and rule-based opponents
- **Parallel Training**: Both multiprocessing and sequential training modes for different hardware setups
- **Scenario-Based Learning**: Multiple training scenarios with different player counts (3-4 players)

### Visualization & Monitoring
- **Real-time Pygame Rendering**: Watch AI agents play with smooth animations and card graphics
- **DPI-Aware Display**: Automatic scaling for high-resolution displays
- **Training Statistics**: Comprehensive logging of win rates, episode lengths, and scenario performance
- **Asset Management**: High-quality card images with dynamic scaling support

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/daniel3303/UnoAI.git
cd UnoAI

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

#### Train a PPO Agent
```bash
# Basic PPO training
uno-train-ppo

# Multi-agent training with self-play
uno-train-multi-agent

# Parallel training (recommended for faster training)
uno-train-parallel --num-envs 8 --timesteps 2000000
```

#### Evaluate Trained Models
```bash
# Evaluate a trained model
uno-evaluate --model-path uno_ppo_final_model.pt --episodes 100

# Demo with visualization
uno-demo --model-path uno_ppo_final_model.pt --episodes 5 --render
```

#### Test Environment
```python
python test_uno_game.py
```

## 🏗️ Architecture

### Environment (`uno_ai.environment`)
- **UNOEnv**: Main Gymnasium environment with observation/action spaces
- **UNOGame**: Core game logic implementing standard UNO rules
- **MultiAgentUNOEnv**: Extended environment supporting multiple AI agents

### Model (`uno_ai.model`)
- **UNOTransformer**: Custom transformer architecture for UNO gameplay
- **UNOTokens**: Specialized vocabulary system for representing game states
- **Multi-Head Attention**: Attention mechanism with RoPE embeddings

### Training (`uno_ai.training`)
- **PPOTrainer**: Single-agent PPO implementation
- **MultiAgentPPOTrainer**: Multi-agent training with scenario sampling
- **ParallelMultiAgentTrainer**: Parallel training with multiprocessing support

### Visualization
- **Pygame Renderer**: Real-time game visualization with card graphics
- **Asset Manager**: High-resolution image loading and scaling
- **DPI Scaling**: Automatic adaptation to different screen resolutions

## 🎯 Training Scenarios

The training system supports various scenarios to create robust AI players:

### 4-Player Scenarios
- **Self-Play**: All agents learning together
- **Mixed Training**: Agents vs random/rule-based opponents
- **Graduated Difficulty**: Progressive opponent strength

### 3-Player Scenarios
- **Compact Games**: Faster training iterations
- **Strategic Focus**: Enhanced decision-making in smaller groups

## 🖥️ Gymnasium Environment

The UNO environment follows OpenAI Gym standards:

```python
import gymnasium as gym
from uno_ai.environment.uno_env import UNOEnv

# Create environment
env = UNOEnv(num_players=4, render_mode=\"human\")

# Standard gym interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Render the game
env.render()  # Opens Pygame window with real-time visualization
```

### Action Space
- **0-6**: Play card from hand (index-based)
- **7**: Draw card from deck

### Observation Space
- **Token Sequence**: Game state represented as sequence of specialized UNO tokens
- **Fixed Length**: Padded sequences for consistent input size
- **Rich Information**: Hand cards, discard pile history, opponent hand sizes

## 🎨 Pygame Visualization

The Pygame renderer provides:
- **High-Quality Card Graphics**: Asset-based rendering with fallback to programmatic drawing
- **DPI Scaling**: Automatic scaling for different screen resolutions
- **Real-time Updates**: Smooth animations and state transitions
- **Player Information**: Hand sizes, current player indicators, game statistics
- **Interactive Display**: Visual feedback for valid moves and game progression

### Visualization Features
- Circular player layout with clear current player indication
- Center discard pile and deck with card counts
- Direction indicators (clockwise/counter-clockwise)
- Win condition displays and game statistics
- Responsive design that adapts to different screen sizes

## 📊 Performance

Training metrics include:
- **Win Rate**: Agent performance against various opponents
- **Episode Length**: Game duration and efficiency
- **Scenario Statistics**: Performance breakdown by training scenario
- **Learning Curves**: Reward progression and convergence tracking

## 🛠️ Development

### Project Structure
```
uno_ai/
├── environment/     # Gym environment and game logic
├── model/          # Neural network architectures
├── training/       # PPO and multi-agent trainers
├── utils/          # Asset management and utilities
└── assets/         # Card images and graphics
```

### Custom Components
- **Token Vocabulary**: 85-token vocabulary representing all UNO cards and actions
- **Reward Shaping**: Custom reward calculation encouraging strategic play
- **Scenario Sampling**: Weighted sampling of training scenarios for balanced learning

## 📋 Requirements

- Python 3.12+
- PyTorch (with MPS/CUDA support)
- Pygame
- Gymnasium
- NumPy
- Additional dependencies in `requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License