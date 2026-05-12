# Deep Reinforcement Learning: DQN and Its Variants in GridWorld

This repository contains the implementation of a Deep Q-Network (DQN) and its advanced variants to solve a custom 4x4 GridWorld environment. The project is structured across three progressive levels of difficulty, exploring the capabilities and stabilization techniques of Reinforcement Learning algorithms.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Configurations](#environment-configurations)
- [Implemented Algorithms](#implemented-algorithms)
- [Results](#results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

## Project Overview
The goal of this project is to train an agent to navigate a 4x4 grid. The grid consists of:
- **Player (P)**: The agent.
- **Goal (+)**: The destination (+10 reward).
- **Pit (-)**: A lethal trap (-10 reward).
- **Wall (W)**: An impassable obstacle.

Every regular step incurs a `-1` penalty to encourage finding the shortest path.

## Environment Configurations
The environment (`Gridworld.py`) can be initialized in three different modes:
1. **Static Mode (`static`)**: The initial positions of all entities are fixed.
2. **Player Mode (`player`)**: The Player's starting position is randomized, but the Goal, Pit, and Wall remain fixed.
3. **Random Mode (`random`)**: The starting positions of all entities are completely randomized in every episode.

## Implemented Algorithms
### 1. Naive DQN
- Implemented a standard Deep Q-Network utilizing an **Experience Replay Buffer** to break sample correlation.
- Validated in `static` mode, achieving a 100% success rate.

### 2. Double DQN & Dueling DQN
- **Double DQN**: Mitigates overestimation bias by decoupling action selection (via the online network) from action evaluation (via the target network).
- **Dueling DQN**: Splits the network architecture to independently estimate the State Value $V(s)$ and the Action Advantage $A(s,a)$.
- Both algorithms were tested in `player` mode, demonstrating robust spatial generalization and achieving 100% success rates.

### 3. PyTorch Lightning DQN
- To tackle the highly chaotic `random` mode, the model was refactored using the **PyTorch Lightning** framework for enhanced modularity.
- **Training Techniques Integrated**:
  - Target Networks (hard updates)
  - Gradient Clipping (`gradient_clip_val=1.0`)
  - Learning Rate Scheduling (`StepLR`)
  - Larger Replay Buffers and IterableDatasets
- Successfully extracted functional generalized policies from fully randomized procedural layouts.

## Results
A detailed, comprehensive English report analyzing the reward curves, success rates, and training loss across all experiments is available in the [HW3_Report.md](./HW3_Report.md) file.

## Getting Started
### Prerequisites
Ensure you have Python 3 installed along with the following dependencies:
```bash
pip install torch torchvision torchaudio pytorch-lightning matplotlib numpy
```

### Execution
You can run the training scripts for the different modes directly from the terminal:

**1. Train Naive DQN (Static Mode)**
```bash
python3 train_hw31_static.py
```

**2. Train & Compare Double/Dueling DQN (Player Mode)**
```bash
python3 train_hw32_player.py
```

**3. Train Lightning DQN (Random Mode)**
```bash
python3 train_hw33_random_lightning.py
```

## Project Structure
- `Gridworld.py` & `GridBoard.py`: Custom environment and rendering logic.
- `models.py`: Neural network definitions (DQN, DuelingDQN).
- `replay_buffer.py`: Experience Replay buffer implementation.
- `train_hw3*.py`: Execution and training scripts for the specific assignments.
- `HW3_Report.md`: Full analysis report of the experiments.
