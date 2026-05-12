# Homework 3: DQN and Its Variants

## 1. Introduction
The objective of this assignment is to learn and implement Deep Q-Networks (DQN) and its advanced variants to solve navigation tasks within a GridWorld environment. As the environment's complexity increases progressively, we completed the following tasks in order:
- **HW3-1**: Implemented and verified the basic **Naive DQN** in a **static mode** environment.
- **HW3-2**: Implemented and compared the performance of **Double DQN** and **Dueling DQN** in a **player mode** environment (where the starting position is randomized).
- **HW3-3**: Tackled the most challenging **random mode** (where all entities are randomized) by wrapping our DQN model in **PyTorch Lightning** and integrating several advanced Training Techniques to stabilize the learning process.

**GridWorld Environment Overview**:
The environment is a 4x4 grid maze consisting of four primary entities:
- **Player (P)**: The agent that we are training.
- **Goal (+)**: The destination the agent must reach.
- **Pit (-)**: A trap the agent must avoid at all costs.
- **Wall (W)**: Impassable obstacles blocking movement.

- **Action Space**: The agent can choose from four discrete actions: `up`, `down`, `left`, and `right`.
- **Reward Design**:
  - Reaching the Goal yields a reward of **+10** (Success).
  - Falling into the Pit yields a reward of **-10** (Failure/Game Over).
  - Every other standard movement incurs a step penalty of **-1**, incentivizing the agent to find the shortest possible path to the Goal.

---

## 2. Environment and Code Structure
The GridWorld environment provides three progressive difficulty tiers (Modes):
- **static mode**: The initial positions of all entities (Player, Goal, Pit, Wall) are strictly fixed across all episodes.
- **player mode**: The Player's starting position is randomized at the beginning of each episode, while the Goal, Pit, and Wall remain fixed.
- **random mode**: The starting positions of all entities (Player, Goal, Pit, Wall) are completely randomized at the beginning of each episode.

**State Representation**:
The environment state is retrieved via the `render_np()` function in `GridBoard.py`. It returns a 3D NumPy array of shape `(4, 4, 4)`, representing independent binary masks for the different entities layered on top of each other. We flatten this 3D array into a 64-dimensional 1D vector. Furthermore, a tiny amount of uniform noise (`+ np.random.rand(len(state))/10.0`) is injected into the state representation to prevent the occurrence of purely zero-vectors, which ensures consistent gradient flow during the early stages of Neural Network training.

**Action Mapping**:
The model outputs a discrete integer ranging from 0 to 3. Before passing this action to the environment's `env.makeMove()` function, it is mapped to a string command via the dictionary `action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}`.

**Primary File Structure**:
- `GridBoard.py`: Handles the low-level rendering of the 4x4 grid, tracking entity positions, and managing collision masks.
- `Gridworld.py`: A wrapper around `GridBoard` that provides high-level environment initializations (for the three modes), movement validation (`validateMove`), and reward logic (`reward`).
- `replay_buffer.py`: Implements the `ExperienceReplay` class using Python's `collections.deque` to store past transitions and sample randomized mini-batches.
- `models.py`: Defines the PyTorch Neural Network architectures, including the standard `DQN` and the bifurcated `DuelingDQN`.
- `train_hw31_static.py`: The training and evaluation script for the Naive DQN in static mode.
- `train_hw32_player.py`: The training script for player mode, executing and comparing both Double DQN and Dueling DQN.
- `train_hw33_random_lightning.py`: The training script for random mode, utilizing the PyTorch Lightning framework to manage the training loop and advanced techniques.

---

## 3. HW3-1: Naive DQN for Static Mode

### 3.1 Task Description
In HW3-1, we operate in **static mode**. Because the layout is entirely deterministic (the start and end points never change), the agent only needs to memorize a single optimal trajectory to the Goal. This represents the easiest tier of difficulty.

### 3.2 Basic DQN Method
We implemented a standard Deep Q-Network:
- **Q-network**: An MLP taking the 64-dimensional state vector as input, passing it through two hidden layers (128 neurons each with ReLU activation), and outputting the estimated Q-values for the 4 possible actions.
- **Action Selection**: An $\epsilon$-greedy exploration strategy is used, where $\epsilon$ decays steadily from 1.0 down to a minimum of 0.1, balancing exploration of the grid and exploitation of learned knowledge.
- **Bellman Target**: We use the current online network to estimate the maximum Q-value of the next state to compute the temporal difference target. *(Note: In this most fundamental implementation, we did not use an isolated Target Network; the online network acts as its own estimator).*
- **Loss Function & Optimizer**: We utilize the Mean Squared Error (`MSE Loss`) to compute the difference between predicted Q-values and target Q-values, optimizing the weights via the `Adam` optimizer (Learning Rate = 1e-3).

The fundamental DQN target formula is:
```math
y = r + \gamma \max_{a'} Q(s', a')
```

### 3.3 Experience Replay Buffer
As implemented in `replay_buffer.py`:
- We instantiate an `ExperienceReplay` buffer with a capacity of 1,000 transitions.
- Every interaction with the environment yields a transition tuple: `(state, action, reward, next_state, done)`, which is pushed into the buffer.
- During training, we sample a randomized mini-batch (size = 32) from the buffer to perform Gradient Descent.
- **Benefits**: Sampling randomly from a replay buffer breaks the temporal correlation between consecutive sequence frames. This vastly improves data efficiency (as past experiences can be reused multiple times) and prevents the network weights from oscillating wildly, leading to much more stable convergence.

### 3.4 Result and Analysis
![HW3-1 Static Results](hw31_static_results.png)

**Detailed Analysis of the Results:**
- **Reward per Epoch (Left)**: In the early epochs, the agent receives massive negative rewards (averaging -20 to -10) as it aimlessly hits the walls or falls into the Pit. Very rapidly, the reward curve surges upwards and converges to a stable positive value (around +4 to +6), indicating the agent has discovered the optimal, shortest path to the Goal while avoiding step penalties.
- **Success Rate (Center)**: The success rate perfectly mirrors the reward curve. It begins low but swiftly climbs, maintaining a near **100% Success Rate** through the mid-to-late stages of training.
- **Training Loss (Right)**: The MSE loss experiences massive spikes in the early phases. This occurs when the agent accidentally stumbles into the Goal or Pit for the first time, producing a huge temporal difference error. As the Q-values accurately map the environment's true Bellman expectations, the loss diminishes and settles into a stable, low-variance baseline.

**Note on minor fluctuations**: Even though the environment is fully static, the reward curve exhibits minor, localized dips. This is completely normal and is caused by the residual 10% $\epsilon$-greedy exploration rate, which occasionally forces the agent to take a sub-optimal step (e.g., bumping into a wall) before continuing to the Goal.

### 3.5 HW3-1 Summary
To conclude, the combination of a Naive DQN and an Experience Replay Buffer is more than sufficient to quickly and successfully learn the optimal policy in a static, deterministic environment.

---

## 4. HW3-2: Enhanced DQN Variants for Player Mode

### 4.1 Task Description
In **player mode**, the Goal, Pit, and Walls remain static, but the **Player's starting position is randomized** across the grid at the start of each episode. This drastically increases the complexity; the agent can no longer overfit to a single path. Instead, it must develop spatial generalization to successfully navigate to the Goal from any valid starting tile.

### 4.2 Double DQN
The Basic DQN relies on a `max` operator to simultaneously select an action and estimate that action's Q-value. This mathematically leads to an Overestimation Bias, where the network overly inflates the perceived value of certain states. **Double DQN** resolves this by decoupling Action Selection from Action Evaluation:
- **Online Network**: Responsible for selecting the best possible action $a^*$ based on the next state $s'$.
- **Target Network**: Responsible for evaluating the true Q-value of that specific action $a^*$, minimizing the risk of overestimation.

The core formulas are:
```math
a^* = \arg\max_a Q_{online}(s', a)
```
```math
y = r + \gamma Q_{target}(s', a^*)
```

### 4.3 Dueling DQN
**Dueling DQN** alters the internal architecture of the neural network rather than the algorithmic target. It splits the penultimate hidden layer into two distinct streams:
1. **State Value $V(s)$**: Estimates "how good it is to simply be in this state" (e.g., being one tile away from the Goal is inherently good, being next to the Pit is inherently bad).
2. **Advantage $A(s,a)$**: Estimates the "relative advantage of choosing a specific action over the other available actions" in that state.

This allows the network to learn state values efficiently without needing to evaluate the effect of every single action, which is incredibly useful for states where the choice of action doesn't significantly impact the outcome. The streams are aggregated at the final output layer:
```math
Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|} \sum_{a'} A(s,a')
```

### 4.4 Result and Comparison
![HW3-2 Player Results](hw32_player_results.png)

**Detailed Analysis of the Results:**
- **Reward Comparison (Left)**: Both Double DQN and Dueling DQN exhibit a robust learning curve. Despite starting from highly penalized negative states, both architectures aggressively climb out of the negative zone and eventually cross the zero-threshold, converging towards their maximum theoretical average rewards given randomized spawns. 
- **Success Rate Comparison (Right)**: The success rates for both agents soar from near 0% up to an impeccable **100% Success Rate** during the final evaluation window.

A consolidated comparison based on the visual trends:

| Method | Environment | Reward Trend | Success Rate Trend | Observation |
| :---: | :---: | :---: | :---: | :--- |
| **Double DQN** | player mode | Steady, steep climb converging to a high positive score. | Rises reliably, reaching near perfection. | In our localized experiment, Double DQN reached a slightly higher peak average reward. Mitigating the overestimation bias proved highly effective in stabilizing the value landscape. |
| **Dueling DQN** | player mode | Steady climb, very similar trajectory to Double DQN. | Rises reliably, reaching near perfection. | Successfully solved the generalization problem. However, the Advantage/Value split typically shines brightest in environments with highly redundant action spaces, so its superiority is less pronounced in a small 4x4 grid. |

### 4.5 HW3-2 Summary
- The player mode requires genuine spatial awareness and generalization, making it substantially harder than static mode.
- **Double DQN** successfully mitigates overestimation bias, creating a more stable learning target.
- **Dueling DQN** leverages architectural separation to more efficiently estimate the baseline value of the grid states.
- Both advanced variants successfully mastered the player mode, establishing optimal policies that vastly outperform a random walk.

---

## 5. HW3-3: DQN for Random Mode with PyTorch Lightning

### 5.1 Task Description
In **random mode**, not only is the Player's position randomized, but **the locations of the Goal, Pit, and Walls are entirely randomized** in every single episode. The agent faces a completely novel procedural layout every time it spawns. This is exponentially more difficult; the agent must actively parse the dynamic state representation to deduce the relative positions of hazards and goals on the fly.

### 5.2 PyTorch to PyTorch Lightning Conversion
To manage the escalating complexity, we refactored our training pipeline into the **PyTorch Lightning** framework:
- We encapsulated the network (`DQN`), the loss calculation, the training loop (`training_step`), and the optimizer logic (`configure_optimizers`) into a unified `LightningDQN` class (inheriting from `LightningModule`).
- **Benefits of Lightning**: It massively improves code modularity and readability by abstracting away the boilerplate engineering loops. It allows us to easily plug into native features like hardware acceleration, logging, and callbacks, making it trivial to implement advanced stabilization techniques.

### 5.3 Training Techniques
Because random mode is a highly volatile environment, we integrated several critical Training Tips to prevent the model from collapsing:
1. **Target Network**: We implemented a hard-updated `target_model`. By only synchronizing the target network at the end of every epoch, we provide a fixed, stable target for the Q-value updates, preventing catastrophic forgetting.
2. **Gradient Clipping**: Enabled natively in the Lightning `Trainer` via `gradient_clip_val=1.0`. Since randomizing the map creates chaotic TD-errors (e.g., spawning directly adjacent to a Pit), gradient clipping prevents exploding gradients that would otherwise destroy the network's weights.
3. **Learning Rate Scheduler**: We introduced `StepLR(step_size=10, gamma=0.9)` to gradually decay the learning rate. This allows the model to take large exploratory steps early on, and fine-tune its policy carefully as it approaches convergence.
4. **Larger Replay Buffer & IterableDataset**: The buffer capacity was expanded to 5,000. Combined with a custom `IterableDataset`, this ensures the network samples a highly diverse batch of generalized grid layouts, reducing overfitting to any single map configuration.
5. **Epsilon Decay**: Gradual decay from 1.0 to 0.1 ensures the agent properly transitions from mapping the unknown physics of the random grid to exploiting its generalized knowledge.

### 5.4 Result and Analysis
![HW3-3 Random Lightning Results](hw33_random_lightning_results.png)

**Detailed Analysis of the Results:**
- **Reward in Random Mode (Left)**: The reward trajectory is highly volatile, which is perfectly expected given the constantly shifting layouts. However, looking at the smoothed trend line, there is a distinct, verifiable upward climb. The agent is clearly moving away from catastrophic failures and learning generalized heuristics (e.g., "move away from the Pit mask").
- **Success Rate (Right)**: The success rate climbs from a baseline near 0% up to approximately **56%** by the end of training. 

**Critical Evaluation**: It is important to note that a 100% success rate in a 4x4 fully randomized grid may be mathematically impossible. Procedural generation frequently creates "Spawn Traps" (e.g., the Player spawns entirely boxed in by Walls and a Pit, guaranteeing immediate failure). Despite these unsolvable layouts, achieving a 56% win rate proves that the agent has successfully extracted functional policies. In summary: **performance improved significantly, but the environment remained inherently unstable.**

### 5.5 HW3-3 Summary
- Random mode introduces massive state distribution variance, serving as the ultimate test of the agent's capabilities.
- By leveraging **PyTorch Lightning** and injecting crucial **training techniques** (Gradient Clipping, LR Scheduling, Target Networks), we successfully forced the network to learn generalized logic rather than collapsing under the noise.
- While the results are naturally less stable than deterministic environments, the upward trend validates our methodology. Future improvements could involve combining PyTorch Lightning with Rainbow DQN or Prioritized Experience Replay to squeeze out further efficiency.

---

## 6. Overall Discussion

A comparative summary of the three difficulty tiers:

| Mode | Randomized Components | Difficulty | Result |
| :---: | :--- | :---: | :--- |
| **static** | none | easiest | Basic DQN can learn |
| **player** | Player only | medium | Double/Dueling DQN improve learning |
| **random** | Player, Goal, Pit, Wall | hardest | Learning improves but remains unstable |

- **static mode**: The easiest challenge. Because the environment never shifts, the agent merely needs to overfit to the exact state sequence required to reach the Goal.
- **player mode**: Medium difficulty. Demands spatial awareness and generalization. The agent must map the relative distance to the fixed Goal from any potential starting tile, which advanced architectures like Double/Dueling DQN handle gracefully.
- **random mode**: The hardest challenge. The agent must rely entirely on its Convolutional/Flattened perspective to dynamically evaluate the layout on every single step. Facing an immense state distribution, it struggles to achieve perfect consistency but demonstrates clear learning capability.

---

## 7. Conclusion
In this assignment, we systematically built and scaled Reinforcement Learning solutions for the GridWorld environment:
In **HW3-1**, we successfully established a foundational baseline using a Naive DQN and an Experience Replay Buffer, achieving perfect convergence in static mode.
In **HW3-2**, we upgraded our architecture to include Double DQN and Dueling DQN, proving that separating evaluation from selection, and state value from advantage, drastically improves generalization in player mode.
Finally, in **HW3-3**, we modernized our codebase using the PyTorch Lightning framework. By employing vital stabilization techniques—like Gradient Clipping and LR Scheduling—we managed to extract meaningful learning trends out of the highly chaotic random mode.
Ultimately, the results clearly demonstrate that as environmental randomness and state-space complexity increase, training becomes exponentially harder. While DQN variants and training stabilization techniques are highly effective at combating this entropy, fully randomized environments remain exceptionally difficult and invite the application of even deeper RL methodologies.

---

## 8. Appendix: How to Run
To reproduce the experiments and generate the plots featured in this report, execute the following commands in your terminal:

Run HW3-1 (Static Mode):
```bash
python3 train_hw31_static.py
```

Run HW3-2 (Player Mode with Double/Dueling DQN):
```bash
python3 train_hw32_player.py
```

Run HW3-3 (Random Mode with PyTorch Lightning):
```bash
python3 train_hw33_random_lightning.py
```
