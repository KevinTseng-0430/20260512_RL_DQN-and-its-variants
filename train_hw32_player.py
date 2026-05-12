import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from Gridworld import Gridworld
from replay_buffer import ExperienceReplay
from models import DQN, DuelingDQN

def get_state(env):
    state = env.board.render_np().flatten()
    return state + np.random.rand(len(state))/10.0 

def is_done(env):
    player = env.board.components['Player'].pos
    goal = env.board.components['Goal'].pos
    pit = env.board.components['Pit'].pos
    return player == goal or player == pit

action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def evaluate(model, mode, num_episodes=100):
    print(f"\nEvaluating in {mode} mode for {num_episodes} episodes...")
    success_count = 0
    total_rewards = []
    
    for _ in range(num_episodes):
        env = Gridworld(size=4, mode=mode)
        state = get_state(env)
        done = False
        step_count = 0
        ep_reward = 0
        
        while not done and step_count < 50:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(q_values).item()
            env.makeMove(action_map[action])
            state = get_state(env)
            reward = env.reward()
            done = is_done(env)
            ep_reward += reward
            step_count += 1
            if done and reward == 10:
                success_count += 1
                
        total_rewards.append(ep_reward)
        
    print(f"Success Rate: {success_count / num_episodes * 100:.2f}%")
    print(f"Average Reward: {np.mean(total_rewards):.2f}\n")
    return success_count / num_episodes, np.mean(total_rewards)

def train_agent(agent_type, epochs=200, batch_size=32, update_freq=10):
    env = Gridworld(size=4, mode='player')
    
    if agent_type == 'double_dqn':
        model = DQN(state_dim=64, action_dim=4)
        target_model = copy.deepcopy(model)
    elif agent_type == 'dueling_dqn':
        model = DuelingDQN(state_dim=64, action_dim=4)
        target_model = copy.deepcopy(model)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    replay = ExperienceReplay(capacity=2000)

    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    rewards_history = []
    successes_history = []

    for epoch in range(epochs):
        env = Gridworld(size=4, mode='player')
        state = get_state(env)
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 50:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()

            env.makeMove(action_map[action])
            next_state = get_state(env)
            reward = env.reward()
            done = is_done(env)

            replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                successes_history.append(1 if reward == 10 else 0)
            elif step_count >= 50:
                successes_history.append(0)

            if len(replay) >= batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                
                q_vals = model(s)
                q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if agent_type == 'double_dqn':
                        # Double DQN: action from model, value from target_model
                        next_actions = model(ns).argmax(dim=1, keepdim=True)
                        max_next_q_val = target_model(ns).gather(1, next_actions).squeeze(1)
                    else:
                        # Dueling DQN: standard target calculation
                        max_next_q_val = target_model(ns).max(1)[0]
                        
                    target_q_val = r + gamma * max_next_q_val * (1 - d)

                loss = loss_fn(q_val, target_q_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epoch % update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (epoch + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_success = np.mean(successes_history[-100:]) * 100
            print(f"[{agent_type}] Epoch: {epoch+1}, Avg Reward: {avg_reward:.2f}, Success Rate: {avg_success:.1f}%, Epsilon: {epsilon:.2f}")

    return rewards_history, successes_history, model

def main():
    print("Training Double DQN...")
    double_dqn_rewards, double_dqn_success, model_double = train_agent('double_dqn')
    
    print("Training Dueling DQN...")
    dueling_dqn_rewards, dueling_dqn_success, model_dueling = train_agent('dueling_dqn')

    print("\n--- Final Evaluations ---")
    print("Double DQN:")
    evaluate(model_double, 'player', num_episodes=100)
    print("Dueling DQN:")
    evaluate(model_dueling, 'player', num_episodes=100)

    # Smoothing function for better visualization
    def smooth(data, window=20):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(smooth(double_dqn_rewards), label='Double DQN')
    plt.plot(smooth(dueling_dqn_rewards), label='Dueling DQN')
    plt.title('Reward Comparison (Player Mode)')
    plt.xlabel('Epochs')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(smooth(double_dqn_success), label='Double DQN')
    plt.plot(smooth(dueling_dqn_success), label='Dueling DQN')
    plt.title('Success Rate Comparison (Smoothed)')
    plt.xlabel('Epochs')
    plt.ylabel('Success Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hw32_player_results.png')
    print("Saved plot to hw32_player_results.png")

if __name__ == '__main__':
    main()
