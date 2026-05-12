import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from Gridworld import Gridworld
from replay_buffer import ExperienceReplay
from models import DQN

def get_state(env):
    # render_np returns shape (4, 4, 4) tensor
    state = env.board.render_np().flatten()
    return state + np.random.rand(len(state))/10.0 # Add slight noise to prevent identically 0 vectors if any

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
        
    print(f"Evaluation Results ({mode} mode):")
    print(f"Success Rate: {success_count / num_episodes * 100:.2f}%")
    print(f"Average Reward: {np.mean(total_rewards):.2f}\n")
    return success_count / num_episodes, np.mean(total_rewards)

def train_static():
    env = Gridworld(size=4, mode='static')
    model = DQN(state_dim=64, action_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    replay = ExperienceReplay(capacity=1000)

    epochs = 200
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    batch_size = 32

    losses = []
    rewards_history = []
    successes_history = []

    for epoch in range(epochs):
        # Reset environment
        env = Gridworld(size=4, mode='static')
        state = get_state(env)
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 50: # Max steps per episode
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()

            # Take step
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

            # Train
            if len(replay) >= batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                
                # Compute Q(s, a)
                q_vals = model(s)
                q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

                # Compute max Q(s', a')
                with torch.no_grad():
                    max_next_q_val = model(ns).max(1)[0]
                    target_q_val = r + gamma * max_next_q_val * (1 - d)

                loss = loss_fn(q_val, target_q_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (epoch + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_success = np.mean(successes_history[-100:]) * 100
            print(f"Epoch: {epoch+1}, Avg Reward: {avg_reward:.2f}, Success Rate: {avg_success:.1f}%, Epsilon: {epsilon:.2f}")

    evaluate(model, 'static', num_episodes=100)

    # Plotting
    def smooth(data, window=20):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Reward per Epoch (Static)')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 2)
    plt.plot(smooth(successes_history))
    plt.title('Success Rate (Smoothed)')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Training Steps')

    plt.tight_layout()
    plt.savefig('hw31_static_results.png')
    print("Saved plot to hw31_static_results.png")

if __name__ == '__main__':
    train_static()
