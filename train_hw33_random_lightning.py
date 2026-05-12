import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import random
import matplotlib.pyplot as plt
from Gridworld import Gridworld
from replay_buffer import ExperienceReplay
from models import DQN

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

class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=32):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for _ in range(len(self.buffer) // self.sample_size):
            s, a, r, ns, d = self.buffer.sample(self.sample_size)
            for i in range(self.sample_size):
                yield s[i], a[i], r[i], ns[i], d[i]

class LightningDQN(pl.LightningModule):
    def __init__(self, state_dim=64, action_dim=4, batch_size=32, lr=1e-3, gamma=0.9):
        super(LightningDQN, self).__init__()
        self.save_hyperparameters()
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_fn = nn.MSELoss()
        
        self.env = Gridworld(size=4, mode='random')
        self.replay_buffer = ExperienceReplay(capacity=5000)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        self.episode_reward = 0
        self.step_count = 0
        self.state = get_state(self.env)
        
        self.rewards_history = []
        self.successes_history = []
        
        # Pre-fill buffer
        self.populate_buffer(1000)

    def populate_buffer(self, steps):
        for _ in range(steps):
            action = random.randint(0, 3)
            self.env.makeMove(action_map[action])
            next_state = get_state(self.env)
            reward = self.env.reward()
            done = is_done(self.env)
            self.replay_buffer.push(self.state, action, reward, next_state, done)
            
            if done or self.step_count >= 50:
                self.env = Gridworld(size=4, mode='random')
                self.state = get_state(self.env)
                self.step_count = 0
            else:
                self.state = next_state
                self.step_count += 1

    def forward(self, x):
        return self.model(x)

    def play_step(self):
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(self.state).unsqueeze(0).to(self.device))
                action = torch.argmax(q_values).item()

        self.env.makeMove(action_map[action])
        next_state = get_state(self.env)
        reward = self.env.reward()
        done = is_done(self.env)

        self.replay_buffer.push(self.state, action, reward, next_state, done)
        self.episode_reward += reward
        self.step_count += 1

        if done or self.step_count >= 50:
            if done:
                self.successes_history.append(1 if reward == 10 else 0)
            else:
                self.successes_history.append(0)
                
            self.rewards_history.append(self.episode_reward)
            self.env = Gridworld(size=4, mode='random')
            self.state = get_state(self.env)
            self.episode_reward = 0
            self.step_count = 0
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.state = next_state

    def training_step(self, batch, batch_idx):
        self.play_step()
        
        s, a, r, ns, d = batch
        
        q_vals = self.model(s)
        q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q_val = self.target_model(ns).max(1)[0]
            target_q_val = r + self.hparams.gamma * max_next_q_val * (1 - d)

        loss = self.loss_fn(q_val, target_q_val)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Update target network
        self.target_model.load_state_dict(self.model.state_dict())
        if len(self.rewards_history) > 0:
            avg_reward = np.mean(self.rewards_history[-100:])
            avg_success = np.mean(self.successes_history[-100:]) * 100
            self.log('avg_reward', avg_reward, prog_bar=True)
            print(f"Epoch End - Avg Reward: {avg_reward:.2f}, Success: {avg_success:.1f}%, Epsilon: {self.epsilon:.2f}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        # Add learning rate scheduler (Training Tip)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = RLDataset(self.replay_buffer, sample_size=self.hparams.batch_size)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

def main():
    model = LightningDQN()
    
    # Gradient clipping is applied here (Training Tip)
    trainer = pl.Trainer(
        max_epochs=50, 
        gradient_clip_val=1.0, 
        callbacks=[LearningRateMonitor(logging_interval='step')],
        enable_progress_bar=False
    )
    
    trainer.fit(model)

    print("\n--- Final Evaluation ---")
    evaluate(model.model, 'random', num_episodes=100)

    # Plot results
    def smooth(data, window=20):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(smooth(model.rewards_history), label='Random Mode (Lightning)')
    plt.title('Reward in Random Mode (Smoothed)')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(smooth(model.successes_history), label='Random Mode (Lightning)')
    plt.title('Success Rate (Smoothed)')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hw33_random_lightning_results.png')
    print("Saved plot to hw33_random_lightning_results.png")

if __name__ == '__main__':
    main()
