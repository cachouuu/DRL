import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from environment import RandomGridWorld
from dqn_model import DuelingDQN
from utils import plot_comparison_curves
from prioritized_replay_buffer import PrioritizedReplayBuffer

def train_simplified_rainbow():
    print("Starting Simplified Rainbow DQN Training (HW3-4 Bonus)...")
    env = RandomGridWorld()
    
    # Hyperparameters
    num_episodes = 500
    batch_size = 32
    gamma = 0.99
    lr = 0.001
    target_update = 10
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    n_step = 3  # N-step return
    
    # Component 1 & 2: Dueling Architecture & Double DQN
    policy_net = DuelingDQN(env.observation_space_n, env.action_space_n)
    target_net = DuelingDQN(env.observation_space_n, env.action_space_n)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    # Component 3: Prioritized Experience Replay (PER)
    replay_buffer = PrioritizedReplayBuffer(10000)
    
    # Component 4: N-Step Return
    # We use a temporary deque to accumulate N steps before pushing to the main buffer
    n_step_buffer = deque(maxlen=n_step)
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        n_step_buffer.clear()
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, env.action_space_n - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()
                    
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Store immediate step in n-step buffer
            n_step_buffer.append((state, action, reward, next_state, done))
            
            # If we have accumulated n steps, process and push to PER
            if len(n_step_buffer) == n_step:
                n_state, n_action, _, _, _ = n_step_buffer[0]
                n_reward = 0
                # Accumulate discounted reward: R_t + gamma*R_{t+1} + gamma^2*R_{t+2}
                for i in range(n_step):
                    n_reward += (gamma ** i) * n_step_buffer[i][2]
                
                _, _, _, n_next_state, n_done = n_step_buffer[-1]
                replay_buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
            
            state = next_state
            steps += 1
            
            # End episode logic and flush remaining steps in n_step_buffer
            if done or steps > 100:
                while len(n_step_buffer) > 0:
                    n_state, n_action, _, _, _ = n_step_buffer[0]
                    n_reward = 0
                    current_n = len(n_step_buffer)
                    for i in range(current_n):
                        n_reward += (gamma ** i) * n_step_buffer[i][2]
                    _, _, _, n_next_state, n_done = n_step_buffer[-1]
                    replay_buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
                    n_step_buffer.popleft()
                break

        # Train network
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
            
            # Current Q-values
            current_q_values = policy_net(states).gather(1, actions)
            
            # Double DQN Logic with N-Step Returns
            with torch.no_grad():
                next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                target_q_values_next = target_net(next_states).gather(1, next_actions)
                # Discount by gamma^n
                target_q_values = rewards + ((gamma ** n_step) * target_q_values_next * (1 - dones))
            
            # Calculate TD Error for PER updates
            td_errors = torch.abs(target_q_values - current_q_values).detach().numpy()
            replay_buffer.update_priorities(indices, td_errors)
            
            # Optimize (weighted by Importance Sampling)
            loss = (weights * (current_q_values - target_q_values)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.2f}")
            
    print("Simplified Rainbow Training Complete!")
    
    # Save Plot
    results = {'Simplified Rainbow DQN': episode_rewards}
    plot_comparison_curves(
        results, 
        save_dir="../../results", 
        filename="hw3_4_rainbow_bonus_curve.png",
        title="HW3-4 Simplified Rainbow DQN (Random Mode)"
    )

if __name__ == "__main__":
    train_simplified_rainbow()
