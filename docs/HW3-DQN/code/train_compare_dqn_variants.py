import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from environment import StaticGridWorld
from dqn_model import DQN, DuelingDQN
from replay_buffer import ReplayBuffer
from utils import plot_comparison_curves

def train_agent(env, model_class, is_double_dqn=False, num_episodes=500):
    """
    Trains a specific DQN variant and returns the episode rewards.
    """
    # Hyperparameters
    batch_size = 32
    gamma = 0.99
    lr = 0.001
    target_update = 10
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    buffer_capacity = 10000

    # Initialize networks and buffer
    policy_net = model_class(env.observation_space_n, env.action_space_n)
    target_net = model_class(env.observation_space_n, env.action_space_n)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # 1. Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, env.action_space_n - 1)
            else:
                with torch.no_grad():
                    # Add a batch dimension of size 1 so shape is [1, state_size]
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            # 2. Step environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            
            # 4. Train network
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Current Q-values
                current_q_values = policy_net(states).gather(1, actions)
                
                # Target Q-values
                with torch.no_grad():
                    if is_double_dqn:
                        # --- DOUBLE DQN LOGIC ---
                        # 1. Use the POLICY network to select the best action in the next state
                        next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                        # 2. Use the TARGET network to evaluate that specific action
                        target_q_values_next = target_net(next_states).gather(1, next_actions)
                    else:
                        # --- BASIC DQN LOGIC ---
                        # Use the TARGET network to both select and evaluate the maximum action
                        target_q_values_next = target_net(next_states).max(1)[0].unsqueeze(1)
                        
                    target_q_values = rewards + (gamma * target_q_values_next * (1 - dones))
                
                # Optimize
                loss = loss_fn(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if steps > 100: # Prevent infinite loops
                break
                
        # 5. Decay epsilon and update target network
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        episode_rewards.append(total_reward)
        
    return episode_rewards

def run_comparison():
    print("Starting Comparison of DQN Variants (HW3-2)...")
    env = StaticGridWorld() # Using the same environment for fair comparison
    
    results = {}
    
    print("\n--- Training Basic DQN ---")
    results['Basic DQN'] = train_agent(env, DQN, is_double_dqn=False)
    
    print("\n--- Training Double DQN ---")
    results['Double DQN'] = train_agent(env, DQN, is_double_dqn=True)
    
    print("\n--- Training Dueling DQN ---")
    # Note: Dueling DQN is an architecture change, but we can also combine it with Double DQN.
    # Here we just evaluate the Dueling architecture with standard Q-learning updates to compare purely structural differences.
    results['Dueling DQN'] = train_agent(env, DuelingDQN, is_double_dqn=False)
    
    print("\nTraining Complete! Generating plot...")
    plot_comparison_curves(results)

if __name__ == "__main__":
    run_comparison()
