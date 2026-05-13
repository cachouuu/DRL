import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from environment import StaticGridWorld
from dqn_model import DQN
from replay_buffer import ReplayBuffer
from utils import plot_learning_curve

def train_dqn():
    """
    Main training loop for Naive DQN with Experience Replay.
    """
    # Hyperparameters
    num_episodes = 500
    batch_size = 32
    gamma = 0.99           # Discount factor
    lr = 0.001             # Learning rate
    target_update = 10     # How often to update target network
    epsilon_start = 1.0    # Exploration rate start
    epsilon_end = 0.01     # Exploration rate minimum
    epsilon_decay = 0.995  # Decay rate per episode
    buffer_capacity = 10000

    # Initialize environment, networks, and buffer
    env = StaticGridWorld()
    
    # Policy network (the one we train)
    policy_net = DQN(env.observation_space_n, env.action_space_n)
    
    # Target network (provides stable Q-value targets)
    target_net = DQN(env.observation_space_n, env.action_space_n)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target network is not trained directly
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    epsilon = epsilon_start
    episode_rewards = []
    episode_losses = []

    print("Starting Training...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_loss = 0
        steps = 0
        
        while not done:
            # 1. Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, env.action_space_n - 1) # Explore
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item() # Exploit
            
            # 2. Take action in environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
            
            # 4. Train network if buffer has enough samples
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Compute current Q values: Q(s, a)
                # policy_net(states) gives Q-values for all actions
                # gather(1, actions) selects the Q-values for the specific actions taken
                current_q_values = policy_net(states).gather(1, actions)
                
                # Compute target Q values: r + gamma * max(Q(s', a'))
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    # If done is 1.0 (True), the future reward is 0
                    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
                
                # Compute Loss and Optimize
                loss = loss_fn(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_loss += loss.item()
                
            # Prevent infinite loops if agent gets stuck
            if steps > 100:
                break
                
        # 5. Decay Epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 6. Update Target Network periodically
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Logging
        episode_rewards.append(total_reward)
        avg_loss = episode_loss / steps if steps > 0 else 0
        episode_losses.append(avg_loss)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.2f} | Avg Loss: {avg_loss:.4f}")

    print("Training Complete!")
    
    # Save the learning curve plot
    plot_learning_curve(episode_rewards, episode_losses)

if __name__ == "__main__":
    train_dqn()
