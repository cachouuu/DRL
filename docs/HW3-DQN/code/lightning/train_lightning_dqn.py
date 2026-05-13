import sys
import os
# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from environment import RandomGridWorld
from replay_buffer import ReplayBuffer
from lightning_dqn import LitDQN, RLDataset
from utils import plot_comparison_curves

def train_lightning_agent():
    """
    Main loop connecting RL environment steps with PyTorch Lightning optimization.
    """
    print("Starting Training for PyTorch Lightning DQN (HW3-3)...")
    
    env = RandomGridWorld()
    
    # Hyperparameters
    num_episodes = 500
    batch_size = 32
    buffer_capacity = 10000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    target_update = 10
    
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # Initialize Lightning Module
    model = LitDQN(
        state_size=env.observation_space_n, 
        action_size=env.action_space_n, 
        batch_size=batch_size
    )
    
    # Initialize Lightning Trainer
    # Gradient clipping is applied here (gradient_clip_val=1.0)
    trainer = pl.Trainer(
        max_epochs=1, # We train manually per episode to control RL interaction
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=1.0 # Training Tip: Gradient Clipping
    )
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # 1. Fill the buffer / Interacting with the environment
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, env.action_space_n - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
                    
            next_state, reward, done = env.step(action)
            total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
            if steps > 100:
                break
                
        # 2. PyTorch Lightning Optimization Phase
        if len(replay_buffer) >= batch_size:
            # We wrap the buffer in our IterableDataset
            dataset = RLDataset(replay_buffer, sample_size=batch_size)
            # DataLoader feeds the Lightning module
            dataloader = DataLoader(dataset, batch_size=batch_size)
            
            # This triggers model.training_step() and updates the network
            # Because max_epochs=1, it will process the dataloader exactly once
            trainer.fit(model, train_dataloaders=dataloader)
            
        # 3. Post-episode updates
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Sync Target Network
        if episode % target_update == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())
            
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.2f}")

    print("Lightning Training Complete!")
    
    # Save Plot
    results = {'Lightning Random DQN': episode_rewards}
    plot_comparison_curves(
        results, 
        save_dir="../../results", 
        filename="hw3_3_lightning_random_curve.png",
        title="HW3-3 Lightning DQN Training Curve (Random Mode)"
    )

if __name__ == "__main__":
    train_lightning_agent()
