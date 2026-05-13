import sys
import os
# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

from dqn_model import DuelingDQN
from replay_buffer import ReplayBuffer
from environment import RandomGridWorld
import random

class RLDataset(IterableDataset):
    """
    Iterable Dataset that samples from the Replay Buffer.
    Required to bridge standard RL loops with PyTorch Lightning's DataLoader system.
    """
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 32):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.buffer.sample(self.sample_size)
        
        # Yield them one by one to form a batch in the DataLoader
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], next_states[i], dones[i]

class LitDQN(pl.LightningModule):
    """
    PyTorch Lightning Module for Dueling Double DQN with Random GridWorld.
    Incorporates training tips:
    - Gradient Clipping (handled by Lightning Trainer)
    - Learning Rate Scheduling (StepLR)
    - Target Network Syncing
    """
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, target_update=10, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        
        # Reusing DuelingDQN architecture from HW3-2
        self.policy_net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.loss_fn = nn.MSELoss()
        
        # We don't train the target net
        for param in self.target_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.policy_net(x)

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        
        # Current Q Values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN target computation
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            target_q_values_next = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.hparams.gamma * target_q_values_next * (1 - dones))
            
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Log the training loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hparams.lr)
        
        # Learning Rate Scheduling: Decrease LR by 10% every 50 epochs (episodes)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        return [optimizer], [scheduler]
