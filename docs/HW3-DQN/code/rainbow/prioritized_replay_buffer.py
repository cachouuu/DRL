import random
import numpy as np
import torch
from collections import deque

class PrioritizedReplayBuffer:
    """
    Simplified Prioritized Experience Replay (PER).
    Stores transitions with priorities and samples them proportionally.
    Uses basic array operations for simplicity instead of complex segment trees.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=1000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        """Saves a transition with maximum priority."""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch of transitions according to their priorities."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate Importance-Sampling Weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights).unsqueeze(1)

    def update_priorities(self, batch_indices, batch_priorities):
        """Updates the priorities of sampled transitions based on TD error."""
        for idx, prio in zip(batch_indices, batch_priorities):
            # Add a small epsilon to prevent priority from becoming exactly zero
            self.priorities[idx] = float(prio) + 1e-5

    def __len__(self):
        return len(self.buffer)
