import random
from collections import deque
import torch

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    This breaks the correlation between consecutive samples and stabilizes training.
    """
    def __init__(self, capacity):
        # deque automatically removes oldest elements when maxlen is reached
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions.
        This provides i.i.d. samples to the neural network.
        """
        # Randomly choose 'batch_size' transitions from the buffer
        transitions = random.sample(self.buffer, batch_size)
        
        # Unzip the batch of transitions into separate lists
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert to PyTorch tensors for neural network processing
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1) # Shape: (batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
