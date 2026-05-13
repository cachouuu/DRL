import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Basic Deep Q-Network (MLP).
    Approximates the Q-value function: Q(s, a; theta)
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        
        # A simple two-hidden-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass mapping state to Q-values for all actions.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network.
    Splits the network into a Value stream and an Advantage stream.
    Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Linear(input_dim, hidden_dim)
        
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Ensure input has a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = F.relu(self.feature_layer(x))
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine streams using the standard Dueling formula
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
