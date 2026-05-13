import numpy as np

class StaticGridWorld:
    """
    A simple 5x5 GridWorld environment for Static Mode DQN testing.
    The agent starts at (0, 0) and wants to reach the goal at (4, 4).
    There are some static obstacles.
    """
    def __init__(self, size=5):
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        
        # Static obstacles
        self.obstacles = [(2, 2), (2, 3), (3, 1)]
        
        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space_n = 4
        # State space: One-hot encoding of the grid (size * size)
        self.observation_space_n = size * size
        
        self.current_pos = self.start_pos
        
    def reset(self):
        """Resets the environment to the starting state."""
        self.current_pos = self.start_pos
        return self._get_state()
        
    def step(self, action):
        """
        Takes an action and returns (next_state, reward, done).
        """
        row, col = self.current_pos
        
        if action == 0:   # Up
            row = max(0, row - 1)
        elif action == 1: # Down
            row = min(self.size - 1, row + 1)
        elif action == 2: # Left
            col = max(0, col - 1)
        elif action == 3: # Right
            col = min(self.size - 1, col + 1)
            
        next_pos = (row, col)
        
        # Determine reward and if episode is done
        done = False
        reward = -0.1 # Small step penalty to encourage shortest path
        
        if next_pos == self.goal_pos:
            reward = 10.0
            done = True
            self.current_pos = next_pos
        elif next_pos in self.obstacles:
            reward = -5.0
            # If we hit an obstacle, we bounce back to current_pos (don't update it)
            # You can also choose to end the episode: done = True
        else:
            self.current_pos = next_pos
            
        return self._get_state(), reward, done
        
    def _get_state(self):
        """Returns the state as a one-hot encoded numpy array."""
        state = np.zeros(self.observation_space_n, dtype=np.float32)
        idx = self.current_pos[0] * self.size + self.current_pos[1]
        state[idx] = 1.0
        return state

import random

class RandomGridWorld(StaticGridWorld):
    """
    A harder GridWorld environment where start and goal positions are randomized 
    every episode, requiring the agent to generalize rather than memorize.
    """
    def __init__(self, size=5):
        super(RandomGridWorld, self).__init__(size=size)
        # State space is larger: agent position (25) + goal position (25) = 50 dimensions
        self.observation_space_n = size * size * 2
        
    def reset(self):
        """Randomizes start and goal positions."""
        self.start_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        self.goal_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        
        # Ensure start and goal are different and not on obstacles
        while self.start_pos == self.goal_pos or self.start_pos in self.obstacles or self.goal_pos in self.obstacles:
            self.start_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            self.goal_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            
        self.current_pos = self.start_pos
        return self._get_state()
        
    def _get_state(self):
        """Returns state as concatenated one-hot arrays of [agent_pos, goal_pos]."""
        state = np.zeros(self.observation_space_n, dtype=np.float32)
        
        # Agent position
        agent_idx = self.current_pos[0] * self.size + self.current_pos[1]
        state[agent_idx] = 1.0
        
        # Goal position (offset by size*size)
        goal_idx = self.goal_pos[0] * self.size + self.goal_pos[1]
        state[(self.size * self.size) + goal_idx] = 1.0
        
        return state
