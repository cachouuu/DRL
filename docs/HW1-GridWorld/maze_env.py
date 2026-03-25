"""
maze_env.py
-----------
Defines the GridWorld environment for Q-learning.

The grid is an n×n matrix. The agent can move up, down, left, right.
Special cells:
  - Start  : where the agent begins each episode
  - Goal   : reaching here gives reward +1 and ends the episode
  - Obstacle: stepping on one gives reward -1 and ends the episode
  - Normal  : reward 0, episode continues
"""

import numpy as np


class GridWorld:
    """
    GridWorld environment.

    Parameters
    ----------
    n         : int  - grid size (n × n)
    start_pos : (row, col) tuple
    end_pos   : (row, col) tuple  (goal)
    block_pos : list of (row, col) tuples  (obstacles)
    """

    # Available actions and their (Δrow, Δcol) effect on the agent
    ACTIONS = {
        'up':    (-1,  0),
        'down':  ( 1,  0),
        'left':  ( 0, -1),
        'right': ( 0,  1),
    }
    ACTION_LIST = ['up', 'down', 'left', 'right']

    def __init__(self, n, start_pos, end_pos, block_pos):
        self.n = n                      # grid dimension
        self.start_pos = start_pos      # (row, col) of starting cell
        self.end_pos = end_pos          # (row, col) of goal cell
        self.block_pos = block_pos      # list of (row, col) obstacles
        self.state = None               # current agent position

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Place the agent at the start position and return the initial state."""
        self.state = list(self.start_pos)   # [row, col]
        return self._state_to_str(self.state)

    def step(self, action):
        """
        Apply `action` and return (next_state, reward, done).

        Parameters
        ----------
        action : str  - one of 'up', 'down', 'left', 'right'

        Returns
        -------
        next_state : str representation of new (row, col)
        reward     : float
        done       : bool  - True when episode should end
        """
        dr, dc = self.ACTIONS[action]
        row, col = self.state

        # Compute tentative next position
        new_row = row + dr
        new_col = col + dc

        # Clip to grid boundaries (agent stays in grid)
        new_row = max(0, min(self.n - 1, new_row))
        new_col = max(0, min(self.n - 1, new_col))

        self.state = [new_row, new_col]
        next_state = self._state_to_str(self.state)

        # Determine reward and termination
        if tuple(self.state) == tuple(self.end_pos):
            reward = 1          # reached goal
            done = True
        elif tuple(self.state) in [tuple(b) for b in self.block_pos]:
            reward = -1         # hit obstacle
            done = True
        else:
            reward = 0          # normal step
            done = False

        return next_state, reward, done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_str(self, state):
        """Convert [row, col] list to a hashable string key for the Q-table."""
        return f"{state[0]},{state[1]}"

    def get_state_str(self):
        """Return the current state as a string."""
        return self._state_to_str(self.state)

    def get_current_pos(self):
        """Return current agent position as a (row, col) tuple."""
        return tuple(self.state)
