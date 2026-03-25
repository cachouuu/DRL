"""
maze_env.py
-----------
GridWorld environment with stochastic MDP transitions (HW 變化).

HW changes applied:
  2. Environment random variable reward  p(r | s, a)
       → Gaussian noise N(0, reward_noise) added to every reward signal
  3. Environment random variable next state  p(s' | s, a)
       → With probability `noise` the intended action is replaced by a
         uniformly random action (slip / transition noise)
"""

import numpy as np


class GridWorld:
    """
    GridWorld environment.

    Parameters
    ----------
    n            : int   - grid size (n × n)
    start_pos    : (row, col)
    end_pos      : (row, col)   goal cell
    block_pos    : list of (row, col)   obstacle cells
    noise        : float  - P(slip): probability of executing a random
                            action instead of the intended one  [HW 變化 3]
    reward_noise : float  - std-dev of Gaussian noise added to rewards [HW 變化 2]
    """

    ACTIONS = {
        'up':    (-1,  0),
        'down':  ( 1,  0),
        'left':  ( 0, -1),
        'right': ( 0,  1),
    }
    ACTION_LIST = ['up', 'down', 'left', 'right']

    def __init__(self, n, start_pos, end_pos, block_pos,
                 noise=0.1, reward_noise=0.1):
        self.n            = n
        self.start_pos    = start_pos
        self.end_pos      = end_pos
        self.block_pos    = block_pos
        self.noise        = noise          # transition slip probability
        self.reward_noise = reward_noise   # reward Gaussian std-dev
        self.state        = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Place the agent at start and return the initial state string."""
        self.state = list(self.start_pos)
        return self._state_to_str(self.state)

    def step(self, action):
        """
        Apply `action` (possibly slipping) and return (next_state, reward, done).

        HW 變化 3 – stochastic transition p(s'|s,a):
            With probability self.noise, override action with a random one.

        HW 變化 2 – stochastic reward p(r|s,a):
            Add Gaussian noise N(0, reward_noise) to the base reward.

        Parameters
        ----------
        action : str  – one of 'up', 'down', 'left', 'right'

        Returns
        -------
        next_state : str
        reward     : float
        done       : bool
        """
        # ── HW 變化 3: slip / transition noise ──────────────────────
        if np.random.random() < self.noise:
            action = np.random.choice(self.ACTION_LIST)

        dr, dc  = self.ACTIONS[action]
        row, col = self.state

        new_row = max(0, min(self.n - 1, row + dr))
        new_col = max(0, min(self.n - 1, col + dc))

        self.state  = [new_row, new_col]
        next_state  = self._state_to_str(self.state)

        # ── Base reward ──────────────────────────────────────────────
        if tuple(self.state) == tuple(self.end_pos):
            base_reward = 1.0
            done        = True
        elif tuple(self.state) in [tuple(b) for b in self.block_pos]:
            base_reward = -1.0
            done        = True
        else:
            base_reward = 0.0
            done        = False

        # ── HW 變化 2: stochastic reward p(r|s,a) ───────────────────
        reward = base_reward + np.random.normal(0.0, self.reward_noise)

        return next_state, reward, done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_str(self, state):
        return f"{state[0]},{state[1]}"

    def get_state_str(self):
        return self._state_to_str(self.state)

    def get_current_pos(self):
        return tuple(self.state)
