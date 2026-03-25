"""
RL_brain.py
-----------
Implements a tabular Q-Learning agent.

The Q-table is a pandas DataFrame where:
  - rows   = state strings (e.g. "2,3")
  - columns= action strings ('up', 'down', 'left', 'right')
  - values = Q-values, initialised to 0

Key hyper-parameters (matching the spec):
  learning_rate (α) = 0.01
  gamma         (γ) = 0.9   (discount factor)
  epsilon       (ε) = 0.9   (greedy probability)
"""

import numpy as np
import pandas as pd


class QLearningTable:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    actions       : list of str  - available action names
    learning_rate : float        - α in the Bellman update
    reward_decay  : float        - γ, discount factor for future rewards
    e_greedy      : float        - ε, probability of choosing the greedy action
    """

    def __init__(self,
                 actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9):
        self.actions = actions              # list of action strings
        self.lr = learning_rate             # α
        self.gamma = reward_decay           # γ
        self.epsilon = e_greedy             # ε

        # Q-table: states × actions, all zeros initially
        self.q_table = pd.DataFrame(
            columns=self.actions, dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def choose_action(self, observation):
        """
        ε-greedy action selection.

        With probability ε  → choose the action with the highest Q-value
                              (ties broken randomly).
        With probability 1-ε→ choose a random action (exploration).

        Parameters
        ----------
        observation : str  - current state key

        Returns
        -------
        action : str
        """
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # Exploit: pick the action(s) with maximum Q-value
            state_action = self.q_table.loc[observation, :]
            # Shuffle to break ties randomly, then take argmax
            action = np.random.choice(
                state_action[state_action == state_action.max()].index
            )
        else:
            # Explore: random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        """
        Q-Learning (off-policy) update rule:

            Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

        Parameters
        ----------
        s  : str   - current state
        a  : str   - action taken
        r  : float - reward received
        s_ : str   - next state (use 'terminal' if episode ended)
        """
        self.check_state_exist(s_)

        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            # Non-terminal: bootstrap from next state
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            # Terminal: no future reward
            q_target = r

        # Bellman update
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        """
        Add a new row of zeros to the Q-table if `state` has not been
        seen before. This implements lazy initialisation of states.

        Parameters
        ----------
        state : str  - state key to check / insert
        """
        if state not in self.q_table.index:
            new_row = pd.Series(
                [0.0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = pd.concat(
                [self.q_table, new_row.to_frame().T]
            )
