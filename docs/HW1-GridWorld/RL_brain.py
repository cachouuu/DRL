"""
RL_brain.py
-----------
Tabular Q-Learning agent with Boltzmann (softmax) stochastic policy (HW 變化).

HW change applied:
  1. Agent stochastic policy p(a|s)
       → Boltzmann policy: p(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)
       → τ (temperature) replaces ε; high τ → exploration, low τ → exploitation

Key hyper-parameters:
  learning_rate (α) = 0.01
  gamma         (γ) = 0.9   (discount factor)
  temperature   (τ) = 1.0   (initial Boltzmann temperature, decays to 0.1)
"""

import numpy as np
import pandas as pd


class QLearningTable:
    """
    Tabular Q-Learning agent with Boltzmann (softmax) action selection.

    Parameters
    ----------
    actions       : list of str
    learning_rate : float  – α
    reward_decay  : float  – γ
    temperature   : float  – τ (initial Boltzmann temperature)  [HW 變化 1]
    temp_min      : float  – minimum τ after decay
    temp_decay    : float  – multiplicative decay applied each episode
    """

    def __init__(self,
                 actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 temperature=1.0,
                 temp_min=0.1,
                 temp_decay=0.995):
        self.actions    = actions
        self.lr         = learning_rate
        self.gamma      = reward_decay
        self.tau        = temperature      # current temperature  [HW 變化 1]
        self.tau_min    = temp_min
        self.tau_decay  = temp_decay

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def choose_action(self, observation):
        """
        HW 變化 1 – Boltzmann (softmax) stochastic policy p(a|s).

        p(a|s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)

        High τ  → near-uniform distribution (exploration)
        Low  τ  → near-deterministic greedy (exploitation)
        """
        self.check_state_exist(observation)
        q_vals = self.q_table.loc[observation, :].values.astype(float)

        # Numerical stability: subtract max before exp
        q_shifted = q_vals - q_vals.max()
        exp_q     = np.exp(q_shifted / max(self.tau, 1e-8))
        probs     = exp_q / exp_q.sum()

        action = np.random.choice(self.actions, p=probs)
        return action

    def decay_temperature(self):
        """Call once per episode to reduce τ towards τ_min."""
        self.tau = max(self.tau_min, self.tau * self.tau_decay)

    def learn(self, s, a, r, s_):
        """
        Q-Learning (off-policy) Bellman update:
            Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        """Lazily add a row of zeros for unseen states."""
        if state not in self.q_table.index:
            new_row = pd.Series(
                [0.0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])
