"""
run_this.py
-----------
Training script for Q-Learning on the GridWorld.

HW 變化 applied:
  1. Agent stochastic policy p(a|s)   → Boltzmann policy with temperature τ
  2. Stochastic reward  p(r|s,a)      → Gaussian noise on rewards
  3. Stochastic transition p(s'|s,a)  → slip probability (noise)

Usage (called by app.py via subprocess):
    python run_this.py <n> <start_row> <start_col> <end_row> <end_col> <block_list>

  where <block_list> is a JSON-encoded list of [row, col] pairs, e.g.
      '[[1,2],[3,4]]'

Console output shows:
  S  = agent / start
  G  = goal
  X  = obstacle
  .  = free cell
  A  = current agent position (if different from start)
"""

import sys
import json
import time
import numpy as np

from maze_env import GridWorld
from RL_brain import QLearningTable


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MAX_EPISODES  = 100    # number of training episodes
MAX_STEPS     = 200    # max steps per episode
RENDER_EVERY  = 10     # print the grid every N episodes
SLEEP_SEC     = 0.05   # pause between rendered steps (seconds)

# Stochastic MDP hyper-parameters (HW 變化)
TRANS_NOISE   = 0.1    # p(s'|s,a) slip probability
REWARD_NOISE  = 0.1    # p(r|s,a)  reward Gaussian std-dev
INIT_TEMP     = 1.0    # τ initial Boltzmann temperature
TEMP_MIN      = 0.1    # τ floor
TEMP_DECAY    = 0.995  # τ decay per episode


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def render_grid(env, agent_pos):
    """
    Print a text representation of the grid to stdout.

    Symbols:
      S  = start
      G  = goal
      X  = obstacle
      A  = current agent position (overwrites S if at start)
      .  = free cell
    """
    grid = []
    for r in range(env.n):
        row = []
        for c in range(env.n):
            cell = (r, c)
            if cell == tuple(agent_pos):
                row.append(' A ')
            elif cell == tuple(env.end_pos):
                row.append(' G ')
            elif cell in [tuple(b) for b in env.block_pos]:
                row.append(' X ')
            elif cell == tuple(env.start_pos):
                row.append(' S ')
            else:
                row.append(' . ')
        grid.append(''.join(row))

    print('\n'.join(grid))
    print()


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def run(n, start_pos, end_pos, block_pos):
    """
    Initialise environment + Q-table, then run Q-Learning training.

    Parameters
    ----------
    n         : int
    start_pos : (row, col)
    end_pos   : (row, col)
    block_pos : list of (row, col)
    """
    # Build stochastic environment (HW 變化 2 & 3)
    env = GridWorld(n=n,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    block_pos=block_pos,
                    noise=TRANS_NOISE,
                    reward_noise=REWARD_NOISE)

    # Build Q-Learning agent with Boltzmann policy (HW 變化 1)
    RL = QLearningTable(
        actions=GridWorld.ACTION_LIST,
        temperature=INIT_TEMP,
        temp_min=TEMP_MIN,
        temp_decay=TEMP_DECAY,
    )

    print(f"=== GridWorld Q-Learning (Stochastic MDP) ===")
    print(f"Grid size        : {n}×{n}")
    print(f"Start            : {start_pos}")
    print(f"Goal             : {end_pos}")
    print(f"Obstacles        : {block_pos}")
    print(f"Episodes         : {MAX_EPISODES}")
    print(f"Policy           : Boltzmann (τ₀={INIT_TEMP}, min={TEMP_MIN}, decay={TEMP_DECAY})")
    print(f"Transition noise : {TRANS_NOISE} (slip probability)")
    print(f"Reward noise σ  : {REWARD_NOISE} (Gaussian std-dev)\n")

    successes = 0

    for episode in range(MAX_EPISODES):
        state      = env.reset()
        step_count = 0
        total_reward = 0.0

        should_render = (episode % RENDER_EVERY == 0) or (episode == MAX_EPISODES - 1)

        while True:
            if should_render:
                print(f"--- Episode {episode + 1}, Step {step_count + 1} | τ={RL.tau:.3f} ---")
                render_grid(env, env.get_current_pos())
                time.sleep(SLEEP_SEC)

            action = RL.choose_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                RL.learn(state, action, reward, 'terminal')
            else:
                RL.learn(state, action, reward, next_state)

            state      = next_state
            step_count += 1

            if done or step_count >= MAX_STEPS:
                # Determine outcome by checking position vs goal/obstacle
                pos = tuple(env.get_current_pos())
                if pos == tuple(env.end_pos):
                    outcome = "GOAL"
                    successes += 1
                elif pos in [tuple(b) for b in env.block_pos]:
                    outcome = "OBSTACLE"
                else:
                    outcome = "TIMEOUT"

                print(f"Episode {episode + 1:>4d} | Steps: {step_count:>4d} | "
                      f"Reward: {total_reward:+.2f} | τ: {RL.tau:.3f} | Outcome: {outcome}")
                break

        # Decay temperature at end of episode
        RL.decay_temperature()

    success_rate = successes / MAX_EPISODES * 100
    print(f"\n=== Training Complete ===")
    print(f"Success rate: {successes}/{MAX_EPISODES} ({success_rate:.1f}%)")
    print("Final Q-table:")
    print(RL.q_table.to_string())


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

if __name__ == '__main__':
    # Parse command-line arguments passed by app.py
    # argv: n start_row start_col end_row end_col block_json
    if len(sys.argv) < 7:
        print("Usage: python run_this.py n start_row start_col "
              "end_row end_col block_json")
        sys.exit(1)

    n_arg     = int(sys.argv[1])
    s_row     = int(sys.argv[2])
    s_col     = int(sys.argv[3])
    e_row     = int(sys.argv[4])
    e_col     = int(sys.argv[5])
    blocks    = json.loads(sys.argv[6])   # list of [row, col]

    start = (s_row, s_col)
    end   = (e_row, e_col)
    blk   = [tuple(b) for b in blocks]

    run(n_arg, start, end, blk)
