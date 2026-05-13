# HW3-4: Simplified Rainbow DQN (Bonus)

This directory contains the code for the bonus assignment: implementing a Simplified Rainbow DQN to solve the Random Mode GridWorld.

## Components Included
*   **Double DQN:** Decouples action selection and evaluation.
*   **Dueling DQN:** Separate value and advantage streams.
*   **Prioritized Experience Replay (PER):** A simplified array-based implementation that samples transitions proportionally to their TD error.
*   **n-step returns:** Accumulates multi-step rewards to propagate values faster.

## Components Excluded
*   Distributional DQN
*   Noisy Networks
These were excluded to keep the implementation beginner-friendly, computationally light, and safe to run on CPUs.

## How to Run

1. Make sure you are in this directory:
   ```bash
   cd /Users/cachou/Documents/DRL/docs/HW3-DQN/code/rainbow
   ```
2. Run the script:
   ```bash
   python3 train_rainbow_dqn.py
   ```
3. The resulting plot will be saved to `docs/HW3-DQN/results/hw3_4_rainbow_bonus_curve.png`.
