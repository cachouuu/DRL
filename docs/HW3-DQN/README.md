# HW3: Deep Q-Network & Variants

This folder contains the report and source code for HW3.

## Structure

*   `index.html`: The main web report summarizing the homework.
*   `report.md`: A markdown version of the homework report.
*   `code/`: Python source code implementations.
    *   `environment.py`: Static GridWorld environment.
    *   `replay_buffer.py`: Experience Replay Buffer class.
    *   `dqn_model.py`: Neural network implementation for DQN.
    *   `train_naive_dqn.py`: Main training loop for HW3-1.
    *   `utils.py`: Helper functions for plotting and logging.
*   `results/`: Directory for storing training plots and logs.

## How to Run HW3-1 (Naive DQN)

1. Make sure you have PyTorch and Matplotlib installed.
   ```bash
   pip install torch matplotlib numpy
   ```
2. Navigate to the `code` directory.
   ```bash
   cd /Users/cachou/Documents/DRL/docs/HW3-DQN/code
   ```
3. Run the training script.
   ```bash
   python3 train_naive_dqn.py
   ```
4. The learning curve plot will be saved in `docs/HW3-DQN/results/hw3_1_learning_curve.png`.

## How to Run HW3-2 (Compare Variants)

1. Navigate to the `code` directory.
   ```bash
   cd /Users/cachou/Documents/DRL/docs/HW3-DQN/code
   ```
2. Run the comparison script.
   ```bash
   python3 train_compare_dqn_variants.py
   ```
   
*(Note: If your local environment defaults to Python 3 with the `python` command, you can replace `python3` with `python` in the commands above.)*
3. The comparison plot will be saved in `docs/HW3-DQN/results/hw3_2_comparison_curve.png`.

## How to Run HW3-3 (PyTorch Lightning Random Mode)

1. Navigate to the `lightning` directory.
   ```bash
   cd /Users/cachou/Documents/DRL/docs/HW3-DQN/code/lightning
   ```
2. Run the PyTorch Lightning script.
   ```bash
   python3 train_lightning_dqn.py
   ```
3. The training plot will be saved in `docs/HW3-DQN/results/hw3_3_lightning_random_curve.png` (using the comparison utility).

## How to Run HW3-4 (Bonus: Simplified Rainbow DQN)

1. Navigate to the `rainbow` directory.
   ```bash
   cd /Users/cachou/Documents/DRL/docs/HW3-DQN/code/rainbow
   ```
2. Run the Simplified Rainbow DQN script.
   ```bash
   python3 train_rainbow_dqn.py
   ```
3. The bonus training plot will be saved in `docs/HW3-DQN/results/hw3_4_rainbow_bonus_curve.png`.
