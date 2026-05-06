# HW2: Cliff Walking (Q-learning vs SARSA)

This is an interactive web-based implementation of the classic Reinforcement Learning environment **Cliff Walking**.

## Overview
This project compares two fundamental Temporal Difference (TD) learning algorithms:
- **Q-Learning** (Off-policy): Learns the optimal, shortest path but suffers from catastrophic failures during exploration.
- **SARSA** (On-policy): Learns a safer, slightly longer path by accounting for its own random exploration, resulting in better online performance during training.

## How to use
No backend server is required. This project uses vanilla JavaScript and Chart.js.
Just open `index.html` in your web browser.

You can interactively adjust the following parameters:
- Algorithm (Q-learning, SARSA, or Compare both)
- $\epsilon$ (Epsilon)
- $\alpha$ (Learning Rate)
- $\gamma$ (Discount Factor)
- Number of episodes
- Animation speed

## File Structure
- `index.html`: Main layout and explanation text (in Chinese).
- `style.css`: Styles for the grid environment, controls, and layout.
- `script.js`: Core RL logic (Agent/Environment) and DOM manipulation.
