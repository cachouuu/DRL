# HW3: DQN & Its Variants Report

## HW3-1: Naive DQN for Static Mode
We implemented a Deep Q-Network from scratch for a static GridWorld environment. The core components of our implementation include:

*   **Basic DQN Model:** We used a simple Multi-Layer Perceptron (MLP) with PyTorch to approximate the Q-value function. It takes the one-hot encoded state as input and outputs the predicted Q-values for all 4 possible actions.
*   **Experience Replay Buffer:** Instead of learning strictly online, we store transitions $(S, A, R, S', done)$ in a `deque`-based buffer. We randomly sample mini-batches during training. This breaks the temporal correlation between consecutive samples and provides i.i.d. data to stabilize the neural network.
*   **Epsilon-Greedy Action Selection:** To balance exploration and exploitation, the agent explores random actions with a probability of $\epsilon$. This $\epsilon$ decays exponentially over time to allow the agent to exploit its learned policy once it discovers the goal.
*   **Target Network:** We maintain a separate, frozen copy of the DQN (the Target Network) to compute the target Q-values in the Bellman equation. Its weights are periodically synced with the main policy network. This prevents the "moving target" problem and stabilizes the loss.
*   **Loss Function:** We use Mean Squared Error (MSE) loss between the current predicted Q-values and the target Q-values: `Target = R + gamma * max(Q_target(S'))`.

## HW3-2: Double DQN and Dueling DQN for Player Mode
To handle the increased variability of Player Mode, we upgraded the agent with two architectural enhancements. We compared their performance using the `train_compare_dqn_variants.py` script.

*   **Double DQN:** Standard DQN suffers from overestimation bias because the same network is used to both select the best action and evaluate it (using the `max` operator). Double DQN addresses this by decoupling these steps:
    1.  The *Policy Network* selects the best action for the next state.
    2.  The *Target Network* evaluates the Q-value of that selected action.
    This small change prevents the propagation of overly optimistic Q-value estimates.

*   **Dueling DQN:** Instead of estimating $Q(s,a)$ directly in one stream, the Dueling Architecture splits the network into two streams:
    1.  A **State-Value stream $V(s)$**: Estimates how good it is to be in a particular state.
    2.  An **Action-Advantage stream $A(s,a)$**: Estimates the relative advantage of taking a specific action compared to others in that state.
    These are combined in the final layer: $Q(s,a) = V(s) + A(s,a) - mean(A(s,a))$. This improves sample efficiency because the network can learn which states are valuable without needing to evaluate the effect of every action in those states.

## HW3-3: DQN for Random Mode with Training Tips
In Random Mode, the start and goal positions change every episode. This is significantly harder than Static Mode because the agent can no longer simply memorize a fixed path. Instead, it must learn to generalize its understanding of the state space (e.g., "moving closer to the goal is good, regardless of where I started").

DQN training is notoriously unstable. Q-values can easily diverge, and the policy can collapse. To address this, we refactored the codebase to use **PyTorch Lightning** to enforce a cleaner training loop structure and integrated several training tips:
*   **PyTorch Lightning Conversion:** Moving from plain PyTorch to Lightning abstracted away boilerplate code (like calling `loss.backward()`, `optimizer.step()`, and manual `device` management). We wrapped the Replay Buffer in a custom `IterableDataset` to seamlessly integrate with Lightning's `DataLoader` ecosystem.
*   **Gradient Clipping:** Limits the maximum size of gradients during backpropagation. This prevents exploding gradients, which are common early in training when Q-value targets (derived from the random initial weights of the target network) are highly unstable.
*   **Learning Rate Scheduling:** We applied a `StepLR` scheduler. As the agent gets closer to the optimal policy, a large learning rate might cause it to overshoot. Scheduling decreases the learning rate over time, allowing for fine-tuning and stable convergence.

## Bonus: Simplified Rainbow DQN
Rainbow DQN is a state-of-the-art reinforcement learning agent that combines several independent, orthogonal improvements to the original DQN algorithm into a single, powerful architecture. For this bonus, we implemented a **Simplified Rainbow DQN** to tackle the Random Mode GridWorld.

### Included Components
1.  **Double DQN:** Addresses overestimation bias by decoupling action selection and evaluation.
2.  **Dueling DQN:** Separate value and advantage streams for better sample efficiency.
3.  **Prioritized Experience Replay (PER):** Standard replay buffers sample transitions uniformly. PER assigns a "priority" to each transition based on its Temporal Difference (TD) error ($|R + \gamma \max Q - Q|$). Transitions that the network is highly "surprised" by (large TD error) are sampled more frequently. We used Importance-Sampling weights to correct the bias introduced by non-uniform sampling.
4.  **N-Step Returns:** Instead of calculating the target using just the immediate next state ($R_{t+1} + \gamma \max Q(S_{t+1})$), n-step returns accumulate rewards over $n$ steps before bootstrapping: $\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max Q(S_{t+n})$. This helps propagate reward signals backwards much faster, which is critical in Random Mode where the goal moves every episode.

### Excluded Components
To keep the implementation computationally accessible for CPU training and beginner-friendly, we **did not** include:
*   **Distributional RL:** Predicts the full probability distribution of returns instead of just the expected mean. This requires significantly more complex network outputs and loss functions.
*   **Noisy Nets:** Replaces standard $\epsilon$-greedy exploration with learned parametric noise added directly to the network weights.

### Results Discussion
By combining PER (focusing on surprising transitions) and N-step returns (faster reward propagation) with our existing Double/Dueling architecture, the Simplified Rainbow agent is structurally much better equipped to handle the shifting goals of Random Mode. When running `train_rainbow_dqn.py`, the agent learns to generalize finding the goal significantly faster than the Naive DQN, demonstrating the compounding benefits of these architectural enhancements.

## Homework Questions
1.  **Why does standard Q-learning struggle in environments with large state spaces?**
    It suffers from the curse of dimensionality and cannot generalize knowledge across similar states.
2.  **How does the Experience Replay Buffer improve training?**
    It breaks temporal correlation, providing i.i.d. samples, and allows the agent to learn from past experiences multiple times (improving data efficiency).
3.  **What is the primary benefit of the Dueling Architecture?**
    It explicitly separates the estimation of state value and action advantage, allowing the network to learn which states are valuable without needing to evaluate every action.
