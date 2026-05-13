import matplotlib.pyplot as plt
import os

def plot_learning_curve(rewards, losses, save_dir="../results"):
    """
    Plots the episode rewards and training losses and saves them to a file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Rewards
    ax1.plot(rewards, color='blue', label='Episode Reward')
    # Simple moving average for smoothed reward
    window = min(10, len(rewards) // 10)
    if window > 0:
        smoothed_rewards = [sum(rewards[i:i+window])/window for i in range(len(rewards)-window)]
        ax1.plot(range(window, len(rewards)), smoothed_rewards, color='orange', label='Smoothed Reward')
    
    ax1.set_title("Training Rewards")
    ax1.set_ylabel("Reward")
    ax1.legend()
    
    # Plot Losses
    ax2.plot(losses, color='red', label='Loss')
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hw3_1_learning_curve.png"))
    print(f"Learning curve saved to {save_dir}/hw3_1_learning_curve.png")

def plot_comparison_curves(results_dict, save_dir="../results", filename="hw3_2_comparison_curve.png", title="DQN Variants Comparison (Player Mode)"):
    """
    Plots the smoothed learning curves for multiple DQN variants for comparison.
    results_dict: dict of format {'Variant Name': [rewards_list]}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for name, rewards in results_dict.items():
        # Calculate moving average
        window = min(10, len(rewards) // 10)
        if window > 0:
            smoothed_rewards = [sum(rewards[i:i+window])/window for i in range(len(rewards)-window)]
            plt.plot(range(window, len(rewards)), smoothed_rewards, label=f"{name} (Smoothed)")
        else:
            plt.plot(rewards, label=name)
            
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"Lightning training curve saved to {save_path}" if "hw3_3" in filename else f"Comparison curve saved to {save_path}")
