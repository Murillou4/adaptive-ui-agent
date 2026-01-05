"""
Visualization utilities for the Adaptive UI Agent.

Provides visual inspection tools:
- Side-by-side original/reconstruction
- Multi-one-hot heatmap
- Click prediction visualization
- Training curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
import torch
from typing import Optional, Tuple, List


class Visualizer:
    """
    Visualization tools for agent inspection.
    
    Creates visual representations of:
    - Environment screenshots
    - VQ-VAE reconstructions
    - Multi-one-hot latent representations
    - Training metrics
    """
    
    def __init__(self, output_dir: str = "data/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Style settings
        plt.style.use('dark_background')
        
    def compare_reconstruction(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Original vs Reconstruction"
    ) -> str:
        """
        Create side-by-side comparison of original and reconstruction.
        
        Args:
            original: Original image (H, W, C)
            reconstruction: Reconstructed image (H, W, C)
            save_path: Optional path to save
            title: Plot title
            
        Returns:
            Path to saved image
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Reconstruction
        axes[1].imshow(reconstruction)
        axes[1].set_title("Reconstruction")
        axes[1].axis('off')
        
        # Difference
        diff = np.abs(original.astype(float) - reconstruction.astype(float))
        diff_normalized = (diff / 255.0 * 3).clip(0, 1)  # Amplify for visibility
        axes[2].imshow(diff_normalized)
        axes[2].set_title("Difference (3x)")
        axes[2].axis('off')
        
        # MSE
        mse = np.mean((original.astype(float) - reconstruction.astype(float)) ** 2)
        fig.suptitle(f"{title} (MSE: {mse:.2f})")
        
        plt.tight_layout()
        
        save_path = save_path or os.path.join(self.output_dir, "reconstruction_compare.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_multi_one_hot(
        self,
        multi_one_hot: np.ndarray,
        grid_size: int = 6,
        codebook_size: int = 512,
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualize multi-one-hot representation as heatmap.
        
        Shows which codebook entries are active at each grid position.
        
        Args:
            multi_one_hot: Binary vector (grid_size^2 * codebook_size,)
            grid_size: Latent grid size
            codebook_size: Codebook size
            save_path: Optional path to save
            
        Returns:
            Path to saved image
        """
        # Reshape to (grid_size, grid_size, codebook_size)
        reshaped = multi_one_hot.reshape(grid_size, grid_size, codebook_size)
        
        # Get active indices for each position
        active_indices = np.argmax(reshaped, axis=2)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Active codebook indices
        im1 = axes[0].imshow(active_indices, cmap='viridis')
        axes[0].set_title("Active Codebook Indices")
        axes[0].set_xlabel("Grid X")
        axes[0].set_ylabel("Grid Y")
        plt.colorbar(im1, ax=axes[0], label="Index")
        
        # Add index annotations
        for i in range(grid_size):
            for j in range(grid_size):
                axes[0].annotate(
                    str(active_indices[i, j]),
                    (j, i),
                    ha='center', va='center',
                    fontsize=8, color='white'
                )
        
        # Sparsity per position
        sparsity = np.sum(reshaped > 0, axis=2)
        im2 = axes[1].imshow(sparsity, cmap='hot')
        axes[1].set_title("Active Bits per Position")
        axes[1].set_xlabel("Grid X")
        axes[1].set_ylabel("Grid Y")
        plt.colorbar(im2, ax=axes[1], label="Count")
        
        plt.tight_layout()
        
        save_path = save_path or os.path.join(self.output_dir, "multi_one_hot.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_action_distribution(
        self,
        action_probs: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualize action probability distribution.
        
        Args:
            action_probs: Action probabilities (9,)
            save_path: Optional path to save
            
        Returns:
            Path to saved image
        """
        action_names = [
            'Up', 'Down', 'Left', 'Right',
            'Up-Left', 'Up-Right', 'Down-Left', 'Down-Right',
            'Click'
        ]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = ['#3498db'] * 8 + ['#e74c3c']  # Blue for moves, red for click
        bars = ax.bar(action_names, action_probs, color=colors)
        
        ax.set_ylabel("Probability")
        ax.set_title("Action Distribution")
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, prob in zip(bars, action_probs):
            height = bar.get_height()
            ax.annotate(
                f'{prob:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9
            )
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = save_path or os.path.join(self.output_dir, "action_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_training_curves(
        self,
        rewards: List[float],
        success_rates: List[float],
        losses: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        window: int = 100
    ) -> str:
        """
        Plot training curves.
        
        Args:
            rewards: Episode rewards
            success_rates: Success rates over time
            losses: Optional loss values
            save_path: Optional path to save
            window: Smoothing window size
            
        Returns:
            Path to saved image
        """
        num_plots = 2 if losses is None else 3
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
        
        # Smooth function
        def smooth(x, w):
            if len(x) < w:
                return x
            return np.convolve(x, np.ones(w) / w, mode='valid')
        
        # Reward curve
        if len(rewards) > 0:
            smoothed_rewards = smooth(rewards, window)
            axes[0].plot(rewards, alpha=0.3, color='#3498db', label='Raw')
            axes[0].plot(
                range(window - 1, len(rewards)),
                smoothed_rewards,
                color='#2ecc71',
                linewidth=2,
                label=f'Smoothed (w={window})'
            )
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Reward")
            axes[0].set_title("Episode Rewards")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Success rate curve
        if len(success_rates) > 0:
            axes[1].plot(success_rates, color='#e74c3c', linewidth=2)
            axes[1].set_xlabel("Logging Interval")
            axes[1].set_ylabel("Success Rate")
            axes[1].set_title("Success Rate")
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)
        
        # Loss curve
        if losses is not None and len(losses) > 0:
            axes[2].plot(losses, color='#9b59b6', linewidth=1)
            axes[2].set_xlabel("Update")
            axes[2].set_ylabel("Loss")
            axes[2].set_title("Training Loss")
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = save_path or os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_click_heatmap(
        self,
        click_positions: List[Tuple[int, int]],
        env_size: int = 64,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create heatmap of click positions.
        
        Args:
            click_positions: List of (x, y) click coordinates
            env_size: Environment size
            save_path: Optional path to save
            
        Returns:
            Path to saved image
        """
        heatmap = np.zeros((env_size, env_size))
        
        for x, y in click_positions:
            if 0 <= x < env_size and 0 <= y < env_size:
                heatmap[y, x] += 1
        
        # Apply Gaussian smoothing for better visualization
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=2)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        ax.set_title("Click Position Heatmap")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="Click Frequency")
        
        plt.tight_layout()
        
        save_path = save_path or os.path.join(self.output_dir, "click_heatmap.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_agent_dashboard(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        multi_one_hot: np.ndarray,
        action_probs: np.ndarray,
        info: dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive dashboard showing agent state.
        
        Args:
            original: Original screenshot
            reconstruction: VQ-VAE reconstruction
            multi_one_hot: Multi-one-hot representation
            action_probs: Action probability distribution
            info: Environment info dict
            save_path: Optional path to save
            
        Returns:
            Path to saved image
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title("Environment")
        ax1.axis('off')
        
        # Mark cursor and targets
        cursor = info.get('cursor_pos', [32, 32])
        target = info.get('target_pos', [0, 0])
        obstacle = info.get('obstacle_pos', [0, 0])
        
        ax1.scatter([cursor[0]], [cursor[1]], c='white', marker='+', s=100, linewidths=2)
        ax1.scatter([target[0]], [target[1]], c='cyan', marker='o', s=50, alpha=0.5)
        ax1.scatter([obstacle[0]], [obstacle[1]], c='yellow', marker='x', s=50, alpha=0.5)
        
        # Reconstruction
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(reconstruction)
        ax2.set_title("VQ-VAE Reconstruction")
        ax2.axis('off')
        
        # Difference
        ax3 = fig.add_subplot(gs[0, 2])
        diff = np.abs(original.astype(float) - reconstruction.astype(float))
        diff_normalized = (diff / 255.0 * 3).clip(0, 1)
        ax3.imshow(diff_normalized)
        ax3.set_title("Difference (3x)")
        ax3.axis('off')
        
        # Multi-one-hot
        ax4 = fig.add_subplot(gs[1, 0])
        grid_size = 6
        codebook_size = len(multi_one_hot) // (grid_size * grid_size)
        reshaped = multi_one_hot.reshape(grid_size, grid_size, codebook_size)
        active_indices = np.argmax(reshaped, axis=2)
        
        im = ax4.imshow(active_indices, cmap='viridis')
        ax4.set_title("Latent Codes")
        plt.colorbar(im, ax=ax4, label="Index")
        
        # Action distribution
        ax5 = fig.add_subplot(gs[1, 1])
        action_names = ['↑', '↓', '←', '→', '↖', '↗', '↙', '↘', '●']
        colors = ['#3498db'] * 8 + ['#e74c3c']
        ax5.bar(action_names, action_probs, color=colors)
        ax5.set_title("Action Probabilities")
        ax5.set_ylim(0, 1)
        
        # Info panel
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        info_text = f"""
Step: {info.get('step_count', 0)}
Total Reward: {info.get('total_reward', 0):.2f}
Cursor: {info.get('cursor_pos', [0, 0])}
Target: {'Blue' if info.get('target_is_blue', True) else 'Red'}

Predicted Action: {action_names[np.argmax(action_probs)]}
Confidence: {np.max(action_probs):.2%}
"""
        ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#2c3e50', alpha=0.8))
        ax6.set_title("Agent State")
        
        plt.suptitle("Adaptive UI Agent Dashboard", fontsize=14, fontweight='bold')
        
        save_path = save_path or os.path.join(self.output_dir, "dashboard.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        return save_path


def create_visualizer(output_dir: str = "data/visualizations") -> Visualizer:
    """Create a visualizer instance."""
    return Visualizer(output_dir)


if __name__ == "__main__":
    # Quick test
    print("Testing Visualizer...")
    
    viz = Visualizer()
    
    # Test reconstruction comparison
    original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    reconstruction = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    path = viz.compare_reconstruction(original, reconstruction)
    print(f"Saved reconstruction comparison to: {path}")
    
    # Test multi-one-hot visualization
    multi_one_hot = np.zeros(6 * 6 * 512)
    for i in range(36):
        multi_one_hot[i * 512 + np.random.randint(0, 512)] = 1
    path = viz.visualize_multi_one_hot(multi_one_hot)
    print(f"Saved multi-one-hot visualization to: {path}")
    
    # Test action distribution
    action_probs = np.random.dirichlet(np.ones(9))
    path = viz.visualize_action_distribution(action_probs)
    print(f"Saved action distribution to: {path}")
    
    # Test training curves
    rewards = np.random.randn(1000).cumsum()
    success_rates = np.clip(np.linspace(0, 1, 100) + np.random.randn(100) * 0.1, 0, 1)
    path = viz.plot_training_curves(list(rewards), list(success_rates))
    print(f"Saved training curves to: {path}")
    
    print("Visualizer test passed!")
