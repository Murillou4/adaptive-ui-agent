"""
Training Orchestration for Adaptive UI Agent

Manages the complete training pipeline:
1. Dataset generation
2. VQ-VAE pre-training
3. PPO training with VQ-VAE encoding
4. Continual RL phase transitions
"""

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import time

from env.sandbox_env import SandboxEnv, EnvConfig
from vision.vqvae import VQVAE, create_vqvae
from agent.ppo import PPOAgent, create_ppo_agent


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)
    adaptation_episodes: List[int] = field(default_factory=list)
    
    # Per-update metrics
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)


class Trainer:
    """
    Training orchestrator for the Adaptive UI Agent.
    
    Implements the training pipeline described in paper 2312.01203v3:
    1. Pre-train VQ-VAE on environment screenshots
    2. Wait for VQ-VAE convergence (the "delay" mentioned in paper)
    3. Train PPO using multi-one-hot representations
    4. Handle continual RL with rule changes
    """
    
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
            device: Device to train on
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        
        # Create components
        self.env = self._create_env()
        self.vqvae = None
        self.agent = None
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # TensorBoard
        log_dir = self.config.get('training', {}).get('log_dir', 'runs')
        self.writer = SummaryWriter(os.path.join(log_dir, 'training'))
        
        # Continual RL state
        self.current_phase = 0
        self.phase_switch_episodes = []
        
    def _create_env(self) -> SandboxEnv:
        """Create environment from config."""
        env_config = self.config.get('env', {})
        rewards = env_config.get('rewards', {})
        
        config = EnvConfig(
            size=env_config.get('size', 64),
            target_size=env_config.get('target_size', 12),
            cursor_size=env_config.get('cursor_size', 4),
            max_steps=env_config.get('max_steps', 200),
            click_target=rewards.get('click_target', 1.0),
            click_obstacle=rewards.get('click_obstacle', -1.0),
            click_background=rewards.get('click_background', -0.1),
            step_penalty=rewards.get('step_penalty', -0.01)
        )
        
        return SandboxEnv(config=config)
    
    def load_vqvae(self, checkpoint_path: Optional[str] = None) -> VQVAE:
        """
        Load pre-trained VQ-VAE.
        
        Args:
            checkpoint_path: Path to VQ-VAE checkpoint
            
        Returns:
            Loaded VQ-VAE model
        """
        self.vqvae = create_vqvae(self.config)
        
        if checkpoint_path is None:
            checkpoint_dir = self.config.get('training', {}).get('checkpoint_dir', 'data/checkpoints')
            checkpoint_path = os.path.join(checkpoint_dir, 'vqvae_best.pt')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.vqvae.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded VQ-VAE from {checkpoint_path}")
        else:
            print(f"Warning: VQ-VAE checkpoint not found at {checkpoint_path}")
            print("Training will start with random VQ-VAE weights")
        
        self.vqvae = self.vqvae.to(self.device)
        self.vqvae.eval()
        
        return self.vqvae
    
    def create_agent(self) -> PPOAgent:
        """Create PPO agent with VQ-VAE encoder."""
        if self.vqvae is None:
            self.load_vqvae()
            
        self.agent = create_ppo_agent(self.vqvae, self.config, self.device)
        self.agent.freeze_encoder()  # Freeze VQ-VAE during PPO training
        
        return self.agent
    
    def run_episode(self) -> Dict:
        """
        Run a single training episode.
        
        Returns:
            Episode metrics
        """
        obs, info = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        success = False
        
        while True:
            # Get action from agent
            action, log_prob, value = self.agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(
                obs, action, reward, next_obs, done, log_prob, value
            )
            
            episode_reward += reward
            episode_length += 1
            
            if terminated and reward > 0:  # Clicked on target
                success = True
            
            if done:
                break
                
            obs = next_obs
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'success': success
        }
    
    def train(
        self,
        num_episodes: int = 10000,
        steps_per_update: int = 128,
        log_interval: int = 100,
        save_interval: int = 1000,
        continual_rl_switch_at: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            num_episodes: Total number of episodes
            steps_per_update: Steps between PPO updates
            log_interval: Episodes between logging
            save_interval: Episodes between checkpoints
            continual_rl_switch_at: Episode to trigger rule change (optional)
        """
        if self.agent is None:
            self.create_agent()
        
        training_config = self.config.get('training', {})
        continual_rl_switch_at = continual_rl_switch_at or training_config.get('rule_switch_at')
        
        print(f"Starting training for {num_episodes} episodes on {self.device}")
        print(f"Continual RL switch at episode: {continual_rl_switch_at}")
        
        total_steps = 0
        recent_rewards = []
        recent_successes = []
        
        for episode in tqdm(range(1, num_episodes + 1)):
            # Run episode
            episode_metrics = self.run_episode()
            
            total_steps += episode_metrics['length']
            recent_rewards.append(episode_metrics['reward'])
            recent_successes.append(float(episode_metrics['success']))
            
            # Store metrics
            self.metrics.episode_rewards.append(episode_metrics['reward'])
            self.metrics.episode_lengths.append(episode_metrics['length'])
            
            # PPO update when buffer is full
            if total_steps >= steps_per_update:
                obs, _ = self.env.reset()
                update_metrics = self.agent.update(obs)
                
                self.metrics.policy_losses.append(update_metrics['policy_loss'])
                self.metrics.value_losses.append(update_metrics['value_loss'])
                self.metrics.entropies.append(update_metrics['entropy'])
                
                # Log update metrics
                self.writer.add_scalar('ppo/policy_loss', update_metrics['policy_loss'], episode)
                self.writer.add_scalar('ppo/value_loss', update_metrics['value_loss'], episode)
                self.writer.add_scalar('ppo/entropy', update_metrics['entropy'], episode)
                
                total_steps = 0
            
            # Continual RL: Switch rules
            if continual_rl_switch_at and episode == continual_rl_switch_at:
                self._trigger_rule_change(episode)
            
            # Logging
            if episode % log_interval == 0:
                avg_reward = np.mean(recent_rewards[-log_interval:])
                avg_success = np.mean(recent_successes[-log_interval:])
                
                self.writer.add_scalar('episode/reward', avg_reward, episode)
                self.writer.add_scalar('episode/success_rate', avg_success, episode)
                self.writer.add_scalar('episode/length', np.mean(self.metrics.episode_lengths[-log_interval:]), episode)
                
                self.metrics.success_rates.append(avg_success)
                
                print(f"\nEpisode {episode}: reward={avg_reward:.2f}, success_rate={avg_success:.2%}")
            
            # Save checkpoint
            if episode % save_interval == 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint(num_episodes)
        self.writer.close()
        
        print("\nTraining complete!")
        return self.metrics
    
    def _trigger_rule_change(self, episode: int):
        """Trigger a rule change for continual RL."""
        print(f"\n=== CONTINUAL RL: Rule change at episode {episode} ===")
        self.env.swap_targets()
        self.current_phase += 1
        self.phase_switch_episodes.append(episode)
        
        # Log phase change
        self.writer.add_scalar('continual_rl/phase', self.current_phase, episode)
        
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = self.config.get('training', {}).get('checkpoint_dir', 'data/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save agent
        agent_path = os.path.join(checkpoint_dir, f"ppo_agent_ep{episode}.pt")
        self.agent.save(agent_path)
        
        # Save latest
        latest_path = os.path.join(checkpoint_dir, "ppo_agent_latest.pt")
        self.agent.save(latest_path)
        
        print(f"Saved checkpoint at episode {episode}")
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate the current agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call create_agent() first.")
        
        rewards = []
        successes = []
        lengths = []
        
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            success = False
            
            while True:
                action, _, _ = self.agent.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated and reward > 0:
                    success = True
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            successes.append(float(success))
            lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(successes),
            'mean_length': np.mean(lengths)
        }
    
    def close(self):
        """Clean up resources."""
        self.env.close()
        self.writer.close()


def train_from_config(config_path: str = "configs/default.yaml"):
    """Train agent using configuration file."""
    trainer = Trainer(config_path)
    
    training_config = trainer.config.get('training', {})
    
    metrics = trainer.train(
        num_episodes=training_config.get('total_episodes', 10000),
        steps_per_update=trainer.config.get('ppo', {}).get('steps_per_update', 128),
        log_interval=training_config.get('log_interval', 100),
        save_interval=training_config.get('save_interval', 1000)
    )
    
    # Final evaluation
    eval_metrics = trainer.evaluate()
    print(f"\nFinal evaluation: {eval_metrics}")
    
    trainer.close()
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Adaptive UI Agent")
    parser.add_argument('--config', '-c', default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                        help='Override number of episodes')
    
    args = parser.parse_args()
    
    train_from_config(args.config)
