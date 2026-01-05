"""
PPO (Proximal Policy Optimization) Agent for Discrete Visual RL

Takes multi-one-hot representations from VQ-VAE as input and outputs
discrete actions for the sandbox environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    hidden_dim: int = 256
    num_layers: int = 2
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 32


class PolicyNetwork(nn.Module):
    """
    MLP Policy Network.
    
    Takes multi-one-hot vector from VQ-VAE and outputs action logits.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_actions: int = 9
    ):
        """
        Initialize policy network.
        
        Args:
            input_dim: Size of multi-one-hot input (6*6*512 = 18432)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            num_actions: Number of discrete actions (9: 8 directions + click)
        """
        super().__init__()
        
        layers = []
        
        # Input projection
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Multi-one-hot input (B, input_dim)
            
        Returns:
            Action logits (B, num_actions)
        """
        return self.network(x)
    
    def get_action_distribution(self, x: torch.Tensor) -> Categorical:
        """Get action distribution for given state."""
        logits = self.forward(x)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """
    MLP Value Network.
    
    Estimates state value from multi-one-hot representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        
        # Input projection
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        
        # Output layer (single value)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Multi-one-hot input (B, input_dim)
            
        Returns:
            State value (B, 1)
        """
        return self.network(x)


@dataclass
class Transition:
    """Single transition in a trajectory."""
    state: np.ndarray      # Multi-one-hot state
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class RolloutBuffer:
    """Buffer for storing rollout trajectories."""
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        
        # Computed after rollout
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        
    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        self.states.append(transition.state)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.dones.append(transition.done)
        self.log_probs.append(transition.log_prob)
        self.values.append(transition.value)
        
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float):
        """
        Compute GAE advantages and returns.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value estimate for final state
        """
        n = len(self.rewards)
        self.advantages = np.zeros(n)
        self.returns = np.zeros(n)
        
        last_gae = 0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = (self.rewards[t] + 
                    gamma * next_value * next_non_terminal - 
                    self.values[t])
            
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            
        self.returns = self.advantages + np.array(self.values)
        
    def get_batches(self, batch_size: int):
        """
        Generate mini-batches for PPO update.
        
        Yields:
            Batch of (states, actions, old_log_probs, returns, advantages)
        """
        n = len(self.states)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            yield (
                np.array([self.states[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices]),
                self.returns[batch_indices],
                self.advantages[batch_indices]
            )
            
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None


class PPOAgent:
    """
    PPO Agent with VQ-VAE integration.
    
    Receives raw pixel observations, encodes them via VQ-VAE to
    multi-one-hot representations, and learns a policy over discrete actions.
    """
    
    def __init__(
        self,
        vqvae: nn.Module,
        input_dim: int,
        num_actions: int = 9,
        config: Optional[PPOConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize PPO agent.
        
        Args:
            vqvae: Pre-trained VQ-VAE model for encoding
            input_dim: Dimension of multi-one-hot vectors
            num_actions: Number of discrete actions
            config: PPO configuration
            device: Device to run on
        """
        self.config = config or PPOConfig()
        self.device = device
        
        # VQ-VAE encoder (frozen by default)
        self.vqvae = vqvae.to(device)
        self.vqvae.eval()
        self.encoder_frozen = True
        
        # Policy and value networks
        self.policy = PolicyNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_actions=num_actions
        ).to(device)
        
        self.value = ValueNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.config.learning_rate
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
    def encode_observation(self, obs: np.ndarray) -> torch.Tensor:
        """
        Encode pixel observation to multi-one-hot.
        
        Args:
            obs: Pixel observation (H, W, C) or (B, H, W, C), values 0-255
            
        Returns:
            Multi-one-hot tensor
        """
        with torch.no_grad():
            # Convert to tensor
            if obs.ndim == 3:
                obs = obs[np.newaxis, ...]  # Add batch dim
            
            # HWC -> CHW, normalize to [0, 1]
            obs = torch.from_numpy(obs).float() / 255.0
            obs = obs.permute(0, 3, 1, 2).to(self.device)
            
            # Get multi-one-hot
            multi_one_hot = self.vqvae.get_multi_one_hot(obs)
            
        return multi_one_hot
    
    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Get action for given observation.
        
        Args:
            obs: Pixel observation (H, W, C), values 0-255
            
        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            value: State value estimate
        """
        # Encode observation
        state = self.encode_observation(obs)
        
        # Get action distribution
        dist = self.policy.get_action_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Get value estimate
        value = self.value(state)
        
        return (
            action.cpu().item(),
            log_prob.cpu().item(),
            value.cpu().item()
        )
    
    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate for observation."""
        state = self.encode_observation(obs)
        value = self.value(state)
        return value.cpu().item()
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store a transition in the buffer."""
        state = self.encode_observation(obs).cpu().numpy().squeeze()
        next_state = self.encode_observation(next_obs).cpu().numpy().squeeze()
        
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        
        self.buffer.add(transition)
    
    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """
        Perform PPO update on collected trajectories.
        
        Args:
            last_obs: Final observation for value bootstrapping
            
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages
        last_value = self.get_value(last_obs)
        self.buffer.compute_advantages(
            self.config.gamma,
            self.config.gae_lambda,
            last_value
        )
        
        # Normalize advantages
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.get_batches(self.config.mini_batch_size):
                states, actions, old_log_probs, returns, advantages = batch
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)
                advantages = torch.FloatTensor(advantages).to(self.device)
                
                # Get current policy distribution
                dist = self.policy.get_action_distribution(states)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_ratio, 
                    1 + self.config.clip_ratio
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value(states).squeeze()
                value_loss = F.mse_loss(values, returns)
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_coef * value_loss -
                       self.config.entropy_coef * entropy)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def freeze_encoder(self):
        """Freeze VQ-VAE encoder weights."""
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
        
    def unfreeze_encoder(self):
        """Unfreeze VQ-VAE encoder for end-to-end training."""
        for param in self.vqvae.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
        
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']


def create_ppo_agent(
    vqvae: nn.Module,
    config: Optional[dict] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> PPOAgent:
    """
    Create PPO agent from configuration.
    
    Args:
        vqvae: Pre-trained VQ-VAE model
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        Initialized PPO agent
    """
    if config is None:
        config = {}
    
    ppo_config_dict = config.get('ppo', {})
    
    ppo_config = PPOConfig(
        hidden_dim=ppo_config_dict.get('hidden_dim', 256),
        num_layers=ppo_config_dict.get('num_layers', 2),
        clip_ratio=ppo_config_dict.get('clip_ratio', 0.2),
        learning_rate=ppo_config_dict.get('learning_rate', 3e-4),
        gamma=ppo_config_dict.get('gamma', 0.99),
        gae_lambda=ppo_config_dict.get('gae_lambda', 0.95),
        value_coef=ppo_config_dict.get('value_coef', 0.5),
        entropy_coef=ppo_config_dict.get('entropy_coef', 0.01),
        max_grad_norm=ppo_config_dict.get('max_grad_norm', 0.5),
        ppo_epochs=ppo_config_dict.get('ppo_epochs', 4),
        mini_batch_size=ppo_config_dict.get('mini_batch_size', 32)
    )
    
    return PPOAgent(
        vqvae=vqvae,
        input_dim=vqvae.multi_one_hot_dim,
        num_actions=9,
        config=ppo_config,
        device=device
    )


if __name__ == "__main__":
    # Quick test
    print("Testing PPO Agent...")
    
    # Create dummy VQ-VAE
    from vision.vqvae import VQVAE
    vqvae = VQVAE()
    print(f"VQ-VAE multi-one-hot dim: {vqvae.multi_one_hot_dim}")
    
    # Create agent
    agent = PPOAgent(vqvae, input_dim=vqvae.multi_one_hot_dim)
    
    # Test with dummy observation
    obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    action, log_prob, value = agent.get_action(obs)
    print(f"Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
    
    # Test storing transitions
    for _ in range(10):
        next_obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        agent.store_transition(obs, action, 0.1, next_obs, False, log_prob, value)
        obs = next_obs
        action, log_prob, value = agent.get_action(obs)
    
    # Test update
    metrics = agent.update(obs)
    print(f"Update metrics: {metrics}")
    
    print("PPO Agent test passed!")
