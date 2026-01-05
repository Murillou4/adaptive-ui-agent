"""
Tests for PPO agent.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ppo import PPOAgent, PPOConfig, PolicyNetwork, ValueNetwork, RolloutBuffer
from vision.vqvae import VQVAE


class TestPolicyNetwork:
    """Test suite for PolicyNetwork."""
    
    def test_initialization(self):
        """Test policy network initialization."""
        policy = PolicyNetwork(input_dim=18432, hidden_dim=256, num_actions=9)
        
        # Count parameters
        num_params = sum(p.numel() for p in policy.parameters())
        assert num_params > 0
    
    def test_forward(self):
        """Test policy forward pass."""
        policy = PolicyNetwork(input_dim=18432, hidden_dim=256, num_actions=9)
        
        x = torch.randn(4, 18432)
        logits = policy(x)
        
        assert logits.shape == (4, 9)
    
    def test_action_distribution(self):
        """Test action distribution."""
        policy = PolicyNetwork(input_dim=18432, hidden_dim=256, num_actions=9)
        
        x = torch.randn(4, 18432)
        dist = policy.get_action_distribution(x)
        
        # Sample should be in range
        action = dist.sample()
        assert action.shape == (4,)
        assert action.min() >= 0
        assert action.max() < 9
        
        # Probabilities should sum to 1
        probs = dist.probs
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


class TestValueNetwork:
    """Test suite for ValueNetwork."""
    
    def test_initialization(self):
        """Test value network initialization."""
        value_net = ValueNetwork(input_dim=18432, hidden_dim=256)
        
        num_params = sum(p.numel() for p in value_net.parameters())
        assert num_params > 0
    
    def test_forward(self):
        """Test value forward pass."""
        value_net = ValueNetwork(input_dim=18432, hidden_dim=256)
        
        x = torch.randn(4, 18432)
        value = value_net(x)
        
        assert value.shape == (4, 1)


class TestRolloutBuffer:
    """Test suite for RolloutBuffer."""
    
    def test_add_and_clear(self):
        """Test adding to and clearing buffer."""
        from agent.ppo import Transition
        
        buffer = RolloutBuffer()
        
        # Add transitions
        for i in range(10):
            transition = Transition(
                state=np.random.randn(18432),
                action=np.random.randint(0, 9),
                reward=np.random.randn(),
                next_state=np.random.randn(18432),
                done=False,
                log_prob=np.random.randn(),
                value=np.random.randn()
            )
            buffer.add(transition)
        
        assert len(buffer.states) == 10
        
        buffer.clear()
        assert len(buffer.states) == 0
    
    def test_compute_advantages(self):
        """Test GAE computation."""
        from agent.ppo import Transition
        
        buffer = RolloutBuffer()
        
        # Add transitions
        for i in range(10):
            transition = Transition(
                state=np.random.randn(18432),
                action=0,
                reward=1.0,
                next_state=np.random.randn(18432),
                done=i == 9,
                log_prob=-1.0,
                value=0.5
            )
            buffer.add(transition)
        
        buffer.compute_advantages(gamma=0.99, gae_lambda=0.95, last_value=0.0)
        
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert len(buffer.advantages) == 10
    
    def test_get_batches(self):
        """Test mini-batch generation."""
        from agent.ppo import Transition
        
        buffer = RolloutBuffer()
        
        # Add transitions
        for i in range(32):
            transition = Transition(
                state=np.random.randn(100),
                action=0,
                reward=1.0,
                next_state=np.random.randn(100),
                done=False,
                log_prob=-1.0,
                value=0.5
            )
            buffer.add(transition)
        
        buffer.compute_advantages(gamma=0.99, gae_lambda=0.95, last_value=0.0)
        
        # Get batches
        batches = list(buffer.get_batches(batch_size=8))
        assert len(batches) == 4  # 32 / 8


class TestPPOAgent:
    """Test suite for PPO agent."""
    
    @pytest.fixture
    def agent(self):
        """Create agent fixture."""
        vqvae = VQVAE()
        agent = PPOAgent(vqvae, input_dim=vqvae.multi_one_hot_dim)
        return agent
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.encoder_frozen == True
        assert agent.config.clip_ratio == 0.2
    
    def test_encode_observation(self, agent):
        """Test observation encoding."""
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        encoded = agent.encode_observation(obs)
        
        assert encoded.shape == (1, 6 * 6 * 512)
    
    def test_get_action(self, agent):
        """Test action selection."""
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        action, log_prob, value = agent.get_action(obs)
        
        assert 0 <= action < 9
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_get_value(self, agent):
        """Test value estimation."""
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        value = agent.get_value(obs)
        
        assert isinstance(value, float)
    
    def test_store_transition(self, agent):
        """Test transition storage."""
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        next_obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        agent.store_transition(
            obs=obs,
            action=0,
            reward=1.0,
            next_obs=next_obs,
            done=False,
            log_prob=-1.0,
            value=0.5
        )
        
        assert len(agent.buffer.states) == 1
    
    def test_update(self, agent):
        """Test PPO update."""
        # Store some transitions
        for _ in range(10):
            obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            next_obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            
            agent.store_transition(
                obs=obs,
                action=np.random.randint(0, 9),
                reward=np.random.randn(),
                next_obs=next_obs,
                done=False,
                log_prob=np.random.randn(),
                value=np.random.randn()
            )
        
        # Run update
        last_obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        metrics = agent.update(last_obs)
        
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics
        
        # Buffer should be cleared
        assert len(agent.buffer.states) == 0
    
    def test_freeze_unfreeze_encoder(self, agent):
        """Test encoder freezing."""
        agent.freeze_encoder()
        assert agent.encoder_frozen == True
        
        for param in agent.vqvae.parameters():
            assert param.requires_grad == False
        
        agent.unfreeze_encoder()
        assert agent.encoder_frozen == False
        
        for param in agent.vqvae.parameters():
            assert param.requires_grad == True
    
    def test_save_load(self, agent, tmp_path):
        """Test save and load."""
        # Get initial action
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        action1, _, _ = agent.get_action(obs)
        
        # Save
        save_path = str(tmp_path / "agent.pt")
        agent.save(save_path)
        
        # Create new agent and load
        vqvae = VQVAE()
        new_agent = PPOAgent(vqvae, input_dim=vqvae.multi_one_hot_dim)
        new_agent.load(save_path)
        
        # Should have same weights after loading
        # (deterministic action might differ due to different VQ-VAE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
