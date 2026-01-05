"""
Tests for sandbox environment.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sandbox_env import SandboxEnv, EnvConfig


class TestSandboxEnv:
    """Test suite for SandboxEnv."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = SandboxEnv()
        assert env is not None
        assert env.action_space_n == 9
        assert env.observation_shape == (64, 64, 3)
        env.close()
    
    def test_reset(self):
        """Test environment reset."""
        env = SandboxEnv()
        obs, info = env.reset(seed=42)
        
        # Check observation shape
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        
        # Check info
        assert 'cursor_pos' in info
        assert 'target_pos' in info
        assert 'obstacle_pos' in info
        
        env.close()
    
    def test_step_movement(self):
        """Test movement actions."""
        env = SandboxEnv()
        env.reset(seed=42)
        
        initial_pos = env.state.cursor_pos.copy()
        
        # Move right (action 3)
        obs, reward, terminated, truncated, info = env.step(3)
        
        # Cursor should have moved
        assert not np.array_equal(env.state.cursor_pos, initial_pos)
        assert obs.shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        env.close()
    
    def test_step_click(self):
        """Test click action."""
        env = SandboxEnv()
        env.reset(seed=42)
        
        # Click action (action 8)
        obs, reward, terminated, truncated, info = env.step(8)
        
        # Reward should be one of: click_target, click_obstacle, click_background
        assert reward in [1.0, -1.0, -0.1, -0.11]  # Including step penalty
        
        env.close()
    
    def test_rewards(self):
        """Test reward values."""
        config = EnvConfig(
            click_target=1.0,
            click_obstacle=-1.0,
            click_background=-0.1,
            step_penalty=-0.01
        )
        env = SandboxEnv(config=config)
        
        assert env.config.click_target == 1.0
        assert env.config.click_obstacle == -1.0
        assert env.config.click_background == -0.1
        
        env.close()
    
    def test_swap_targets(self):
        """Test target swapping for continual RL."""
        env = SandboxEnv()
        env.reset()
        
        initial_state = env.target_is_blue
        env.swap_targets()
        
        assert env.target_is_blue != initial_state
        
        env.close()
    
    def test_set_rule(self):
        """Test rule setting."""
        env = SandboxEnv()
        env.reset()
        
        env.set_rule("blue_bad")
        assert env.target_is_blue == False
        
        env.set_rule("blue_good")
        assert env.target_is_blue == True
        
        env.close()
    
    def test_set_reward_override(self):
        """Test reward override."""
        env = SandboxEnv()
        env.reset()
        
        env.set_reward("click_target", 10.0)
        assert env.reward_overrides["click_target"] == 10.0
        
        env.reset_rewards()
        assert len(env.reward_overrides) == 0
        
        env.close()
    
    def test_max_steps_truncation(self):
        """Test episode truncation at max steps."""
        config = EnvConfig(max_steps=10)
        env = SandboxEnv(config=config)
        env.reset()
        
        for _ in range(15):  # More than max_steps
            _, _, terminated, truncated, _ = env.step(0)  # Move up
            if terminated or truncated:
                break
        
        assert truncated  # Should be truncated at max_steps
        
        env.close()
    
    def test_screenshot(self):
        """Test screenshot capture."""
        env = SandboxEnv()
        env.reset()
        
        screenshot = env.get_screenshot()
        
        assert screenshot.shape == (64, 64, 3)
        assert screenshot.dtype == np.uint8
        
        env.close()
    
    def test_deterministic_reset(self):
        """Test that reset with same seed gives same result."""
        env = SandboxEnv()
        
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        
        assert np.array_equal(obs1, obs2)
        assert info1['target_pos'] == info2['target_pos']
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
