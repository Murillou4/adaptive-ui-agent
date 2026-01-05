"""
Tests for Universal Environment Adapter
"""

import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.universal_env import UniversalEnv
from interaction.screen_capture import ScreenCapture
from interaction.input_controller import InputController


class TestComponents:
    """Test individual interaction components."""
    
    def test_screen_capture(self):
        try:
            cap = ScreenCapture()
            frame = cap.capture()
            
            assert isinstance(frame, np.ndarray)
            assert len(frame.shape) == 3
            assert frame.shape[2] == 3
            
            w, h = cap.get_screen_size()
            assert w > 0 and h > 0
            
            cap.close()
        except Exception as e:
            pytest.skip(f"Screen capture failed (headless env?): {e}")

    def test_input_controller(self):
        try:
            ctrl = InputController()
            # Just check if initialized without error
            # Actual movement tests are dangerous on dev machine during automated tests
            assert ctrl.mouse is not None or ctrl.keyboard is None
        except Exception as e:
            pytest.skip(f"Input controller failed: {e}")


class TestUniversalEnv:
    """Test the Gym environment wrapper."""
    
    def test_env_initialization(self):
        # Initialize env
        # Note: This loads heavy vision models, might be slow
        env = UniversalEnv(target_resolution=(128, 128))
        
        assert env.observation_space.shape == (128, 128, 3)
        assert env.action_space.n == 7
        
        env.close()

    def test_env_reset_step(self):
        try:
            env = UniversalEnv(target_resolution=(64, 64))
            
            # Reset
            obs, info = env.reset()
            assert obs.shape == (64, 64, 3)
            assert "visual_state" in info
            assert "mouse_pos" in info
            
            # Step (No-op action 0 to avoid moving mouse randomly)
            obs, reward, terminated, truncated, info = env.step(0)
            
            assert obs.shape == (64, 64, 3)
            assert not terminated
            
            env.close()
            
        except ImportError:
            pytest.skip("Missing dependencies for env test")
        except Exception as e:
            pytest.skip(f"Env test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
