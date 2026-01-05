"""
Universal Environment for Adaptive UI Agent

This is the Gymnasium-compatible environment that wraps ANY application.
It connects:
1. Screen Capture (Observation)
2. Input Controller (Action)
3. Vision System (State Extraction)

It allows the RL agent to interact with the OS as a human would.
"""

import gymnasium as gym
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, Union
import logging

from interaction.screen_capture import ScreenCapture
from interaction.input_controller import InputController
from vision.universal_detector import UniversalVisionDetector
from vision.state_extractor import StateExtractor, VisualState

logger = logging.getLogger(__name__)

class UniversalEnv(gym.Env):
    """
    Gym environment for interacting with any application/OS.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    def __init__(
        self,
        target_resolution: Tuple[int, int] = (640, 480),
        action_delay: float = 0.1,
        headless: bool = False
    ):
        super().__init__()
        
        self.target_resolution = target_resolution
        self.action_delay = action_delay
        
        # Initialize components
        self.capture = ScreenCapture()
        self.input = InputController()
        
        # Initialize vision (lazy load heavy models)
        self.detector = UniversalVisionDetector()
        self.state_extractor = StateExtractor(self.detector)
        
        # Define Action Space
        # Hybrid discrete/continuous is ideal, but starting with Discrete for simplicity
        # 0: No-op
        # 1-4: Move (Up, Down, Left, Right)
        # 5: Click
        # 6: Type generic input
        self.action_space = gym.spaces.Discrete(7)
        
        # Define Observation Space
        # VQ-VAE expects specific resolution
        h, w = target_resolution
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )
        
        self.current_state: Optional[VisualState] = None
        self._last_obs: Optional[np.ndarray] = None
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # In a real app environment, 'reset' might imply killing/restarting the app
        # For now, we just capture fresh state
        
        return self._get_observation()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        
        # Execute action
        if action == 1: self._move_relative(0, -50)  # Up
        elif action == 2: self._move_relative(0, 50) # Down
        elif action == 3: self._move_relative(-50, 0) # Left
        elif action == 4: self._move_relative(50, 0) # Right
        elif action == 5: self.input.click()
        # Action 0 is no-op
        
        # Wait for app response
        time.sleep(self.action_delay)
        
        # Capture new state
        obs, info = self._get_observation()
        
        # Calculate reward (placeholder - real reward comes from planner/task)
        reward = 0.0
        
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Capture screen and extract state."""
        # 1. Capture frame
        raw_frame = self.capture.capture()
        
        # 2. Resize to target resolution (simple resize for now)
        # In production, use high-quality resize
        # For RL, we need consistent shape
        import cv2
        resized_frame = cv2.resize(raw_frame, self.target_resolution)
        
        # 3. Extract semantic state (using full res frame for accuracy)
        # Note: This is expensive! In training loop, might run less frequently
        self.current_state = self.state_extractor.extract(raw_frame)
        
        self._last_obs = resized_frame
        
        info = {
            "visual_state": self.current_state,
            "mouse_pos": self.input.get_mouse_position()
        }
        
        return resized_frame, info
        
    def _move_relative(self, dx: int, dy: int):
        """Move mouse relative to current position."""
        curr_x, curr_y = self.input.get_mouse_position()
        
        # Get screen bounds
        screen_w, screen_h = self.capture.get_screen_size()
        
        new_x = max(0, min(screen_w - 1, curr_x + dx))
        new_y = max(0, min(screen_h - 1, curr_y + dy))
        
        self.input.move_to(new_x, new_y, duration=0.0)
        
    def render(self):
        """Render environment."""
        if self.render_mode == "rgb_array":
            return self._last_obs
    
    def close(self):
        """Cleanup."""
        self.capture.close()
