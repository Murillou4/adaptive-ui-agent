"""
App Sandbox

A wrapper for UniversalEnv that restricts the agent to a specific application window.
Instead of a heavy VM, this ensures the agent only sees and interacts with the target app.
"""

import logging
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import time

# Wraps the env
from env.universal_env import UniversalEnv

logger = logging.getLogger(__name__)

class AppSandbox(UniversalEnv):
    """
    Restricts observation and action space to a specific application window.
    """
    
    def __init__(
        self, 
        app_title: str,
        target_resolution: Tuple[int, int] = (640, 480),
        action_delay: float = 0.1
    ):
        # Initialize parent
        super().__init__(target_resolution, action_delay)
        
        self.app_title = app_title
        self._window_bounds = None
        
        logger.info(f"AppSandbox initialized for: {app_title}")
        
    def reset(self, seed=None, options=None):
        return self._get_observation()
        
    def _get_observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Capture ONLY the target window."""
        
        # Capture using window title
        # This returns the cropped window directly
        raw_frame = self.capture.capture(window_title=self.app_title)
        
        # Update bounds for action mapping (not perfect, but needed for clicks)
        try:
            import pygetwindow as gw
            win = gw.getWindowsWithTitle(self.app_title)[0]
            self._window_bounds = (win.left, win.top, win.width, win.height)
        except Exception:
            pass

        # Resize for observation
        import cv2
        resized_frame = cv2.resize(raw_frame, self.target_resolution)
        
        # Extract state
        self.current_state = self.state_extractor.extract(raw_frame)
        self._last_obs = resized_frame
        
        info = {
            "visual_state": self.current_state,
            "window_bounds": self._window_bounds
        }
        
        return resized_frame, info
        
    def _move_relative(self, dx: int, dy: int):
        """
        Move relative effectively, but we might want to clamp to window bounds.
        For now, we trust the standard move, but a real sandbox would clamp.
        """
        super()._move_relative(dx, dy)
        
    def click_at_percent(self, x_pct: float, y_pct: float):
        """Click at percentage coordinates within the window."""
        if not self._window_bounds:
            logger.warning("Window bounds unknown, cannot reference click")
            return
            
        left, top, w, h = self._window_bounds
        
        abs_x = left + int(x_pct * w)
        abs_y = top + int(y_pct * h)
        
        self.input.click(abs_x, abs_y)
