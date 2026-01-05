"""
Screen Capture Module

Provides high-performance screen capture capabilities using MSS.
Supports capturing full screen, specific monitors, or regions.
"""

import numpy as np
import mss
from typing import Tuple, Optional, Dict, Any, List
import logging
import platform

logger = logging.getLogger(__name__)

class ScreenCapture:
    """
    Cross-platform screen capture using MSS.
    Optimized for speed to support real-time RL loops.
    """
    
    def __init__(self, monitor_index: int = 1):
        """
        Initialize screen capture.
        
        Args:
            monitor_index: Index of monitor to capture (1 = primary)
        """
        self.monitor_index = monitor_index
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[monitor_index]
        
        logger.info(f"Screen capture initialized on monitor {monitor_index}: {self._monitor}")
        
    def capture(self, region: Optional[Tuple[int, int, int, int]] = None, window_title: Optional[str] = None) -> np.ndarray:
        """
        Capture current frame.
        
        Args:
            region: Optional (left, top, width, height) to capture.
            window_title: Optional title of window to capture (overrides region).
                    
        Returns:
            RGB numpy array (H, W, 3)
        """
        try:
            target_monitor = self._monitor
            
            if window_title:
                try:
                    import pygetwindow as gw
                    win = gw.getWindowsWithTitle(window_title)[0]
                    if win:
                        # Ensure window is visible/active (optional, might need activation)
                        # win.activate() 
                        target_monitor = {
                            "top": int(win.top), 
                            "left": int(win.left), 
                            "width": int(win.width), 
                            "height": int(win.height)
                        }
                except ImportError:
                    logger.warning("pygetwindow not installed, ignoring window_title")
                except IndexError:
                    logger.warning(f"Window '{window_title}' not found")
            
            if region and not window_title:
                left, top, width, height = region
                target_monitor = {
                    "top": top, 
                    "left": left, 
                    "width": width, 
                    "height": height
                }
                
            # Capture
            sct_img = self._sct.grab(target_monitor)
            
            # Convert to numpy array (BGRA -> RGB)
            # MSS returns BGRA, we need RGB for vision models
            img = np.array(sct_img)
            img_rgb = img[:, :, :3][:, :, ::-1]  # Drop alpha, swap BGR to RGB
            
            return np.ascontiguousarray(img_rgb)
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            # Return black frame on error to prevent crash
            w = region[2] if region else self._monitor["width"]
            h = region[3] if region else self._monitor["height"]
            return np.zeros((h, w, 3), dtype=np.uint8)
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get (width, height) of the captured monitor."""
        return self._monitor["width"], self._monitor["height"]
        
    def close(self):
        """Clean up resources."""
        self._sct.close()

    @property
    def monitors(self) -> List[Dict[str, Any]]:
        """Get list of available monitors."""
        return self._sct.monitors[1:]  # Skip index 0 (all monitors combined)
