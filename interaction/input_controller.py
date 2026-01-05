"""
Input Controller Module

Simulates mouse and keyboard input for interacting with OS/Apps.
Uses pynput for precise control and pyautogui for high-level actions.
"""

import time
import math
import logging
from typing import Optional, Tuple, Union

try:
    import pyautogui
    from pynput.mouse import Button, Controller as MouseController
    from pynput.keyboard import Key, Controller as KeyboardController
    INPUT_AVAILABLE = True
except ImportError:
    INPUT_AVAILABLE = False
    pyautogui = None
    MouseController = None
    KeyboardController = None

logger = logging.getLogger(__name__)

# Configure pyautogui safety
if pyautogui:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

class InputController:
    """
    Controls mouse and keyboard to interact with the environment.
    """
    
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        
        if not INPUT_AVAILABLE:
            logger.warning("Input libraries not found. Input simulation will not work.")
            self.mouse = None
            self.keyboard = None
            return
            
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        
        logger.info("Input controller initialized")
        
    def move_to(self, x: int, y: int, duration: float = 0.1, tweet: bool = False):
        """
        Move mouse to coordinate (x, y).
        
        Args:
            x, y: Target coordinates
            duration: Duration of movement in seconds (0 for instant)
            tweet: Whether to use tweening (human-like movement)
        """
        if not self._check_available(): return
        
        x, y = int(x), int(y)
        
        if duration > 0 and pyautogui:
            # Use pyautogui for smooth movement
            tween = pyautogui.easeInOutQuad if tweet else pyautogui.linear
            pyautogui.moveTo(x, y, duration=duration, tween=tween)
        else:
            # Use pynput for instant movement
            self.mouse.position = (x, y)
            
    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", double: bool = False):
        """
        Click mouse button.
        
        Args:
            x, y: Optional coordinates to move to before clicking
            button: "left", "right", or "middle"
            double: Whether to double click
        """
        if not self._check_available(): return
        
        if x is not None and y is not None:
            self.move_to(x, y, duration=0.0) # Instant move for click
            
        btn = getattr(Button, button, Button.left)
        
        self.mouse.click(btn, 2 if double else 1)
        
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """
        Drag from start to end position.
        """
        if not self._check_available(): return
        
        self.move_to(start_x, start_y)
        self.mouse.press(Button.left)
        time.sleep(0.05)
        self.move_to(end_x, end_y, duration=duration)
        time.sleep(0.05)
        self.mouse.release(Button.left)
        
    def type_text(self, text: str, interval: float = 0.05):
        """
        Type text string.
        """
        if not self._check_available(): return
        
        if pyautogui and interval > 0:
            pyautogui.write(text, interval=interval)
        else:
            self.keyboard.type(text)
            
    def press_key(self, key_name: str):
        """
        Press a specific key (e.g., 'enter', 'esc', 'ctrl').
        """
        if not self._check_available(): return
        
        if pyautogui:
            pyautogui.press(key_name)
        else:
            # Fallback mapping for common keys if pyautogui missing
            k = getattr(Key, key_name, None)
            if k:
                self.keyboard.press(k)
                self.keyboard.release(k)
                
    def hotkey(self, *keys):
        """
        Press key combination (e.g., 'ctrl', 'c').
        """
        if not self._check_available(): return
        
        if pyautogui:
            pyautogui.hotkey(*keys)
        else:
            # Manual hotkey with pynput constraints
            # (Simplified logic, robust implementation is complex without pyautogui)
            logger.warning("Complex hotkeys require pyautogui")
            
    def scroll(self, clicks: int):
        """Scroll mouse wheel."""
        if not self._check_available(): return
        self.mouse.scroll(0, clicks)
        
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse (x, y)."""
        if not self._check_available(): return (0, 0)
        return self.mouse.position
        
    def _check_available(self) -> bool:
        return self.mouse is not None and self.keyboard is not None
