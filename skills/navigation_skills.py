"""
Navigation Skills

Skills for navigating the interface:
- Scroll: Scroll up/down
- SwitchWindow: Switch focus to another app
"""

import time
import logging
from typing import Optional, Literal

from skills import Skill, SkillResult, register_skill
from vision.state_extractor import VisualState

logger = logging.getLogger(__name__)


@register_skill
class ScrollSkill(Skill):
    """
    Scroll the mouse wheel.
    """
    name = "scroll"
    description = "Scroll up or down"
    
    def __init__(
        self, 
        clicks: int = -3,
        direction: Literal["up", "down"] = "down"
    ):
        # Allow specifying direction string for ease of use
        if direction == "up" and clicks < 0:
            clicks = abs(clicks)
        elif direction == "down" and clicks > 0:
            clicks = -clicks
            
        super().__init__(clicks=clicks)
        self.clicks = clicks
        
    def is_applicable(self, state: VisualState) -> bool:
        return True
        
    def execute(self, env) -> SkillResult:
        env.input.scroll(self.clicks)
        time.sleep(0.5) # Wait for UI to update
        return SkillResult(True, f"Scrolled {self.clicks} clicks")


@register_skill
class SwitchWindowSkill(Skill):
    """
    Switch to another window (using Alt+Tab or clicking taskbar).
    Currently implemented as Alt+Tab for simplicity.
    """
    name = "switch_window"
    description = "Switch active window"
    
    def __init__(self, times: int = 1):
        super().__init__(times=times)
        self.times = times
        
    def is_applicable(self, state: VisualState) -> bool:
        return True
        
    def execute(self, env) -> SkillResult:
        # Hold Alt
        env.input.press_key("alt")
        
        # Press Tab N times
        for _ in range(self.times):
            env.input.press_key("tab")
            time.sleep(0.2)
            
        # Release Alt
        env.input.press_key("alt") # pynput press/release
        
        time.sleep(0.5)
        return SkillResult(True, "Switched window")
