"""
Desktop Skills

Skills for OS-level interactions:
- OpenApp: Open an application via Start Menu
- MinimizeAll: Show desktop
"""

import time
import logging

from skills import Skill, SkillResult, register_skill
from vision.state_extractor import VisualState

logger = logging.getLogger(__name__)


@register_skill
class OpenAppSkill(Skill):
    """
    Open an application using the Start Menu search.
    """
    name = "open_app"
    description = "Open app via Start Menu"
    
    def __init__(self, app_name: str):
        super().__init__(app_name=app_name)
        self.app_name = app_name
        
    def is_applicable(self, state: VisualState) -> bool:
        return True
        
    def execute(self, env) -> SkillResult:
        # Press Windows key
        env.input.press_key("win")
        time.sleep(1.0) # Wait for animation
        
        # Type app name
        env.input.type_text(self.app_name)
        time.sleep(1.0) # Wait for search results
        
        # Press Enter
        env.input.press_key("enter")
        
        # Wait for app to launch
        time.sleep(3.0) 
        
        return SkillResult(True, f"Opened app: {self.app_name}")


@register_skill
class MinimizeAllSkill(Skill):
    """
    Minimize all windows (Win+D).
    """
    name = "minimize_all"
    description = "Show Desktop"
    
    def __init__(self):
        super().__init__()
        
    def is_applicable(self, state: VisualState) -> bool:
        return True
        
    def execute(self, env) -> SkillResult:
        env.input.hotkey("win", "d")
        time.sleep(0.5)
        return SkillResult(True, "Minimized all windows")
