"""
Keyboard Skills

Skills for keyboard interaction:
- TypeText: Type a string
- Hotkey: Press key combination
"""

import time
import logging
from typing import Optional

from skills import Skill, SkillResult, register_skill
from vision.state_extractor import VisualState

logger = logging.getLogger(__name__)


@register_skill
class TypeTextSkill(Skill):
    """
    Type text (optionally clicking an input field first).
    """
    name = "type_text"
    description = "Type string into field"
    
    def __init__(
        self, 
        text: str, 
        input_query: Optional[str] = None,
        clear_first: bool = False
    ):
        super().__init__(text=text, input=input_query)
        self.text = text
        self.input_query = input_query
        self.clear_first = clear_first
        
    def is_applicable(self, state: VisualState) -> bool:
        if self.input_query:
            return state.find_element(self.input_query) is not None
        return True # Can always type if no specific target
        
    def execute(self, env) -> SkillResult:
        # Focus input if query provided
        if self.input_query:
            # Import here to avoid circular dependency loop if defined at top
            from skills.cursor_skills import ClickSkill
            
            click_action = ClickSkill(element_query=self.input_query)
            result = click_action.execute(env)
            if not result.success:
                return result
                
            time.sleep(0.2)
            
        # Clear field if requested (ctrl+a, backspace)
        if self.clear_first:
            env.input.hotkey("ctrl", "a")
            env.input.press_key("backspace")
            time.sleep(0.1)
            
        # Type text
        env.input.type_text(self.text)
        
        return SkillResult(True, f"Typed: '{self.text}'")


@register_skill
class HotkeySkill(Skill):
    """
    Press a key or key combination.
    """
    name = "hotkey"
    description = "Press key(s)"
    
    def __init__(self, *keys):
        super().__init__(keys=keys)
        self.keys = keys
        
    def is_applicable(self, state: VisualState) -> bool:
        return True
        
    def execute(self, env) -> SkillResult:
        if len(self.keys) == 1:
            env.input.press_key(self.keys[0])
        else:
            env.input.hotkey(*self.keys)
            
        return SkillResult(True, f"Pressed: {'+'.join(self.keys)}")
