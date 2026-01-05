"""
Cursor Skills

Fundamental skills for mouse interaction:
- MoveTo: Move cursor to a target (coordinates or element)
- Click: Click on a target
- Drag: Drag and drop
"""

import time
import numpy as np
from typing import Optional, Union, Tuple
import logging

from skills import Skill, SkillResult, register_skill
from vision.state_extractor import VisualState
from vision.yolo_backend import Detection, BoundingBox

logger = logging.getLogger(__name__)


@register_skill
class MoveToSkill(Skill):
    """
    Move cursor to a target position or element.
    """
    name = "move_to"
    description = "Move cursor to x,y or element"
    
    def __init__(
        self, 
        x: Optional[int] = None, 
        y: Optional[int] = None, 
        element_query: Optional[str] = None
    ):
        super().__init__(x=x, y=y, element_query=element_query)
        self.target_pos = (x, y) if x is not None and y is not None else None
        self.element_query = element_query
        
    def is_applicable(self, state: VisualState) -> bool:
        if self.target_pos:
            return True
        if self.element_query:
            return state.find_element(self.element_query) is not None
        return False
        
    def execute(self, env) -> SkillResult:
        target = self.target_pos
        
        # If targeting by query, find element first
        if not target and self.element_query:
            # Note: We use env.current_state which should be fresh
            # If not, we might need env.reset() or env.step(0)
            if not env.current_state:
                env.step(0) # Capture state
                
            element = env.current_state.find_element(self.element_query)
            if element:
                target = element.bbox.center
            else:
                return SkillResult(False, f"Element '{self.element_query}' not found")
        
        if not target:
            return SkillResult(False, "No target specified")
            
        x, y = target
        
        # Move execution
        env.input.move_to(x, y, duration=0.5, tweet=True)
        # Wait a bit for stability
        time.sleep(0.1)
        
        return SkillResult(True, f"Moved to {x}, {y}")


@register_skill
class ClickSkill(Skill):
    """
    Click on a target (or current position).
    """
    name = "click"
    description = "Click on x,y or element"
    
    def __init__(
        self, 
        x: Optional[int] = None, 
        y: Optional[int] = None, 
        element_query: Optional[str] = None,
        double: bool = False
    ):
        super().__init__(x=x, y=y, query=element_query, double=double)
        self.move_skill = MoveToSkill(x, y, element_query)
        self.double = double
        
    def is_applicable(self, state: VisualState) -> bool:
        return self.move_skill.is_applicable(state)
        
    def execute(self, env) -> SkillResult:
        # First move to target if needed
        if self.move_skill.params.get('x') or self.move_skill.params.get('element_query'):
            result = self.move_skill.execute(env)
            if not result.success:
                return result
        
        # Then click
        env.input.click(double=self.double)
        
        return SkillResult(True, f"Clicked {'double' if self.double else ''}")


@register_skill
class DragSkill(Skill):
    """
    Drag from start to end.
    """
    name = "drag"
    description = "Drag from A to B"
    
    def __init__(
        self,
        start_query: str,
        end_query: str
    ):
        super().__init__(start=start_query, end=end_query)
        self.start_query = start_query
        self.end_query = end_query
        
    def is_applicable(self, state: VisualState) -> bool:
        return (state.find_element(self.start_query) is not None and 
                state.find_element(self.end_query) is not None)
                
    def execute(self, env) -> SkillResult:
        if not env.current_state:
            env.step(0)
            
        start_el = env.current_state.find_element(self.start_query)
        end_el = env.current_state.find_element(self.end_query)
        
        if not start_el or not end_el:
            return SkillResult(False, "Start or end element not found")
            
        sx, sy = start_el.bbox.center
        ex, ey = end_el.bbox.center
        
        env.input.drag(sx, sy, ex, ey)
        
        return SkillResult(True, f"Dragged {self.start_query} to {self.end_query}")
