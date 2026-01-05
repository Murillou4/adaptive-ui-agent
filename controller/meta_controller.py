"""
Meta-Controller

The "Brain" of the agent. Orchestrates the execution of skills to achieve high-level goals.
Uses LLMPlanner to decompose goals into a sequence of skills (TaskGraph).
Monitors execution and triggers re-planning on failure.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from planner.llm_planner import LLMPlanner, StructuredPlan
from skills import create_skill, SkillResult
from env.universal_env import UniversalEnv
from vision.state_extractor import VisualState

logger = logging.getLogger(__name__)


@dataclass
class GoalResult:
    """Result of a high-level goal execution."""
    success: bool
    message: str
    steps_executed: int
    executed_skills: List[str]


class MetaController:
    """
    Hierarchical controller that:
    1. Receives high-level goal (e.g., "Search for cats on Google")
    2. Plans sequence of skills (OpenBrowser -> TypeText -> Click)
    3. Executes skills sequentially
    4. Handles failures (Re-plan or abort)
    """
    
    def __init__(self, llm_provider: str = "openai", verbose: bool = True):
        self.planner = LLMPlanner(provider=None) # Uses default/mock for now, configurable later
        # In real usage, pass actual provider
        
        self.verbose = verbose
        
    async def achieve_goal(self, goal: str, env: UniversalEnv) -> GoalResult:
        """
        Achieve a high-level goal.
        
        Args:
            goal: Natural language goal
            env: Universal environment
            
        Returns:
            GoalResult
        """
        logger.info(f"Targeting goal: {goal}")
        
        # 1. Initial observation
        if env.current_state is None:
            env.step(0) # Capture state
            
        # 2. Generate Plan
        # Note: The current LLMPlanner generates a DSL for the RL agent (rectangles, etc.)
        # We need to adapt it or create a new planner mode for "Skill Planning".
        # For this Phase 5, we will implement a simplified heuristic planner 
        # that maps common goals to skills, or uses the LLM to generate a list of skill names.
        
        # TODO: Upgrade LLMPlanner to support "Skill Mode"
        # For now, we simulate planning for demonstration or use a simple parser
        plan_skills = self._plan_heuristic(goal, env.current_state)
        
        steps = 0
        executed_skills = []
        
        # 3. Execution Loop with Recovery
        max_retries = 2
        
        for skill_name, params in plan_skills:
            logger.info(f"Executing skill: {skill_name} with {params}")
            
            success = False
            retries = 0
            
            while not success and retries <= max_retries:
                # Create skill instance
                skill = create_skill(skill_name, **params)
                if not skill:
                    return GoalResult(False, f"Failed to create skill: {skill_name}", steps, executed_skills)
                
                # Check applicability
                if not skill.is_applicable(env.current_state):
                    logger.warning(f"Skill {skill_name} not applicable. Attempting recovery...")
                    # Simple recovery: Wait and refresh
                    env.step(0)
                    if not skill.is_applicable(env.current_state):
                         # Try pressing ESC to close popups?
                         from skills.keyboard_skills import HotkeySkill
                         HotkeySkill("esc").execute(env)
                         env.step(0)
                         
                    if not skill.is_applicable(env.current_state):
                        logger.error(f"Skill {skill_name} still not applicable after recovery.")
                        return GoalResult(False, f"Skill {skill_name} not applicable", steps, executed_skills)
                
                # Execute
                result = skill.execute(env)
                
                if result.success:
                    success = True
                    steps += 1
                    executed_skills.append(skill_name)
                    # Update state
                    if env.current_state is None:
                        env.step(0)
                else:
                    retries += 1
                    logger.warning(f"Skill {skill_name} failed: {result.message}. Retry {retries}/{max_retries}")
                    # Brief pause before retry
                    import time
                    time.sleep(1.0)
                    env.step(0) # Refresh state
            
            if not success:
                logger.error(f"Skill {skill_name} failed after {max_retries} retries.")
                return GoalResult(False, f"Skill {skill_name} failed: {result.message}", steps, executed_skills)
                
        return GoalResult(True, "Goal achieved", steps, executed_skills)

    def _plan_heuristic(self, goal: str, state: VisualState) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Simple heuristic planner for testing.
        In production, this would be the LLM call.
        """
        goal_lower = goal.lower()
        plan = []
        
        if "google" in goal_lower and "search" in goal_lower:
            # Example: "Search for cats on Google"
            
            # 1. Open Default Browser via Run Dialog (Win+R)
            # This is universal: works for Chrome, Edge, Firefox, whatever is default.
            plan.append(("hotkey", {"keys": ("win", "r")}))
            
            # Brief wait for Run dialog
            # (We rely on retry/wait in execution, or we can add a dummy wait skill if needed,
            # but usually type_text has a small delay)
            
            # 2. Type URL
            plan.append(("type_text", {"text": "https://www.google.com", "clear_first": False}))
            plan.append(("hotkey", {"keys": ("enter",)}))
            
            # 3. Wait for browser to load (heuristic wait could be improved with visual check, 
            # but for now we rely on the next skill's applicability check retry)
            
            # 4. Click search bar (input field)
            # The page usually auto-focuses, but we included this for robustness.
            # If auto-focused, typing immediately works.
            # Let's try typing explicitly.
            
            search_query = goal_lower.replace("search for", "").replace("on google", "").strip()
            
            # We add a "wait" using a visual check would be best. 
            # For now, we assume the next step will retry until applicable.
            
            plan.append(("click", {"element_query": "input field"}))
            plan.append(("type_text", {"text": search_query}))
            plan.append(("hotkey", {"keys": ("enter",)}))
            
        elif "open" in goal_lower and ("notepad" in goal_lower or "calculator" in goal_lower or "app" in goal_lower):
            # General "Open [App]" handler
            app_name = goal_lower.replace("open", "").strip()
            plan.append(("open_app", {"app_name": app_name}))
            
            # If text is involved (e.g. "Open notepad and type...")
            if "type" in goal_lower:
                text_content = goal_lower.split("type")[-1].strip()
                plan.append(("type_text", {"text": text_content}))
            
        elif "click" in goal_lower:
            # "Click the submit button"
            target = goal_lower.replace("click", "").strip()
            plan.append(("click", {"element_query": target}))
            
        else:
            # Default fallback for testing
            plan.append(("move_to", {"x": 100, "y": 100}))
            
        return plan
