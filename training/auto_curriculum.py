"""
Auto-Curriculum Generator

Generates new learning tasks automatically using LLM.
Analyzes current agent performance and suggests the "frontier" of learnable tasks.
"""

import logging
from typing import List, Optional
from dataclasses import asdict

from training.curriculum_manager import TaskDefinition, CurriculumManager
from planner.llm_provider import LLMProvider, OpenAIProvider

logger = logging.getLogger(__name__)

class AutoCurriculum:
    """
    Generates new tasks to expand the curriculum.
    """
    
    def __init__(self, manager: CurriculumManager, llm_provider: Optional[LLMProvider] = None):
        self.manager = manager
        self.llm = llm_provider or OpenAIProvider() # Default to OpenAI if available
        
    def generate_next_tasks(self, n: int = 3) -> List[TaskDefinition]:
        """
        Generate N new tasks based on current progress.
        """
        # 1. Get current context
        completed = list(self.manager.completed_tasks.keys())
        current_level = self.manager.levels[self.manager.current_level_idx]
        
        prompt = f"""
        I am training an AI agent to use a computer.
        
        The agent has mastered these tasks: {completed}
        
        Current Learning Level: {current_level.name}
        Existing tasks in level: {[t.name for t in current_level.tasks]}
        
        Generate {n} NEW tasks that are slightly harder than the mastered ones, but achievable.
        Return ONLY valid YAML format for tasks.
        """
        
        # In a real implementation, we would call LLM here:
        # response = self.llm.generate(prompt)
        # new_tasks = parse_yaml(response)
        
        logger.info("Auto-generating tasks (Simulated)")
        
        # Simulating generation for Phase 6 MVP
        generated_tasks = [
            TaskDefinition(
                id=f"auto_gen_{i}",
                name=f"Generated Task {i}",
                description="Automatically generated task",
                goal_prompt="Do something new",
                difficulty=current_level.tasks[0].difficulty + 1 if current_level.tasks else 1
            )
            for i in range(n)
        ]
        
        return generated_tasks
        
    def extend_curriculum(self):
        """Add generated tasks to the manager."""
        new_tasks = self.generate_next_tasks()
        
        # Add to current level (simplified logic)
        current_level = self.manager.levels[self.manager.current_level_idx]
        current_level.tasks.extend(new_tasks)
        
        logger.info(f"Extended curriculum with {len(new_tasks)} new tasks")
