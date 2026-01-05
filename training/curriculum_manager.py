"""
Curriculum Manager

Manages the progression of tasks for the agent.
Loads curriculum from YAML files and serves tasks appropriate for the agent's current level.
"""

import yaml
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import os

logger = logging.getLogger(__name__)

@dataclass
class TaskDefinition:
    """Definition of a single learning task."""
    id: str
    name: str
    description: str
    goal_prompt: str  # The goal string passed to the agent
    difficulty: int
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timeout_steps: int = 100
    env_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumLevel:
    """A level containing a set of tasks."""
    name: str
    tasks: List[TaskDefinition]
    completion_threshold: float = 0.8  # % of tasks passed to unlock next level


class CurriculumManager:
    """
    Manages task progression.
    
    Features:
    - Load curriculum from YAML
    - Track progress (completed tasks)
    - Serve next task based on skills
    """
    
    def __init__(self, curriculum_path: Optional[str] = None):
        self.levels: List[CurriculumLevel] = []
        self.completed_tasks: Dict[str, float] = {}  # task_id -> success_rate
        self.current_level_idx: int = 0
        
        if curriculum_path and os.path.exists(curriculum_path):
            self.load_curriculum(curriculum_path)
            
    def load_curriculum(self, path: str):
        """Load curriculum from YAML file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            self.levels = []
            for level_data in data.get('levels', []):
                tasks = []
                for task_data in level_data.get('tasks', []):
                    tasks.append(TaskDefinition(
                        id=task_data['id'],
                        name=task_data['name'],
                        description=task_data.get('description', ''),
                        goal_prompt=task_data['goal'],
                        difficulty=task_data.get('difficulty', 1),
                        prerequisites=task_data.get('requires', []),
                        timeout_steps=task_data.get('timeout', 100)
                    ))
                
                self.levels.append(CurriculumLevel(
                    name=level_data['name'],
                    tasks=tasks,
                    completion_threshold=level_data.get('threshold', 0.8)
                ))
            
            logger.info(f"Loaded curriculum with {len(self.levels)} levels")
            
        except Exception as e:
            logger.error(f"Failed to load curriculum: {e}")
            
    def get_next_task(self) -> Optional[TaskDefinition]:
        """Get the next appropriate task for the agent."""
        if not self.levels:
            return None
            
        level = self.levels[self.current_level_idx]
        
        # Filter tasks in current level
        available_tasks = []
        for task in level.tasks:
            # Check if already mastered
            if self.is_task_mastered(task.id):
                continue
                
            # Check prerequisites (from previous levels or same level)
            if self.are_prerequisites_met(task):
                available_tasks.append(task)
        
        if not available_tasks:
            # Maybe level up?
            if self.can_advance_level():
                self.current_level_idx += 1
                if self.current_level_idx < len(self.levels):
                    logger.info(f"Advancing to level: {self.levels[self.current_level_idx].name}")
                    return self.get_next_task()
                else:
                    logger.info("Curriculum completed!")
                    return None
            else:
                # Level not complete but no tasks? (Shouldn't happen unless prereqs loops)
                return None
                
        # Simple strategy: return first available. 
        # In future: return tasks with lowest success rate that are unlocked
        return available_tasks[0]
        
    def update_task_result(self, task_id: str, success: bool):
        """Update progress after a task attempt."""
        # Simple moving average or just boolean for now
        # Ideally, we track history. Here we just set 1.0 if success.
        current = self.completed_tasks.get(task_id, 0.0)
        
        if success:
            new_score = 1.0  # Instant mastery for simple implementation
        else:
            new_score = current # Don't degrade for now, or decay slightly
            
        self.completed_tasks[task_id] = new_score
        
    def is_task_mastered(self, task_id: str) -> bool:
        return self.completed_tasks.get(task_id, 0.0) >= 1.0
        
    def are_prerequisites_met(self, task: TaskDefinition) -> bool:
        for prereq_id in task.prerequisites:
            if not self.is_task_mastered(prereq_id):
                return False
        return True
        
    def can_advance_level(self) -> bool:
        if self.current_level_idx >= len(self.levels):
            return False
            
        level = self.levels[self.current_level_idx]
        mastered_count = sum(1 for t in level.tasks if self.is_task_mastered(t.id))
        rate = mastered_count / len(level.tasks) if level.tasks else 1.0
        
        return rate >= level.completion_threshold
