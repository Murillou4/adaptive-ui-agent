"""
Tests for Curriculum System
"""

import pytest
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.curriculum_manager import CurriculumManager, TaskDefinition, CurriculumLevel


class TestCurriculum:
    
    @pytest.fixture
    def sample_curriculum_file(self, tmp_path):
        """Create a temporary curriculum YAML file."""
        data = {
            "levels": [
                {
                    "name": "Level 1",
                    "threshold": 1.0,
                    "tasks": [
                        {"id": "task1", "name": "Task 1", "goal": "Do 1"},
                        {"id": "task2", "name": "Task 2", "goal": "Do 2", "requires": ["task1"]}
                    ]
                },
                {
                    "name": "Level 2",
                    "tasks": [
                        {"id": "task3", "name": "Task 3", "goal": "Do 3"}
                    ]
                }
            ]
        }
        
        p = tmp_path / "curriculum.yaml"
        with open(p, "w") as f:
            yaml.dump(data, f)
        return str(p)

    def test_load_curriculum(self, sample_curriculum_file):
        manager = CurriculumManager(sample_curriculum_file)
        
        assert len(manager.levels) == 2
        assert len(manager.levels[0].tasks) == 2
        assert manager.levels[0].tasks[0].id == "task1"
        assert manager.levels[0].tasks[1].prerequisites == ["task1"]

    def test_progression(self, sample_curriculum_file):
        manager = CurriculumManager(sample_curriculum_file)
        
        # Initial: Task 1 available
        task = manager.get_next_task()
        assert task.id == "task1"
        
        # Complete Task 1
        manager.update_task_result("task1", True)
        
        # Next: Task 2 (unlocked by Task 1)
        task = manager.get_next_task()
        assert task.id == "task2"
        
        # Complete Task 2
        manager.update_task_result("task2", True)
        
        # Level complete (threshold 1.0). Should advance to Level 2.
        task = manager.get_next_task()
        assert task.id == "task3"
        assert manager.current_level_idx == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
