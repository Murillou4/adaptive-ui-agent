"""
Tests for Skill Library
"""

import pytest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills import create_skill, SKILL_REGISTRY
from skills.cursor_skills import MoveToSkill, ClickSkill
from skills.keyboard_skills import TypeTextSkill
from vision.state_extractor import VisualState


class TestSkillRegistry:
    def test_registry(self):
        # Ensure skills are registered (imports triggers registration)
        assert "move_to" in SKILL_REGISTRY
        assert "click" in SKILL_REGISTRY
        assert "type_text" in SKILL_REGISTRY
        
    def test_factory(self):
        skill = create_skill("move_to", x=10, y=20)
        assert isinstance(skill, MoveToSkill)
        assert skill.target_pos == (10, 20)


class TestSkills:
    @pytest.fixture
    def mock_env(self):
        """Mock environment for skills testing."""
        env = MagicMock()
        env.input = MagicMock()
        env.current_state = MagicMock()
        
        # Mock finding an element
        mock_bbox = MagicMock()
        mock_bbox.center = (100, 200)
        
        mock_elem = MagicMock()
        mock_elem.bbox = mock_bbox
        
        env.current_state.find_element.return_value = mock_elem
        
        return env

    def test_move_to_coords(self, mock_env):
        skill = MoveToSkill(x=50, y=60)
        res = skill.execute(mock_env)
        
        assert res.success
        mock_env.input.move_to.assert_called_with(50, 60, duration=0.5, tweet=True)

    def test_move_to_element(self, mock_env):
        skill = MoveToSkill(element_query="button")
        res = skill.execute(mock_env)
        
        assert res.success
        # Should move to center of mock element (100, 200)
        mock_env.input.move_to.assert_called_with(100, 200, duration=0.5, tweet=True)

    def test_click(self, mock_env):
        skill = ClickSkill(x=50, y=60)
        res = skill.execute(mock_env)
        
        assert res.success
        mock_env.input.move_to.assert_called() # First move
        mock_env.input.click.assert_called()   # Then click

    def test_type_text(self, mock_env):
        skill = TypeTextSkill(text="hello")
        res = skill.execute(mock_env)
        
        assert res.success
        mock_env.input.type_text.assert_called_with("hello")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
