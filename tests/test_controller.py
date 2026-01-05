"""
Tests for Hierarchical Controller
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.meta_controller import MetaController, GoalResult
from skills import SkillResult


class TestMetaController:
    @pytest.mark.asyncio
    async def test_achieve_goal_heuristic(self):
        controller = MetaController()
        
        # Mock environment
        env = MagicMock()
        env.current_state = MagicMock()
        env.step = MagicMock()
        
        # Mock skill creation and execution
        with patch('controller.meta_controller.create_skill') as mock_create:
            mock_skill = MagicMock()
            mock_skill.is_applicable.return_value = True
            mock_skill.execute.return_value = SkillResult(True, "Success")
            
            mock_create.return_value = mock_skill
            
            # Test "search" goal (triggers heuristic)
            result = await controller.achieve_goal("Search for cats on Google", env)
            
            assert result.success
            assert len(result.executed_skills) == 3 # Click, Type, Limit
            assert mock_create.call_count == 3
            
    @pytest.mark.asyncio
    async def test_skill_failure(self):
        controller = MetaController()
        env = MagicMock()
        env.current_state = MagicMock()
        
        with patch('controller.meta_controller.create_skill') as mock_create:
            mock_skill = MagicMock()
            mock_skill.is_applicable.return_value = True
            # Fail the skill
            mock_skill.execute.return_value = SkillResult(False, "Failed")
            
            mock_create.return_value = mock_skill
            
            result = await controller.achieve_goal("Click something", env)
            
            assert not result.success
            assert "failed" in result.message.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
