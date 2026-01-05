"""
Extended Sandbox Environment with Dynamic Element Creation

Extends the base sandbox to support:
- Creating multiple elements
- Different element types
- Dynamic reward functions from LLM plans
"""

import numpy as np
import pygame
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field

from env.sandbox_env import SandboxEnv, EnvConfig
from planner.goal_dsl import Color, ElementType
from planner.visual_detectors import BoundingBox


@dataclass
class VisualElement:
    """An element in the extended environment."""
    type: ElementType
    bbox: BoundingBox
    color: Tuple[int, int, int]
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.bbox.x, self.bbox.y,
            self.bbox.width, self.bbox.height
        )


class ExtendedSandboxEnv(SandboxEnv):
    """
    Extended sandbox environment for LLM-RL integration.
    
    Supports:
    - Creating elements via actions
    - Dynamic reward functions
    - Multiple element types
    """
    
    # Extended action space
    # 0-7: Movement (same as parent)
    # 8: Click/Create element
    # 9-14: Change current tool/element type
    ACTION_NAMES = [
        'up', 'down', 'left', 'right',
        'up-left', 'up-right', 'down-left', 'down-right',
        'create',  # Creates element at cursor
        'tool_rectangle', 'tool_circle', 'tool_button',
        'tool_color_blue', 'tool_color_red', 'tool_color_green'
    ]
    
    TOOL_COLORS = {
        'blue': (60, 120, 220),
        'red': (220, 60, 60),
        'green': (60, 180, 60),
        'yellow': (220, 220, 60),
    }
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: str = "rgb_array",
        enable_creation: bool = True
    ):
        super().__init__(config, render_mode)
        
        self.enable_creation = enable_creation
        
        # Extended action space
        self.action_space_n = 15 if enable_creation else 9
        
        # Current tool state
        self.current_tool = ElementType.RECTANGLE
        self.current_color = 'blue'
        
        # Created elements
        self.elements: List[VisualElement] = []
        
        # Dynamic reward function (set by ObjectiveTranslator)
        self.reward_fn: Optional[Callable] = None
        self.prev_obs: Optional[np.ndarray] = None
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment, clearing all created elements."""
        obs, info = super().reset(seed)
        
        # Clear created elements
        self.elements.clear()
        
        # Reset tool state
        self.current_tool = ElementType.RECTANGLE
        self.current_color = 'blue'
        
        # Store for reward computation
        self.prev_obs = obs.copy()
        
        info['elements'] = []
        info['current_tool'] = self.current_tool.value
        info['current_color'] = self.current_color
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action with extended capabilities."""
        
        if not self.enable_creation or action < 9:
            # Use parent step for basic actions
            if action == 8:
                # Create element instead of click
                return self._create_element()
            else:
                obs, reward, terminated, truncated, info = super().step(action)
                
                # Apply dynamic reward if set
                if self.reward_fn and self.prev_obs is not None:
                    reward, reward_info = self.reward_fn(self.prev_obs, action, obs)
                    info['reward_info'] = reward_info
                
                self.prev_obs = obs.copy()
                return obs, reward, terminated, truncated, info
        
        # Extended actions
        reward = self.config.step_penalty
        terminated = False
        truncated = False
        
        if action == 9:  # Switch to rectangle tool
            self.current_tool = ElementType.RECTANGLE
        elif action == 10:  # Switch to circle tool
            self.current_tool = ElementType.CIRCLE
        elif action == 11:  # Switch to button tool
            self.current_tool = ElementType.BUTTON
        elif action == 12:  # Blue color
            self.current_color = 'blue'
        elif action == 13:  # Red color
            self.current_color = 'red'
        elif action == 14:  # Green color
            self.current_color = 'green'
        
        self.state.step_count += 1
        self.state.total_reward += reward
        truncated = self.state.step_count >= self.config.max_steps
        
        obs = self._get_observation()
        info = self._get_info()
        info['current_tool'] = self.current_tool.value
        info['current_color'] = self.current_color
        
        if self.reward_fn and self.prev_obs is not None:
            reward, reward_info = self.reward_fn(self.prev_obs, action, obs)
            info['reward_info'] = reward_info
        
        self.prev_obs = obs.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _create_element(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Create an element at the current cursor position."""
        # Default element size
        size = 12
        
        # Create bounding box centered at cursor
        cx, cy = self.state.cursor_pos
        x = max(0, cx - size // 2)
        y = max(0, cy - size // 2)
        
        # Ensure within bounds
        x = min(x, self.config.size - size)
        y = min(y, self.config.size - size)
        
        bbox = BoundingBox(x, y, size, size)
        color = self.TOOL_COLORS.get(self.current_color, (60, 120, 220))
        
        element = VisualElement(
            type=self.current_tool,
            bbox=bbox,
            color=color
        )
        
        self.elements.append(element)
        
        # Get observation and compute reward
        self.state.step_count += 1
        obs = self._get_observation()
        
        # Base reward for creating something
        reward = 0.1
        
        # Apply dynamic reward if set
        if self.reward_fn and self.prev_obs is not None:
            reward, reward_info = self.reward_fn(self.prev_obs, 8, obs)
        else:
            reward_info = {}
        
        self.state.total_reward += reward
        
        terminated = False
        truncated = self.state.step_count >= self.config.max_steps
        
        info = self._get_info()
        info['element_created'] = True
        info['total_elements'] = len(self.elements)
        info['reward_info'] = reward_info
        
        self.prev_obs = obs.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _render_frame(self):
        """Render with created elements."""
        # Call parent render first
        super()._render_frame()
        
        # Draw created elements
        for element in self.elements:
            if element.type == ElementType.CIRCLE:
                center = (element.bbox.cx, element.bbox.cy)
                radius = element.bbox.width // 2
                pygame.draw.circle(self.screen, element.color, center, radius)
            else:
                # Default to rectangle
                pygame.draw.rect(self.screen, element.color, element.rect)
    
    def set_dynamic_reward(self, reward_fn: Callable):
        """
        Set a dynamic reward function.
        
        Args:
            reward_fn: Function(prev_obs, action, curr_obs) -> (reward, info)
        """
        self.reward_fn = reward_fn
    
    def clear_elements(self):
        """Clear all created elements."""
        self.elements.clear()
    
    def _get_info(self) -> dict:
        """Extended info with element data."""
        info = super()._get_info()
        info['total_elements'] = len(self.elements)
        info['elements'] = [
            {
                'type': e.type.value,
                'bbox': (e.bbox.x, e.bbox.y, e.bbox.width, e.bbox.height),
                'color': e.color
            }
            for e in self.elements
        ]
        return info


def make_extended_env(
    config: Optional[EnvConfig] = None,
    enable_creation: bool = True
) -> ExtendedSandboxEnv:
    """Create an extended sandbox environment."""
    return ExtendedSandboxEnv(config=config, enable_creation=enable_creation)


if __name__ == "__main__":
    # Test extended environment
    print("Testing Extended Sandbox Environment...")
    
    env = ExtendedSandboxEnv()
    obs, info = env.reset()
    
    print(f"Action space: {env.action_space_n}")
    print(f"Observation shape: {obs.shape}")
    print(f"Initial elements: {info['total_elements']}")
    
    # Move and create some elements
    actions = [3, 3, 3, 8, 3, 3, 8, 3, 3, 8]  # Move right and create 3 times
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if action == 8:
            print(f"Created element! Total: {info['total_elements']}")
    
    print(f"\nFinal elements: {info['total_elements']}")
    print(f"Total reward: {info['total_reward']:.2f}")
    
    env.close()
    print("\nExtended Environment test passed!")
