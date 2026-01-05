"""
Sandbox Environment for Adaptive UI Agent
A pygame-based environment where an agent learns to click targets using visual input.

Based on paper 2312.01203v3: Discrete Representations for Continual RL
"""

import numpy as np
import pygame
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Configuration for the sandbox environment."""
    size: int = 64
    target_size: int = 12
    cursor_size: int = 4
    max_steps: int = 200
    
    # Rewards
    click_target: float = 1.0
    click_obstacle: float = -1.0
    click_background: float = -0.1
    step_penalty: float = -0.01


@dataclass
class GameState:
    """Current state of the game."""
    cursor_pos: np.ndarray = field(default_factory=lambda: np.array([32, 32]))
    target_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    obstacle_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    step_count: int = 0
    total_reward: float = 0.0
    done: bool = False


class SandboxEnv:
    """
    Pygame sandbox environment for visual RL.
    
    The agent observes the screen as pixels and learns to:
    1. Move a cursor (8 directions)
    2. Click on targets (+reward) while avoiding obstacles (-penalty)
    
    Supports continual RL through rule swapping (target becomes obstacle).
    """
    
    # Action space: 8 directions + click
    ACTIONS = {
        0: (0, -1),   # Up
        1: (0, 1),    # Down
        2: (-1, 0),   # Left
        3: (1, 0),    # Right
        4: (-1, -1),  # Up-Left
        5: (1, -1),   # Up-Right
        6: (-1, 1),   # Down-Left
        7: (1, 1),    # Down-Right
        8: None,      # Click
    }
    
    # Colors (RGB)
    COLORS = {
        'background': (50, 50, 55),
        'target': (60, 120, 220),      # Blue
        'obstacle': (220, 60, 60),     # Red
        'cursor': (255, 255, 255),     # White
        'cursor_outline': (0, 0, 0),   # Black
    }
    
    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str = "rgb_array"):
        """
        Initialize the sandbox environment.
        
        Args:
            config: Environment configuration
            render_mode: "rgb_array" for headless, "human" for display
        """
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Game state
        self.state = GameState()
        
        # Target/obstacle role tracking for continual RL
        self.target_is_blue = True  # Blue is target, Red is obstacle
        
        # Custom reward overrides
        self.reward_overrides: Dict[str, float] = {}
        
        # Initialize pygame
        pygame.init()
        self._init_display()
        
        # Action and observation space info
        self.action_space_n = 9  # 8 directions + click
        self.observation_shape = (self.config.size, self.config.size, 3)
        
    def _init_display(self):
        """Initialize pygame display."""
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.config.size * 8, self.config.size * 8)  # Scaled for visibility
            )
            pygame.display.set_caption("Adaptive UI Agent - Sandbox")
        else:
            self.screen = pygame.Surface((self.config.size, self.config.size))
        
        self.clock = pygame.time.Clock()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: RGB image of the environment
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Reset cursor to center
        center = self.config.size // 2
        self.state.cursor_pos = np.array([center, center])
        
        # Place target and obstacle at random non-overlapping positions
        self._place_objects()
        
        # Reset counters
        self.state.step_count = 0
        self.state.total_reward = 0.0
        self.state.done = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _place_objects(self):
        """Place target and obstacle at random positions."""
        margin = self.config.target_size + 2
        max_pos = self.config.size - margin
        
        # Place target
        self.state.target_pos = np.array([
            np.random.randint(margin, max_pos),
            np.random.randint(margin, max_pos)
        ])
        
        # Place obstacle (ensure it doesn't overlap with target)
        min_distance = self.config.target_size * 2
        for _ in range(100):  # Max attempts
            self.state.obstacle_pos = np.array([
                np.random.randint(margin, max_pos),
                np.random.randint(margin, max_pos)
            ])
            distance = np.linalg.norm(self.state.target_pos - self.state.obstacle_pos)
            if distance >= min_distance:
                break
                
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action index (0-8)
            
        Returns:
            observation: RGB image
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was cut short (max steps)
            info: Additional information
        """
        assert 0 <= action < 9, f"Invalid action: {action}"
        
        reward = self._get_reward('step_penalty')
        terminated = False
        
        if action == 8:  # Click action
            reward += self._handle_click()
            terminated = self._check_click_result()
        else:
            # Move cursor
            direction = self.ACTIONS[action]
            new_pos = self.state.cursor_pos + np.array(direction) * 4  # Move 4 pixels
            
            # Clamp to bounds
            new_pos = np.clip(new_pos, 0, self.config.size - 1)
            self.state.cursor_pos = new_pos
            
        self.state.step_count += 1
        self.state.total_reward += reward
        
        # Check for truncation
        truncated = self.state.step_count >= self.config.max_steps
        self.state.done = terminated or truncated
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _handle_click(self) -> float:
        """
        Handle click action and return reward.
        
        Returns:
            reward: Reward based on what was clicked
        """
        cursor = self.state.cursor_pos
        half_size = self.config.target_size // 2
        
        # Check if cursor is on target
        target = self.state.target_pos
        if self._point_in_rect(cursor, target, half_size):
            return self._get_reward('click_target')
            
        # Check if cursor is on obstacle
        obstacle = self.state.obstacle_pos
        if self._point_in_rect(cursor, obstacle, half_size):
            return self._get_reward('click_obstacle')
            
        # Clicked on background
        return self._get_reward('click_background')
    
    def _check_click_result(self) -> bool:
        """Check if click resulted in episode termination."""
        cursor = self.state.cursor_pos
        half_size = self.config.target_size // 2
        
        # Episode ends on target click (success) or obstacle click (failure)
        target = self.state.target_pos
        obstacle = self.state.obstacle_pos
        
        return (self._point_in_rect(cursor, target, half_size) or 
                self._point_in_rect(cursor, obstacle, half_size))
    
    @staticmethod
    def _point_in_rect(point: np.ndarray, rect_center: np.ndarray, half_size: int) -> bool:
        """Check if point is inside rectangle."""
        return (abs(point[0] - rect_center[0]) <= half_size and 
                abs(point[1] - rect_center[1]) <= half_size)
    
    def _get_reward(self, reward_type: str) -> float:
        """Get reward value, considering overrides."""
        if reward_type in self.reward_overrides:
            return self.reward_overrides[reward_type]
        return getattr(self.config, reward_type)
    
    def _get_observation(self) -> np.ndarray:
        """Render and return the current observation as RGB array."""
        self._render_frame()
        
        # Get pixels from surface
        pixels = pygame.surfarray.array3d(self.screen)
        # Transpose to (H, W, C) format
        pixels = np.transpose(pixels, (1, 0, 2))
        
        return pixels.astype(np.uint8)
    
    def _render_frame(self):
        """Render the current frame to the pygame surface."""
        # Clear background
        self.screen.fill(self.COLORS['background'])
        
        # Determine colors based on current rules
        if self.target_is_blue:
            target_color = self.COLORS['target']
            obstacle_color = self.COLORS['obstacle']
        else:
            target_color = self.COLORS['obstacle']
            obstacle_color = self.COLORS['target']
        
        # Draw target
        half_size = self.config.target_size // 2
        target_rect = pygame.Rect(
            self.state.target_pos[0] - half_size,
            self.state.target_pos[1] - half_size,
            self.config.target_size,
            self.config.target_size
        )
        pygame.draw.rect(self.screen, target_color, target_rect)
        
        # Draw obstacle
        obstacle_rect = pygame.Rect(
            self.state.obstacle_pos[0] - half_size,
            self.state.obstacle_pos[1] - half_size,
            self.config.target_size,
            self.config.target_size
        )
        pygame.draw.rect(self.screen, obstacle_color, obstacle_rect)
        
        # Draw cursor (small crosshair)
        cx, cy = self.state.cursor_pos
        cs = self.config.cursor_size
        
        # Cursor outline
        pygame.draw.line(self.screen, self.COLORS['cursor_outline'], 
                        (cx - cs - 1, cy), (cx + cs + 1, cy), 3)
        pygame.draw.line(self.screen, self.COLORS['cursor_outline'], 
                        (cx, cy - cs - 1), (cx, cy + cs + 1), 3)
        
        # Cursor
        pygame.draw.line(self.screen, self.COLORS['cursor'], 
                        (cx - cs, cy), (cx + cs, cy), 1)
        pygame.draw.line(self.screen, self.COLORS['cursor'], 
                        (cx, cy - cs), (cx, cy + cs), 1)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        return {
            'step_count': self.state.step_count,
            'total_reward': self.state.total_reward,
            'cursor_pos': self.state.cursor_pos.tolist(),
            'target_pos': self.state.target_pos.tolist(),
            'obstacle_pos': self.state.obstacle_pos.tolist(),
            'target_is_blue': self.target_is_blue,
        }
    
    # ==================== Continual RL Methods ====================
    
    def swap_targets(self):
        """
        Swap target and obstacle roles (Continual RL).
        Blue becomes obstacle, Red becomes target.
        """
        self.target_is_blue = not self.target_is_blue
        
    def set_rule(self, rule: str):
        """
        Set environment rule for continual RL.
        
        Args:
            rule: Rule string (e.g., "blue_bad", "blue_good")
        """
        if rule == "blue_bad":
            self.target_is_blue = False
        elif rule == "blue_good":
            self.target_is_blue = True
        else:
            raise ValueError(f"Unknown rule: {rule}")
            
    def set_reward(self, reward_type: str, value: float):
        """
        Override a reward value.
        
        Args:
            reward_type: Type of reward to override
            value: New reward value
        """
        valid_types = ['click_target', 'click_obstacle', 'click_background', 'step_penalty']
        if reward_type not in valid_types:
            raise ValueError(f"Invalid reward type: {reward_type}. Must be one of {valid_types}")
        self.reward_overrides[reward_type] = value
        
    def reset_rewards(self):
        """Reset all reward overrides to default."""
        self.reward_overrides.clear()
    
    # ==================== Utility Methods ====================
    
    def render(self):
        """Render for human viewing."""
        if self.render_mode == "human":
            # Scale up for visibility
            scaled = pygame.transform.scale(
                self.screen, 
                (self.config.size * 8, self.config.size * 8)
            )
            self.screen.blit(scaled, (0, 0))
            pygame.display.flip()
            self.clock.tick(30)
            
    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
        
    def get_screenshot(self) -> np.ndarray:
        """Get current screenshot as numpy array."""
        return self._get_observation()
    
    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)


# Module-level convenience function
def make_env(config: Optional[EnvConfig] = None, render_mode: str = "rgb_array") -> SandboxEnv:
    """Create a new sandbox environment."""
    return SandboxEnv(config=config, render_mode=render_mode)


if __name__ == "__main__":
    # Quick test
    env = SandboxEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Take a few random actions
    for i in range(5):
        action = np.random.randint(0, 9)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, terminated={terminated}")
        
    env.close()
    print("Environment test passed!")
