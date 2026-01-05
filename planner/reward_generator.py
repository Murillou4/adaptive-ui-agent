"""
Reward Generator - Converts Structured Plans to Reward Functions

This is the CRITICAL glue between LLM (planner) and RL (executor).
It translates high-level goals into scalar reward signals.

Key principle: If you can't measure it, you can't reward it.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field

from planner.goal_dsl import (
    StructuredPlan,
    ElementSpec,
    ConstraintSpec,
    ElementType,
    ConstraintType,
    Color,
)
from planner.visual_detectors import (
    VisualDetector,
    DetectedElement,
    detect_alignment,
    detect_spacing,
    detect_centering,
)


# =============================================================================
# REWARD CONFIGURATION
# =============================================================================

@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Element rewards
    element_created: float = 1.0
    element_correct_color: float = 0.5
    
    # Constraint rewards
    constraint_satisfied: float = 0.5
    constraint_improved: float = 0.2
    
    # Penalties
    step_penalty: float = -0.01
    useless_action: float = -0.1
    constraint_violated: float = -0.5
    
    # Completion bonus
    success_bonus: float = 5.0
    
    # Shaping
    enable_shaping: bool = True  # Reward for partial progress


@dataclass
class RewardState:
    """Tracks state for reward shaping."""
    prev_element_count: int = 0
    prev_alignment_score: float = 0.0
    prev_spacing_score: float = 0.0
    prev_constraint_scores: Dict[str, float] = field(default_factory=dict)
    steps: int = 0
    
    def reset(self):
        self.prev_element_count = 0
        self.prev_alignment_score = 0.0
        self.prev_spacing_score = 0.0
        self.prev_constraint_scores.clear()
        self.steps = 0


# =============================================================================
# DYNAMIC REWARD FUNCTION
# =============================================================================

class DynamicRewardFunction:
    """
    Generates reward signals from a structured plan.
    
    This is the translator layer:
    Plan (from LLM) → Reward signals (for RL)
    """
    
    def __init__(
        self,
        plan: StructuredPlan,
        detector: VisualDetector,
        config: Optional[RewardConfig] = None
    ):
        """
        Initialize reward function from plan.
        
        Args:
            plan: Structured plan from LLM
            detector: Visual detector for element detection
            config: Reward configuration
        """
        self.plan = plan
        self.detector = detector
        self.config = config or RewardConfig()
        self.state = RewardState()
        
        # Pre-compute required elements and constraints
        self._parse_requirements()
    
    def _parse_requirements(self):
        """Parse plan into requirement checkers."""
        self.required_elements = []
        self.required_constraints = []
        
        # Element requirements
        for elem_spec in self.plan.elements:
            self.required_elements.append({
                "type": elem_spec.type,
                "count": elem_spec.count,
                "color": elem_spec.color,
            })
        
        # Constraint requirements
        for const_spec in self.plan.constraints:
            self.required_constraints.append({
                "type": const_spec.type,
                "value": const_spec.value,
                "tolerance": const_spec.tolerance,
            })
    
    def reset(self):
        """Reset reward state for new episode."""
        self.state.reset()
    
    def compute(
        self,
        prev_obs: np.ndarray,
        action: int,
        curr_obs: np.ndarray
    ) -> tuple[float, Dict[str, Any]]:
        """
        Compute reward for a single step.
        
        Args:
            prev_obs: Previous observation
            action: Action taken
            curr_obs: Current observation after action
            
        Returns:
            reward: Scalar reward
            info: Dictionary with reward breakdown
        """
        reward = 0.0
        info = {"components": {}}
        
        # Step penalty
        reward += self.config.step_penalty
        info["components"]["step_penalty"] = self.config.step_penalty
        self.state.steps += 1
        
        # Detect elements in both frames
        curr_elements = self.detector.detect_all(curr_obs)
        curr_count = len(curr_elements)
        
        # Reward for creating new elements
        if curr_count > self.state.prev_element_count:
            element_reward = self.config.element_created * (curr_count - self.state.prev_element_count)
            reward += element_reward
            info["components"]["new_elements"] = element_reward
        
        # Check element requirements
        elem_score, elem_info = self._evaluate_elements(curr_obs, curr_elements)
        if self.config.enable_shaping:
            reward += elem_score * 0.1  # Small shaping reward
        info["element_requirements"] = elem_info
        
        # Check constraint satisfaction
        const_score, const_info = self._evaluate_constraints(curr_obs, curr_elements)
        if self.config.enable_shaping:
            reward += const_score * 0.1  # Small shaping reward
        info["constraint_satisfaction"] = const_info
        
        # Update state
        self.state.prev_element_count = curr_count
        
        info["total_reward"] = reward
        info["steps"] = self.state.steps
        
        return reward, info
    
    def _evaluate_elements(
        self,
        obs: np.ndarray,
        elements: List[DetectedElement]
    ) -> tuple[float, Dict]:
        """Evaluate how well element requirements are met."""
        score = 0.0
        info = {}
        
        for i, req in enumerate(self.required_elements):
            req_type = req["type"]
            req_count = req["count"]
            req_color = req["color"]
            
            # Count matching elements
            if req_color:
                matching = [e for e in elements if e.color == req_color]
            else:
                matching = [e for e in elements if e.type == req_type]
            
            actual_count = len(matching)
            
            # Score: ratio of actual to required (capped at 1.0)
            ratio = min(actual_count / req_count, 1.0) if req_count > 0 else 1.0
            score += ratio
            
            info[f"element_{i}"] = {
                "required": req_count,
                "actual": actual_count,
                "color": req_color.value if req_color else None,
                "satisfied": actual_count >= req_count,
            }
        
        # Normalize score
        if self.required_elements:
            score /= len(self.required_elements)
        
        return score, info
    
    def _evaluate_constraints(
        self,
        obs: np.ndarray,
        elements: List[DetectedElement]
    ) -> tuple[float, Dict]:
        """Evaluate constraint satisfaction."""
        score = 0.0
        info = {}
        
        if len(elements) < 2:
            # Not enough elements to check constraints
            return 0.0, {"insufficient_elements": True}
        
        alignment = detect_alignment(elements)
        spacing = detect_spacing(elements)
        centered = detect_centering(elements, self.detector.canvas_size)
        
        for i, const in enumerate(self.required_constraints):
            const_type = const["type"]
            satisfied = False
            
            if const_type == ConstraintType.ALIGNED_HORIZONTAL:
                satisfied = alignment.is_aligned_horizontal
                if not satisfied:
                    print(f"DEBUG: H-Align Fail (var={alignment.horizontal_variance:.1f})")
            elif const_type == ConstraintType.ALIGNED_VERTICAL:
                satisfied = alignment.is_aligned_vertical
                if not satisfied:
                    print(f"DEBUG: V-Align Fail (var={alignment.vertical_variance:.1f})")
            elif const_type == ConstraintType.CENTERED:
                satisfied = centered
            elif const_type == ConstraintType.EQUAL_SPACING:
                satisfied = spacing.is_equal
                if not satisfied:
                    print(f"DEBUG: Spacing Fail (var={spacing.variance:.1f})")
            
            if satisfied:
                score += 1.0
            
            info[f"constraint_{i}"] = {
                "type": const_type.value,
                "satisfied": satisfied,
            }
        
        # Normalize score
        if self.required_constraints:
            score /= len(self.required_constraints)
        
        return score, info
    
    def is_success(self, obs: np.ndarray) -> bool:
        """
        Check if the goal is achieved.
        
        Args:
            obs: Current observation
            
        Returns:
            True if all success conditions are met
        """
        elements = self.detector.detect_all(obs)
        
        # Check minimum elements
        conditions = self.plan.success_conditions
        if conditions.min_elements:
            if len(elements) < conditions.min_elements:
                return False
        
        # Check element requirements
        elem_score, elem_info = self._evaluate_elements(obs, elements)
        for key, val in elem_info.items():
            if isinstance(val, dict) and not val.get("satisfied", True):
                return False
        
        # Check constraints if required
        if conditions.all_constraints_met and self.required_constraints:
            const_score, const_info = self._evaluate_constraints(obs, elements)
            for key, val in const_info.items():
                if isinstance(val, dict) and not val.get("satisfied", True):
                    return False
        
        return True
    
    def get_progress(self, obs: np.ndarray) -> float:
        """
        Get progress toward goal (0.0 to 1.0).
        
        Args:
            obs: Current observation
            
        Returns:
            Progress percentage
        """
        elements = self.detector.detect_all(obs)
        
        elem_score, _ = self._evaluate_elements(obs, elements)
        const_score, _ = self._evaluate_constraints(obs, elements)
        
        # Weighted average
        if self.required_constraints:
            return 0.6 * elem_score + 0.4 * const_score
        else:
            return elem_score


# =============================================================================
# OBJECTIVE TRANSLATOR
# =============================================================================

class ObjectiveTranslator:
    """
    Translates structured plans into reward functions.
    
    This is the main entry point for the LLM -> RL translation.
    """
    
    def __init__(
        self,
        canvas_size: tuple[int, int] = (64, 64),
        config: Optional[RewardConfig] = None
    ):
        self.canvas_size = canvas_size
        self.config = config or RewardConfig()
        self.detector = VisualDetector(canvas_size=canvas_size)
    
    def translate(self, plan: StructuredPlan) -> DynamicRewardFunction:
        """
        Translate a plan into a reward function.
        
        Args:
            plan: Structured plan from LLM
            
        Returns:
            DynamicRewardFunction ready for RL training
        """
        return DynamicRewardFunction(
            plan=plan,
            detector=self.detector,
            config=self.config
        )
    
    def get_detector(self) -> VisualDetector:
        """Get the visual detector."""
        return self.detector


if __name__ == "__main__":
    # Test reward generator
    print("Testing Reward Generator...")
    
    from planner.goal_dsl import EXAMPLE_PLANS
    
    # Create plan
    plan = StructuredPlan.from_dict(EXAMPLE_PLANS["three_aligned_rectangles"])
    print(f"Plan: {plan.goal}")
    print(f"Required elements: {len(plan.elements)}")
    print(f"Required constraints: {len(plan.constraints)}")
    
    # Create translator
    translator = ObjectiveTranslator()
    reward_fn = translator.translate(plan)
    
    # Create test observations
    empty_obs = np.ones((64, 64, 3), dtype=np.uint8) * 50
    
    # Obs with 3 blue rectangles
    obs_with_rects = empty_obs.copy()
    obs_with_rects[10:22, 10:22] = [60, 120, 220]  # Blue
    obs_with_rects[10:22, 25:37] = [60, 120, 220]  # Blue
    obs_with_rects[10:22, 40:52] = [60, 120, 220]  # Blue
    
    # Test reward computation
    reward, info = reward_fn.compute(empty_obs, 0, obs_with_rects)
    print(f"\nReward: {reward:.3f}")
    print(f"Element requirements: {info.get('element_requirements', {})}")
    print(f"Constraint satisfaction: {info.get('constraint_satisfaction', {})}")
    
    # Test success check
    success = reward_fn.is_success(obs_with_rects)
    print(f"\nSuccess: {success}")
    
    # Test progress
    progress = reward_fn.get_progress(obs_with_rects)
    print(f"Progress: {progress:.1%}")
    
    print("\nReward Generator test passed!")
