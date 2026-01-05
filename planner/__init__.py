"""Planner package for LLM-RL integration."""

from planner.goal_dsl import (
    StructuredPlan,
    ElementSpec,
    ConstraintSpec,
    SuccessConditions,
    ElementType,
    ConstraintType,
    Color,
    validate_plan,
    sanitize_plan,
)
from planner.visual_detectors import (
    VisualDetector,
    DetectedElement,
    BoundingBox,
    detect_rectangles,
    detect_alignment,
    detect_spacing,
)
from planner.reward_generator import (
    ObjectiveTranslator,
    DynamicRewardFunction,
    RewardConfig,
)
from planner.llm_planner import (
    LLMPlanner,
    create_planner,
    PlanResult,
)


__all__ = [
    # Goal DSL
    'StructuredPlan', 'ElementSpec', 'ConstraintSpec', 'SuccessConditions',
    'ElementType', 'ConstraintType', 'Color', 'validate_plan', 'sanitize_plan',
    # Visual Detectors
    'VisualDetector', 'DetectedElement', 'BoundingBox',
    'detect_rectangles', 'detect_alignment', 'detect_spacing',
    # Reward Generator
    'ObjectiveTranslator', 'DynamicRewardFunction', 'RewardConfig',
    # LLM Planner
    'LLMPlanner', 'create_planner', 'PlanResult',
    # Integration
    # 'LLMRLIntegration', 'IntegrationResult',
]
