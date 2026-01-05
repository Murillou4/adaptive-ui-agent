"""
Goal DSL (Domain Specific Language) for LLM-RL Integration

Defines the constrained visual vocabulary that:
- LLM can generate
- Translator can understand
- RL can learn to achieve

Key principle: If it's not in the DSL, it can't be a goal.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
from jsonschema import validate, ValidationError


# =============================================================================
# VOCABULARY DEFINITIONS
# =============================================================================

class ElementType(str, Enum):
    """Visual elements that can be created/detected."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    BUTTON = "button"
    INPUT = "input"
    FRAME = "frame"
    TEXT = "text"
    LINE = "line"


class ConstraintType(str, Enum):
    """Spatial/visual constraints between elements."""
    ALIGNED_HORIZONTAL = "aligned_horizontal"
    ALIGNED_VERTICAL = "aligned_vertical"
    CENTERED = "centered"
    EQUAL_SPACING = "equal_spacing"
    INSIDE = "inside"
    ADJACENT = "adjacent"
    STACKED = "stacked"


class PropertyType(str, Enum):
    """Properties that can be specified for elements."""
    COLOR = "color"
    SIZE = "size"
    POSITION = "position"
    COUNT = "count"
    WIDTH = "width"
    HEIGHT = "height"


class Color(str, Enum):
    """Supported colors for detection."""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    WHITE = "white"
    BLACK = "black"
    GRAY = "gray"


# Color RGB mappings for detection
COLOR_RGB = {
    Color.RED: (220, 60, 60),
    Color.BLUE: (60, 120, 220),
    Color.GREEN: (60, 180, 60),
    Color.YELLOW: (220, 220, 60),
    Color.WHITE: (240, 240, 240),
    Color.BLACK: (20, 20, 20),
    Color.GRAY: (128, 128, 128),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ElementSpec:
    """Specification for a visual element."""
    type: ElementType
    count: int = 1
    color: Optional[Color] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    id: Optional[str] = None  # For referencing in constraints
    
    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "count": self.count,
            "color": self.color.value if self.color else None,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "id": self.id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ElementSpec":
        return cls(
            type=ElementType(data["type"]),
            count=data.get("count", 1),
            color=Color(data["color"]) if data.get("color") else None,
            min_size=data.get("min_size"),
            max_size=data.get("max_size"),
            id=data.get("id"),
        )


@dataclass
class ConstraintSpec:
    """Specification for a constraint between elements."""
    type: ConstraintType
    value: Optional[float] = None  # e.g., spacing value
    tolerance: float = 5.0  # pixel tolerance for detection
    target_ids: List[str] = field(default_factory=list)  # element IDs this applies to
    
    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "value": self.value,
            "tolerance": self.tolerance,
            "target_ids": self.target_ids,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConstraintSpec":
        return cls(
            type=ConstraintType(data["type"]),
            value=data.get("value"),
            tolerance=data.get("tolerance", 5.0),
            target_ids=data.get("target_ids", []),
        )


@dataclass
class SuccessConditions:
    """Conditions that define task completion."""
    min_elements: Optional[int] = None
    all_constraints_met: bool = True
    max_steps: int = 500
    custom_checks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "min_elements": self.min_elements,
            "all_constraints_met": self.all_constraints_met,
            "max_steps": self.max_steps,
            "custom_checks": self.custom_checks,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SuccessConditions":
        return cls(
            min_elements=data.get("min_elements"),
            all_constraints_met=data.get("all_constraints_met", True),
            max_steps=data.get("max_steps", 500),
            custom_checks=data.get("custom_checks", []),
        )


@dataclass
class StructuredPlan:
    """Complete structured plan from LLM."""
    goal: str
    elements: List[ElementSpec]
    constraints: List[ConstraintSpec] = field(default_factory=list)
    success_conditions: SuccessConditions = field(default_factory=SuccessConditions)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "elements": [e.to_dict() for e in self.elements],
            "constraints": [c.to_dict() for c in self.constraints],
            "success_conditions": self.success_conditions.to_dict(),
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "StructuredPlan":
        return cls(
            goal=data["goal"],
            elements=[ElementSpec.from_dict(e) for e in data["elements"]],
            constraints=[ConstraintSpec.from_dict(c) for c in data.get("constraints", [])],
            success_conditions=SuccessConditions.from_dict(
                data.get("success_conditions", {})
            ),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "StructuredPlan":
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# JSON SCHEMA FOR VALIDATION
# =============================================================================

PLAN_SCHEMA = {
    "type": "object",
    "required": ["goal", "elements"],
    "properties": {
        "goal": {"type": "string", "minLength": 1},
        "elements": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"enum": [e.value for e in ElementType]},
                    "count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "color": {"enum": [c.value for c in Color] + [None]},
                    "min_size": {"type": "integer", "minimum": 1},
                    "max_size": {"type": "integer", "minimum": 1},
                    "id": {"type": "string"},
                }
            }
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"enum": [c.value for c in ConstraintType]},
                    "value": {"type": "number"},
                    "tolerance": {"type": "number", "minimum": 0},
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                }
            }
        },
        "success_conditions": {
            "type": "object",
            "properties": {
                "min_elements": {"type": "integer", "minimum": 0},
                "all_constraints_met": {"type": "boolean"},
                "max_steps": {"type": "integer", "minimum": 1},
            }
        },
        "metadata": {"type": "object"}
    }
}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_plan(plan: Union[dict, StructuredPlan]) -> tuple[bool, Optional[str]]:
    """
    Validate a plan against the DSL schema.
    
    Args:
        plan: Plan dictionary or StructuredPlan object
        
    Returns:
        (is_valid, error_message)
    """
    if isinstance(plan, StructuredPlan):
        plan = plan.to_dict()
    
    try:
        validate(instance=plan, schema=PLAN_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def sanitize_plan(plan: dict) -> dict:
    """
    Sanitize a plan by removing invalid fields and setting defaults.
    
    Args:
        plan: Raw plan dictionary
        
    Returns:
        Sanitized plan dictionary
    """
    sanitized = {
        "goal": plan.get("goal", "unknown"),
        "elements": [],
        "constraints": [],
        "success_conditions": {
            "all_constraints_met": True,
            "max_steps": 500,
        },
        "metadata": {}
    }
    
    # Sanitize elements
    for elem in plan.get("elements", []):
        if elem.get("type") in [e.value for e in ElementType]:
            sanitized["elements"].append({
                "type": elem["type"],
                "count": min(max(elem.get("count", 1), 1), 20),
                "color": elem.get("color") if elem.get("color") in [c.value for c in Color] else None,
            })
    
    # Sanitize constraints
    for const in plan.get("constraints", []):
        if const.get("type") in [c.value for c in ConstraintType]:
            sanitized["constraints"].append({
                "type": const["type"],
                "value": const.get("value"),
                "tolerance": const.get("tolerance", 5.0),
            })
    
    return sanitized


# =============================================================================
# EXAMPLES
# =============================================================================

EXAMPLE_PLANS = {
    "three_aligned_rectangles": {
        "goal": "create_aligned_layout",
        "elements": [
            {"type": "rectangle", "count": 3, "color": "blue"}
        ],
        "constraints": [
            {"type": "aligned_horizontal"},
            {"type": "equal_spacing", "value": 10}
        ],
        "success_conditions": {
            "min_elements": 3,
            "all_constraints_met": True
        }
    },
    "simple_form": {
        "goal": "create_form",
        "elements": [
            {"type": "input", "count": 2, "id": "inputs"},
            {"type": "button", "count": 1, "color": "blue", "id": "submit"}
        ],
        "constraints": [
            {"type": "aligned_vertical", "target_ids": ["inputs"]},
            {"type": "stacked"}
        ],
        "success_conditions": {
            "min_elements": 3
        }
    },
    "centered_button": {
        "goal": "create_centered_button",
        "elements": [
            {"type": "button", "count": 1, "color": "green"}
        ],
        "constraints": [
            {"type": "centered"}
        ]
    }
}


if __name__ == "__main__":
    # Test DSL
    print("Testing Goal DSL...")
    
    # Create a plan programmatically
    plan = StructuredPlan(
        goal="test_layout",
        elements=[
            ElementSpec(type=ElementType.RECTANGLE, count=3, color=Color.BLUE),
        ],
        constraints=[
            ConstraintSpec(type=ConstraintType.ALIGNED_HORIZONTAL),
        ],
    )
    
    print(f"Plan JSON:\n{plan.to_json()}")
    
    # Validate
    is_valid, error = validate_plan(plan)
    print(f"Valid: {is_valid}, Error: {error}")
    
    # Test from dict
    plan2 = StructuredPlan.from_dict(EXAMPLE_PLANS["three_aligned_rectangles"])
    print(f"\nLoaded plan: {plan2.goal}")
    print(f"Elements: {len(plan2.elements)}")
    print(f"Constraints: {len(plan2.constraints)}")
    
    print("\nGoal DSL test passed!")
