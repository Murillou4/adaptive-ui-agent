"""
Skill Library Foundation

Defines the base interface for all skills (reusable policies).
A Skill is a self-contained ability that:
1. Checks preconditions (can I do this?)
2. Executes actions (do it)
3. Checks effects (did it work?)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import logging

# Circular imports handling
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env.universal_env import UniversalEnv
    from vision.state_extractor import VisualState

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result of a skill execution."""
    success: bool
    message: str
    frames_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """
    Abstract base class for all skills.
    """
    name: str = "base_skill"
    description: str = "Base skill"
    
    def __init__(self, **kwargs):
        self.params = kwargs
        
    @abstractmethod
    def is_applicable(self, state: "VisualState") -> bool:
        """
        Check if the skill can be applied in the current state.
        For example: "Click button" requires a button to be visible.
        """
        pass
        
    @abstractmethod
    def execute(self, env: "UniversalEnv") -> SkillResult:
        """
        Execute the skill in the environment.
        This is a blocking call that may run for multiple steps.
        """
        pass
    
    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize skill."""
        return {
            "name": self.name,
            "version": getattr(self, "version", "1.0.0"),
            "params": self.params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Deserialize skill."""
        # Typically handled by registry, but good for direct loading
        if "params" in data:
            return cls(**data["params"])
        return cls()

# =============================================================================
# SKILL REGISTRY
# =============================================================================

SKILL_REGISTRY = {}

def register_skill(cls):
    """Decorator to register a skill class."""
    SKILL_REGISTRY[cls.name] = cls
    return cls

def create_skill(name: str, **params) -> Optional[Skill]:
    """Factory to create a skill by name."""
    if name not in SKILL_REGISTRY:
        logger.error(f"Unknown skill: {name}")
        return None
        
    return SKILL_REGISTRY[name](**params)

def save_skill_library(path: str):
    """Save all registered skills metadata to disk."""
    import json
    data = {}
    for name, cls in SKILL_REGISTRY.items():
        data[name] = {
            "description": cls.description,
            "version": getattr(cls, "version", "1.0.0")
        }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_skill_config(path: str):
    """Load skill configuration/overrides."""
    # Placeholder for loading custom parameters or overrides
    pass
