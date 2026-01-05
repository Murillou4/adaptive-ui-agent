"""
LLM Planner - Decomposes User Intent into Structured Plans

The LLM acts as the "symbolic brain":
- Interprets natural language goals
- Decomposes into observable tasks
- Generates structured plans using DSL vocabulary

Key principle: LLM NEVER touches the mouse. Only generates plans.
"""

import json
import os
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    PLAN_SCHEMA,
)


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a visual task planner for a UI creation agent.

Your job is to translate user requests into structured plans that an RL agent can execute.

CRITICAL RULES:
1. You can ONLY use the provided vocabulary (elements, constraints, colors)
2. You output ONLY valid JSON - no explanations, no markdown
3. Break complex tasks into simple, measurable sub-goals
4. If something can't be measured visually, you can't plan for it

AVAILABLE VOCABULARY:

Elements (what can be created):
- rectangle: A rectangular shape
- circle: A circular shape  
- button: A clickable button element
- input: A text input field
- frame: A container frame
- text: Text element
- line: A line element

Colors (for elements):
- red, blue, green, yellow, white, black, gray

Constraints (spatial relationships):
- aligned_horizontal: Elements are in the same row
- aligned_vertical: Elements are in the same column
- centered: Elements are centered in the canvas
- equal_spacing: Elements have equal gaps between them
- inside: Element is contained within another
- adjacent: Elements are next to each other
- stacked: Elements are stacked vertically

OUTPUT FORMAT (JSON only):
{
  "goal": "brief description",
  "elements": [
    {"type": "element_type", "count": N, "color": "color_name"}
  ],
  "constraints": [
    {"type": "constraint_type", "value": optional_number}
  ],
  "success_conditions": {
    "min_elements": N,
    "all_constraints_met": true
  }
}

Remember: If you can't describe it with this vocabulary, simplify it until you can."""


DECOMPOSITION_PROMPT = """Given this user request, create a structured plan.

User request: "{user_request}"

Think about:
1. What visual elements need to exist?
2. How should they be arranged?
3. What colors matter?
4. What defines success?

Output ONLY the JSON plan, nothing else."""


# =============================================================================
# LLM PROVIDER INTERFACE
# =============================================================================

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response from prompt."""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM for testing without API calls."""
    
    def __init__(self):
        # Pre-defined responses for common requests
        self.responses = {
            "3 quadrados": {
                "goal": "create_three_squares",
                "elements": [{"type": "rectangle", "count": 3, "color": "blue"}],
                "constraints": [{"type": "aligned_horizontal"}],
                "success_conditions": {"min_elements": 3}
            },
            "botão": {
                "goal": "create_button",
                "elements": [{"type": "button", "count": 1, "color": "blue"}],
                "constraints": [{"type": "centered"}],
                "success_conditions": {"min_elements": 1}
            },
            "formulário": {
                "goal": "create_form",
                "elements": [
                    {"type": "input", "count": 2},
                    {"type": "button", "count": 1, "color": "green"}
                ],
                "constraints": [{"type": "aligned_vertical"}, {"type": "stacked"}],
                "success_conditions": {"min_elements": 3}
            },
            "alinhado": {
                "goal": "create_aligned_layout",
                "elements": [{"type": "rectangle", "count": 3}],
                "constraints": [{"type": "aligned_horizontal"}, {"type": "equal_spacing"}],
                "success_conditions": {"min_elements": 3, "all_constraints_met": True}
            }
        }
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate mock response based on keywords."""
        prompt_lower = prompt.lower()
        
        for keyword, response in self.responses.items():
            if keyword in prompt_lower:
                return json.dumps(response)
        
        # Default response
        return json.dumps({
            "goal": "create_elements",
            "elements": [{"type": "rectangle", "count": 2, "color": "blue"}],
            "constraints": [],
            "success_conditions": {"min_elements": 2}
        })


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response using Ollama."""
        try:
            import requests
            
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.2}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"Ollama error: {response.status_code}")
                
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")


# =============================================================================
# LLM PLANNER
# =============================================================================

@dataclass
class PlanResult:
    """Result of planning operation."""
    success: bool
    plan: Optional[StructuredPlan]
    raw_response: str
    error: Optional[str] = None


class LLMPlanner:
    """
    High-level planner that uses LLM to decompose goals.
    
    Flow:
    1. User provides natural language request
    2. LLM generates structured plan using DSL
    3. Plan is validated
    4. Ready for ObjectiveTranslator
    """
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        auto_sanitize: bool = True
    ):
        """
        Initialize planner.
        
        Args:
            provider: LLM provider (defaults to MockLLMProvider)
            auto_sanitize: Whether to auto-fix invalid plans
        """
        self.provider = provider or MockLLMProvider()
        self.auto_sanitize = auto_sanitize
        self.last_raw_response = ""
    
    def plan(self, user_request: str) -> PlanResult:
        """
        Generate a structured plan from user request.
        
        Args:
            user_request: Natural language goal description
            
        Returns:
            PlanResult with plan or error
        """
        # Build prompt
        prompt = DECOMPOSITION_PROMPT.format(user_request=user_request)
        
        # Get LLM response
        try:
            raw_response = self.provider.generate(prompt, SYSTEM_PROMPT)
            self.last_raw_response = raw_response
        except Exception as e:
            return PlanResult(
                success=False,
                plan=None,
                raw_response="",
                error=f"LLM error: {str(e)}"
            )
        
        # Parse JSON
        try:
            # Try to extract JSON from response
            plan_dict = self._extract_json(raw_response)
        except json.JSONDecodeError as e:
            return PlanResult(
                success=False,
                plan=None,
                raw_response=raw_response,
                error=f"JSON parse error: {str(e)}"
            )
        
        # Validate
        is_valid, error = validate_plan(plan_dict)
        
        if not is_valid:
            if self.auto_sanitize:
                # Try to fix the plan
                plan_dict = sanitize_plan(plan_dict)
                is_valid, error = validate_plan(plan_dict)
            
            if not is_valid:
                return PlanResult(
                    success=False,
                    plan=None,
                    raw_response=raw_response,
                    error=f"Validation error: {error}"
                )
        
        # Create StructuredPlan
        try:
            plan = StructuredPlan.from_dict(plan_dict)
        except Exception as e:
            return PlanResult(
                success=False,
                plan=None,
                raw_response=raw_response,
                error=f"Plan creation error: {str(e)}"
            )
        
        return PlanResult(
            success=True,
            plan=plan,
            raw_response=raw_response
        )
    
    def _extract_json(self, response: str) -> dict:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        response = response.strip()
        
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return json.loads(response[start:end].strip())
        
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return json.loads(response[start:end].strip())
        
        # Try to find JSON object directly
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        
        raise json.JSONDecodeError("No JSON found", response, 0)
    
    def adjust_plan(
        self,
        plan: StructuredPlan,
        feedback: Dict[str, Any]
    ) -> PlanResult:
        """
        Adjust a plan based on execution feedback.
        
        Args:
            plan: Current plan
            feedback: Feedback from RL execution (what failed, progress, etc.)
            
        Returns:
            Adjusted plan
        """
        # Build adjustment prompt
        adjustment_prompt = f"""The following plan was partially executed but needs adjustment.

Current plan:
{plan.to_json()}

Execution feedback:
{json.dumps(feedback, indent=2)}

Please provide an adjusted plan that addresses the issues.
Output ONLY the adjusted JSON plan."""

        return self.plan(adjustment_prompt)
    
    def validate_and_fix(self, plan_dict: dict) -> tuple[bool, dict, Optional[str]]:
        """
        Validate a plan and attempt to fix if invalid.
        
        Args:
            plan_dict: Plan dictionary to validate
            
        Returns:
            (is_valid, fixed_plan, error_message)
        """
        is_valid, error = validate_plan(plan_dict)
        
        if is_valid:
            return True, plan_dict, None
        
        # Try to sanitize
        fixed = sanitize_plan(plan_dict)
        is_valid, error = validate_plan(fixed)
        
        if is_valid:
            return True, fixed, None
        
        return False, plan_dict, error


def create_planner(
    provider_type: str = "mock",
    **kwargs
) -> LLMPlanner:
    """
    Factory function to create a planner.
    
    Args:
        provider_type: "mock", "openai", or "ollama"
        **kwargs: Provider-specific arguments
        
    Returns:
        Configured LLMPlanner
    """
    if provider_type == "mock":
        provider = MockLLMProvider()
    elif provider_type == "openai":
        provider = OpenAIProvider(**kwargs)
    elif provider_type == "ollama":
        provider = OllamaProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return LLMPlanner(provider=provider)


if __name__ == "__main__":
    # Test LLM Planner
    print("Testing LLM Planner...")
    
    # Create planner with mock provider
    planner = create_planner("mock")
    
    # Test various requests
    test_requests = [
        "Cria 3 quadrados azuis alinhados",
        "Cria um botão centralizado",
        "Cria um formulário com 2 inputs e um botão",
    ]
    
    for request in test_requests:
        print(f"\n{'='*50}")
        print(f"Request: {request}")
        
        result = planner.plan(request)
        
        if result.success:
            print(f"✅ Success!")
            print(f"Goal: {result.plan.goal}")
            print(f"Elements: {len(result.plan.elements)}")
            print(f"Constraints: {len(result.plan.constraints)}")
            print(f"JSON:\n{result.plan.to_json()}")
        else:
            print(f"❌ Failed: {result.error}")
    
    print("\n" + "="*50)
    print("LLM Planner test passed!")
