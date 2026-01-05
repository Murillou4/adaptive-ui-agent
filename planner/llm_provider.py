"""
Universal LLM Provider using LiteLLM

Supports 100+ LLM providers through a unified interface:
- OpenAI (GPT-5.2)
- Anthropic (Claude 4.5)
- Google (Gemini 2.5/3)
- xAI (Grok 4)
- Meta (Llama 4 via Ollama)
- Alibaba (Qwen 3)
- DeepSeek, Groq, Together, and more
"""

import os
import yaml
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path


def load_llm_config(config_path: Optional[str] = None) -> dict:
    """Load LLM configuration from YAML file."""
    if config_path is None:
        # Try common locations
        locations = [
            "configs/llm_config.yaml",
            "llm_config.yaml",
            os.path.expanduser("~/.adaptive-ui-agent/llm_config.yaml"),
        ]
        for loc in locations:
            if os.path.exists(loc):
                config_path = loc
                break
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables
        config = _expand_env_vars(config)
        return config
    
    return {}


def _expand_env_vars(obj):
    """Recursively expand ${VAR} patterns in config."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


class UniversalLLMProvider:
    """
    Universal LLM provider using LiteLLM.
    
    Supports all major LLM providers through a single interface.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the universal LLM provider.
        
        Args:
            config_path: Path to YAML config file
            provider: Override default provider (openai, anthropic, google, etc.)
            **kwargs: Override specific settings
        """
        self.config = load_llm_config(config_path)
        self.provider = provider or self.config.get("default_provider", "openai")
        self.overrides = kwargs
        
        # Import litellm
        try:
            import litellm
            self.litellm = litellm
            
            # Set API keys from config
            self._configure_api_keys()
            
        except ImportError:
            print("⚠️ LiteLLM not installed. Install with: pip install litellm")
            self.litellm = None
    
    def _configure_api_keys(self):
        """Configure API keys for all providers."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
        }
        
        for provider, env_var in key_mapping.items():
            if provider in self.config:
                api_key = self.config[provider].get("api_key", "")
                if api_key and not os.environ.get(env_var):
                    os.environ[env_var] = api_key
    
    def _get_model_string(self) -> str:
        """Get the LiteLLM model string for current provider."""
        provider_config = self.config.get(self.provider, {})
        model = self.overrides.get("model") or provider_config.get("model", "")
        
        # LiteLLM model format mapping
        model_prefixes = {
            "openai": "",  # OpenAI is default, no prefix needed
            "anthropic": "anthropic/",
            "google": "gemini/",
            "xai": "xai/",
            "ollama": "ollama/",
            "litellm": "",  # Already in correct format
        }
        
        prefix = model_prefixes.get(self.provider, "")
        
        # If already has provider prefix, don't add
        if "/" in model:
            return model
        
        return f"{prefix}{model}"
    
    def _get_params(self) -> dict:
        """Get parameters for the API call."""
        provider_config = self.config.get(self.provider, {})
        
        params = {
            "temperature": self.overrides.get("temperature") or provider_config.get("temperature", 0.2),
            "max_tokens": self.overrides.get("max_tokens") or provider_config.get("max_tokens", 2000),
        }
        
        # Add base_url if specified
        base_url = self.overrides.get("base_url") or provider_config.get("base_url")
        if base_url:
            params["api_base"] = base_url
        
        return params
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        if self.litellm is None:
            raise ImportError("LiteLLM not installed")
        
        model = self._get_model_string()
        params = self._get_params()
        params.update(kwargs)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.litellm.completion(
                model=model,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
            
        except Exception as e:
            # Try fallback if enabled
            if self._should_fallback():
                return self._try_fallback(messages, params, str(e))
            raise
    
    def _should_fallback(self) -> bool:
        """Check if fallback is enabled."""
        fallback = self.config.get("fallback", {})
        return fallback.get("enabled", False)
    
    def _try_fallback(
        self,
        messages: list,
        params: dict,
        original_error: str
    ) -> str:
        """Try fallback providers."""
        fallback = self.config.get("fallback", {})
        providers = fallback.get("providers", [])
        max_retries = fallback.get("max_retries", 3)
        
        for provider in providers:
            if provider == self.provider:
                continue
            
            for attempt in range(max_retries):
                try:
                    self.provider = provider
                    model = self._get_model_string()
                    
                    response = self.litellm.completion(
                        model=model,
                        messages=messages,
                        **params
                    )
                    return response.choices[0].message.content
                    
                except Exception:
                    continue
        
        raise Exception(f"All providers failed. Original error: {original_error}")
    
    def list_available_models(self) -> list:
        """List available models for current provider."""
        provider_models = {
            "openai": [
                "gpt-5.2-instant", "gpt-5.2-thinking", "gpt-5.2-pro",
                "gpt-4o", "gpt-4o-mini", "o1", "o1-mini"
            ],
            "anthropic": [
                "claude-4.5-opus", "claude-4.5-sonnet", "claude-4.5-haiku",
                "claude-3.5-sonnet", "claude-3-opus"
            ],
            "google": [
                "gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro",
                "gemini-2.0-flash-exp"
            ],
            "xai": [
                "grok-4", "grok-4.1", "grok-3", "grok-3-mini"
            ],
            "ollama": [
                "llama4:scout", "llama4:maverick", "qwen3", "gemma3",
                "deepseek-r1", "mistral-large"
            ],
        }
        
        return provider_models.get(self.provider, [])


class LiteLLMPlannerProvider:
    """
    LLM Provider for the planner using LiteLLM.
    
    Drop-in replacement for the existing planner providers.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.llm = UniversalLLMProvider(
            config_path=config_path,
            provider=provider,
            model=model
        )
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response for planner."""
        return self.llm.generate(prompt, system_prompt)


def create_llm_provider(
    provider: str = "auto",
    config_path: Optional[str] = None,
    **kwargs
) -> LiteLLMPlannerProvider:
    """
    Factory to create an LLM provider.
    
    Args:
        provider: Provider name or "auto" to use config default
        config_path: Path to config file
        **kwargs: Additional settings
        
    Returns:
        Configured LLM provider
    """
    if provider == "auto":
        provider = None
    
    return LiteLLMPlannerProvider(
        config_path=config_path,
        provider=provider,
        **kwargs
    )


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Universal LLM Provider...")
    
    # Check if litellm is installed
    try:
        import litellm
        print("✅ LiteLLM installed")
    except ImportError:
        print("❌ LiteLLM not installed. Run: pip install litellm")
        exit(1)
    
    # Create provider
    provider = UniversalLLMProvider(provider="openai")
    
    print(f"\nProvider: {provider.provider}")
    print(f"Model: {provider._get_model_string()}")
    print(f"Available models: {provider.list_available_models()}")
    
    # Try a test call (will fail without API key, but shows setup works)
    print("\nTo test with a real API:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python planner/llm_provider.py")
