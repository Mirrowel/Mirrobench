"""
Configuration loader for the benchmark system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class ConfigLoader:
    """Load and validate configuration from config.yaml."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file. See config.example.yaml for reference."
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        # Validate required fields
        self._validate()

    def _validate(self):
        """Validate configuration."""
        if not self.config.get("models"):
            raise ValueError(
                "Configuration must include at least one model in 'models' list"
            )

        if not self.config.get("judge_model"):
            raise ValueError("Configuration must include a 'judge_model'")

    @property
    def models(self) -> List[str]:
        """Get list of models to benchmark."""
        return self.config.get("models", [])

    @property
    def judge_model(self) -> str:
        """Get judge model."""
        return self.config.get("judge_model", "anthropic/claude-3-5-sonnet-20241022")

    @property
    def categories(self) -> Optional[List[str]]:
        """Get categories to test (None means all)."""
        cats = self.config.get("categories", [])
        return cats if cats else None

    @property
    def question_ids(self) -> Optional[List[str]]:
        """Get specific question IDs to test (None means all)."""
        ids = self.config.get("question_ids", [])
        return ids if ids else None

    @property
    def max_concurrent(self) -> int:
        """Get max concurrent requests (global default)."""
        return self.config.get("max_concurrent", 3)

    @property
    def provider_concurrency(self) -> Dict[str, int]:
        """Get per-provider concurrency limits."""
        provider_limits = self.config.get("provider_concurrency", {})
        return provider_limits if provider_limits is not None else {}

    def get_provider_concurrency(self, provider: str) -> int:
        """Get concurrency limit for a specific provider.

        Falls back to global max_concurrent if provider not specified.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'gemini')

        Returns:
            Concurrency limit for this provider
        """
        provider_limits = self.provider_concurrency
        return provider_limits.get(provider, self.max_concurrent)

    @property
    def questions_dir(self) -> str:
        """Get questions directory."""
        return self.config.get("questions_dir", "questions")

    @property
    def results_dir(self) -> str:
        """Get results directory."""
        return self.config.get("results_dir", "results")

    @property
    def pass_threshold(self) -> float:
        """Get passing score threshold."""
        return self.config.get("evaluation", {}).get("pass_threshold", 70.0)

    @property
    def code_timeout(self) -> int:
        """Get code execution timeout."""
        return self.config.get("evaluation", {}).get("code_timeout", 10)

    @property
    def viewer_host(self) -> str:
        """Get viewer host."""
        return self.config.get("viewer", {}).get("host", "0.0.0.0")

    @property
    def viewer_port(self) -> int:
        """Get viewer port."""
        return self.config.get("viewer", {}).get("port", 8000)

    @property
    def fixer_model(self) -> str:
        """Get fixer model for reformatting responses."""
        return self.config.get("fixer_model", "anthropic/claude-3-5-sonnet-20241022")

    @property
    def model_system_instructions(self) -> Dict[str, str]:
        """Get per-model system instructions (legacy)."""
        instructions = self.config.get("model_system_instructions", {})
        return instructions if instructions is not None else {}

    @property
    def model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get per-model configurations (instructions and options)."""
        return self.config.get("model_configs", {})

    @property
    def model_display_names(self) -> Dict[str, str]:
        """Get friendly display names for models.

        Returns a dictionary mapping model identifiers to friendly display names.
        If a model is not in the mapping, the full model identifier should be used.
        """
        return self.config.get("model_display_names", {})

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get full config for a specific model."""
        return self.model_configs.get(model_name, {})

    def get_model_system_instruction(self, model_name: str) -> Optional[str]:
        """Get system instruction for a specific model.

        Checks model_configs first, then falls back to model_system_instructions.
        This ensures backward compatibility while preferring the new unified config.
        """
        # Check new model_configs first (takes precedence)
        model_config = self.get_model_config(model_name)
        if "system_instruction" in model_config:
            return model_config["system_instruction"]

        # Fall back to legacy model_system_instructions
        return self.model_system_instructions.get(model_name)

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """Get additional API body fields (options) for a specific model.

        These options will be merged into the API request body when calling this model.
        Example: {"reasoning_effort": "high"} for OpenAI o1 models.
        """
        model_config = self.get_model_config(model_name)
        return model_config.get("options", {})

    def get_model_system_instruction_position(self, model_name: str) -> str:
        """Get system instruction position for a specific model.

        Returns "prepend" or "append" (default) to control where the model-specific
        system instruction is placed relative to other system content.
        """
        model_config = self.get_model_config(model_name)
        return model_config.get("system_instruction_position", "append")

    @property
    def all_model_system_instructions(self) -> Dict[str, str]:
        """Get all model system instructions (merged from both sources).

        Merges legacy model_system_instructions with model_configs.
        model_configs takes precedence if both specify an instruction for the same model.
        """
        instructions = {}

        # Start with legacy model_system_instructions (handle None case)
        legacy_instructions = self.model_system_instructions
        if legacy_instructions:
            instructions.update(legacy_instructions)

        # Override with model_configs (takes precedence)
        for model_name, config in self.model_configs.items():
            if "system_instruction" in config:
                instructions[model_name] = config["system_instruction"]

        return instructions

    @property
    def all_model_options(self) -> Dict[str, Dict[str, Any]]:
        """Get all model options from model_configs."""
        options = {}
        for model_name, config in self.model_configs.items():
            if "options" in config:
                options[model_name] = config["options"]
        return options

    @property
    def all_model_system_instruction_positions(self) -> Dict[str, str]:
        """Get all model system instruction positions from model_configs."""
        positions = {}
        for model_name, config in self.model_configs.items():
            positions[model_name] = config.get("system_instruction_position", "append")
        return positions

    @property
    def code_formatting_enabled(self) -> bool:
        """Check if code formatting instructions are enabled."""
        return self.config.get("code_formatting_instructions", {}).get("enabled", True)

    @property
    def code_formatting_instruction(self) -> str:
        """Get code formatting instruction text."""
        default_instruction = (
            "When providing code, use markdown code blocks with language tags. "
            "For multi-file apps, you may use ```language:filename format."
        )
        return self.config.get("code_formatting_instructions", {}).get(
            "instruction", default_instruction
        )

    @property
    def max_retries_per_key(self) -> int:
        """Get maximum retry attempts per API key."""
        return self.config.get("retry_settings", {}).get("max_retries_per_key", 2)

    @property
    def global_timeout(self) -> int:
        """Get global timeout for API requests (seconds)."""
        return self.config.get("retry_settings", {}).get("global_timeout", 45)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
