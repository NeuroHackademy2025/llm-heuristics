"""Configuration management utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field


class ModelConfig(BaseModel):
    """Configuration for LLM model settings."""

    name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    use_quantization: bool = True
    max_length: int = 8192
    temperature: float = 0.7
    device: str | None = None


class HeuristicConfig(BaseModel):
    """Configuration for heuristic generation settings."""

    include_session: bool = True
    use_run_numbers: bool = True
    exclude_motion_corrected: bool = True
    default_task_name: str = "rest"


class Config(BaseModel):
    """Main configuration class."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    heuristic: HeuristicConfig = Field(default_factory=HeuristicConfig)
    log_level: str = "INFO"
    cache_dir: Path | None = None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @classmethod
    def from_file(cls, config_file: Path) -> Config:
        """
        Load configuration from file.

        Parameters
        ----------
        config_file : Path
            Path to configuration file

        Returns
        -------
        Config
            Configuration instance
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file) as f:
            config_data = json.load(f)

        return cls(**config_data)

    @classmethod
    def from_env(cls) -> Config:
        """
        Load configuration from environment variables.

        Returns
        -------
        Config
            Configuration instance
        """
        config_data = {}

        # Model configuration
        model_config = {}
        if "LLM_MODEL_NAME" in os.environ:
            model_config["name"] = os.environ["LLM_MODEL_NAME"]
        if "LLM_USE_QUANTIZATION" in os.environ:
            model_config["use_quantization"] = os.environ["LLM_USE_QUANTIZATION"].lower() == "true"
        if "LLM_MAX_LENGTH" in os.environ:
            model_config["max_length"] = int(os.environ["LLM_MAX_LENGTH"])
        if "LLM_TEMPERATURE" in os.environ:
            model_config["temperature"] = float(os.environ["LLM_TEMPERATURE"])
        if "LLM_DEVICE" in os.environ:
            model_config["device"] = os.environ["LLM_DEVICE"]

        if model_config:
            config_data["model"] = model_config

        # Heuristic configuration
        heuristic_config = {}
        if "HEURISTIC_INCLUDE_SESSION" in os.environ:
            heuristic_config["include_session"] = (
                os.environ["HEURISTIC_INCLUDE_SESSION"].lower() == "true"
            )
        if "HEURISTIC_USE_RUN_NUMBERS" in os.environ:
            heuristic_config["use_run_numbers"] = (
                os.environ["HEURISTIC_USE_RUN_NUMBERS"].lower() == "true"
            )
        if "HEURISTIC_EXCLUDE_MOTION_CORRECTED" in os.environ:
            heuristic_config["exclude_motion_corrected"] = (
                os.environ["HEURISTIC_EXCLUDE_MOTION_CORRECTED"].lower() == "true"
            )
        if "HEURISTIC_DEFAULT_TASK_NAME" in os.environ:
            heuristic_config["default_task_name"] = os.environ["HEURISTIC_DEFAULT_TASK_NAME"]

        if heuristic_config:
            config_data["heuristic"] = heuristic_config

        # General configuration
        if "LOG_LEVEL" in os.environ:
            config_data["log_level"] = os.environ["LOG_LEVEL"]
        if "CACHE_DIR" in os.environ:
            config_data["cache_dir"] = Path(os.environ["CACHE_DIR"])

        return cls(**config_data)

    def to_file(self, config_file: Path) -> None:
        """
        Save configuration to file.

        Parameters
        ----------
        config_file : Path
            Path to save configuration file
        """
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable dict
        config_dict = self.dict()

        # Handle Path objects
        if config_dict.get("cache_dir"):
            config_dict["cache_dir"] = str(config_dict["cache_dir"])

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    def get_cache_dir(self) -> Path:
        """
        Get the cache directory.

        Returns
        -------
        Path
            Cache directory path
        """
        if self.cache_dir:
            return self.cache_dir

        # Default cache directory
        cache_dir = Path.home() / ".cache" / "llm-heuristics"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns
    -------
    Config
        Default configuration instance
    """
    return Config()


def load_config(config_file: Path | None = None) -> Config:
    """
    Load configuration from file or environment.

    Parameters
    ----------
    config_file : Path | None
        Optional configuration file path

    Returns
    -------
    Config
        Configuration instance
    """
    if config_file and config_file.exists():
        return Config.from_file(config_file)

    # Try to load from environment
    try:
        return Config.from_env()
    except Exception:
        # Fall back to defaults
        return get_default_config()
