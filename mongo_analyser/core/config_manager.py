import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR_NAME = "mongo_analyser"
DEFAULT_CONFIG_FILE_NAME = "config.json"


DEFAULT_THEME_NAME = "textual-dark"
VALID_THEMES = [
    "textual-dark",
    "textual-light",
    "nord",
    "gruvbox",
    "catppuccin-mocha",
    "dracula",
    "tokyo-night",
    "monokai",
    "flexoki",
    "catppuccin-latte",
    "solarized-light",
]

DEFAULT_SETTINGS = {
    "theme": DEFAULT_THEME_NAME,
    "default_log_level": "INFO",
    "schema_analysis_default_sample_size": 1000,
    "data_explorer_default_sample_size": 10,
    "llm_default_provider": "ollama",
    "llm_default_model_ollama": "gemma3:4b",
    "llm_default_model_openai": "gpt-4.1-nano",
    "llm_default_model_google": "models/gemini-2.0-flash-lite",
    "llm_default_temperature": 0.7,
    "llm_default_max_history": 20,
}


class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        self._config_path: Path = config_path or self._get_default_config_path()
        self._config: Dict[str, Any] = {}
        self.load_config()

    def _get_default_config_path(self) -> Path:
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            base_path = Path(xdg_data_home)
        else:
            base_path = Path.home() / ".local" / "share"

        config_dir = base_path / DEFAULT_CONFIG_DIR_NAME
        return config_dir / DEFAULT_CONFIG_FILE_NAME

    def load_config(self) -> None:
        loaded_settings = {}
        if self._config_path.exists() and self._config_path.is_file():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    loaded_settings = json.load(f)
                logger.info(f"Configuration loaded from {self._config_path}")
            except (IOError, json.JSONDecodeError) as e:
                logger.error(
                    f"Error loading configuration from {self._config_path}: {e}. Using defaults."
                )
        else:
            logger.info(
                f"Configuration file not found at {self._config_path}. Using defaults and will create on save."
            )

        self._config = DEFAULT_SETTINGS.copy()
        self._config.update(loaded_settings)

        if self._config.get("theme") not in VALID_THEMES:
            logger.warning(
                f"Loaded theme '{self._config.get('theme')}' is not in VALID_THEMES. Resetting to default '{DEFAULT_THEME_NAME}'."
            )
            self._config["theme"] = DEFAULT_THEME_NAME

    def save_config(self) -> bool:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {self._config_path}")
            return True
        except IOError as e:
            logger.error(f"Error saving configuration to {self._config_path}: {e}")
            return False

    def get_setting(self, key: str, default: Optional[Any] = None) -> Any:
        if key == "theme":
            theme_value = self._config.get(key)
            if theme_value not in VALID_THEMES:
                return default if default is not None else DEFAULT_THEME_NAME
            return theme_value

        return self._config.get(key, default if default is not None else DEFAULT_SETTINGS.get(key))

    def update_setting(self, key: str, value: Any) -> None:
        if key == "theme" and value not in VALID_THEMES:
            logger.warning(
                f"Attempted to set invalid theme '{value}'. Using default '{DEFAULT_THEME_NAME}' instead."
            )
            self._config[key] = DEFAULT_THEME_NAME
        else:
            self._config[key] = value

    def get_all_settings(self) -> Dict[str, Any]:
        current_theme = self._config.get("theme")
        if current_theme not in VALID_THEMES:
            self._config["theme"] = DEFAULT_THEME_NAME
        return self._config.copy()

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        if "theme" in new_settings and new_settings["theme"] not in VALID_THEMES:
            logger.warning(
                f"Invalid theme '{new_settings['theme']}' in update_settings. Using default '{DEFAULT_THEME_NAME}'."
            )
            new_settings["theme"] = DEFAULT_THEME_NAME
        self._config.update(new_settings)
