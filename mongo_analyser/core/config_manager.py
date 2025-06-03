import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

APP_DIR_NAME = "mongo_analyser"
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
    "default_log_level": "OFF",
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
    def __init__(self, base_app_data_dir_override: Optional[Path] = None):
        if base_app_data_dir_override:
            self._base_app_data_dir = base_app_data_dir_override.expanduser().resolve()
        else:
            self._base_app_data_dir = self._get_default_base_app_data_dir()

        try:
            logger.debug(
                f"ConfigManager: Ensuring base application data directory exists: {self._base_app_data_dir}"
            )
            self._base_app_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"CRITICAL FAILURE: Could not create or access base application data directory "
                f"{self._base_app_data_dir}: {e}. Configs, logs, and exports will likely fail.",
                exc_info=True,
            )

        self._config_file_path = self._base_app_data_dir / DEFAULT_CONFIG_FILE_NAME
        self._config: Dict[str, Any] = {}
        logger.debug(
            f"ConfigManager initialized. Base data dir: {self._base_app_data_dir}, Config file: {self._config_file_path}"
        )
        self.load_config()

    def _get_default_base_app_data_dir(self) -> Path:
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            base_path = Path(xdg_data_home) / APP_DIR_NAME
        else:
            base_path = Path.home() / ".local" / "share" / APP_DIR_NAME
        return base_path

    def get_base_app_data_dir(self) -> Path:
        """Returns the root directory used by the application for its data."""
        return self._base_app_data_dir

    def get_config_file_path(self) -> Path:
        """Returns the full path to the config.json file."""
        return self._config_file_path

    def _get_or_create_subdir(self, subdir_name: str) -> Path:
        """Helper to get a subdirectory path within the base app data dir, creating it if necessary."""
        subdir = self._base_app_data_dir / subdir_name
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Could not create subdirectory {subdir}: {e}. Operations in this directory might fail."
            )

        return subdir

    def get_logs_dir(self) -> Path:
        return self._get_or_create_subdir("logs")

    def get_chats_dir(self) -> Path:
        return self._get_or_create_subdir("chats")

    def get_exports_dir(self) -> Path:
        return self._get_or_create_subdir("exports")

    def load_config(self) -> None:
        loaded_settings = {}
        if self._config_file_path.exists() and self._config_file_path.is_file():
            try:
                with open(self._config_file_path, "r", encoding="utf-8") as f:
                    loaded_settings = json.load(f)
                logger.info(f"Configuration loaded from {self._config_file_path}")
            except (IOError, json.JSONDecodeError) as e:
                logger.error(
                    f"Error loading configuration from {self._config_file_path}: {e}. Using defaults."
                )
        else:
            logger.info(
                f"Configuration file not found at {self._config_file_path}. Using defaults and will create on save."
            )

        self._config = DEFAULT_SETTINGS.copy()
        self._config.update(loaded_settings)

        if self._config.get("theme") not in VALID_THEMES:
            logger.warning(
                f"Loaded theme '{self._config.get('theme')}' is invalid. Resetting to default."
            )
            self._config["theme"] = DEFAULT_SETTINGS["theme"]

        loaded_log_level = self._config.get("default_log_level")
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]
        if not (isinstance(loaded_log_level, str) and loaded_log_level.upper() in valid_log_levels):
            logger.warning(
                f"Loaded default_log_level '{loaded_log_level}' is invalid. Resetting to default."
            )
            self._config["default_log_level"] = DEFAULT_SETTINGS["default_log_level"]
        else:
            self._config["default_log_level"] = loaded_log_level.upper()

    def save_config(self) -> bool:
        try:
            self._base_app_data_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Attempting to save configuration to: {self._config_file_path}")
            with open(self._config_file_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved successfully to {self._config_file_path}")
            return True
        except OSError as e:
            logger.error(
                f"OSError saving configuration to {self._config_file_path}. "
                f"Error: {e.strerror} (errno {e.errno}). Path involved: {e.filename}",
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error saving configuration to {self._config_file_path}: {e}",
                exc_info=True,
            )
            return False

    def get_setting(self, key: str, default: Optional[Any] = None) -> Any:
        if key == "theme":
            theme_value = self._config.get(key, DEFAULT_SETTINGS.get(key))
            return theme_value if theme_value in VALID_THEMES else DEFAULT_THEME_NAME

        if key in self._config:
            return self._config[key]

        return DEFAULT_SETTINGS.get(key, default)

    def update_setting(self, key: str, value: Any) -> None:
        if key == "theme":
            if value not in VALID_THEMES:
                logger.warning(
                    f"Attempted to set invalid theme '{value}'. Using default '{DEFAULT_THEME_NAME}'."
                )
                self._config[key] = DEFAULT_THEME_NAME
            else:
                self._config[key] = value
            return

        if key == "default_log_level":
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]
            if isinstance(value, str) and value.upper() in valid_log_levels:
                self._config[key] = value.upper()
            else:
                logger.warning(
                    f"Attempted to set invalid default_log_level '{value}'. Using default '{DEFAULT_SETTINGS['default_log_level']}'."
                )
                self._config[key] = DEFAULT_SETTINGS["default_log_level"]
            return

        self._config[key] = value

    def get_all_settings(self) -> Dict[str, Any]:
        effective_settings = DEFAULT_SETTINGS.copy()

        effective_settings.update(self._config)

        if effective_settings.get("theme") not in VALID_THEMES:
            effective_settings["theme"] = DEFAULT_THEME_NAME

        loaded_log_level = effective_settings.get("default_log_level")
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]
        if not (isinstance(loaded_log_level, str) and loaded_log_level.upper() in valid_log_levels):
            effective_settings["default_log_level"] = DEFAULT_SETTINGS["default_log_level"]
        else:
            effective_settings["default_log_level"] = loaded_log_level.upper()

        return effective_settings

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        for key, value in new_settings.items():
            self.update_setting(key, value)
