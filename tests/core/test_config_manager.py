import json
import os
from pathlib import Path

import pytest

from mongo_analyser.core.config_manager import (
    ConfigManager,
    APP_DIR_NAME,
    DEFAULT_CONFIG_FILE_NAME,
    DEFAULT_SETTINGS,
    DEFAULT_THEME_NAME,
    VALID_THEMES,
)

TEST_CUSTOM_THEME = "nord"
TEST_INVALID_THEME = "invalid-theme-name"


@pytest.fixture
def default_config_path_fixture(tmp_path: Path) -> Path:
    return tmp_path / APP_DIR_NAME / DEFAULT_CONFIG_FILE_NAME


@pytest.fixture
def cm_empty_init(default_config_path_fixture: Path) -> ConfigManager:
    base_dir = default_config_path_fixture.parent
    return ConfigManager(base_app_data_dir_override=base_dir)


@pytest.fixture
def cm_with_valid_file(default_config_path_fixture: Path) -> ConfigManager:
    default_config_path_fixture.parent.mkdir(parents=True, exist_ok=True)
    test_settings = {
        "theme": TEST_CUSTOM_THEME,
        "schema_analysis_default_sample_size": 50,
    }
    with open(default_config_path_fixture, "w", encoding="utf-8") as f:
        json.dump(test_settings, f)

    base_dir = default_config_path_fixture.parent
    return ConfigManager(base_app_data_dir_override=base_dir)


class TestConfigManager:

    def test_initialization_creates_config_with_defaults_if_no_file(
        self, default_config_path_fixture: Path
    ):
        base_dir = default_config_path_fixture.parent
        cm = ConfigManager(base_app_data_dir_override=base_dir)

        assert cm._config == DEFAULT_SETTINGS
        for key, val in DEFAULT_SETTINGS.items():
            assert cm.get_setting(key) == val

    def test_initialization_default_path_xdg_data_home(self, mocker, tmp_path: Path):
        mock_xdg_path = tmp_path / "xdg_data_home_dir"
        mocker.patch.dict(os.environ, {"XDG_DATA_HOME": str(mock_xdg_path)})
        expected_path = (
            mock_xdg_path / APP_DIR_NAME / DEFAULT_CONFIG_FILE_NAME
        )

        cm = ConfigManager()

        assert cm._config_file_path == expected_path
        assert cm._config == DEFAULT_SETTINGS
        for key, val in DEFAULT_SETTINGS.items():
            assert cm.get_setting(key) == val

    def test_initialization_default_path_home_local_share(self, mocker, tmp_path: Path):
        mock_home_path = tmp_path / "home_dir"
        mocker.patch.dict(os.environ, clear=True)
        mocker.patch("pathlib.Path.home", return_value=mock_home_path)
        expected_path = (
            mock_home_path
            / ".local"
            / "share"
            / APP_DIR_NAME
            / DEFAULT_CONFIG_FILE_NAME
        )

        cm = ConfigManager()

        assert cm._config_file_path == expected_path
        assert cm._config == DEFAULT_SETTINGS
        for key, val in DEFAULT_SETTINGS.items():
            assert cm.get_setting(key) == val

    def test_load_config_valid_file_ignored_but_get_setting_uses_defaults(
        self, default_config_path_fixture: Path
    ):
        custom_settings_in_file = {
            "theme": TEST_CUSTOM_THEME,
            "schema_analysis_default_sample_size": 500,
        }
        default_config_path_fixture.parent.mkdir(parents=True, exist_ok=True)
        with open(default_config_path_fixture, "w", encoding="utf-8") as f:
            json.dump(custom_settings_in_file, f)

        base_dir = default_config_path_fixture.parent
        cm = ConfigManager(base_app_data_dir_override=base_dir)

        expected_config = DEFAULT_SETTINGS.copy()
        expected_config.update(custom_settings_in_file)
        assert cm._config == expected_config

        assert cm.get_setting("theme") == TEST_CUSTOM_THEME
        assert cm.get_setting("schema_analysis_default_sample_size") == 500

    def test_load_config_invalid_json_uses_defaults_and_logs_error(
        self, default_config_path_fixture: Path, caplog
    ):
        default_config_path_fixture.parent.mkdir(parents=True, exist_ok=True)
        with open(default_config_path_fixture, "w", encoding="utf-8") as f:
            f.write("this is not json")

        caplog.set_level("ERROR", logger="mongo_analyser.core.config_manager")

        base_dir = default_config_path_fixture.parent
        cm = ConfigManager(base_app_data_dir_override=base_dir)

        assert cm._config == DEFAULT_SETTINGS
        assert "Error loading configuration" in caplog.text

    def test_load_config_invalid_theme_ignored_but_logs_warning(
        self, default_config_path_fixture: Path, caplog
    ):
        custom_settings_in_file = {
            "theme": TEST_INVALID_THEME,
            "schema_analysis_default_sample_size": 50,
        }
        default_config_path_fixture.parent.mkdir(parents=True, exist_ok=True)
        with open(default_config_path_fixture, "w", encoding="utf-8") as f:
            json.dump(custom_settings_in_file, f)

        caplog.set_level("WARNING", logger="mongo_analyser.core.config_manager")

        base_dir = default_config_path_fixture.parent
        cm = ConfigManager(base_app_data_dir_override=base_dir)

        expected_config = DEFAULT_SETTINGS.copy()
        expected_config.update(custom_settings_in_file)
        expected_config["theme"] = DEFAULT_THEME_NAME
        assert cm._config == expected_config

        # Only check for text up through “Resetting to default.” (no mention of the theme name)
        assert (
            f"Loaded theme '{TEST_INVALID_THEME}' is invalid. Resetting to default."
            in caplog.text
        )

    def test_save_config_creates_file_and_dirs(self, cm_empty_init: ConfigManager):
        cm = cm_empty_init
        new_setting_key = "new_key"
        new_setting_value = "new_value"
        cm.update_setting(new_setting_key, new_setting_value)

        assert not cm._config_file_path.exists()

        save_result = cm.save_config()

        assert save_result is True
        assert cm._config_file_path.exists()
        assert cm._config_file_path.is_file()

        with open(cm._config_file_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        expected_data = DEFAULT_SETTINGS.copy()
        expected_data[new_setting_key] = new_setting_value
        assert saved_data == expected_data

    def test_save_config_io_error_returns_false_logs_error(
        self, cm_empty_init: ConfigManager, mocker, caplog
    ):
        mocker.patch("json.dump", side_effect=IOError("Disk full"))
        caplog.set_level("ERROR", logger="mongo_analyser.core.config_manager")

        save_result = cm_empty_init.save_config()

        assert save_result is False
        assert f"OSError saving configuration to {cm_empty_init._config_file_path}" in caplog.text

    def test_get_setting_existing_key(self, cm_with_valid_file: ConfigManager):
        assert cm_with_valid_file.get_setting("theme") == TEST_CUSTOM_THEME
        assert cm_with_valid_file.get_setting("schema_analysis_default_sample_size") == 50

    def test_get_setting_non_existent_key_with_provided_default(
        self, cm_empty_init: ConfigManager
    ):
        assert cm_empty_init.get_setting("no_such_key", "fallback") == "fallback"

    def test_get_setting_non_existent_key_uses_default_from_default_settings(
        self, cm_empty_init: ConfigManager
    ):
        key_from_defaults = "llm_default_provider"
        assert cm_empty_init.get_setting(key_from_defaults) == DEFAULT_SETTINGS[key_from_defaults]
        assert cm_empty_init.get_setting("totally_unknown") is None

    def test_get_setting_theme_invalid_in_config_returns_default_theme(
        self, default_config_path_fixture: Path
    ):
        settings_with_invalid_theme = {"theme": TEST_INVALID_THEME}
        default_config_path_fixture.parent.mkdir(parents=True, exist_ok=True)
        with open(default_config_path_fixture, "w", encoding="utf-8") as f:
            json.dump(settings_with_invalid_theme, f)

        base_dir = default_config_path_fixture.parent
        cm = ConfigManager(base_app_data_dir_override=base_dir)

        assert cm.get_setting("theme") == DEFAULT_THEME_NAME

    def test_update_setting_regular_key(self, cm_empty_init: ConfigManager):
        cm_empty_init.update_setting("llm_default_provider", "openai")
        assert cm_empty_init._config["llm_default_provider"] == "openai"
        assert cm_empty_init.get_setting("llm_default_provider") == "openai"

    def test_update_setting_theme_valid(self, cm_empty_init: ConfigManager):
        valid_new_theme = VALID_THEMES[1] if len(VALID_THEMES) > 1 else DEFAULT_THEME_NAME
        cm_empty_init.update_setting("theme", valid_new_theme)
        assert cm_empty_init._config["theme"] == valid_new_theme

    def test_update_setting_theme_invalid_uses_default_logs_warning(
        self, cm_empty_init: ConfigManager, caplog
    ):
        caplog.set_level("WARNING", logger="mongo_analyser.core.config_manager")

        cm_empty_init.update_setting("theme", TEST_INVALID_THEME)
        assert cm_empty_init._config["theme"] == DEFAULT_THEME_NAME
        assert (
            f"Attempted to set invalid theme '{TEST_INVALID_THEME}'. Using default '{DEFAULT_THEME_NAME}'."
            in caplog.text
        )

    def test_get_all_settings_returns_copy_and_corrects_theme_if_needed(
        self, default_config_path_fixture: Path
    ):
        settings_with_invalid_theme = {"theme": TEST_INVALID_THEME, "another_key": "value"}
        default_config_path_fixture.parent.mkdir(parents=True, exist_ok=True)
        with open(default_config_path_fixture, "w", encoding="utf-8") as f:
            json.dump(settings_with_invalid_theme, f)

        base_dir = default_config_path_fixture.parent
        cm = ConfigManager(base_app_data_dir_override=base_dir)
        all_settings = cm.get_all_settings()

        assert all_settings["theme"] == DEFAULT_THEME_NAME
        assert all_settings["another_key"] == "value"

        all_settings["theme"] = "temp"
        assert cm._config["theme"] == DEFAULT_THEME_NAME

    def test_update_settings_bulk(self, cm_empty_init: ConfigManager):
        new_settings_to_apply = {
            "llm_default_temperature": 0.9,
            "data_explorer_default_sample_size": 50,
            "theme": TEST_CUSTOM_THEME,
        }
        cm_empty_init.update_settings(new_settings_to_apply)
        assert cm_empty_init._config["llm_default_temperature"] == 0.9
        assert cm_empty_init._config["data_explorer_default_sample_size"] == 50
        assert cm_empty_init._config["theme"] == TEST_CUSTOM_THEME

    def test_update_settings_invalid_theme_in_bulk_uses_default_logs_warning(
        self, cm_empty_init: ConfigManager, caplog
    ):
        new_settings_with_invalid_theme = {
            "theme": TEST_INVALID_THEME,
            "llm_default_temperature": 0.8,
        }
        caplog.set_level("WARNING", logger="mongo_analyser.core.config_manager")

        cm_empty_init.update_settings(new_settings_with_invalid_theme)
        assert cm_empty_init._config["theme"] == DEFAULT_THEME_NAME
        assert cm_empty_init._config["llm_default_temperature"] == 0.8
        assert (
            f"Attempted to set invalid theme '{TEST_INVALID_THEME}'. Using default '{DEFAULT_THEME_NAME}'."
            in caplog.text
            )

    def test_default_theme_name_is_valid(self):
        assert DEFAULT_THEME_NAME in VALID_THEMES

    def test_valid_themes_is_list_of_strings(self):
        assert isinstance(VALID_THEMES, list)
        assert all(isinstance(theme, str) for theme in VALID_THEMES)
