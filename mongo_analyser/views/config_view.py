import logging
from typing import Any, Dict, Optional

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static

from mongo_analyser.core.config_manager import (
    DEFAULT_SETTINGS,
    DEFAULT_THEME_NAME,
    VALID_THEMES,
    ConfigManager,
)
from mongo_analyser.dialogs import ConfirmDialog, ErrorDialog

logger = logging.getLogger(__name__)


class ConfigView(Container):
    config_save_feedback = reactive(Text(""))
    _feedback_timer_cv: Optional[Any] = None

    def __init__(
        self,
        *children: Widget,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        disabled: bool = False,
    ):
        super().__init__(*children, name=name, id=id, classes=classes, disabled=disabled)
        self._config_manager: Optional[ConfigManager] = None

    def on_mount(self) -> None:
        if hasattr(self.app, "config_manager"):
            self._config_manager = self.app.config_manager
            self.load_settings_to_ui()
        else:
            logger.error("ConfigManager not found on app. Cannot initialize ConfigView.")
            self.config_save_feedback = Text.from_markup(
                "[red]Error: ConfigManager not available.[/]"
            )

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="config_view_scroll_panel"):
            yield Label("Theme:", classes="config_label")
            yield Select(
                [(name.replace("-", " ").title(), name) for name in VALID_THEMES],
                id="config_theme_select",
                value=DEFAULT_THEME_NAME,
            )

            yield Label("Default Log Level:", classes="config_label")
            yield Select(
                [
                    ("DEBUG", "DEBUG"),
                    ("INFO", "INFO"),
                    ("WARNING", "WARNING"),
                    ("ERROR", "ERROR"),
                    ("CRITICAL", "CRITICAL"),
                    ("OFF", "OFF"),
                ],
                id="config_log_level_select",
                value="INFO",
            )
            yield Label("Schema Analysis - Default Sample Size:", classes="config_label")
            yield Input(
                id="config_schema_sample_size_input",
                value=str(DEFAULT_SETTINGS["schema_analysis_default_sample_size"]),
                placeholder="e.g., 1000",
                type="integer",
            )
            yield Label("Data Explorer - Default Sample Size:", classes="config_label")
            yield Input(
                id="config_explorer_sample_size_input",
                value=str(DEFAULT_SETTINGS["data_explorer_default_sample_size"]),
                placeholder="e.g., 10",
                type="integer",
            )

            yield Label("LLM Default Provider:", classes="config_label")
            yield Select(
                [("Ollama", "ollama"), ("OpenAI", "openai"), ("Google", "google")],
                id="config_llm_default_provider_select",
                value=DEFAULT_SETTINGS["llm_default_provider"],
                allow_blank=False,
            )
            yield Label("LLM Default Model - Ollama:", classes="config_label")
            yield Input(
                id="config_llm_model_ollama_input",
                value=DEFAULT_SETTINGS["llm_default_model_ollama"] or "",
                placeholder="e.g., llama3:8b",
            )
            yield Label("LLM Default Model - OpenAI:", classes="config_label")
            yield Input(
                id="config_llm_model_openai_input",
                value=DEFAULT_SETTINGS["llm_default_model_openai"] or "",
                placeholder="e.g., gpt-4o-mini",
            )
            yield Label("LLM Default Model - Google:", classes="config_label")
            yield Input(
                id="config_llm_model_google_input",
                value=DEFAULT_SETTINGS["llm_default_model_google"] or "",
                placeholder="e.g., gemini-1.5-flash-latest",
            )

            yield Label("LLM Default Temperature:", classes="config_label")
            yield Input(
                id="config_llm_temperature_input",
                value=str(DEFAULT_SETTINGS["llm_default_temperature"]),
                placeholder="e.g., 0.7",
                type="number",
            )
            yield Label("LLM Default Max History Messages:", classes="config_label")
            yield Input(
                id="config_llm_max_history_input",
                value=str(DEFAULT_SETTINGS["llm_default_max_history"]),
                placeholder="e.g., 20 (0=all, -1=none)",
                type="integer",
            )

            yield Button("Save Configuration", variant="primary", id="config_save_button")
            yield Button("Reset to Defaults", variant="warning", id="config_reset_button")
            yield Static(self.config_save_feedback, id="config_save_feedback_label")

    def load_settings_to_ui(self) -> None:
        if not self._config_manager:
            return
        try:
            theme_select = self.query_one("#config_theme_select", Select)
            theme_to_set = self._config_manager.get_setting("theme")

            current_options = theme_select._options
            if not current_options:
                theme_select.set_options(
                    [(name.replace("-", " ").title(), name) for name in VALID_THEMES]
                )
                current_options = theme_select._options

            if theme_to_set in [opt_val for _, opt_val in current_options]:
                theme_select.value = theme_to_set
            else:
                logger.warning(
                    f"Theme '{theme_to_set}' from config not in Select options. Defaulting to {DEFAULT_THEME_NAME}"
                )
                theme_select.value = DEFAULT_THEME_NAME

            self.query_one(
                "#config_log_level_select", Select
            ).value = self._config_manager.get_setting("default_log_level")
            self.query_one("#config_schema_sample_size_input", Input).value = str(
                self._config_manager.get_setting("schema_analysis_default_sample_size")
            )
            self.query_one("#config_explorer_sample_size_input", Input).value = str(
                self._config_manager.get_setting("data_explorer_default_sample_size")
            )

            self.query_one(
                "#config_llm_default_provider_select", Select
            ).value = self._config_manager.get_setting("llm_default_provider")
            self.query_one("#config_llm_model_ollama_input", Input).value = (
                self._config_manager.get_setting("llm_default_model_ollama") or ""
            )
            self.query_one("#config_llm_model_openai_input", Input).value = (
                self._config_manager.get_setting("llm_default_model_openai") or ""
            )
            self.query_one("#config_llm_model_google_input", Input).value = (
                self._config_manager.get_setting("llm_default_model_google") or ""
            )

            self.query_one("#config_llm_temperature_input", Input).value = str(
                self._config_manager.get_setting("llm_default_temperature")
            )
            self.query_one("#config_llm_max_history_input", Input).value = str(
                self._config_manager.get_setting("llm_default_max_history")
            )

            logger.info("ConfigView: Settings loaded into UI.")
        except NoMatches as e:
            logger.error(f"ConfigView: Error finding widget to load settings: {e}")
        except Exception as e:
            logger.error(f"ConfigView: Unexpected error loading settings to UI: {e}", exc_info=True)

    def _collect_settings_from_ui(self) -> Dict[str, Any]:
        settings = {}
        try:
            theme_value = self.query_one("#config_theme_select", Select).value
            settings["theme"] = (
                str(theme_value)
                if theme_value != Select.BLANK and theme_value is not None
                else DEFAULT_THEME_NAME
            )
            if settings["theme"] not in VALID_THEMES:
                logger.warning(
                    f"Invalid theme '{settings['theme']}' collected from UI. Using default."
                )
                settings["theme"] = DEFAULT_THEME_NAME

            settings["default_log_level"] = str(
                self.query_one("#config_log_level_select", Select).value
            )

            schema_sample_size_str = self.query_one("#config_schema_sample_size_input", Input).value
            try:
                settings["schema_analysis_default_sample_size"] = int(schema_sample_size_str)
            except ValueError:
                settings["schema_analysis_default_sample_size"] = DEFAULT_SETTINGS[
                    "schema_analysis_default_sample_size"
                ]
                logger.warning(
                    f"Invalid schema sample size '{schema_sample_size_str}', using default."
                )

            explorer_sample_size_str = self.query_one(
                "#config_explorer_sample_size_input", Input
            ).value
            try:
                settings["data_explorer_default_sample_size"] = int(explorer_sample_size_str)
            except ValueError:
                settings["data_explorer_default_sample_size"] = DEFAULT_SETTINGS[
                    "data_explorer_default_sample_size"
                ]
                logger.warning(
                    f"Invalid explorer sample size '{explorer_sample_size_str}', using default."
                )

            settings["llm_default_provider"] = str(
                self.query_one("#config_llm_default_provider_select", Select).value
            )

            ollama_model_val = self.query_one("#config_llm_model_ollama_input", Input).value.strip()
            settings["llm_default_model_ollama"] = ollama_model_val if ollama_model_val else None

            openai_model_val = self.query_one("#config_llm_model_openai_input", Input).value.strip()
            settings["llm_default_model_openai"] = openai_model_val if openai_model_val else None

            google_model_val = self.query_one("#config_llm_model_google_input", Input).value.strip()
            settings["llm_default_model_google"] = google_model_val if google_model_val else None

            llm_temp_str = self.query_one("#config_llm_temperature_input", Input).value
            try:
                settings["llm_default_temperature"] = float(llm_temp_str)
            except ValueError:
                settings["llm_default_temperature"] = DEFAULT_SETTINGS["llm_default_temperature"]
                logger.warning(f"Invalid LLM temperature '{llm_temp_str}', using default.")

            llm_hist_str = self.query_one("#config_llm_max_history_input", Input).value
            try:
                settings["llm_default_max_history"] = int(llm_hist_str)
            except ValueError:
                settings["llm_default_max_history"] = DEFAULT_SETTINGS["llm_default_max_history"]
                logger.warning(f"Invalid LLM max history '{llm_hist_str}', using default.")

        except NoMatches as e:
            logger.error(f"ConfigView: Error finding widget to collect settings: {e}")
            self.app.notify(
                f"UI Error: Could not read setting for {e.widget}",
                severity="error",
                title="Config Error",
            )
        except Exception as e:
            logger.error(
                f"ConfigView: Unexpected error collecting settings from UI: {e}", exc_info=True
            )
            self.app.notify(
                "Unexpected error reading settings.", severity="error", title="Config Error"
            )
        return settings

    @on(Button.Pressed, "#config_save_button")
    async def save_configuration_button_pressed(self) -> None:
        if not self._config_manager:
            self.config_save_feedback = Text.from_markup(
                "[red]Error: ConfigManager not available for saving.[/]"
            )
            return

        settings_from_ui = self._collect_settings_from_ui()
        if not settings_from_ui:
            self.config_save_feedback = Text.from_markup(
                "[orange3]Warning: No settings collected from UI to save.[/]"
            )
            return

        self._config_manager.update_settings(settings_from_ui)

        if self._config_manager.save_config():
            self.config_save_feedback = Text.from_markup(
                "[green]Configuration saved successfully![/]"
            )
            self.app.notify("Configuration saved.", title="Save Success")

            if hasattr(self.app, "config_manager"):
                new_theme_name = self._config_manager.get_setting("theme")
                if self.app.theme != new_theme_name:
                    self.app.theme = new_theme_name
                    logger.info(f"Theme changed to '{new_theme_name}' and applied from config.")
        else:
            self.config_save_feedback = Text.from_markup(
                "[red]Error saving configuration. Check logs.[/]"
            )
            await self.app.push_screen(
                ErrorDialog("Save Error", "Could not save configuration. See logs for details.")
            )

    @on(Button.Pressed, "#config_reset_button")
    async def reset_configuration_button_pressed(self) -> None:
        if not self._config_manager:
            self.config_save_feedback = Text.from_markup(
                "[red]Error: ConfigManager not available for reset.[/]"
            )
            return

        confirm = await self.app.push_screen_wait(
            ConfirmDialog(
                "Reset Configuration",
                "Are you sure you want to reset all settings to their defaults?",
            )
        )
        if confirm:
            self._config_manager.update_settings(DEFAULT_SETTINGS.copy())
            self.load_settings_to_ui()
            if self._config_manager.save_config():
                self.config_save_feedback = Text.from_markup(
                    "[green]Configuration reset to defaults and saved.[/]"
                )
                self.app.notify("Configuration reset to defaults.", title="Reset Success")
                if hasattr(self.app, "config_manager"):
                    default_theme_name = self._config_manager.get_setting("theme")
                    if self.app.theme != default_theme_name:
                        self.app.theme = default_theme_name
                        logger.info(f"Theme reset to default '{default_theme_name}' and applied.")
            else:
                self.config_save_feedback = Text.from_markup(
                    "[red]Error saving defaults after reset. Check logs.[/]"
                )

    def watch_config_save_feedback(self, new_feedback: Text) -> None:
        if self.is_mounted:
            try:
                lbl = self.query_one("#config_save_feedback_label", Static)
                lbl.update(new_feedback)
                if self._feedback_timer_cv is not None:
                    try:
                        self._feedback_timer_cv.stop()
                    except AttributeError:
                        if hasattr(self._feedback_timer_cv, "stop_no_wait"):
                            self._feedback_timer_cv.stop_no_wait()
                    self._feedback_timer_cv = None
                if new_feedback.plain:
                    self._feedback_timer_cv = self.set_timer(
                        4, lambda: setattr(self, "config_save_feedback", Text(""))
                    )
            except NoMatches:
                logger.warning(
                    "#config_save_feedback_label NOT FOUND in watch_config_save_feedback!"
                )
            except Exception as e:
                logger.error(
                    f"Error in watch_config_save_feedback (ConfigView): {e}", exc_info=True
                )

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#config_theme_select", Select).focus()
        except NoMatches:
            logger.debug("ConfigView: Could not focus default select.")
