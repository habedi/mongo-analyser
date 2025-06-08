import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Type, Union

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.core.config_manager import DEFAULT_THEME_NAME, VALID_THEMES, ConfigManager
from mongo_analyser.views import (
    ChatView,
    ConfigView,
    DataExplorerView,
    DBConnectionView,
    SchemaAnalysisView,
)
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.driver import Driver
from textual.reactive import reactive
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Tab,
    Tabs,
    TextArea,
)

logger = logging.getLogger(__name__)

CSSPathType = Union[str, Path, List[Union[str, Path]]]
APP_LOGGER_NAME = "mongo_analyser"


class MongoAnalyserApp(App[None]):
    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("q", "quit_app_action", "Quit", show=True, priority=True),
        Binding("ctrl+t", "change_theme", "Change Theme", show=True),
        Binding("ctrl+c", "app_copy", "Copy Text", show=True, key_display="Ctrl+C"),
        Binding("ctrl+insert", "app_copy", "Copy Text (Alt)", show=False, priority=True),
        Binding("ctrl+v", "app_paste", "Paste Text", show=True, key_display="Ctrl+V"),
        Binding("shift+insert", "app_paste", "Paste Text (Alt)", show=False, priority=True),
    ]

    current_mongo_uri: reactive[Optional[str]] = reactive(None)
    current_db_name: reactive[Optional[str]] = reactive(None)
    available_collections: reactive[List[str]] = reactive([])
    active_collection: reactive[Optional[str]] = reactive(None)
    current_schema_analysis_results: reactive[Optional[dict]] = reactive(None)

    base_app_data_dir: Path

    def __init__(
        self,
        driver_class: Optional[Type[Driver]] = None,
        css_path: Optional[CSSPathType] = None,
        watch_css: bool = False,
        initial_mongo_uri: Optional[str] = None,
        initial_db_name: Optional[str] = None,
        base_app_data_dir_override: Optional[Path] = None,
    ):
        self.config_manager = ConfigManager(base_app_data_dir_override=base_app_data_dir_override)

        self.base_app_data_dir = self.config_manager.get_base_app_data_dir()

        configured_theme_name = self.config_manager.get_setting("theme", DEFAULT_THEME_NAME)
        if configured_theme_name not in VALID_THEMES:
            configured_theme_name = DEFAULT_THEME_NAME

        super().__init__(driver_class, css_path, watch_css)

        self.theme = configured_theme_name
        self._initial_mongo_uri = initial_mongo_uri
        self._initial_db_name = initial_db_name

        if logger.isEnabledFor(logging.DEBUG):
            safe_uri = core_db_manager.redact_uri_password(self._initial_mongo_uri or "N/A")
            logger.debug(
                "MongoAnalyserApp initialized. Initial URI: '%s', DB: '%s', Theme: '%s', Base App Data Dir: '%s'",
                safe_uri,
                self._initial_db_name,
                self.theme,
                self.base_app_data_dir,
            )

    def on_mount(self) -> None:
        try:
            schema_view = self.query_one(SchemaAnalysisView)
            schema_view.query_one("#schema_sample_size_input", Input).value = str(
                self.config_manager.get_setting("schema_analysis_default_sample_size")
            )
        except NoMatches:
            logger.warning("Could not find SchemaAnalysisView or its sample size input on mount.")
        try:
            explorer_view = self.query_one(DataExplorerView)
            explorer_view.query_one("#data_explorer_sample_size_input", Input).value = str(
                self.config_manager.get_setting("data_explorer_default_sample_size")
            )
        except NoMatches:
            logger.warning("Could not find DataExplorerView or its sample size input on mount.")
        try:
            chat_view = self.query_one(ChatView)
            llm_config_panel = chat_view.query_one("#chat_llm_config_panel")
            default_provider = self.config_manager.get_setting("llm_default_provider")
            default_temp = self.config_manager.get_setting("llm_default_temperature")
            default_hist = self.config_manager.get_setting("llm_default_max_history")
            provider_select = llm_config_panel.query_one("#llm_config_provider_select", Select)
            if provider_select.value != default_provider:
                provider_select.value = default_provider
            temp_input = llm_config_panel.query_one("#llm_config_temperature", Input)
            if temp_input.value != str(default_temp):
                temp_input.value = str(default_temp)
            hist_input = llm_config_panel.query_one("#llm_config_max_history", Input)
            if hist_input.value != str(default_hist):
                hist_input.value = str(default_hist)
        except NoMatches:
            logger.warning(
                "Could not find ChatView or LLMConfigPanel elements on mount for defaults."
            )
        except Exception as e:
            logger.error(f"Error applying LLM defaults from config on mount: {e}", exc_info=True)

    def watch_available_collections(self) -> None:
        for view_cls in (SchemaAnalysisView, DataExplorerView):
            try:
                view = self.query_one(view_cls)
                if view.is_mounted:
                    view.update_collection_select()
            except NoMatches:
                pass

    def watch_active_collection(self, old: Optional[str], new: Optional[str]) -> None:
        if old != new:
            self.current_schema_analysis_results = None
        for view_cls in (SchemaAnalysisView, DataExplorerView):
            try:
                view = self.query_one(view_cls)
                if view.is_mounted:
                    view.update_collection_select()
            except NoMatches:
                pass

    def compose(self) -> ComposeResult:
        yield Header()
        yield Tabs(
            Tab("DB Connection", id="tab_db_connection"),
            Tab("Schema Analysis", id="tab_schema_analysis"),
            Tab("Data Explorer", id="tab_data_explorer"),
            Tab("Chat", id="tab_chat"),
            Tab("Config", id="tab_config"),
        )
        with ContentSwitcher(initial="view_db_connection_content"):
            yield DBConnectionView(id="view_db_connection_content")
            yield SchemaAnalysisView(id="view_schema_analysis_content")
            yield DataExplorerView(id="view_data_explorer_content")
            yield ChatView(id="view_chat_content")
            yield ConfigView(id="view_config_content")
        yield Footer()

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if not event.tab or not event.tab.id:
            return
        suffix = event.tab.id.split("_", 1)[1]
        view_id = f"view_{suffix}_content"
        try:
            switcher = self.query_one(ContentSwitcher)
            switcher.current = view_id
            widget_to_focus = switcher.get_widget_by_id(view_id)
            if hasattr(widget_to_focus, "focus_default_widget"):
                widget_to_focus.focus_default_widget()
            if view_id == "view_config_content" and hasattr(widget_to_focus, "load_settings_to_ui"):
                widget_to_focus.load_settings_to_ui()
        except NoMatches:
            if logger.isEnabledFor(logging.ERROR):
                logger.error("Could not switch to view '%s' or find its content widget.", view_id)

    async def action_app_copy(self) -> None:
        focused = self.focused
        text_to_copy: Optional[str] = None
        source_widget_type = "unknown"
        if isinstance(focused, Input):
            source_widget_type = "Input"
            text_to_copy = focused.selected_text if focused.selected_text else focused.value
        elif isinstance(focused, TextArea):
            source_widget_type = "TextArea"
            text_to_copy = focused.selected_text if focused.selected_text else focused.text
        elif isinstance(focused, DataTable):
            source_widget_type = "DataTable"
            if focused.show_cursor and focused.cursor_coordinate:
                try:
                    cell_renderable = focused.get_cell_at(focused.cursor_coordinate)
                    if isinstance(cell_renderable, Text):
                        text_to_copy = cell_renderable.plain
                    elif isinstance(cell_renderable, str):
                        text_to_copy = cell_renderable
                    else:
                        text_to_copy = str(cell_renderable)
                except Exception as e:
                    if logger.isEnabledFor(logging.ERROR):
                        logger.error(
                            "Error getting DataTable cell content for copy: %s", e, exc_info=True
                        )
                    self.notify(
                        "Failed to get cell content.",
                        title="Copy Error",
                        severity="error",
                        timeout=3,
                    )
                    return
            else:
                self.notify(
                    "DataTable has no active cursor or cell selected.",
                    title="Copy Info",
                    severity="information",
                    timeout=3,
                )
                return
        elif isinstance(focused, (Static, Label)):
            source_widget_type = focused.__class__.__name__
            widget_content = getattr(focused, "renderable", None)
            if isinstance(widget_content, Text):
                text_to_copy = widget_content.plain
            elif isinstance(widget_content, str):
                text_to_copy = widget_content
        if text_to_copy is not None:
            if text_to_copy.strip():
                try:
                    self.copy_to_clipboard(text_to_copy)
                    display_text = text_to_copy.replace("\n", "↵")
                    self.notify(
                        f"Copied from {source_widget_type}: '{display_text[:30]}{'...' if len(display_text) > 30 else ''}'",
                        title="Copy Success",
                        timeout=3,
                    )
                except Exception as e:
                    if logger.isEnabledFor(logging.ERROR):
                        logger.error("Copy to clipboard failed: %s", e, exc_info=True)
                    self.notify(
                        "Copy to system clipboard failed. Check logs/permissions.",
                        title="Copy Error",
                        severity="error",
                        timeout=5,
                    )
                return
            else:
                self.notify(
                    f"Focused {source_widget_type} is empty or contains only whitespace.",
                    title="Copy Info",
                    severity="information",
                    timeout=3,
                )
                return
        self.notify(
            "No text selected or suitable widget focused to copy.",
            title="Copy Info",
            severity="information",
            timeout=3,
        )

    async def action_app_paste(self) -> None:
        focused_widget = self.focused
        if isinstance(focused_widget, (Input, TextArea)):
            self.notify(
                "Paste action initiated. Relies on terminal/widget paste support.",
                title="Paste Action",
                severity="information",
                timeout=2,
            )
        else:
            self.notify(
                "Cannot paste here. Focus an input field (Input, TextArea).",
                title="Paste Info",
                severity="warning",
            )

    def action_change_theme(self) -> None:
        current_theme_name = self.theme
        if not VALID_THEMES:
            logger.warning("VALID_THEMES is empty. Cannot change theme.")
            return
        try:
            current_index = VALID_THEMES.index(current_theme_name)
            next_index = (current_index + 1) % len(VALID_THEMES)
            new_theme_name = VALID_THEMES[next_index]
        except ValueError:
            logger.warning(
                f"Current theme '{current_theme_name}' not in VALID_THEMES. Defaulting to first in list."
            )
            new_theme_name = VALID_THEMES[0] if VALID_THEMES else DEFAULT_THEME_NAME
        self.theme = new_theme_name
        if hasattr(self, "config_manager") and self.config_manager:
            self.config_manager.update_setting("theme", new_theme_name)
            logger.info(f"Theme changed to: {new_theme_name} and updated in config manager.")
        else:
            logger.info(f"Theme changed to: {new_theme_name} (config manager not available).")

    async def action_quit_app_action(self) -> None:
        if hasattr(self, "config_manager") and self.config_manager:
            logger.info("Attempting to save configuration on quit...")
            if self.config_manager.save_config():
                logger.info("Configuration saved successfully on quit.")
            else:
                logger.error("Failed to save configuration on quit. Check logs.")
        await self.action_quit()


def main_interactive_tui(
    initial_mongo_uri: Optional[str] = None,
    initial_db_name: Optional[str] = None,
    base_app_data_dir_override: Optional[Path] = None,
):
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    startup_config_manager = ConfigManager(base_app_data_dir_override=base_app_data_dir_override)

    configured_log_level_str = startup_config_manager.get_setting(
        "default_log_level", "INFO"
    ).upper()
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]
    if configured_log_level_str not in valid_log_levels:
        print(
            f"Warning: Invalid default_log_level '{configured_log_level_str}' in configuration. Defaulting to INFO.",
            file=sys.stderr,
        )
        configured_log_level_str = "INFO"

    for handler in list(app_logger.handlers):
        app_logger.removeHandler(handler)
    root_logger_for_cleanup = logging.getLogger()
    for handler in list(root_logger_for_cleanup.handlers):
        root_logger_for_cleanup.removeHandler(handler)

    if configured_log_level_str == "OFF":
        logging.disable(logging.CRITICAL + 1)
        if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
            print("CLI Debug: Application logging configured to OFF.", file=sys.stderr)
    else:
        if logging.getLogger().manager.disable > 0:
            logging.disable(logging.NOTSET)
        app_logger.setLevel(configured_log_level_str)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s"
        )

        try:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(log_formatter)
            console_handler.setLevel(configured_log_level_str)
            app_logger.addHandler(console_handler)
        except Exception as e_ch:
            print(
                f"CRITICAL WARNING: Failed to set up console logging for {APP_LOGGER_NAME}: {e_ch}",
                file=sys.stderr,
            )

        log_dir = startup_config_manager.get_logs_dir()
        log_file = log_dir / f"{APP_LOGGER_NAME}_tui.log"
        file_handler_active = False
        try:
            if not any(
                isinstance(h, logging.FileHandler)
                and getattr(h, "baseFilename", None) == str(log_file.resolve())
                for h in app_logger.handlers
            ):
                file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                file_handler.setFormatter(log_formatter)
                file_handler.setLevel(configured_log_level_str)
                app_logger.addHandler(file_handler)
                file_handler_active = True
            else:
                file_handler_active = True
        except Exception as e_fh:
            print(
                f"WARNING: Could not set up file logging to {log_file}: {e_fh}. App logs may only go to console.",
                file=sys.stderr,
            )

        app_logger.propagate = False

        if app_logger.getEffectiveLevel() > logging.DEBUG:
            libraries_to_silence = [
                "httpx",
                "httpcore",
                "openai",
                "google.generativeai",
                "pymongo.command",
                "urllib3.connectionpool",
                "textual",
            ]
            for lib_logger_name in libraries_to_silence:
                lib_logger = logging.getLogger(lib_logger_name)
                if lib_logger.getEffectiveLevel() < logging.WARNING:
                    lib_logger.setLevel(logging.WARNING)

        if app_logger.isEnabledFor(logging.INFO):
            if file_handler_active:
                app_logger.info(
                    f"File logging for '{APP_LOGGER_NAME}' initialized at level {configured_log_level_str} to: {log_file}"
                )
            else:
                app_logger.warning(
                    f"File logging for '{APP_LOGGER_NAME}' to {log_file} FAILED. Current app log level: {configured_log_level_str}"
                )
            app_logger.info(
                f"Console logging for '{APP_LOGGER_NAME}' initialized at level {configured_log_level_str}."
            )

    module_tui_logger = logging.getLogger(__name__)
    if module_tui_logger.isEnabledFor(logging.INFO):
        module_tui_logger.info(
            f"--- Starting Mongo Analyser TUI (Log Level for '{APP_LOGGER_NAME}': {configured_log_level_str}) ---"
        )

    enable_devtools = os.getenv("MONGO_ANALYSER_DEBUG", "0") == "1"
    if (
        enable_devtools
        and configured_log_level_str != "OFF"
        and app_logger.isEnabledFor(logging.INFO)
    ):
        app_logger.info("Textual Devtools are enabled via MONGO_ANALYSER_DEBUG=1.")

    try:
        app = MongoAnalyserApp(
            initial_mongo_uri=initial_mongo_uri,
            initial_db_name=initial_db_name,
            base_app_data_dir_override=base_app_data_dir_override,
        )
        if enable_devtools and hasattr(app, "devtools"):
            try:
                app.devtools = True
            except Exception as e_devtools:
                if configured_log_level_str != "OFF" and app_logger.isEnabledFor(logging.WARNING):
                    app_logger.warning("Failed to enable Textual Devtools: %s.", e_devtools)
        app.run()
    except Exception as e_run:
        crit_msg = f"MongoAnalyserApp crashed during run: {e_run}"
        if configured_log_level_str != "OFF" and app_logger.isEnabledFor(logging.CRITICAL):
            app_logger.critical(crit_msg, exc_info=True)
        else:
            print(f"\nCRITICAL ERROR: {crit_msg}", file=sys.stderr)
            if configured_log_level_str == "OFF":
                import traceback

                traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if configured_log_level_str != "OFF" and app_logger.isEnabledFor(logging.INFO):
            app_logger.info("--- Exiting Mongo Analyser TUI ---")
        core_db_manager.disconnect_all_mongo()


if __name__ == "__main__":
    print(
        "Running tui.py directly. For CLI arguments (including --app-data-dir) and proper execution, use 'mongo_analyser' or 'python -m mongo_analyser.cli'."
    )

    base_dir_override_for_direct_run: Optional[Path] = None
    custom_dir_env_var = os.getenv("MONGO_ANALYSER_HOME_DIR")
    if custom_dir_env_var:
        try:
            resolved_dir = Path(custom_dir_env_var).expanduser().resolve()
            resolved_dir.mkdir(parents=True, exist_ok=True)
            base_dir_override_for_direct_run = resolved_dir
            if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
                print(
                    f"Direct tui.py run: Using MONGO_ANALYSER_HOME_DIR for base data directory: {base_dir_override_for_direct_run}",
                    file=sys.stderr,
                )
        except OSError as e:
            print(
                f"Direct tui.py run: ERROR creating custom base data directory '{custom_dir_env_var}': {e}. Using default.",
                file=sys.stderr,
            )

    main_interactive_tui(
        initial_mongo_uri=os.getenv("MONGO_URI"),
        initial_db_name=os.getenv("MONGO_DATABASE"),
        base_app_data_dir_override=base_dir_override_for_direct_run,
    )
