import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Type, Union

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
    Static,
    Tab,
    Tabs,
)

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.views import ChatView, DataExplorerView, DBConnectionView, SchemaAnalysisView

logger = logging.getLogger(__name__)

CSSPathType = Union[str, Path, List[Union[str, Path]]]


class MongoAnalyserApp(App[None]):
    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme", show=True),
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

    dark: bool

    def __init__(
        self,
        driver_class: Optional[Type[Driver]] = None,
        css_path: Optional[CSSPathType] = None,
        watch_css: bool = False,
    ):
        super().__init__(driver_class, css_path, watch_css)
        self.dark = True

    def on_mount(self) -> None:
        chosen = "dracula" if self.dark else "textual-dark"
        try:
            self.theme = chosen
            logger.info(f"Starting with theme: {chosen}")
        except Exception as e:
            logger.error(f"Failed to set initial theme '{chosen}': {e}. Falling back to default.")

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
        )
        with ContentSwitcher(initial="view_db_connection_content"):
            yield DBConnectionView(id="view_db_connection_content")
            yield SchemaAnalysisView(id="view_schema_analysis_content")
            yield DataExplorerView(id="view_data_explorer_content")
            yield ChatView(id="view_chat_content")
        yield Footer()

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if not event.tab or not event.tab.id:
            return
        suffix = event.tab.id.split("_", 1)[1]
        view_id = f"view_{suffix}_content"
        try:
            switcher = self.query_one(ContentSwitcher)
            switcher.current = view_id
            widget = switcher.get_widget_by_id(view_id)
            if hasattr(widget, "focus_default_widget"):
                widget.focus_default_widget()
        except NoMatches:
            logger.error(f"Could not switch to view '{view_id}'")

    async def action_app_copy(self) -> None:
        """Handles copying text from the focused Input, DataTable cell, Static, or Label."""
        focused = self.focused
        text_to_copy: Optional[str] = None
        source_widget_type = "unknown"

        if isinstance(focused, Input):
            source_widget_type = "Input"
            text_to_copy = focused.selected_text if focused.selected_text else focused.value
        elif isinstance(focused, DataTable):
            if focused.show_cursor and focused.cursor_coordinate:
                try:
                    cell_renderable = focused.get_cell_at(focused.cursor_coordinate)
                    if isinstance(cell_renderable, Text):
                        text_to_copy = cell_renderable.plain
                    elif isinstance(cell_renderable, str):
                        text_to_copy = cell_renderable
                    else:
                        text_to_copy = str(cell_renderable)
                    logger.debug(
                        f"DataTable copy: cell at {focused.cursor_coordinate} gave '{text_to_copy}'"
                    )
                except Exception as e:
                    logger.error(
                        f"Error getting DataTable cell content for copy: {e}", exc_info=True
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
                    logger.error("Copy to clipboard failed", exc_info=e)
                    self.notify(
                        "Copy to system clipboard failed. Check logs/permissions.",
                        title="Copy Error",
                        severity="error",
                        timeout=5,
                    )
                return
            else:
                self.notify(
                    f"Focused {source_widget_type} is empty.",
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

    def action_toggle_theme(self) -> None:
        self.dark = not self.dark
        chosen = "dracula" if self.dark else "textual-dark"
        try:
            self.theme = chosen
            logger.info(f"Switched to theme: {chosen}")
        except Exception as e:
            logger.error(
                f"Failed to switch theme to '{chosen}': {e}. Theme may not be registered or CSS is missing."
            )


def main_interactive_tui():
    log_dir = Path(os.getenv("MONGO_ANALYSER_LOG_DIR", Path.cwd()))
    log_file = log_dir / "mongo_analyser_tui.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    enable_devtools = os.getenv("MONGO_ANALYSER_DEBUG", "0") == "1"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        handlers=handlers,
        force=True,
        format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)

    logger.info("--- Starting Mongo Analyser TUI ---")
    if enable_devtools:
        logger.info("Textual Devtools are enabled (MONGO_ANALYSER_DEBUG=1).")
    try:
        app = MongoAnalyserApp()
        if enable_devtools:
            app.devtools = True
        app.run()
    except Exception:
        logger.critical("MongoAnalyserApp crashed", exc_info=True)
    finally:
        logger.info("--- Exiting Mongo Analyser TUI ---")
        core_db_manager.disconnect_all_mongo()


if __name__ == "__main__":
    main_interactive_tui()
