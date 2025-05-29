# tui.py

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Type, Union

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.driver import Driver
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, Footer, Header, Input, Tab, Tabs

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.views import ChatView, DataExplorerView, DBConnectionView, SchemaAnalysisView

logger = logging.getLogger(__name__)

CSSPathType = Union[str, Path, List[Union[str, Path]]]


class MongoAnalyserApp(App[None]):
    """
    Main TUI application for Mongo Analyser.
    Toggles based on the 'dark' attribute on Ctrl+T.
    """

    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss" # Ensure this file, or others in CSS_PATH, define styles for themes if needed

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme", show=True),
        Binding("ctrl+c", "app_copy", "Copy Selection", show=False, priority=True),
        Binding("ctrl+insert", "app_copy", "Copy Selection (Alt)", show=False, priority=True),
        Binding("ctrl+v", "app_paste", "Paste", show=False, priority=True),
        Binding("shift+insert", "app_paste", "Paste (Alt)", show=False, priority=True),
    ]

    current_mongo_uri: reactive[Optional[str]] = reactive(None)
    current_db_name: reactive[Optional[str]] = reactive(None)
    available_collections: reactive[List[str]] = reactive([])
    active_collection: reactive[Optional[str]] = reactive(None)
    current_schema_analysis_results: reactive[Optional[dict]] = reactive(None)

    dark: bool # Flag to indicate current theme state

    def __init__(
        self,
        driver_class: Optional[Type[Driver]] = None,
        css_path: Optional[CSSPathType] = None,
        watch_css: bool = False,
    ):
        super().__init__(driver_class, css_path, watch_css)
        self.dark = True # Default to dark theme state on startup

    def on_mount(self) -> None:
        """Apply the initial theme on mount."""
        # For this to work, "dracula" and "textual-dark" must be valid theme names
        # recognized by Textual, either as built-ins in your Textual version,
        # registered themes (via App.register_theme), or through CSS selectors
        # like Screen[theme="dracula"] in your loaded CSS.
        chosen = "dracula" if self.dark else "textual-dark"
        try:
            self.theme = chosen
            logger.info(f"Starting with theme: {chosen}")
        except Exception as e: # Catch potential errors if themes are not found
            logger.error(f"Failed to set initial theme '{chosen}': {e}. Falling back to default.")
            # Fallback or further error handling might be needed if self.theme assignment fails


    def watch_available_collections(self, old: List[str], new: List[str]) -> None:
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
        focused = self.focused
        if isinstance(focused, Input):
            text = (
                focused.selected_text if getattr(focused, "selected_text", None) else focused.value
            )
            if text:
                try:
                    self.copy_to_clipboard(text)
                    self.notify("Copied to clipboard", title="Copy Success")
                except Exception as e:
                    logger.error("Copy failed", exc_info=e)
                    self.notify("Copy failed", title="Copy Error", severity="error")
                return
        self.notify("No text to copy", title="Copy Info", severity="info")

    async def action_app_paste(self) -> None:
        focused = self.focused
        if isinstance(focused, Input):
            try:
                await self.request_paste_from_clipboard()
                return
            except Exception as e:
                logger.error("Paste failed", exc_info=e)
                self.notify("Paste failed", title="Paste Error", severity="error")
                return
        self.notify("Cannot paste here", title="Paste Info", severity="warning")

    def action_toggle_theme(self) -> None:
        self.dark = not self.dark
        chosen = "dracula" if self.dark else "textual-dark"
        try:
            self.theme = chosen # This assigns the string name to App.theme
            logger.info(f"Switched to theme: {chosen}")
        except Exception as e: # Catch potential errors if themes are not found
            logger.error(f"Failed to switch theme to '{chosen}': {e}. Theme may not be registered or CSS is missing.")
            self.dark = True # This itself might trigger Textual's default dark if self.theme failed.


def main_interactive_tui():
    log_dir = Path(os.getenv("MONGO_ANALYSER_LOG_DIR", Path.cwd()))
    log_file = log_dir / "mongo_analyser_tui.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        handlers=handlers,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logger.info("--- Starting Mongo Analyser TUI ---")

    try:
        app = MongoAnalyserApp()
        app.run()
    except Exception as e:
        logger.critical("App crashed", exc_info=True)
        logger.error("Unhandled exception in TUI: %s", e)
    finally:
        logger.info("--- Exiting Mongo Analyser TUI ---")
        core_db_manager.disconnect_all_mongo()


if __name__ == "__main__":
    main_interactive_tui()
