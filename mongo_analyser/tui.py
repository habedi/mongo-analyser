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
from textual.widgets import (
    ContentSwitcher,
    Footer,
    Header,
    Input,
    Tab,
    Tabs,
)

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.views import (
    ChatView,
    DataExplorerView,
    DBConnectionView,
    SchemaAnalysisView,
)

logger = logging.getLogger(__name__)

CSSPathType = Union[str, Path, List[Union[str, Path]]]


class MongoAnalyserApp(App[None]):
    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, key_display=None, priority=True),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme", show=True),
        Binding(  # For Textual 0.48+
            "ctrl+c", "app_copy", "Copy Selection", show=False, key_display="Ctrl+C", priority=True
        ),
        Binding(  # Fallback for older terminals/systems or if Ctrl+C is an issue
            "ctrl+insert",
            "app_copy",
            "Copy Selection (Alt)",
            show=False,
            key_display="Ctrl+Ins",
            priority=True,
        ),
        Binding(  # For Textual 0.48+
            "ctrl+v", "app_paste", "Paste", show=False, key_display="Ctrl+V", priority=True
        ),
        Binding(  # Fallback
            "shift+insert",
            "app_paste",
            "Paste (Alt)",
            show=False,
            key_display="Shift+Ins",
            priority=True,
        ),
    ]

    current_mongo_uri: reactive[str | None] = reactive(None)
    current_db_name: reactive[str | None] = reactive(None)
    available_collections: reactive[List[str]] = reactive([])
    active_collection: reactive[str | None] = reactive(None)

    # To store schema analysis results for chat injection
    current_schema_analysis_results: reactive[dict | None] = reactive(None)

    def __init__(
        self,
        driver_class: Type[Driver] | None = None,
        css_path: CSSPathType | None = None,
        watch_css: bool = False,
    ):
        super().__init__(driver_class, css_path, watch_css)
        self.dark = True  # Default to dark theme

    def watch_available_collections(
        self, old_collections: List[str], new_collections: List[str]
    ) -> None:
        logger.debug(
            f"App: available_collections changed from {len(old_collections)} items to: {len(new_collections)} items"
        )
        try:
            schema_view = self.query_one(SchemaAnalysisView)
            if schema_view.is_mounted:
                schema_view.update_collection_select()
        except NoMatches:
            pass  # View might not be mounted yet
        except Exception as e:
            logger.error(f"Error updating schema view collections: {e}", exc_info=True)

        try:
            explorer_view = self.query_one(DataExplorerView)
            if explorer_view.is_mounted:
                explorer_view.update_collection_select()
        except NoMatches:
            pass
        except Exception as e:
            logger.error(f"Error updating data explorer view collections: {e}", exc_info=True)

    def watch_active_collection(self, old_coll: Optional[str], new_coll: Optional[str]) -> None:
        logger.debug(f"App: active_collection changed from '{old_coll}' to '{new_coll}'")
        # Reset schema analysis results if collection changes
        if old_coll != new_coll:
            self.current_schema_analysis_results = None

        # Update relevant views that depend on active_collection
        try:
            schema_view = self.query_one(SchemaAnalysisView)
            if schema_view.is_mounted:
                schema_view.update_collection_select()  # Ensures its select is also in sync
        except NoMatches:
            pass

        try:
            explorer_view = self.query_one(DataExplorerView)
            if explorer_view.is_mounted:
                explorer_view.update_collection_select()
        except NoMatches:
            pass

    def compose(self) -> ComposeResult:
        yield Header()
        yield Tabs(
            Tab(label="DB Connection", id="tab_db_connection"),  # Made this first
            Tab(label="Schema Analysis", id="tab_schema_analysis"),
            Tab(label="Data Explorer", id="tab_data_explorer"),
            Tab(label="Chat", id="tab_chat"),
        )
        with ContentSwitcher(initial="view_db_connection_content"):
            yield DBConnectionView(id="view_db_connection_content")
            yield SchemaAnalysisView(id="view_schema_analysis_content")
            yield DataExplorerView(id="view_data_explorer_content")
            yield ChatView(id="view_chat_content")
        yield Footer()

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        view_id_suffix: str | None = None
        if event.tab and event.tab.id:
            try:
                suffix = event.tab.id.split("_", 1)[
                    1
                ]  # e.g., "db_connection" from "tab_db_connection"
                view_id_suffix = f"view_{suffix}_content"  # e.g., "view_db_connection_content"
            except IndexError:
                logger.warning(f"Could not parse suffix from tab ID: {event.tab.id}")
                return

        if not view_id_suffix:
            logger.warning(
                f"No view_id_suffix could be determined for tab {event.tab.id if event.tab else 'None'}"
            )
            return

        switcher: ContentSwitcher | None = None
        try:
            switcher = self.query_one(ContentSwitcher)
            switcher.current = view_id_suffix
        except NoMatches:
            logger.error(
                f"ContentSwitcher not found, or view '{view_id_suffix}' not found in ContentSwitcher."
            )
            return
        except Exception as e:  # Catch other potential errors during switch
            logger.error(f"Error setting ContentSwitcher to '{view_id_suffix}': {e}", exc_info=True)
            return

        # Focus default widget in the newly activated view
        active_view_id_for_logging = view_id_suffix
        try:
            active_view_id = switcher.current  # Get the ID of the widget now current
            if active_view_id:
                active_view_id_for_logging = active_view_id
                active_view = switcher.get_widget_by_id(active_view_id)  # Get the widget instance
                if hasattr(active_view, "focus_default_widget"):
                    active_view.focus_default_widget()
            else:
                logger.warning(
                    f"ContentSwitcher has no current widget ID after attempting to set to {view_id_suffix}"
                )
        except NoMatches:  # If get_widget_by_id fails
            logger.debug(
                f"No widget with ID '{active_view_id_for_logging}' found in ContentSwitcher for focusing."
            )
        except Exception as e:
            logger.error(
                f"Error focusing input in view '{active_view_id_for_logging}': {e}", exc_info=True
            )

    async def action_app_copy(self) -> None:
        focused_widget = self.focused
        text_to_copy = None

        if isinstance(focused_widget, Input):
            # For Textual 0.48+, selected_text is the way
            if hasattr(focused_widget, "selected_text") and focused_widget.selected_text:
                text_to_copy = focused_widget.selected_text
            elif (
                not focused_widget.selected_text and focused_widget.value
            ):  # If no selection, copy all
                text_to_copy = focused_widget.value
        # Add other copyable widgets here, e.g., DataTable cell, Markdown selection
        # For Markdown, Textual might not have a direct selected_text.
        # For DataTable, you'd get cell at cursor.

        if text_to_copy is not None:
            try:
                self.copy_to_clipboard(text_to_copy)
                logger.info(f"Copied to system clipboard: '{text_to_copy[:100]}...'")
                self.notify("Selected text copied to clipboard.", title="Copy Success")
            except Exception as e:  # Catch driver errors or other issues
                logger.error(f"Failed to copy to system clipboard: {e}", exc_info=True)
                self.notify(f"Failed to copy text: {e!s}", title="Copy Error", severity="error")
                # await self.push_screen(ErrorDialog("Clipboard Error", f"Could not copy to clipboard: {e!s}"))
        else:
            logger.info("No text selected or available in the focused widget to copy.")
            self.notify("No text to copy.", title="Copy Info", severity="information")

    async def action_app_paste(self) -> None:
        focused_widget = self.focused

        if isinstance(focused_widget, Input):
            try:
                # For Textual 0.48+, request_paste handles it
                await self.request_paste_from_clipboard()
                logger.info("Paste requested for focused Input widget.")
            except Exception as e:  # Catch if method not available or other errors
                logger.error(f"Error requesting paste from clipboard: {e}", exc_info=True)
                self.notify(
                    f"Error during paste: {e!s}. Try terminal's native paste (Ctrl+Shift+V).",
                    title="Paste Error",
                    severity="error",
                    timeout=8.0,
                )
                # await self.push_screen(ErrorDialog("Clipboard Error", f"Could not paste from clipboard: {e!s}"))
        else:
            logger.info("Paste action ignored: focused widget is not an input field.")
            self.notify(
                "Cannot paste here. Focus an input field.", title="Paste Info", severity="warning"
            )

    def action_toggle_theme(self) -> None:
        self.dark = not self.dark
        logger.info(f"Textual dark mode toggled to: {self.dark}")


def main_interactive_tui():
    log_file_path = Path(os.getenv("MONGO_ANALYSER_LOG_DIR", Path.cwd())) / "mongo_analyser_tui.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
        )
    )

    # Configure root logger
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),  # Default to INFO
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Also log to stdout for easier debugging if file log fails
            file_handler,
        ],
        force=True,  # Override any existing basicConfig
    )

    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    litellm_logger = logging.getLogger("litellm")  # For LiteLLM library
    litellm_logger.setLevel(logging.WARNING)  # Or INFO if you need more from it

    # Your app's main logger
    logger.info(f"--- Starting Mongo Analyser TUI. Logging to {log_file_path.resolve()} ---")

    # Test if clipboard is available early
    try:
        app_for_clipboard_test = App()
        app_for_clipboard_test.copy_to_clipboard("test")
        logger.info("Clipboard access test successful.")
    except Exception as e:
        logger.warning(f"Clipboard access test failed: {e}. Copy/paste might not work as expected.")

    try:
        app = MongoAnalyserApp()
        app.run()
    except Exception:  # Catch-all for app.run() level crashes
        logger.critical("MongoAnalyserApp failed to run.", exc_info=True)
        print("MongoAnalyserApp crashed. Check logs for details.", file=sys.stderr)
    finally:
        logger.info("--- Mongo Analyser TUI finished. ---")
        core_db_manager.disconnect_all_mongo()


if __name__ == "__main__":
    main_interactive_tui()
