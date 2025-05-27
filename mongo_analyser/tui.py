import logging
import os
from pathlib import Path
from typing import List, Type, Union

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
    DBConnectionView,
    SchemaAnalysisView,
)  # Removed DataExtractionView

logger = logging.getLogger(__name__)

CSSPathType = Union[str, Path, List[Union[str, Path]]]


class MongoAnalyserApp(App[None]):
    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, key_display=None, priority=True),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme (Textual Default)", show=True),
        Binding(
            "ctrl+insert",
            "app_copy",
            "Copy Selection",
            show=False,
            key_display="Ctrl+Ins",
            priority=True,
        ),
        Binding(
            "shift+insert",
            "app_paste",
            "Paste from Clipboard",
            show=False,
            key_display="Shift+Ins",
            priority=True,
        ),
    ]

    current_mongo_uri: reactive[str | None] = reactive(None)
    current_db_name: reactive[str | None] = reactive(None)
    available_collections: reactive[List[str]] = reactive([])
    active_collection: reactive[str | None] = reactive(None)

    def __init__(
        self,
        driver_class: Type[Driver] | None = None,
        css_path: CSSPathType | None = None,
        watch_css: bool = False,
    ):
        super().__init__(driver_class, css_path, watch_css)
        self.dark = True

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
            pass
        except Exception as e:
            logger.error(f"Error updating schema view collections: {e}", exc_info=True)

        # Removed update for DataExtractionView as it's being removed
        # try:
        #     extract_view = self.query_one(DataExtractionView)
        #     if extract_view.is_mounted:
        #         extract_view.update_collection_select()
        # except NoMatches:
        #     pass
        # except Exception as e:
        #     logger.error(f"Error updating extraction view collections: {e}", exc_info=True)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Tabs(
            Tab(label="Chat", id="tab_chat"),
            Tab(label="DB Connection", id="tab_db_connection"),
            Tab(label="Schema Analysis", id="tab_schema_analysis"),
            # Removed Data Extraction Tab
        )
        with ContentSwitcher(initial="view_chat_content"):
            yield ChatView(id="view_chat_content")
            yield DBConnectionView(id="view_db_connection_content")
            yield SchemaAnalysisView(id="view_schema_analysis_content")
            # Removed DataExtractionView instance
        yield Footer()

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        view_id_suffix: str | None = None
        if event.tab and event.tab.id:
            try:
                suffix = event.tab.id.split("_", 1)[1]
                view_id_suffix = f"view_{suffix}_content"
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
        except Exception as e:
            logger.error(f"Error setting ContentSwitcher to '{view_id_suffix}': {e}", exc_info=True)
            return

        active_view_id_for_logging = view_id_suffix
        try:
            active_view_id = switcher.current
            if active_view_id:
                active_view_id_for_logging = active_view_id
                active_view = switcher.get_widget_by_id(active_view_id)
                if (
                    isinstance(active_view, ChatView)
                    or isinstance(active_view, DBConnectionView)
                    or isinstance(active_view, SchemaAnalysisView)
                    # Removed DataExtractionView from isinstance check
                ):
                    active_view.focus_default_widget()
            else:
                logger.warning(
                    f"ContentSwitcher has no current widget ID after attempting to set to {view_id_suffix}"
                )
        except NoMatches:
            logger.debug(
                f"No widget with ID '{active_view_id_for_logging}' found in ContentSwitcher for focusing, or no default focus target in it."
            )
        except Exception as e:
            logger.error(
                f"Error focusing input in view '{active_view_id_for_logging}': {e}", exc_info=True
            )

    async def action_app_copy(self) -> None:
        focused_widget = self.focused
        text_to_copy = None

        if isinstance(focused_widget, Input):
            if hasattr(focused_widget, "selected_text") and focused_widget.selected_text:
                text_to_copy = focused_widget.selected_text

        if text_to_copy is not None:
            try:
                self.copy_to_clipboard(text_to_copy)
                logger.info(f"Copied to system clipboard: '{text_to_copy[:100]}...'")
                self.notify("Selected text copied to clipboard.", title="Copy Success")
            except Exception as e:
                logger.error(f"Failed to copy to system clipboard: {e}")
                self.notify("Failed to copy text.", title="Copy Error", severity="error")
        else:
            logger.info("No text selected in the focused widget to copy.")

    async def action_app_paste(self) -> None:
        focused_widget = self.focused

        if isinstance(focused_widget, Input):
            try:
                await self.request_paste_from_clipboard()
                logger.info("Paste requested for focused widget.")
            except AttributeError:
                logger.error(
                    "App.request_paste_from_clipboard() is not available in this Textual version. "
                    "Try using the terminal's native paste (e.g., Ctrl+Shift+V or Shift+Insert directly)."
                )
                self.notify(
                    "Programmatic paste (Shift+Ins) requires a newer Textual version or specific terminal support. "
                    "Try your terminal's native paste shortcut (e.g., Ctrl+Shift+V).",
                    title="Paste Info",
                    severity="warning",
                    timeout=8.0,
                )
            except Exception as e:
                logger.error(f"Error requesting paste from clipboard: {e}")
                self.notify(f"Error during paste: {e}", title="Paste Error", severity="error")
        else:
            logger.info("Paste action ignored: focused widget is not an input field.")

    def action_toggle_theme(self) -> None:
        self.dark = not self.dark
        logger.info(f"Textual dark mode toggled to: {self.dark}")


def main_interactive_tui():
    log_file_path = "mongo_analyser_tui.log"

    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
        )
    )

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "DEBUG").upper(),
        handlers=[file_handler],
        force=True,
    )

    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.WARNING)

    logger.info(f"Starting Mongo Analyser TUI. Logging to {Path(log_file_path).resolve()}")
    try:
        app = MongoAnalyserApp()
        app.run()
    except Exception:
        logger.critical("MongoAnalyserApp failed to run.", exc_info=True)
    finally:
        logger.info("Mongo Analyser TUI finished.")
        core_db_manager.disconnect_all_mongo()


if __name__ == "__main__":
    main_interactive_tui()
