import logging
import os
from pathlib import Path
from typing import List, Type

from textual._path import CSSPathType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.driver import Driver
from textual.reactive import reactive
from textual.widgets import (
    ContentSwitcher,
    Footer,
    Header,
    Tab,
    Tabs,
)  # Added Input, Select for focus

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.views import (
    ChatView,
    DataExtractionView,
    DBConnectionView,
    SchemaAnalysisView,
)

logger = logging.getLogger(__name__)


class MongoAnalyserApp(App[None]):
    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, key_display=None, priority=True),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme (Textual Default)", show=True),
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
        # ansi_color: bool = False, # Deprecated in newer Textual versions
    ):
        super().__init__(driver_class, css_path, watch_css)  # Removed ansi_color
        self.dark = True  # Default to dark theme for Textual's built-in toggle

    def watch_available_collections(
        self, old_collections: List[str], new_collections: List[str]
    ) -> None:
        logger.debug(
            f"App: available_collections changed from {old_collections} to: {new_collections}"
        )
        try:
            schema_view = self.query_one(SchemaAnalysisView)
            if schema_view.is_mounted:
                schema_view.update_collection_select()
        except NoMatches:
            pass  # View might not be active or exist yet
        except Exception as e:
            logger.error(f"Error updating schema view collections: {e}", exc_info=True)

        try:
            extract_view = self.query_one(DataExtractionView)
            if extract_view.is_mounted:
                extract_view.update_collection_select()
        except NoMatches:
            pass
        except Exception as e:
            logger.error(f"Error updating extraction view collections: {e}", exc_info=True)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Tabs(
            Tab(label="Chat", id="tab_chat"),
            Tab(label="DB Connection", id="tab_db_connection"),
            Tab(label="Schema Analysis", id="tab_schema_analysis"),
            Tab(label="Data Extraction", id="tab_data_extraction"),
        )
        with ContentSwitcher(initial="view_chat_content"):  # Ensure initial matches an ID
            yield ChatView(id="view_chat_content")
            yield DBConnectionView(id="view_db_connection_content")
            yield SchemaAnalysisView(id="view_schema_analysis_content")
            yield DataExtractionView(id="view_data_extraction_content")
        yield Footer()

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        view_id_suffix: str | None = None
        if event.tab and event.tab.id:
            try:
                # Assuming tab IDs are like "tab_chat", "tab_db_connection"
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

        try:
            switcher = self.query_one(ContentSwitcher)
            switcher.current = view_id_suffix
        except NoMatches:
            logger.error(
                f"View '{view_id_suffix}' not found in ContentSwitcher or ContentSwitcher itself not found."
            )
            return
        except Exception as e:
            logger.error(f"Error setting ContentSwitcher to '{view_id_suffix}': {e}", exc_info=True)
            return

        # Focus appropriate input based on the newly activated view
        try:
            active_view = switcher.current_widget
            if (
                isinstance(active_view, ChatView)
                or isinstance(active_view, DBConnectionView)
                or isinstance(active_view, SchemaAnalysisView)
                or isinstance(active_view, DataExtractionView)
            ):
                active_view.focus_default_widget()
        except NoMatches:
            logger.debug(f"No default focus target found in view '{view_id_suffix}'.")
        except Exception as e:
            logger.error(f"Error focusing input in view '{view_id_suffix}': {e}", exc_info=True)

    def action_toggle_theme(self) -> None:
        self.dark = not self.dark
        logger.info(f"Textual dark mode toggled to: {self.dark}")


def main_interactive_tui():  # Renamed to avoid conflict if run directly
    log_file_path = "mongo_analyser_tui.log"
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "DEBUG").upper(),
        filename=log_file_path,
        filemode="a",
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

    logger.info(
        f"Starting Mongo Analyser TUI (Refactored). Logging to {Path(log_file_path).resolve()}"
    )
    try:
        app = MongoAnalyserApp()
        app.run()
    except Exception as e:
        logger.critical("MongoAnalyserApp (Refactored) failed to run.", exc_info=True)
        print(f"A critical error occurred: {e}. Check logs at {Path(log_file_path).resolve()}")
    finally:
        logger.info("Mongo Analyser TUI (Refactored) finished.")
        core_db_manager.disconnect_all_mongo()


if __name__ == "__main__":
    # This would typically be called from cli.py now
    main_interactive_tui()
