import logging
import os
from typing import Any, Dict, List

from rich.text import Text  # For styled status
from textual.app import ComposeResult
from textual.containers import Container  # Removed Horizontal as it's not directly used
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Label, Static
from textual.worker import Worker

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.core.shared import redact_uri_password

logger = logging.getLogger(__name__)


def _format_bytes_tui(size_bytes: Any) -> str:
    # (Keep the existing _format_bytes_tui function here)
    import math

    if size_bytes is None or not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    try:
        i = int(math.floor(math.log(size_bytes, 1024)))
        if i >= len(size_name):
            i = len(size_name) - 1
    except ValueError:
        i = 0
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


class DBConnectionView(Container):
    connection_status = reactive(Text.from_markup("[#D08770]Not Connected[/]"))

    def compose(self) -> ComposeResult:
        yield Label("MongoDB Connection", classes="panel_title")
        yield Label("MongoDB URI:")
        yield Input(
            id="mongo_uri_input", value=os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        )
        yield Label("Database Name (for connection context):")
        yield Input(id="mongo_db_name_input", placeholder="e.g., my_database (optional if in URI)")
        yield Button("Connect & List Collections", variant="primary", id="connect_mongo_button")
        yield Static(self.connection_status, id="mongo_connection_status_label")

        # Potentially use CollectionStatsWidget here
        yield Label("Collections:", classes="panel_title_small", id="collections_title_label")
        yield DataTable(
            id="collections_data_table",
            show_header=True,
            show_cursor=True,
            classes="collections_list_container",  # Ensure this class is in app.tcss
        )

    def on_mount(self) -> None:
        self.query_one("#collections_title_label", Label).visible = False
        table = self.query_one("#collections_data_table", DataTable)
        table.add_columns("Name", "Docs", "Avg Size", "Total Size", "Storage Size")
        table.visible = False

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#mongo_uri_input", Input).focus()
        except NoMatches:
            logger.debug("DBConnectionView: Could not focus default input.")

    def watch_connection_status(self, new_status: Text) -> None:
        if self.is_mounted:
            try:
                self.query_one("#mongo_connection_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #mongo_connection_status_label.")

    async def _connect_and_list_collections_task(
        self,
    ) -> tuple[bool, Text, List[Dict[str, Any]], str | None, str | None]:
        uri = self.query_one("#mongo_uri_input", Input).value
        db_name_from_input = self.query_one("#mongo_db_name_input", Input).value.strip() or None

        collections_with_stats: List[Dict[str, Any]] = []
        self.app.available_collections = []

        core_db_manager.disconnect_mongo(core_db_manager.DEFAULT_ALIAS)
        is_active = core_db_manager.db_connection_active(
            uri=uri, db_name=db_name_from_input, server_timeout_ms=3000
        )

        if not is_active:
            return (
                False,
                Text.from_markup(f"[#BF616A]Connection Failed to {redact_uri_password(uri)}[/]"),
                [],
                None,
                None,
            )

        db_instance = core_db_manager.get_mongo_db()
        actual_db_name = db_instance.name
        collection_names_list: List[str] = []
        try:
            fetched_names = sorted(db_instance.list_collection_names())
            for name in fetched_names:
                collection_names_list.append(name)
                try:
                    coll_stats = db_instance.command("collStats", name)
                    collections_with_stats.append(
                        {
                            "name": name,
                            "count": coll_stats.get("count", "N/A"),
                            "avgObjSize": coll_stats.get("avgObjSize", "N/A"),
                            "size": coll_stats.get("size", "N/A"),
                            "storageSize": coll_stats.get("storageSize", "N/A"),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not get stats for collection {name}: {e}")
                    collections_with_stats.append(
                        {
                            "name": name,
                            "count": "N/A",
                            "avgObjSize": "N/A",
                            "size": "N/A",
                            "storageSize": "N/A",
                        }
                    )
            self.app.available_collections = collection_names_list
        except Exception as e:
            logger.error(f"Failed to list collections or their stats: {e}", exc_info=True)
            return (
                False,
                Text.from_markup(
                    f"[#BF616A]Connected, but failed to list collections/stats for {actual_db_name}: {e}[/]"
                ),
                [],
                uri,
                actual_db_name,
            )

        status_message = Text.from_markup(
            f"[#A3BE8C]Connected to {redact_uri_password(uri)} (DB: {actual_db_name})[/]"
        )
        return True, status_message, collections_with_stats, uri, actual_db_name

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect_mongo_button":
            collections_table = self.query_one("#collections_data_table", DataTable)
            title_label = self.query_one("#collections_title_label", Label)

            self.connection_status = Text.from_markup("[#EBCB8B]Connecting...[/]")
            self.app.available_collections = []  # Triggers watch method
            collections_table.clear()
            title_label.visible = False
            collections_table.visible = False

            try:
                worker: Worker[tuple[bool, Text, List[Dict[str, Any]], str | None, str | None]] = (
                    self.app.run_worker(
                        self._connect_and_list_collections_task, thread=True, group="db_connect"
                    )
                )
                (
                    success,
                    status_msg_text,
                    collections_stats_data,
                    connected_uri,
                    connected_db_name,
                ) = await worker.wait()

                self.connection_status = status_msg_text  # This will be styled Text
                if success and connected_uri and connected_db_name:
                    self.app.current_mongo_uri = connected_uri
                    self.app.current_db_name = connected_db_name

                    if collections_stats_data:
                        title_label.visible = True
                        collections_table.visible = True
                        for coll_data in collections_stats_data:
                            collections_table.add_row(
                                coll_data["name"],
                                str(coll_data["count"]),
                                _format_bytes_tui(coll_data["avgObjSize"]),
                                _format_bytes_tui(coll_data["size"]),
                                _format_bytes_tui(coll_data["storageSize"]),
                            )
                    else:  # Success but no collections
                        title_label.visible = True
                        collections_table.visible = True  # Show table with message
                        collections_table.add_row("No collections found in DB.", "", "", "", "")
                else:  # Failure
                    self.app.current_mongo_uri = None
                    self.app.current_db_name = None
                    # self.app.available_collections = [] # Already cleared
                    collections_table.clear()
                    collections_table.add_row(
                        status_msg_text.plain, "", "", "", ""
                    )  # Show error in table
                    title_label.visible = True
                    collections_table.visible = True
            except Exception as e:
                logger.error(f"DB Connection Op Error: {e}", exc_info=True)
                err_text = Text.from_markup(f"[#BF616A]Error during connection: {str(e)[:100]}[/]")
                self.connection_status = err_text
                collections_table.clear()
                collections_table.add_row(err_text.plain, "", "", "", "")
                self.app.current_mongo_uri = None
                self.app.current_db_name = None
                title_label.visible = True
                collections_table.visible = True
