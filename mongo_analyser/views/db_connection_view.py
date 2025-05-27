import logging
import os
from typing import Any, Dict, List

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Label, Static
from textual.worker import Worker  # Ensure Worker is imported

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.core.shared import redact_uri_password

logger = logging.getLogger(__name__)


def _format_bytes_tui(size_bytes: Any) -> str:
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
    except ValueError:  # math domain error for log(0) or negative
        if size_bytes > 0:  # For very small positive numbers that might cause issues with log
            i = 0
        else:  # Should be caught by initial checks, but as a safeguard
            return "N/A"

    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


class DBConnectionView(Container):
    connection_status = reactive(Text.from_markup("[#D08770]Not Connected[/]"))

    def compose(self) -> ComposeResult:
        yield Label("MongoDB URI:")
        yield Input(
            id="mongo_uri_input", value=os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        )
        yield Label("Database Name (for connection context):")
        yield Input(id="mongo_db_name_input", placeholder="e.g., my_database (optional if in URI)")
        yield Button("Connect to DB", variant="primary", id="connect_mongo_button")
        yield Static(self.connection_status, id="mongo_connection_status_label")

        yield Label(
            "Collections in the DB:",
            classes="panel_title_small",
            id="collections_title_label",
        )
        yield DataTable(
            id="collections_data_table",
            show_header=True,
            show_cursor=True,
            classes="collections_list_container",
        )

    def on_mount(self) -> None:
        self.query_one("#collections_title_label", Label).visible = False
        table = self.query_one("#collections_data_table", DataTable)
        # Add "Indexes" column
        table.add_columns("Name", "Docs", "Avg Size", "Total Size", "Storage Size", "Indexes")
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
        self.app.available_collections = []  # Reset available collections

        # Ensure any previous connection with the default alias is disconnected before a new attempt
        core_db_manager.disconnect_mongo(core_db_manager.DEFAULT_ALIAS)
        is_active = core_db_manager.db_connection_active(
            uri=uri,
            db_name=db_name_from_input,
            server_timeout_ms=3000,  # Use a reasonable timeout
        )

        if not is_active:
            return (
                False,
                Text.from_markup(f"[#BF616A]Connection Failed to {redact_uri_password(uri)}[/]"),
                [],
                None,  # No URI if connection failed
                None,  # No DB name if connection failed
            )

        # If connection is active, get the database instance
        db_instance = core_db_manager.get_mongo_db()  # Uses DEFAULT_ALIAS
        actual_db_name = db_instance.name  # Get the actual DB name from the connection
        collection_names_list: List[str] = []
        try:
            fetched_names = sorted(db_instance.list_collection_names())
            for name in fetched_names:
                collection_names_list.append(name)
                try:
                    # Fetch collStats for each collection
                    coll_stats = db_instance.command("collStats", name)
                    collections_with_stats.append(
                        {
                            "name": name,
                            "count": coll_stats.get("count", "N/A"),
                            "avgObjSize": coll_stats.get("avgObjSize", "N/A"),
                            "size": coll_stats.get("size", "N/A"),
                            "storageSize": coll_stats.get("storageSize", "N/A"),
                            "nindexes": coll_stats.get("nindexes", "N/A"),  # Get nindexes
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not get stats for collection {name}: {e}")
                    collections_with_stats.append(  # Add with N/A if stats fail
                        {
                            "name": name,
                            "count": "N/A",
                            "avgObjSize": "N/A",
                            "size": "N/A",
                            "storageSize": "N/A",
                            "nindexes": "N/A",
                        }
                    )
            self.app.available_collections = collection_names_list  # Update app state
        except Exception as e:
            logger.error(f"Failed to list collections or their stats: {e}", exc_info=True)
            # Return success=False but with URI and DB name as connection itself was okay
            return (
                False,  # Indicate overall failure in this step
                Text.from_markup(
                    f"[#BF616A]Connected, but failed to list collections/stats for {actual_db_name}: {str(e)[:50]}[/]"
                ),
                [],  # No collection data
                uri,  # URI was valid
                actual_db_name,  # DB name was valid
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
            self.app.available_collections = []
            self.app.active_collection = None  # Reset active collection
            collections_table.clear()  # Clear previous rows
            # Columns are set in on_mount, no need to re-add unless they change dynamically

            title_label.visible = False  # Hide until data is ready
            collections_table.visible = False

            try:
                # Type hint for the worker's return value
                worker: Worker[tuple[bool, Text, List[Dict[str, Any]], str | None, str | None]] = (
                    self.app.run_worker(
                        self._connect_and_list_collections_task, thread=True, group="db_connect"
                    )
                )
                (
                    success,  # Overall success of the task (connection AND listing stats)
                    status_msg_text,
                    collections_stats_data,
                    connected_uri,  # URI used for the successful connection attempt
                    connected_db_name,  # DB name from the successful connection
                ) = await worker.wait()

                self.connection_status = (
                    status_msg_text  # Update status label regardless of full success
                )

                if success and connected_uri and connected_db_name:  # Task fully succeeded
                    self.app.current_mongo_uri = connected_uri
                    self.app.current_db_name = connected_db_name
                    # self.app.available_collections is updated within the task

                    if collections_stats_data:
                        title_label.visible = True
                        collections_table.visible = True
                        for coll_data in collections_stats_data:
                            collections_table.add_row(
                                coll_data["name"],
                                str(coll_data.get("count", "N/A")),  # Use .get for safety
                                _format_bytes_tui(coll_data.get("avgObjSize")),
                                _format_bytes_tui(coll_data.get("size")),
                                _format_bytes_tui(coll_data.get("storageSize")),
                                str(coll_data.get("nindexes", "N/A")),  # Add nindexes to the row
                                key=coll_data["name"],  # Use name as row key
                            )
                    else:  # Connected, but no collections found or stats failed for all
                        title_label.visible = True
                        collections_table.visible = True
                        collections_table.add_row(
                            "No collections found or stats unavailable.", "", "", "", "", ""
                        )

                # Handle cases where connection was okay but listing collections/stats failed
                elif (
                    connected_uri and connected_db_name
                ):  # URI and DB name are set, but success is False
                    self.app.current_mongo_uri = connected_uri  # Store URI and DB name
                    self.app.current_db_name = connected_db_name
                    # self.app.available_collections might be empty or partially filled if error occurred mid-loop
                    title_label.visible = True
                    collections_table.visible = True
                    collections_table.add_row(
                        status_msg_text.plain if status_msg_text else "Error listing collections.",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )

                else:  # Connection itself failed (connected_uri or connected_db_name is None)
                    self.app.current_mongo_uri = None
                    self.app.current_db_name = None
                    # self.app.available_collections is already reset
                    title_label.visible = True  # Show title even for error message in table
                    collections_table.visible = True
                    collections_table.add_row(
                        status_msg_text.plain if status_msg_text else "Connection failed.",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )

            except Exception as e:
                logger.error(f"DB Connection Operation Error: {e}", exc_info=True)
                err_text = Text.from_markup(f"[#BF616A]Error: {str(e)[:100]}[/]")
                self.connection_status = err_text
                collections_table.clear()
                collections_table.add_row(err_text.plain, "", "", "", "", "")
                self.app.current_mongo_uri = None
                self.app.current_db_name = None
                title_label.visible = True
                collections_table.visible = True

    @on(DataTable.RowSelected)
    def on_collection_selected(self, event: DataTable.RowSelected) -> None:
        if event.control.id == "collections_data_table":
            if event.row_key and event.row_key.value:  # Check if row_key and its value exist
                selected_collection_name = str(event.row_key.value)
                self.app.active_collection = selected_collection_name
                logger.info(
                    f"DBConnectionView: Collection '{selected_collection_name}' set as active."
                )
                self.app.notify(f"Collection '{selected_collection_name}' set as active.")
            else:
                logger.warning(
                    "DBConnectionView: RowSelected event triggered with no row_key value (empty table or unkeyed row)."
                )
