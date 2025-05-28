import functools  # Import functools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Label, Static
from textual.worker import Worker, WorkerCancelled

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.core.shared import redact_uri_password
from mongo_analyser.dialogs import ErrorDialog

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
        elif i < 0:
            i = 0
    except ValueError:
        i = 0 if size_bytes > 0 else -1
        if i == -1:
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
        yield Label("Database Name (optional, overrides URI's DB if specified):")
        yield Input(id="mongo_db_name_input", placeholder="e.g., my_database")
        yield Button("Connect to DB", variant="primary", id="connect_mongo_button")
        yield Static(self.connection_status, id="mongo_connection_status_label")

        yield Label(
            "Collections in Database:",
            classes="panel_title_small",
            id="collections_title_label",
        )
        yield DataTable(
            id="collections_data_table",
            show_header=True,
            show_cursor=True,
            cursor_type="row",
            classes="collections_list_container",
        )

        yield Label(
            "Indexes for Selected Collection:",
            classes="panel_title_small",
            id="indexes_title_label",
        )
        yield DataTable(
            id="indexes_data_table",
            show_header=True,
            show_cursor=False,
        )

    def on_mount(self) -> None:
        collections_table = self.query_one("#collections_data_table", DataTable)
        if not collections_table.columns:
            collections_table.add_columns(
                "Name", "Docs", "Avg Size", "Total Size", "Storage Size", "Indexes"
            )
        collections_table.visible = False
        self.query_one("#collections_title_label", Label).visible = False

        indexes_table = self.query_one("#indexes_data_table", DataTable)
        if not indexes_table.columns:
            indexes_table.add_columns(
                "Name", "Fields (Key)", "Unique", "Sparse", "Background", "Other Props"
            )
        indexes_table.visible = False
        self.query_one("#indexes_title_label", Label).visible = False

        self.focus_default_widget()

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
                pass

    async def _connect_and_list_collections_task(
        self, uri: str, db_name_from_input: Optional[str]
    ) -> Tuple[bool, Text, List[Dict[str, Any]], Optional[str], Optional[str]]:
        collections_with_stats: List[Dict[str, Any]] = []

        is_active = core_db_manager.db_connection_active(
            uri=uri,
            db_name=db_name_from_input,
            server_timeout_ms=3000,
            force_reconnect=True,
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
                            "nindexes": coll_stats.get("nindexes", "N/A"),
                        }
                    )
                except Exception as e_stats:
                    logger.warning(f"Could not get stats for collection {name}: {e_stats}")
                    collections_with_stats.append(
                        {
                            "name": name,
                            "count": "N/A",
                            "avgObjSize": "N/A",
                            "size": "N/A",
                            "storageSize": "N/A",
                            "nindexes": "N/A",
                        }
                    )

        except Exception as e_list:
            logger.error(
                f"Failed to list collections or their stats for DB '{actual_db_name}': {e_list}",
                exc_info=True,
            )
            return (
                False,
                Text.from_markup(
                    f"[#BF616A]Connected, but failed to list collections/stats for {actual_db_name}: {str(e_list)[:50]}[/]"
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
            collections_title = self.query_one("#collections_title_label", Label)
            indexes_table = self.query_one("#indexes_data_table", DataTable)
            indexes_title = self.query_one("#indexes_title_label", Label)

            uri_input = self.query_one("#mongo_uri_input", Input).value
            db_name_input = self.query_one("#mongo_db_name_input", Input).value.strip() or None

            self.connection_status = Text.from_markup("[#EBCB8B]Connecting...[/]")
            self.app.available_collections = []
            self.app.active_collection = None
            self.app.current_mongo_uri = None
            self.app.current_db_name = None

            collections_table.clear()
            indexes_table.clear()
            collections_title.visible = False
            collections_table.visible = False
            indexes_title.visible = False
            indexes_table.visible = False

            try:
                # Use functools.partial to pre-bind arguments to the task
                task_with_args = functools.partial(
                    self._connect_and_list_collections_task, uri_input, db_name_input
                )

                worker: Worker[
                    Tuple[bool, Text, List[Dict[str, Any]], Optional[str], Optional[str]]
                ] = self.app.run_worker(
                    task_with_args,  # Pass the partial function
                    thread=True,
                    name=f"connect_worker_{db_name_input or 'default'}",
                    # Optional: explicit name for worker
                    group="db_operations",  # Optional: explicit group for worker
                )
                (
                    task_success,
                    status_msg_text,
                    collections_stats_data,
                    connected_uri,
                    connected_db_name_actual,
                ) = await worker.wait()

                if worker.is_cancelled:
                    self.connection_status = Text.from_markup(
                        "[#D08770]Connection attempt cancelled.[/]"
                    )
                    return

                self.connection_status = status_msg_text

                if task_success and connected_uri and connected_db_name_actual:
                    self.app.current_mongo_uri = connected_uri
                    self.app.current_db_name = connected_db_name_actual
                    self.app.available_collections = [
                        item["name"] for item in collections_stats_data
                    ]

                    if collections_stats_data:
                        collections_title.visible = True
                        collections_table.visible = True
                        for coll_data in collections_stats_data:
                            collections_table.add_row(
                                coll_data["name"],
                                str(coll_data.get("count", "N/A")),
                                _format_bytes_tui(coll_data.get("avgObjSize")),
                                _format_bytes_tui(coll_data.get("size")),
                                _format_bytes_tui(coll_data.get("storageSize")),
                                str(coll_data.get("nindexes", "N/A")),
                                key=coll_data["name"],
                            )
                    else:
                        collections_title.visible = True
                        collections_table.visible = True
                        collections_table.add_row(
                            "No collections found or stats unavailable.", "", "", "", "", ""
                        )

                elif connected_uri and connected_db_name_actual:
                    self.app.current_mongo_uri = connected_uri
                    self.app.current_db_name = connected_db_name_actual
                    collections_title.visible = True
                    collections_table.visible = True
                    collections_table.add_row(
                        status_msg_text.plain if status_msg_text else "Error listing collections.",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )

                else:
                    collections_title.visible = True
                    collections_table.visible = True
                    collections_table.add_row(
                        status_msg_text.plain if status_msg_text else "Connection failed.",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )

            except WorkerCancelled:
                self.connection_status = Text.from_markup("[#D08770]Connection task cancelled.[/]")
            except Exception as e:
                logger.error(f"DB Connection Operation Error: {e}", exc_info=True)
                err_text_display = Text.from_markup(f"[#BF616A]Error: {str(e)[:100]}[/]")
                self.connection_status = err_text_display
                collections_table.clear()
                collections_table.add_row(err_text_display.plain, "", "", "", "", "")
                await self.app.push_screen(ErrorDialog("Connection Error", str(e)))

    @on(DataTable.RowSelected, "#collections_data_table")
    async def on_collection_selected_in_table(self, event: DataTable.RowSelected) -> None:
        if event.control.id != "collections_data_table":
            return
        if not event.row_key or not event.row_key.value:
            return
        selected = str(event.row_key.value)
        if selected == self.app.active_collection:
            return
        self.app.active_collection = selected
        await self._load_indexes_for_collection(selected)

    async def _load_indexes_for_collection(self, collection_name: str) -> None:
        uri = self.app.current_mongo_uri
        db_name_app = self.app.current_db_name

        indexes_table = self.query_one("#indexes_data_table", DataTable)
        indexes_title = self.query_one("#indexes_title_label", Label)
        indexes_table.clear()

        if not uri or not db_name_app:
            logger.warning("Cannot load indexes, DB not connected or DB name unknown.")
            indexes_title.visible = False
            indexes_table.visible = False
            return

        indexes_title.visible = True
        indexes_table.visible = True
        indexes_table.add_row("Loading indexes...", "", "", "", "", "")

        try:
            if not core_db_manager.db_connection_active(
                uri=uri, db_name=db_name_app, force_reconnect=False
            ):
                raise ConnectionError(
                    f"Failed to re-verify connection to {db_name_app} for listing indexes."
                )

            db_instance = core_db_manager.get_mongo_db()
            if db_instance.name != db_name_app:
                logger.error(
                    f"Mismatch: Expected DB '{db_name_app}', but got '{db_instance.name}' for indexes."
                )
                raise ConnectionError(
                    f"DB context mismatch for index listing. Expected {db_name_app}."
                )

            collection_obj = db_instance[collection_name]

            # Use functools.partial for the lambda as well for consistency and clarity
            task_with_args = functools.partial(lambda: list(collection_obj.list_indexes()))

            worker: Worker[List[Dict]] = self.app.run_worker(
                task_with_args,
                thread=True,
                name=f"list_indexes_{collection_name}",  # Optional: explicit name
                group="db_operations",  # Optional: explicit group
            )
            raw_indexes = await worker.wait()
            indexes_table.clear()

            if worker.is_cancelled:
                indexes_table.add_row("Index loading cancelled.", "", "", "", "", "")
                return

            if not raw_indexes:
                indexes_table.add_row(
                    "No indexes found for this collection (excluding _id).", "", "", "", "", ""
                )
                return

            for idx_info in raw_indexes:
                key_dict = idx_info.get("key", {})
                key_str = ", ".join([f"{k}: {v}" for k, v in key_dict.items()])

                other_props_list = []
                for p_name, p_val in idx_info.items():
                    if p_name not in ["v", "key", "name", "ns", "unique", "sparse", "background"]:
                        other_props_list.append(f"{p_name}={p_val}")
                other_props_str = ", ".join(other_props_list) if other_props_list else "N/A"

                indexes_table.add_row(
                    idx_info.get("name", "N/A"),
                    key_str,
                    str(idx_info.get("unique", False)),
                    str(idx_info.get("sparse", False)),
                    str(idx_info.get("background", False)),
                    other_props_str,
                )
        except WorkerCancelled:
            indexes_table.clear()
            indexes_table.add_row("Index loading cancelled during operation.", "", "", "", "", "")
        except Exception as e:
            logger.error(f"Error loading indexes for '{collection_name}': {e}", exc_info=True)
            indexes_table.clear()
            indexes_table.add_row(f"Error: {str(e)[:70]}", "", "", "", "", "")
