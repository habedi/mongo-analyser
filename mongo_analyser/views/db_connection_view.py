import functools
import logging
from typing import Any, Dict, List, Optional, Tuple

from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure
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


def _is_auth_error(e: PyMongoOperationFailure) -> bool:
    """Checks if a PyMongoOperationFailure is likely an authorization error."""
    if e.code == 13:
        return True
    error_msg_lower = str(e).lower()
    if "not authorized" in error_msg_lower or "unauthorized" in error_msg_lower:
        return True
    return False


class DBConnectionView(Container):
    connection_status = reactive(Text.from_markup("[#D08770]Not Connected[/]"))

    def compose(self) -> ComposeResult:
        yield Label("MongoDB URI:")
        yield Input(id="mongo_uri_input", value="mongodb://localhost:27017/")
        yield Label("Database Name (optional, overrides URI's DB if specified):")
        yield Input(id="mongo_db_name_input", placeholder="my_database")
        yield Button("Connect to DB", variant="primary", id="connect_mongo_button")
        yield Static(self.connection_status, id="mongo_connection_status_label")

        yield Label(
            "Collections in Database (Click to Select):",
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

        try:
            uri_input_widget = self.query_one("#mongo_uri_input", Input)
            db_name_input_widget = self.query_one("#mongo_db_name_input", Input)

            initial_uri_from_app = getattr(self.app, "_initial_mongo_uri", None)
            initial_db_name_from_app = getattr(self.app, "_initial_db_name", None)

            if initial_uri_from_app:
                uri_input_widget.value = initial_uri_from_app
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "DBConnectionView pre-filled Mongo URI from initial config: %s",
                        redact_uri_password(initial_uri_from_app),
                    )

            if initial_db_name_from_app:
                db_name_input_widget.value = initial_db_name_from_app
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "DBConnectionView pre-filled Mongo Database Name from initial config: %s",
                        initial_db_name_from_app,
                    )

        except NoMatches:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "DBConnectionView: URI or DB Name input widgets not found on mount for pre-filling."
                )
        except Exception as e:
            if logger.isEnabledFor(logging.ERROR):
                logger.error(
                    "Error pre-filling DBConnectionView inputs from initial app config: %s",
                    e,
                    exc_info=True,
                )

        self.focus_default_widget()

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#mongo_uri_input", Input).focus()
        except NoMatches:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("DBConnectionView: Could not focus default input '#mongo_uri_input'.")

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
        final_status_message: Text
        connection_is_meaningful_for_analyser = False
        redacted_uri_for_log = redact_uri_password(uri)
        self._last_connect_error_msg = ""

        if not core_db_manager.db_connection_active(
            uri=uri, db_name=db_name_from_input, server_timeout_ms=3000, force_reconnect=True
        ):
            if "Cannot determine database context" in getattr(
                self, "_internal_db_conn_error_reason", ""
            ):
                final_status_message = Text.from_markup(
                    f"[#BF616A]Connection Failed: MongoDB URI ('{redacted_uri_for_log}') does not specify a default database path, "
                    "and no database name was provided. Please specify a database."
                )
                self._last_connect_error_msg = final_status_message.plain
            else:
                final_status_message = Text.from_markup(
                    f"[#BF616A]Connection Failed: Could not connect to MongoDB server at {redacted_uri_for_log}[/]"
                )
                self._last_connect_error_msg = final_status_message.plain
            return False, final_status_message, [], None, None

        client = core_db_manager.get_mongo_client()
        db_instance = core_db_manager.get_mongo_db()

        if client is None or db_instance is None:
            final_status_message = Text.from_markup(
                "[#BF616A]Internal Error: Could not retrieve active MongoDB client/db after connection attempt.[/]"
            )
            self._last_connect_error_msg = final_status_message.plain
            return (
                False,
                final_status_message,
                [],
                core_db_manager.get_current_uri(),
                core_db_manager.get_current_resolved_db_name(),
            )

        actual_db_name = db_instance.name
        connected_uri = core_db_manager.get_current_uri()

        try:
            collection_names_list = sorted(db_instance.list_collection_names())
            connection_is_meaningful_for_analyser = True

            if collection_names_list:
                for name in collection_names_list:
                    coll_stat_entry = {
                        "name": name,
                        "count": "N/A",
                        "avgObjSize": "N/A",
                        "size": "N/A",
                        "storageSize": "N/A",
                        "nindexes": "N/A",
                    }
                    try:
                        coll_stats = db_instance.command("collStats", name)
                        coll_stat_entry.update(
                            {
                                "count": coll_stats.get("count", "N/A"),
                                "avgObjSize": _format_bytes_tui(coll_stats.get("avgObjSize")),
                                "size": _format_bytes_tui(coll_stats.get("size")),
                                "storageSize": _format_bytes_tui(coll_stats.get("storageSize")),
                                "nindexes": str(coll_stats.get("nindexes", "N/A")),
                            }
                        )
                    except PyMongoOperationFailure as e_stats:
                        logger.warning(
                            "Could not get stats for collection '%s' in DB '%s': %s",
                            name,
                            actual_db_name,
                            e_stats,
                        )
                        if _is_auth_error(e_stats):
                            coll_stat_entry.update(
                                {
                                    k: "Unauthorized"
                                    for k in [
                                        "count",
                                        "avgObjSize",
                                        "size",
                                        "storageSize",
                                        "nindexes",
                                    ]
                                }
                            )

                        else:
                            coll_stat_entry.update(
                                {
                                    k: "Err(Stats)"
                                    for k in [
                                        "count",
                                        "avgObjSize",
                                        "size",
                                        "storageSize",
                                        "nindexes",
                                    ]
                                }
                            )
                    except Exception as e_stats_other:
                        logger.warning(
                            "Unexpected error getting stats for collection '%s' in DB '%s': %s",
                            name,
                            actual_db_name,
                            e_stats_other,
                            exc_info=True,
                        )
                        coll_stat_entry.update(
                            {
                                k: "Err(Stats)"
                                for k in ["count", "avgObjSize", "size", "storageSize", "nindexes"]
                            }
                        )
                    collections_with_stats.append(coll_stat_entry)
                final_status_message = Text.from_markup(
                    f"[#A3BE8C]Connected to {redact_uri_password(connected_uri or 'unknown URI')} (DB: {actual_db_name}). "
                    f"{len(collection_names_list)} collection(s) found.[/]"
                )
            else:
                final_status_message = Text.from_markup(
                    f"[#A3BE8C]Connected to DB: '{actual_db_name}' at {redact_uri_password(connected_uri or 'unknown URI')}. "
                    f"This database is empty (no collections).[/]"
                )
        except PyMongoOperationFailure as e_list_coll:
            connection_is_meaningful_for_analyser = False

            logger.error(
                "MongoDB operation failure listing collections for DB '%s': %s",
                actual_db_name,
                e_list_coll,
                exc_info=not _is_auth_error(e_list_coll),
            )
            if _is_auth_error(e_list_coll):
                final_status_message = Text.from_markup(
                    f"[#BF616A]Connected to server ({redact_uri_password(connected_uri or 'unknown URI')}). However, user is not authorized "
                    f"to list collections in database '{actual_db_name}'. Please check permissions.[/]"
                )
            else:
                err_details = (
                    e_list_coll.details.get("errmsg", str(e_list_coll))
                    if e_list_coll.details
                    else str(e_list_coll)
                )
                final_status_message = Text.from_markup(
                    f"[#BF616A]Connected ({redact_uri_password(connected_uri or 'unknown URI')}, DB: {actual_db_name}), but an error occurred "
                    f"listing collections: {str(err_details)[:70]}[/]"
                )
        except Exception as e_generic_list_coll:
            connection_is_meaningful_for_analyser = False
            logger.error(
                "Unexpected error listing collections for DB '%s': %s",
                actual_db_name,
                e_generic_list_coll,
                exc_info=True,
            )
            final_status_message = Text.from_markup(
                f"[#BF616A]Connected ({redact_uri_password(connected_uri or 'unknown URI')}, DB: {actual_db_name}), but an unexpected error "
                f"occurred while listing collections: {str(e_generic_list_coll)[:70]}[/]"
            )

        if not connection_is_meaningful_for_analyser:
            self._last_connect_error_msg = (
                final_status_message.plain if final_status_message else "Unknown connection error"
            )

        return (
            connection_is_meaningful_for_analyser,
            final_status_message,
            collections_with_stats,
            connected_uri,
            actual_db_name,
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect_mongo_button":
            collections_table = self.query_one("#collections_data_table", DataTable)
            collections_title = self.query_one("#collections_title_label", Label)
            indexes_table = self.query_one("#indexes_data_table", DataTable)
            indexes_title = self.query_one("#indexes_title_label", Label)

            uri_input_value = self.query_one("#mongo_uri_input", Input).value
            db_name_input_value = (
                self.query_one("#mongo_db_name_input", Input).value.strip() or None
            )

            self.connection_status = Text.from_markup("[#EBCB8B]Connecting...[/]")
            self.app.available_collections = []
            self.app.active_collection = None

            collections_table.clear()
            indexes_table.clear()
            collections_title.visible = False
            collections_table.visible = False
            indexes_title.visible = False
            indexes_table.visible = False

            try:
                task_with_args = functools.partial(
                    self._connect_and_list_collections_task, uri_input_value, db_name_input_value
                )

                worker_result_type = Tuple[
                    bool, Text, List[Dict[str, Any]], Optional[str], Optional[str]
                ]
                worker: Worker[worker_result_type] = self.app.run_worker(
                    task_with_args,
                    thread=True,
                    name=f"connect_worker_{db_name_input_value or 'from_uri'}",
                    group="db_operations",
                    description=f"Connecting to MongoDB: {redact_uri_password(uri_input_value)}",
                )
                (
                    task_success,
                    status_msg_text,
                    collections_stats_data,
                    connected_uri_result,
                    connected_db_name_actual,
                ) = await worker.wait()

                if worker.is_cancelled:
                    self.connection_status = Text.from_markup(
                        "[#D08770]Connection attempt cancelled.[/]"
                    )
                    self.app.current_mongo_uri = None
                    self.app.current_db_name = None
                    self.app.available_collections = []
                    self.app.active_collection = None
                    return

                self.connection_status = status_msg_text

                if task_success and connected_uri_result and connected_db_name_actual:
                    self.app.current_mongo_uri = connected_uri_result
                    self.app.current_db_name = connected_db_name_actual
                    self.app.available_collections = [
                        item["name"] for item in collections_stats_data
                    ]

                    collections_title.visible = True
                    collections_table.visible = True
                    if collections_stats_data:
                        for coll_data in collections_stats_data:
                            collections_table.add_row(
                                coll_data["name"],
                                str(coll_data.get("count", "N/A")),
                                coll_data.get("avgObjSize", "N/A"),
                                coll_data.get("size", "N/A"),
                                coll_data.get("storageSize", "N/A"),
                                str(coll_data.get("nindexes", "N/A")),
                                key=coll_data["name"],
                            )
                    else:
                        no_coll_msg = "No collections found in this database."
                        if status_msg_text and (
                            "empty" in status_msg_text.plain.lower()
                            or "no collections" in status_msg_text.plain.lower()
                        ):
                            no_coll_msg = status_msg_text.plain
                        collections_table.add_row(no_coll_msg, "", "", "", "", "")

                elif connected_uri_result and connected_db_name_actual:
                    self.app.current_mongo_uri = connected_uri_result
                    self.app.current_db_name = connected_db_name_actual
                    self.app.available_collections = []
                    self.app.active_collection = None
                    collections_title.visible = True
                    collections_table.visible = True
                    collections_table.add_row(
                        status_msg_text.plain if status_msg_text else "Problematic DB context.",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )
                else:
                    self.app.current_mongo_uri = None
                    self.app.current_db_name = None
                    self.app.available_collections = []
                    self.app.active_collection = None
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
                self.connection_status = Text.from_markup(
                    "[#D08770]Connection task was cancelled.[/]"
                )
            except Exception as e:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error("DB Connection Operation Error in View: %s", e, exc_info=True)
                err_text_display = Text.from_markup(f"[#BF616A]Error: {str(e)[:100]}[/]")
                self.connection_status = err_text_display
                collections_table.clear()
                collections_table.add_row(
                    err_text_display.plain if err_text_display else "Connection Error",
                    "",
                    "",
                    "",
                    "",
                    "",
                )
                collections_title.visible = True
                collections_table.visible = True
                await self.app.push_screen(ErrorDialog("Connection Error", str(e)))

    @on(DataTable.RowSelected, "#collections_data_table")
    async def on_collection_selected_in_table(self, event: DataTable.RowSelected) -> None:
        if event.control.id != "collections_data_table":
            return

        indexes_table = self.query_one("#indexes_data_table", DataTable)
        indexes_title = self.query_one("#indexes_title_label", Label)

        if not event.row_key or not event.row_key.value:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Collection selection event with no row key or value.")
            indexes_table.clear()
            indexes_title.visible = False
            indexes_table.visible = False
            if self.app.active_collection is not None:
                self.app.active_collection = None
            return

        selected_collection_name = str(event.row_key.value)

        if selected_collection_name not in self.app.available_collections:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Clicked row '%s' is not an available collection name (e.g., 'Unauthorized' or message row). Ignoring for index loading.",
                    selected_collection_name,
                )

            indexes_table.clear()
            indexes_title.visible = False
            indexes_table.visible = False
            if self.app.active_collection is not None:
                self.app.active_collection = None
            return

        if selected_collection_name == self.app.active_collection:
            return

        self.app.active_collection = selected_collection_name
        await self._load_indexes_for_collection(selected_collection_name)

    async def _load_indexes_for_collection(self, collection_name: str) -> None:
        uri = self.app.current_mongo_uri
        db_name_app = self.app.current_db_name

        indexes_table = self.query_one("#indexes_data_table", DataTable)
        indexes_title = self.query_one("#indexes_title_label", Label)
        indexes_table.clear()

        if not uri or not db_name_app:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "Cannot load indexes for '%s': MongoDB not connected or DB name unknown.",
                    collection_name,
                )
            indexes_title.visible = False
            indexes_table.visible = False
            return

        indexes_title.visible = True
        indexes_table.visible = True
        indexes_table.add_row(Text("Loading indexes...", style="italic dim"), "", "", "", "", "")

        try:
            if not core_db_manager.db_connection_active(
                uri=uri, db_name=db_name_app, force_reconnect=False, server_timeout_ms=2000
            ):
                raise ConnectionError(
                    f"Failed to re-verify connection to DB '{db_name_app}' for listing indexes of '{collection_name}'."
                )

            db_instance = core_db_manager.get_mongo_db()

            if db_instance.name != db_name_app:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(
                        "DB context mismatch. Expected '%s', got '%s' for index listing of '%s'.",
                        db_name_app,
                        db_instance.name,
                        collection_name,
                    )
                raise ConnectionError(f"DB context mismatch. Expected {db_name_app}.")

            collection_obj = db_instance[collection_name]

            def list_indexes_task():
                return list(collection_obj.list_indexes())

            worker: Worker[List[Dict]] = self.app.run_worker(
                list_indexes_task,
                thread=True,
                name=f"list_indexes_{collection_name}",
                group="db_operations",
                description=f"Loading indexes for {collection_name}",
            )
            raw_indexes = await worker.wait()
            indexes_table.clear()

            if worker.is_cancelled:
                indexes_table.add_row(
                    Text("Index loading cancelled by user.", style="italic yellow"),
                    "",
                    "",
                    "",
                    "",
                    "",
                )
                return

            if not raw_indexes:
                indexes_table.add_row(
                    Text(f"No indexes found for '{collection_name}'.", style="italic"),
                    "",
                    "",
                    "",
                    "",
                    "",
                )
                return

            for idx_info in raw_indexes:
                key_dict = idx_info.get("key", {})
                key_str = ", ".join([f"{k}: {v}" for k, v in key_dict.items()])
                other_props_list = []
                excluded_fields = [
                    "v",
                    "key",
                    "name",
                    "ns",
                    "unique",
                    "sparse",
                    "background",
                    "weights",
                    "default_language",
                    "language_override",
                    "textIndexVersion",
                    "2dsphereIndexVersion",
                    "bits",
                    "min",
                    "max",
                    "bucketSize",
                ]
                for p_name, p_val in idx_info.items():
                    if p_name not in excluded_fields:
                        other_props_list.append(
                            f"{p_name}={str(p_val)[:50]}{'...' if len(str(p_val)) > 50 else ''}"
                        )
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
            indexes_table.add_row(
                Text("Index loading was cancelled during operation.", style="italic yellow"),
                "",
                "",
                "",
                "",
                "",
            )
        except PyMongoOperationFailure as e_op:
            logger.warning(
                f"MongoDB operation failure loading indexes for '{collection_name}': {e_op}"
            )
            indexes_table.clear()
            if _is_auth_error(e_op):
                msg = f"Unauthorized to list indexes for '{collection_name}'."
                indexes_table.add_row(Text(msg, style="italic red"), "", "", "", "", "")
                self.app.notify(msg, title="Permission Denied", severity="error", timeout=5)
            else:
                err_details = e_op.details.get("errmsg", str(e_op)) if e_op.details else str(e_op)
                msg = f"Error listing indexes: {str(err_details)[:70]}"
                indexes_table.add_row(Text(msg, style="italic red"), "", "", "", "", "")
                self.app.notify(
                    f"Error listing indexes for '{collection_name}'. Check logs.",
                    title="Operation Error",
                    severity="error",
                    timeout=5,
                )
        except (PyMongoConnectionFailure, ConnectionError) as e_conn:
            logger.error(
                f"Connection error loading indexes for '{collection_name}': {e_conn}", exc_info=True
            )
            indexes_table.clear()
            msg = f"Connection error loading indexes: {str(e_conn)[:70]}"
            indexes_table.add_row(Text(msg, style="italic red"), "", "", "", "", "")
            self.app.notify(
                "Connection error. Please reconnect.",
                title="Connection Error",
                severity="error",
                timeout=5,
            )
        except Exception as e:
            logger.error(
                "Unexpected error loading indexes for collection '%s': %s",
                collection_name,
                e,
                exc_info=True,
            )
            indexes_table.clear()
            indexes_table.add_row(
                Text(f"Unexpected error loading indexes: {str(e)[:70]}", style="italic red"),
                "",
                "",
                "",
                "",
                "",
            )
            await self.app.push_screen(
                ErrorDialog(
                    "Index Load Error", f"Could not load indexes for '{collection_name}': {e!s}"
                )
            )


def _format_bytes_tui(size_bytes: Any) -> str:
    import math

    if size_bytes is None or not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    try:
        if size_bytes <= 0:
            i = 0
        else:
            i = int(math.floor(math.log(size_bytes, 1024)))

        if i >= len(size_name):
            i = len(size_name) - 1
        elif i < 0:
            i = 0
    except ValueError:
        i = 0

    p = math.pow(1024, i)
    s = round(size_bytes / p, 2) if p > 0 else 0
    return f"{s} {size_name[i]}"
