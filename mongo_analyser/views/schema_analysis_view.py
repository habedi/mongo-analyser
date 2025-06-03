import csv
import functools
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    Static,
)
from textual.worker import Worker, WorkerCancelled

from mongo_analyser.core import SchemaAnalyser
from mongo_analyser.dialogs import ErrorDialog

logger = logging.getLogger(__name__)


NO_DB_CONNECTION_TEXT = Text.from_markup(
    "[#BF616A]MongoDB not connected. Please connect in the 'DB Connection' tab first.[/]"
)
NO_COLLECTION_SELECTED_TEXT = Text.from_markup(
    "[#BF616A]No collection selected. Please select a collection from the dropdown.[/]"
)
UI_ERROR_INPUT_WIDGETS_TEXT = Text.from_markup("[#BF616A]UI Error: Input widgets not found.[/]")


def _is_auth_error_from_op_failure(e: PyMongoOperationFailure) -> bool:
    if e.code == 13:
        return True
    error_msg_lower = str(e).lower()
    if "not authorized" in error_msg_lower or "unauthorized" in error_msg_lower:
        return True
    return False


class SchemaAnalysisView(Container):
    analysis_status = reactive(Text("Select a collection and click Analyze Schema"))
    schema_copy_feedback = reactive(Text(""))
    current_hierarchical_schema: Dict = {}
    _current_schema_json_str: str = "{}"
    _feedback_timer: Optional[Any] = None

    def __init__(
        self,
        *children: Widget,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        disabled: bool = False,
    ):
        super().__init__(*children, name=name, id=id, classes=classes, disabled=disabled)
        self._last_collections: List[str] = []

    def _get_default_save_path(self) -> str:
        db_name = self.app.current_db_name
        collection_name = self.app.active_collection
        if db_name and collection_name:
            return f"output/{db_name}/{collection_name}_schema.json"
        if collection_name:
            return f"output/{collection_name}_schema.json"
        return "output/default_schema.json"

    def compose(self) -> ComposeResult:
        yield Label("Collection:")
        yield Select(
            [],
            prompt="Connect to DB to see collections",
            id="schema_collection_select",
            allow_blank=True,
        )
        yield Label("Sample Size (-1 for all documents):")
        yield Input(
            id="schema_sample_size_input",
            value="1000",
            placeholder="e.g., 1000 or -1",
            tooltip="Number of documents to sample for analysis. -1 processes all documents (can be slow).",
        )
        yield Button("Analyze Schema", variant="primary", id="analyze_schema_button")
        yield LoadingIndicator(id="schema_loading_indicator")
        yield Static(self.analysis_status, id="schema_status_label")
        yield Label("Inferred Metadata:", classes="panel_title_small")
        yield DataTable(
            id="schema_results_table", show_header=True, show_cursor=True, cursor_type="row"
        )
        yield Label("Inferred Schema (JSON):", classes="panel_title_small")
        with VerticalScroll(classes="json_view_container"):
            yield Markdown("```json\n{}\n```", id="schema_json_view")

        with Horizontal(classes="copy_button_container"):
            yield Button("Copy Cell Value", id="copy_cell_button")
            yield Button("Copy Collection Schema", id="copy_json_button")
            yield Button("Copy Collection Metadata", id="copy_table_csv_button")
        yield Static(self.schema_copy_feedback, id="schema_copy_feedback_label")

        yield Label("Save File Path:", classes="panel_title_small")
        yield Input(id="schema_save_path_input", value=self._get_default_save_path())
        yield Button("Save Schema to File", id="save_schema_json_button")

    def on_mount(self) -> None:
        self._last_collections: List[str] = []
        self.update_collection_select()
        try:
            self.query_one("#schema_save_path_input", Input).value = self._get_default_save_path()
            self.query_one("#schema_loading_indicator", LoadingIndicator).display = False
        except NoMatches:
            logger.warning(
                "SchemaAnalysisView: #schema_save_path_input or #schema_loading_indicator not found on mount."
            )
        table = self.query_one("#schema_results_table", DataTable)
        if not table.columns:
            table.add_columns(
                "Field",
                "Type(s)",
                "Cardinality",
                "Missing (%)",
                "Numeric Min",
                "Numeric Max",
                "Date Min",
                "Date Max",
                "Top Values (Field)",
                "Array Elem Types",
                "Array Elem Top Values",
            )

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#schema_collection_select", Select).focus()
        except NoMatches:
            logger.debug("SchemaAnalysisView: Could not focus default select.")

    def update_collection_select(self) -> None:
        try:
            select_widget = self.query_one("#schema_collection_select", Select)
            save_path_input = self.query_one("#schema_save_path_input", Input)
            collections = self.app.available_collections

            current_selection_in_widget = (
                str(select_widget.value) if select_widget.value != Select.BLANK else None
            )
            app_active_collection = self.app.active_collection

            needs_options_update = collections != self._last_collections
            needs_value_update = (
                app_active_collection != current_selection_in_widget
                and app_active_collection is not None
                and app_active_collection in collections
            )

            if not needs_options_update and not needs_value_update:
                if app_active_collection != current_selection_in_widget:
                    save_path_input.value = self._get_path_for_collection(
                        current_selection_in_widget
                    )
                return

            if needs_options_update:
                self._last_collections = list(collections)
                if collections:
                    options = [(c, c) for c in collections]
                    select_widget.set_options(options)
                    select_widget.disabled = False
                    select_widget.prompt = "Select Collection"
                else:
                    select_widget.set_options([])
                    select_widget.prompt = "Connect to DB to see collections"
                    select_widget.disabled = True
                    select_widget.value = Select.BLANK

            if app_active_collection and app_active_collection in self._last_collections:
                select_widget.value = app_active_collection
            elif (
                current_selection_in_widget
                and current_selection_in_widget in self._last_collections
            ):
                pass
            elif self._last_collections:
                select_widget.value = Select.BLANK

            final_selection_for_path = (
                str(select_widget.value) if select_widget.value != Select.BLANK else None
            )
            save_path_input.value = self._get_path_for_collection(final_selection_for_path)

        except NoMatches:
            logger.warning(
                "SchemaAnalysisView: select or input widget not found in update_collection_select."
            )
        except Exception as e:
            logger.error(
                f"Error in SchemaAnalysisView.update_collection_select: {e}", exc_info=True
            )

    def _get_path_for_collection(self, name: Optional[str]) -> str:
        db_name = self.app.current_db_name
        if db_name and name:
            return f"output/{db_name}/{name}_schema.json"
        if name:
            return f"output/{name}_schema.json"
        return "output/default_schema.json"

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "schema_collection_select":
            return
        new_selected_collection = str(event.value) if event.value != Select.BLANK else None

        if new_selected_collection != self.app.active_collection:
            self.app.active_collection = new_selected_collection

        save_path_input = self.query_one("#schema_save_path_input", Input)
        save_path_input.value = self._get_path_for_collection(new_selected_collection)

        self._clear_analysis_results()
        self.analysis_status = Text("Collection changed. Click 'Analyze Schema'")

    def _clear_analysis_results(self):
        try:
            table = self.query_one("#schema_results_table", DataTable)
            md = self.query_one("#schema_json_view", Markdown)
            table.clear()
        except NoMatches:
            logger.warning("SchemaAnalysisView: Table or Markdown view not found during clear.")

        self.current_hierarchical_schema = {}
        self._current_schema_json_str = "{}"
        if self.is_mounted and hasattr(self, "query_one"):
            try:
                md_view_update = self.query_one("#schema_json_view", Markdown)
                md_view_update.update(f"```json\n{self._current_schema_json_str}\n```")
            except NoMatches:
                pass

        self.app.current_schema_analysis_results = None
        self.analysis_status = Text("Results cleared. Select a collection and click Analyze Schema")

    def watch_analysis_status(self, new_status: Text) -> None:
        if self.is_mounted:
            try:
                self.query_one("#schema_status_label", Static).update(new_status)
            except NoMatches:
                pass

    def watch_schema_copy_feedback(self, new_feedback: Text) -> None:
        if self.is_mounted:
            try:
                lbl = self.query_one("#schema_copy_feedback_label", Static)
                lbl.update(new_feedback)
                if self._feedback_timer is not None:
                    try:
                        self._feedback_timer.stop()
                    except AttributeError:
                        if hasattr(self._feedback_timer, "stop_no_wait"):
                            self._feedback_timer.stop_no_wait()
                    self._feedback_timer = None
                if new_feedback.plain:
                    self._feedback_timer = self.set_timer(
                        3, lambda: setattr(self, "schema_copy_feedback", Text(""))
                    )
            except NoMatches:
                logger.warning(
                    "#schema_copy_feedback_label NOT FOUND in watch_schema_copy_feedback!"
                )
            except Exception as e:
                logger.error(f"Error in watch_schema_copy_feedback: {e}", exc_info=True)

    def _validate_sample_size_input(self) -> Tuple[Optional[str], Optional[Text]]:
        """Validates sample size input. Returns (size_str, error_text_or_none)."""
        try:
            size_input_widget = self.query_one("#schema_sample_size_input", Input)
            size_str = size_input_widget.value.strip()
            if not size_str:
                return None, Text.from_markup("[#BF616A]Sample size cannot be empty.[/]")
            int(size_str)
            return size_str, None
        except NoMatches:
            return None, UI_ERROR_INPUT_WIDGETS_TEXT
        except ValueError:
            return None, Text.from_markup("[#BF616A]Sample size must be an integer.[/]")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        loading_indicator = self.query_one("#schema_loading_indicator", LoadingIndicator)

        if btn == "analyze_schema_button":
            self._clear_analysis_results()
            loading_indicator.display = True
            self.analysis_status = Text.from_markup("[#EBCB8B]Preparing analysis...[/]")
            self.schema_copy_feedback = Text("")

            if not self.app.current_mongo_uri or not self.app.current_db_name:
                self.analysis_status = NO_DB_CONNECTION_TEXT
                await self.app.push_screen(
                    ErrorDialog("Connection Required", NO_DB_CONNECTION_TEXT.plain)
                )
                loading_indicator.display = False
                return

            collection_name: Optional[str] = None
            try:
                coll_select = self.query_one("#schema_collection_select", Select)
                if coll_select.value == Select.BLANK:
                    self.analysis_status = NO_COLLECTION_SELECTED_TEXT
                    await self.app.push_screen(
                        ErrorDialog("Collection Required", NO_COLLECTION_SELECTED_TEXT.plain)
                    )
                    loading_indicator.display = False
                    return
                collection_name = str(coll_select.value)
            except NoMatches:
                self.analysis_status = UI_ERROR_INPUT_WIDGETS_TEXT
                await self.app.push_screen(
                    ErrorDialog("UI Error", UI_ERROR_INPUT_WIDGETS_TEXT.plain)
                )
                loading_indicator.display = False
                return

            sample_size_str, err_text = self._validate_sample_size_input()
            if err_text or sample_size_str is None:
                self.analysis_status = err_text or Text.from_markup("[#BF616A]Invalid input.[/]")
                if err_text:
                    await self.app.push_screen(ErrorDialog("Input Error", err_text.plain))
                loading_indicator.display = False
                return

            uri = self.app.current_mongo_uri
            db_name = self.app.current_db_name

            size = int(sample_size_str)
            self.analysis_status = Text.from_markup(
                f"[#EBCB8B]Analyzing content of '{collection_name}' using a sample of {size if size >= 0 else 'all'} docs...[/]"
            )

            try:
                callable_with_args = functools.partial(
                    self._run_analysis_task, uri, db_name, collection_name, size
                )
                worker: Worker[
                    Tuple[
                        Optional[Dict], Optional[Dict], Optional[Dict], Optional[Tuple[str, bool]]
                    ]
                ] = self.app.run_worker(callable_with_args, thread=True, group="schema_analysis")
                result = await worker.wait()
                schema_data, field_stats_data, hierarchical_schema, error_tuple = result

                if worker.is_cancelled:
                    self.analysis_status = Text.from_markup(
                        "[#D08770]Analysis cancelled by user.[/]"
                    )
                elif error_tuple and error_tuple[0]:
                    error_msg_str, is_auth_error = error_tuple
                    self.analysis_status = Text.from_markup(
                        f"[#BF616A]Analysis Error: {error_msg_str}[/]"
                    )
                    dialog_title = "Authorization Error" if is_auth_error else "Analysis Error"
                    await self.app.push_screen(ErrorDialog(dialog_title, error_msg_str))
                elif (
                    schema_data is not None
                    and field_stats_data is not None
                    and hierarchical_schema is not None
                ):
                    self.current_hierarchical_schema = hierarchical_schema
                    self.app.current_schema_analysis_results = {
                        "flat_schema": schema_data,
                        "field_stats": field_stats_data,
                        "hierarchical_schema": hierarchical_schema,
                        "collection_name": collection_name,
                    }
                    table = self.query_one("#schema_results_table", DataTable)
                    md_view = self.query_one("#schema_json_view", Markdown)

                    rows: List[Tuple[Any, ...]] = []
                    for field, details in schema_data.items():
                        stats = field_stats_data.get(field, {})

                        def fmt(v: Any, m: int = 30) -> str:
                            if v is None or v == "N/A":
                                return "N/A"
                            s = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                            return s[:m] + ("..." if len(s) > m else "")

                        arr = stats.get("array_elements", {})
                        rows.append(
                            (
                                field,
                                details.get("type", "N/A"),
                                stats.get("cardinality", "N/A"),
                                f"{stats.get('missing_percentage', 0):.1f}",
                                fmt(stats.get("numeric_min")),
                                fmt(stats.get("numeric_max")),
                                fmt(stats.get("date_min")),
                                fmt(stats.get("date_max")),
                                fmt(stats.get("top_values")),
                                fmt(arr.get("type_distribution")),
                                fmt(arr.get("top_values")),
                            )
                        )
                    if rows:
                        table.add_rows(rows)
                    else:
                        table.add_row(
                            "No schema fields found or analyzed.",
                            *[""] * (len(table.columns) - 1 if table.columns else 10),
                        )

                    try:
                        self._current_schema_json_str = json.dumps(
                            hierarchical_schema, indent=2, default=str
                        )
                        md_view.update(f"```json\n{self._current_schema_json_str}\n```")
                        self.analysis_status = Text.from_markup("[#A3BE8C]Analysis complete.[/]")
                    except TypeError:
                        self._current_schema_json_str = f"// Error: Schema not fully JSON serializable.\n{str(hierarchical_schema)[:1000]}"
                        md_view.update(f"```json\n{self._current_schema_json_str}\n```")
                        self.analysis_status = Text.from_markup(
                            "[#D08770]Analysis complete (schema display partial).[/]"
                        )
                else:
                    self.analysis_status = Text.from_markup(
                        "[#D08770]Analysis completed with no data or an unknown issue.[/]"
                    )

            except WorkerCancelled:
                self.analysis_status = Text.from_markup("[#D08770]Analysis was cancelled.[/]")
            except Exception as e:
                logger.error(f"Schema analysis main handler error: {e}", exc_info=True)
                self.analysis_status = Text.from_markup(
                    f"[#BF616A]Unexpected Error: {str(e)[:70]}[/]"
                )
                await self.app.push_screen(ErrorDialog("Unexpected Analysis Error", str(e)))
            finally:
                if self.is_mounted:
                    loading_indicator.display = False

        elif btn == "save_schema_json_button":
            save_path_str = self.query_one("#schema_save_path_input", Input).value.strip()
            if not save_path_str:
                self.schema_copy_feedback = Text.from_markup(
                    "[#BF616A]Schema save path cannot be empty.[/]"
                )
                self.app.notify("Schema save path empty.", title="Save Error", severity="error")
                return
            if not self.current_hierarchical_schema:
                self.schema_copy_feedback = Text.from_markup(
                    "[#D08770]No schema data to save. Analyze first.[/]"
                )
                self.app.notify("No schema to save.", title="Save Info", severity="warning")
                return
            save_path = Path(save_path_str)
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with save_path.open("w", encoding="utf-8") as f:
                    json.dump(self.current_hierarchical_schema, f, indent=2, default=str)
                logger.info(f"Hierarchical schema has been saved to {save_path}")
                self.schema_copy_feedback = Text.from_markup(
                    f"[#A3BE8C]Schema saved to {save_path.name}[/]"
                )
                self.app.notify(f"Schema saved to {save_path}", title="Save Success")
            except Exception as e:
                logger.error(f"Error saving schema to file: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    f"[#BF616A]Error saving: {str(e)[:50]}[/]"
                )
                await self.app.push_screen(
                    ErrorDialog("Save Error", f"Could not save schema: {e!s}")
                )

        elif btn == "copy_json_button":
            if self._current_schema_json_str and self._current_schema_json_str != "{}":
                self.app.copy_to_clipboard(self._current_schema_json_str)
                self.schema_copy_feedback = Text.from_markup("[#A3BE8C]Full JSON Schema Copied![/]")
                self.app.notify("JSON schema copied.", title="Copy Success")
            else:
                self.schema_copy_feedback = Text.from_markup(
                    "[#D08770]No JSON to copy. Analyze first.[/]"
                )
                self.app.notify("No JSON content to copy.", title="Copy Info", severity="warning")

        elif btn == "copy_cell_button":
            try:
                table = self.query_one("#schema_results_table", DataTable)
                coord = table.cursor_coordinate
                if table.row_count == 0 or not table.show_cursor or coord is None:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No data or cell selected.[/]"
                    )
                    self.app.notify("No cell selected.", title="Copy Info", severity="warning")
                    return
                r, c = coord.row, coord.column
                cell_value = table.get_cell_at(coord)
                val_to_copy = cell_value.plain if isinstance(cell_value, Text) else str(cell_value)
                self.app.copy_to_clipboard(val_to_copy)
                self.schema_copy_feedback = Text.from_markup(
                    f"[#A3BE8C]Cell ({r},{c}) value copied![/]"
                )
                self.app.notify(f"Cell ({r},{c}) copied.", title="Copy Success")
            except Exception as e:
                logger.error(f"Error copying cell value: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    f"[#BF616A]Error copying cell: {str(e)[:50]}[/]"
                )
                await self.app.push_screen(ErrorDialog("Copy Error", f"Could not copy cell: {e!s}"))

        elif btn == "copy_table_csv_button":
            try:
                table = self.query_one("#schema_results_table", DataTable)
                if table.row_count == 0:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No data in table for CSV.[/]"
                    )
                    self.app.notify(
                        "No data in table for CSV.", title="Copy Info", severity="warning"
                    )
                    return
                output = io.StringIO()
                writer = csv.writer(output, quoting=csv.QUOTE_ALL)
                headers = [
                    col_def.label.plain if isinstance(col_def.label, Text) else str(col_def.label)
                    for _, col_def in table.columns.items()
                ]
                writer.writerow(headers)
                for i in range(table.row_count):
                    row_data_renderables = table.get_row_at(i)
                    row_data_plain = [
                        cell.plain if isinstance(cell, Text) else str(cell)
                        for cell in row_data_renderables
                    ]
                    writer.writerow(row_data_plain)
                self.app.copy_to_clipboard(output.getvalue())
                self.schema_copy_feedback = Text.from_markup(
                    "[#A3BE8C]Analysis table copied as CSV![/]"
                )
                self.app.notify("Table copied as CSV.", title="Copy Success")
            except Exception as e:
                logger.error(f"Error copying table as CSV: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    f"[#BF616A]Error copying CSV: {str(e)[:50]}[/]"
                )
                await self.app.push_screen(
                    ErrorDialog("Copy Error", f"Could not copy table as CSV: {e!s}")
                )

    def _run_analysis_task(
        self, uri: str, db_name: str, collection_name: str, sample_size: int
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Tuple[str, bool]]]:
        try:
            coll = SchemaAnalyser.get_collection(uri, db_name, collection_name)
            schema_data, field_stats_data = SchemaAnalyser.infer_schema_and_field_stats(
                coll, sample_size
            )
            if schema_data is None and field_stats_data is None:
                return None, None, None, ("Analysis returned no data.", False)

            hierarchical_schema = SchemaAnalyser.schema_to_hierarchical(
                schema_data if schema_data else {}
            )
            return schema_data, field_stats_data, hierarchical_schema, None
        except PyMongoOperationFailure as e_op:
            logger.warning(
                f"Schema analysis: MongoDB operation failure for '{collection_name}': {e_op}"
            )
            is_auth = _is_auth_error_from_op_failure(e_op)
            err_details = e_op.details.get("errmsg", str(e_op)) if e_op.details else str(e_op)
            err_msg = (
                f"Not authorized to read/analyze collection '{collection_name}'."
                if is_auth
                else f"Operation failed on '{collection_name}': {err_details}"
            )
            return None, None, None, (err_msg, is_auth)
        except (PyMongoConnectionFailure, ConnectionError) as e_conn:
            logger.error(
                f"Schema analysis: DB connection error for '{collection_name}': {e_conn}",
                exc_info=True,
            )
            return None, None, None, (f"Database Connection Error: {e_conn!s}", False)
        except Exception as e:
            logger.exception(
                f"Unexpected error in schema analysis task for '{collection_name}': {e}"
            )
            return None, None, None, (f"Unexpected Analysis Error: {e!s}", False)
