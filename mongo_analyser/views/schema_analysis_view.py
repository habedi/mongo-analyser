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
from textual.widgets import Button, DataTable, Input, Label, Markdown, Select, Static
from textual.worker import Worker, WorkerCancelled

from mongo_analyser.core.analyser import SchemaAnalyser
from mongo_analyser.dialogs import ErrorDialog

logger = logging.getLogger(__name__)


class SchemaAnalysisView(Container):
    analysis_status = reactive(Text("Select a collection and click Analyze Schema"))
    schema_copy_feedback = reactive(Text(""))
    current_hierarchical_schema: Dict = {}
    _current_schema_json_str: str = "{}"

    def _get_default_save_path(self) -> str:
        db_name = self.app.current_db_name
        collection_name = self.app.active_collection
        if db_name and collection_name:
            return f"output/{db_name}/{collection_name}_schema.json"
        if collection_name:
            return f"output/{collection_name}_schema.json"
        return "output/default_schema.json"

    def compose(self) -> ComposeResult:
        yield Label("Collection Name:")
        yield Select(
            [],
            prompt="Connect to DB to see collections",
            id="schema_collection_select",
            allow_blank=True,
        )
        yield Label("Sample Size (-1 for all documents):")
        yield Input(id="schema_sample_size_input", value="1000", placeholder="e.g., 1000 or -1")
        yield Button("Analyze Schema", variant="primary", id="analyze_schema_button")
        yield Static(self.analysis_status, id="schema_status_label")
        yield Label("Collection Field Analysis:", classes="panel_title_small")
        yield DataTable(
            id="schema_results_table", show_header=True, show_cursor=True, cursor_type="row"
        )
        yield Label("Hierarchical Schema (JSON):", classes="panel_title_small")
        with VerticalScroll(classes="json_view_container"):
            yield Markdown("```json\n{}\n```", id="schema_json_view")
        yield Label("Save Schema to File Path:", classes="panel_title_small")
        yield Input(id="schema_save_path_input", value=self._get_default_save_path())
        yield Button("Save Schema to File", id="save_schema_json_button")
        with Horizontal(classes="copy_button_container"):
            yield Button("Copy Cell Value", id="copy_cell_button")
            yield Button("Copy Schema JSON", id="copy_json_button")
            yield Button("Copy Analysis Table (CSV)", id="copy_table_csv_button")
        yield Static(self.schema_copy_feedback, id="schema_copy_feedback_label")

    def on_mount(self) -> None:
        self._last_collections: List[str] = []
        self.update_collection_select()
        try:
            self.query_one("#schema_save_path_input", Input).value = self._get_default_save_path()
        except NoMatches:
            logger.warning("SchemaAnalysisView: #schema_save_path_input not found on mount.")
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
            if collections == self._last_collections:
                return
            self._last_collections = list(collections)
            if collections:
                options = [(c, c) for c in collections]
                select_widget.set_options(options)
                select_widget.disabled = False
                select_widget.prompt = "Select Collection"
                active = self.app.active_collection
                if active in collections and select_widget.value != active:
                    select_widget.value = active
            else:
                select_widget.set_options([])
                select_widget.prompt = "Connect to DB to see collections"
                select_widget.disabled = True
                select_widget.value = Select.BLANK
            current = str(select_widget.value) if select_widget.value != Select.BLANK else None
            save_path_input.value = self._get_path_for_collection(
                current or self.app.active_collection
            )
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
        new = str(event.value) if event.value != Select.BLANK else None
        if new == self.app.active_collection:
            return
        self.app.active_collection = new
        save_path_input = self.query_one("#schema_save_path_input", Input)
        save_path_input.value = self._get_path_for_collection(new)
        self._clear_analysis_results()
        self.analysis_status = Text("Collection changed. Click 'Analyze Schema'")

    def _clear_analysis_results(self):
        table = self.query_one("#schema_results_table", DataTable)
        md = self.query_one("#schema_json_view", Markdown)
        table.clear()
        self.current_hierarchical_schema = {}
        self._current_schema_json_str = "{}"
        md.update(f"```json\n{self._current_schema_json_str}\n```")
        self.app.current_schema_analysis_results = None

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
                if new_feedback.plain:
                    self.set_timer(3, lambda: setattr(self, "schema_copy_feedback", Text("")))
            except NoMatches:
                pass
            except Exception as e:
                logger.error(f"Error in watch_schema_copy_feedback: {e}", exc_info=True)

    def _prepare_analysis_inputs(
        self,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], str, Optional[Text]]:
        uri = self.app.current_mongo_uri
        db_name = self.app.current_db_name
        if not uri or not db_name:
            return (
                None,
                None,
                None,
                "",
                Text.from_markup(
                    "[#BF616A]MongoDB not connected. Connect in 'DB Connection' tab.[/]"
                ),
            )
        try:
            sel = self.query_one("#schema_collection_select", Select)
            if sel.value != Select.BLANK:
                coll = str(sel.value)
            else:
                coll = None
            size = self.query_one("#schema_sample_size_input", Input).value.strip()
        except NoMatches:
            return (
                uri,
                db_name,
                None,
                "",
                Text.from_markup("[#BF616A]UI Error: Input widgets not found.[/]"),
            )
        if not coll:
            return (
                uri,
                db_name,
                None,
                size,
                Text.from_markup("[#BF616A]Please select a collection.[/]"),
            )
        try:
            int(size)
        except ValueError:
            return (
                uri,
                db_name,
                coll,
                "",
                Text.from_markup("[#BF616A]Sample size must be an integer.[/]"),
            )
        return uri, db_name, coll, size, None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        if btn == "analyze_schema_button":
            self._clear_analysis_results()
            self.analysis_status = Text.from_markup("[#EBCB8B]Preparing analysis...[/]")
            self.schema_copy_feedback = Text("")
            uri, db, coll, size_str, err = self._prepare_analysis_inputs()
            if err or not uri or not db or not coll:
                self.analysis_status = err or Text.from_markup(
                    "[#BF616A]Missing DB connection or collection.[/]"
                )
                if err:
                    await self.app.push_screen(ErrorDialog("Input Error", err.plain))
                return
            size = int(size_str)
            self.analysis_status = Text.from_markup(
                f"[#EBCB8B]Analyzing '{coll}' (sample: {size if size >= 0 else 'all'})...[/]"
            )
            try:
                callable_with_args = functools.partial(self._run_analysis_task, uri, db, coll, size)
                worker: Worker = self.app.run_worker(
                    callable_with_args, thread=True, group="schema_analysis"
                )
                schema_data, field_stats_data, hierarchical_schema, error_msg = await worker.wait()
                if worker.is_cancelled:
                    self.analysis_status = Text.from_markup(
                        "[#D08770]Analysis cancelled by user.[/]"
                    )
                    return
                if error_msg:
                    self.analysis_status = Text.from_markup(
                        f"[#BF616A]Analysis Error: {error_msg}[/]"
                    )
                    await self.app.push_screen(ErrorDialog("Analysis Error", error_msg))
                    return
                table = self.query_one("#schema_results_table", DataTable)
                md_view = self.query_one("#schema_json_view", Markdown)
                if schema_data and field_stats_data and hierarchical_schema is not None:
                    self.current_hierarchical_schema = hierarchical_schema
                    self.app.current_schema_analysis_results = {
                        "flat_schema": schema_data,
                        "field_stats": field_stats_data,
                        "hierarchical_schema": hierarchical_schema,
                        "collection_name": coll,
                    }
                    if not table.columns:
                        table.add_columns(
                            "Field",
                            "Type(s)",
                            "Cardinality",
                            "Missing (%)",
                            "Num Min",
                            "Num Max",
                            "Date Min",
                            "Date Max",
                            "Top Values (Field)",
                            "Array Elem Types",
                            "Array Elem Top Values",
                        )
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
                        table.add_row("No schema fields found or analyzed.", *[""] * 10)
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
                        "[#D08770]Analysis completed with no data or partial results.[/]"
                    )
            except WorkerCancelled:
                self.analysis_status = Text.from_markup("[#D08770]Analysis was cancelled.[/]")
            except Exception as e:
                logger.error(f"Schema analysis main handler error: {e}", exc_info=True)
                self.analysis_status = Text.from_markup(
                    f"[#BF616A]Unexpected Error: {str(e)[:70]}[/]"
                )
                await self.app.push_screen(ErrorDialog("Unexpected Analysis Error", str(e)))
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
                if table.row_count == 0 or coord is None:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No data or cell selected.[/]"
                    )
                    self.app.notify("No cell selected.", title="Copy Info", severity="warning")
                    return
                r, c = coord
                cell = table.get_cell_at(coord)
                val = cell.plain if isinstance(cell, Text) else str(cell)
                self.app.copy_to_clipboard(val)
                self.schema_copy_feedback = Text.from_markup(f"[#A3BE8C]Cell ({r},{c}) copied![/]")
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
                    self.schema_copy_feedback = Text.from_markup("[#D08770]No data for CSV.[/]")
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
                    row = table.get_row_at(i)
                    writer.writerow(
                        [cell.plain if isinstance(cell, Text) else str(cell) for cell in row]
                    )
                self.app.copy_to_clipboard(output.getvalue())
                self.schema_copy_feedback = Text.from_markup("[#A3BE8C]Table copied as CSV![/]")
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
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[str]]:
        try:
            coll = SchemaAnalyser.get_collection(uri, db_name, collection_name)
            schema_data, field_stats_data = SchemaAnalyser.infer_schema_and_field_stats(
                coll, sample_size
            )
            if not schema_data and not field_stats_data:
                return {}, {}, {}, None
            hierarchical_schema = SchemaAnalyser.schema_to_hierarchical(schema_data)
            return schema_data, field_stats_data, hierarchical_schema, None
        except (PyMongoConnectionFailure, PyMongoOperationFailure, ConnectionError) as e:
            logger.error(f"Schema analysis: DB error: {e}", exc_info=False)
            return None, None, None, f"Database Error: {e!s}"
        except Exception as e:
            logger.exception(f"Unexpected error in schema analysis task: {e}")
            return None, None, None, f"Unexpected Analysis Error: {e!s}"
