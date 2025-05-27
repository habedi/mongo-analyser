import csv
import functools
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Label, Markdown, Select, Static
from textual.worker import Worker

from mongo_analyser.core.analyser import SchemaAnalyser

logger = logging.getLogger(__name__)


class SchemaAnalysisView(Container):
    analysis_status = reactive(Text("Select collection and click Analyze Schema"))
    schema_copy_feedback = reactive(Text(""))
    current_hierarchical_schema: Dict = {}

    def _get_default_save_path(self) -> str:
        db_name = self.app.current_db_name
        collection_name = self.app.active_collection
        if db_name and collection_name:
            return f"output/{db_name}/{collection_name}_schema.json"
        elif collection_name:  # Fallback if db_name is somehow not set but collection is
            return f"output/{collection_name}_schema.json"
        return "output/schema.json"  # Generic fallback

    def compose(self) -> ComposeResult:
        yield Label("Collection Name:")
        yield Select(
            [],
            prompt="Connect to DB to see collections",
            id="schema_collection_select",
            allow_blank=True,
        )
        yield Label("Sample Size (-1 for all):")
        yield Input(id="schema_sample_size_input", value="1000")
        yield Button("Analyze Schema", variant="primary", id="analyze_schema_button")
        yield Static(self.analysis_status, id="schema_status_label")

        yield Label("Field Schema & Statistics:", classes="panel_title_small")
        yield DataTable(id="schema_results_table", show_header=True, show_cursor=True)

        yield Label("Hierarchical Schema (JSON):", classes="panel_title_small")
        with VerticalScroll(classes="json_view_container"):
            yield Markdown("```json\n{}\n```", id="schema_json_view")

        yield Label(
            "Save Schema JSON to:", classes="panel_title_small"
        )  # Removed placeholder from label
        yield Input(id="schema_save_path_input", value=self._get_default_save_path())  # Use helper
        yield Button("Save Schema to File", id="save_schema_json_button")

        with Horizontal(classes="copy_button_container"):
            yield Button("Copy Cell Value", id="copy_cell_button")
            yield Button("Copy Full JSON", id="copy_json_button")
            yield Button("Copy Table as CSV", id="copy_table_csv_button")
        yield Static(self.schema_copy_feedback, id="schema_copy_feedback_label")

    def on_mount(self) -> None:
        self.update_collection_select()  # This will also update save path via its logic
        # If collections are already available, update_collection_select will handle the save path
        # If not, _get_default_save_path in compose already set a generic one.
        # We can explicitly set it again here if needed after app state might be more ready.
        try:
            self.query_one("#schema_save_path_input", Input).value = self._get_default_save_path()
        except NoMatches:
            logger.warning(
                "SchemaAnalysisView: #schema_save_path_input not found on mount for path update."
            )

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#schema_collection_select", Select).focus()
        except NoMatches:
            logger.debug("SchemaAnalysisView: Could not focus default input.")

    def update_collection_select(self) -> None:
        try:
            select_widget = self.query_one("#schema_collection_select", Select)
            save_path_input = self.query_one("#schema_save_path_input", Input)
            collections = self.app.available_collections
            current_active_collection = self.app.active_collection  # This is app.active_collection
            current_db_name = self.app.current_db_name

            default_save_path = "output/schema.json"  # Generic default
            if current_db_name and current_active_collection:
                default_save_path = (
                    f"output/{current_db_name}/{current_active_collection}_schema.json"
                )
            elif current_active_collection:  # Fallback if db_name somehow not set
                default_save_path = f"output/{current_active_collection}_schema.json"

            if collections:
                options = [(coll, coll) for coll in collections]
                current_select_value = select_widget.value  # Value in the Select widget
                select_widget.set_options(options)
                select_widget.disabled = False
                select_widget.prompt = "Select Collection"

                if current_active_collection and current_active_collection in collections:
                    select_widget.value = current_active_collection
                    # Path already set based on current_active_collection and current_db_name
                elif (
                    current_select_value != Select.BLANK
                    and str(current_select_value) in collections
                ):
                    # User might have selected something different from app.active_collection
                    # Update path based on current_select_value if it's valid
                    if current_db_name:
                        default_save_path = (
                            f"output/{current_db_name}/{current_select_value!s}_schema.json"
                        )
                    else:
                        default_save_path = f"output/{current_select_value!s}_schema.json"
                elif select_widget.value not in collections:  # current value is invalid
                    select_widget.value = Select.BLANK
                    # Path will be the generic default_save_path calculated at the start of this method
            else:  # No collections
                select_widget.set_options([])
                select_widget.prompt = "Connect to DB to see collections"
                select_widget.disabled = True
                select_widget.value = Select.BLANK
                # Path will be the generic default_save_path

            save_path_input.value = default_save_path

        except NoMatches:
            logger.warning(
                "SchemaAnalysisView: #schema_collection_select or #schema_save_path_input not found for update."
            )
        except Exception as e:
            logger.error(f"Error in update_collection_select: {e}", exc_info=True)

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "schema_collection_select":
            new_collection_name = (
                str(event.value)
                if event.value is not None and event.value != Select.BLANK
                else None
            )
            self.app.active_collection = new_collection_name  # Update app's active collection

            # Update save path based on the new selection and current_db_name
            try:
                save_path_input = self.query_one("#schema_save_path_input", Input)
                current_db_name = self.app.current_db_name
                if current_db_name and new_collection_name:
                    save_path_input.value = (
                        f"output/{current_db_name}/{new_collection_name}_schema.json"
                    )
                elif new_collection_name:  # Fallback if db_name not set
                    save_path_input.value = f"output/{new_collection_name}_schema.json"
                else:  # No collection selected
                    save_path_input.value = "output/schema.json"
            except NoMatches:
                logger.warning(
                    "SchemaAnalysisView: #schema_save_path_input not found during on_select_changed."
                )

    def watch_analysis_status(self, new_status: Text) -> None:
        if self.is_mounted:
            try:
                self.query_one("#schema_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #schema_status_label.")

    def watch_schema_copy_feedback(self, new_feedback: Text) -> None:
        if self.is_mounted:
            try:
                feedback_label = self.query_one("#schema_copy_feedback_label", Static)
                feedback_label.update(new_feedback)
                if new_feedback.plain:
                    clear_feedback_callback = functools.partial(feedback_label.update, Text(""))
                    feedback_label.set_timer(2, clear_feedback_callback)
            except NoMatches:
                logger.warning("Could not find #schema_copy_feedback_label.")
            except Exception as e:
                logger.error(f"Error in watch_schema_copy_feedback: {e}", exc_info=True)

    def _prepare_analysis_inputs(
        self,
    ) -> tuple[str | None, str | None, str | None, str, Text | None]:
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

        collection_select = self.query_one("#schema_collection_select", Select)
        collection_name = (
            str(collection_select.value) if collection_select.value != Select.BLANK else None
        )
        sample_size_str = self.query_one("#schema_sample_size_input", Input).value.strip()

        if not collection_name:
            return (
                uri,
                db_name,
                None,
                sample_size_str,
                Text.from_markup("[#BF616A]Please select a collection.[/]"),
            )
        try:
            int(sample_size_str)
        except ValueError:
            return (
                uri,
                db_name,
                collection_name,
                "",
                Text.from_markup("[#BF616A]Sample size must be an integer.[/]"),
            )
        return uri, db_name, collection_name, sample_size_str, None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze_schema_button":
            table = self.query_one("#schema_results_table", DataTable)
            md_view = self.query_one("#schema_json_view", Markdown)

            self.analysis_status = Text.from_markup("[#EBCB8B]Preparing analysis...[/]")
            self.schema_copy_feedback = Text("")
            table.clear(columns=True)
            md_view.update("```json\n{}\n```")
            self.current_hierarchical_schema = {}

            uri, db_name, collection_name, sample_size_str, prep_error_text = (
                self._prepare_analysis_inputs()
            )
            if prep_error_text or not uri or not db_name or not collection_name:
                self.analysis_status = prep_error_text or Text.from_markup(
                    "[#BF616A]Missing DB connection or collection.[/]"
                )
                return

            sample_size = int(sample_size_str)
            self.analysis_status = Text.from_markup(
                f"[#EBCB8B]Analyzing '{collection_name}' (sample: {sample_size})...[/]"
            )

            try:
                callable_with_args = functools.partial(
                    self._run_analysis_task, uri, db_name, collection_name, sample_size
                )
                worker: Worker[Tuple[Dict | None, Dict | None, Dict | None, str | None]] = (
                    self.app.run_worker(callable_with_args, thread=True, group="schema_analysis")
                )
                (
                    schema_data,
                    field_stats_data,
                    hierarchical_schema,
                    error_msg_str,
                ) = await worker.wait()

                if error_msg_str:
                    self.analysis_status = Text.from_markup(f"[#BF616A]Error: {error_msg_str}[/]")
                    return

                if schema_data and field_stats_data and hierarchical_schema is not None:
                    self.current_hierarchical_schema = hierarchical_schema
                    table.add_columns("Field", "Type", "Cardinality", "Missing (%)")
                    rows_for_table: List[Tuple[Any, ...]] = []
                    for field, details in schema_data.items():
                        field_s_data = field_stats_data.get(field, {})
                        cardinality = field_s_data.get("cardinality", "N/A")
                        missing = field_s_data.get("missing_percentage", "N/A")
                        rows_for_table.append(
                            (
                                field,
                                details["type"],
                                cardinality,
                                f"{missing:.1f}" if isinstance(missing, (float, int)) else "N/A",
                            )
                        )
                    if rows_for_table:
                        table.add_rows(rows_for_table)
                    else:
                        table.add_row("No schema fields found or analyzed.")

                    try:
                        json_output = json.dumps(hierarchical_schema, indent=4)
                        md_view.update(f"```json\n{json_output}\n```")
                        self.analysis_status = Text.from_markup("[#A3BE8C]Analysis complete.[/]")
                    except TypeError as te:
                        logger.error(f"JSON serialization error: {te}", exc_info=True)
                        md_view.update("```json\n// Error: Schema non-serializable.\n{}\n```")
                        self.analysis_status = Text.from_markup(
                            "[#D08770]Analysis complete (schema display error).[/]"
                        )
                else:
                    self.analysis_status = Text.from_markup(
                        "[#D08770]Analysis completed with no data.[/]"
                    )
            except Exception as e:
                logger.error(f"Schema analysis handler error: {e}", exc_info=True)
                self.analysis_status = Text.from_markup(f"[#BF616A]Display Error: {e!s}[/]")

        elif event.button.id == "save_schema_json_button":
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
                # Ensure parent directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)
                SchemaAnalyser.save_schema_to_json(self.current_hierarchical_schema, save_path)
                self.schema_copy_feedback = Text.from_markup(
                    f"[#A3BE8C]Schema saved to {save_path}[/]"
                )
                self.app.notify(f"Schema saved to {save_path}", title="Save Success")
            except Exception as e:
                logger.error(f"Error saving schema to file: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    f"[#BF616A]Error saving schema: {e!s}[/]"
                )
                self.app.notify(f"Error saving schema: {e!s}", title="Save Error", severity="error")

        elif event.button.id == "copy_json_button":
            try:
                md_view = self.query_one("#schema_json_view", Markdown)
                markdown_content = ""
                if hasattr(md_view, "document"):
                    markdown_content = md_view.document
                elif hasattr(md_view, "_markdown"):
                    markdown_content = md_view._markdown
                else:
                    logger.warning(
                        "Could not determine how to get markdown content from Markdown widget."
                    )
                    raise AttributeError("Cannot get markdown content.")

                json_to_copy = "{}"
                if markdown_content.startswith("```json\n") and markdown_content.endswith("\n```"):
                    extracted_json = markdown_content[len("```json\n") : -len("\n```")]
                    if extracted_json.strip() and not extracted_json.strip().startswith("// Error"):
                        json_to_copy = extracted_json
                    elif extracted_json.strip().startswith("// Error"):
                        json_to_copy = "// Error in schema data"
                elif markdown_content.strip() == "{}":
                    json_to_copy = "{}"
                elif "Error: Schema non-serializable" in markdown_content:
                    json_to_copy = "// Error: Schema non-serializable."

                if json_to_copy:
                    self.app.copy_to_clipboard(json_to_copy)
                    self.schema_copy_feedback = Text.from_markup(
                        "[#A3BE8C]Full JSON Schema Copied![/]"
                    )
                    self.app.notify("JSON schema copied to clipboard.", title="Copy Success")
                else:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No JSON content to copy.[/]"
                    )
                    self.app.notify(
                        "No JSON content to copy.", title="Copy Info", severity="warning"
                    )
            except AttributeError as ae:
                logger.error(f"Error accessing Markdown content: {ae}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    "[#BF616A]Error accessing Markdown content.[/]"
                )
                self.app.notify(
                    "Error accessing Markdown content.", title="Copy Error", severity="error"
                )
            except Exception as e:
                logger.error(f"Error copying JSON: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup("[#BF616A]Error copying JSON.[/]")
                self.app.notify("Error copying JSON.", title="Copy Error", severity="error")

        elif event.button.id == "copy_cell_button":
            try:
                table = self.query_one("#schema_results_table", DataTable)
                if table.row_count == 0:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No data in table to copy.[/]"
                    )
                    self.app.notify(
                        "No data in table to copy.", title="Copy Info", severity="warning"
                    )
                    return

                cursor_row, cursor_col = table.cursor_coordinate
                if cursor_row < 0 or cursor_col < 0:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No cell selected in table (or table empty).[/]"
                    )
                    self.app.notify(
                        "No cell selected in table.", title="Copy Info", severity="warning"
                    )
                    return

                cell_value = table.get_cell_at(table.cursor_coordinate)
                self.app.copy_to_clipboard(str(cell_value))
                self.schema_copy_feedback = Text.from_markup(
                    f"[#A3BE8C]Cell ({cursor_row},{cursor_col}) value copied![/]"
                )
                self.app.notify(f"Cell ({cursor_row},{cursor_col}) copied.", title="Copy Success")
            except Exception as e:
                logger.error(f"Error copying cell value: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    "[#BF616A]Error copying cell value.[/]"
                )
                self.app.notify("Error copying cell value.", title="Copy Error", severity="error")

        elif event.button.id == "copy_table_csv_button":
            try:
                table = self.query_one("#schema_results_table", DataTable)
                if table.row_count == 0:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No data in table to copy as CSV.[/]"
                    )
                    self.app.notify(
                        "No data in table for CSV.", title="Copy Info", severity="warning"
                    )
                    return

                output = io.StringIO()
                csv_writer = csv.writer(output, quoting=csv.QUOTE_ALL)
                headers = []
                for column_key in table.columns:
                    column_label = table.columns[column_key].label
                    headers.append(
                        str(column_label.plain if hasattr(column_label, "plain") else column_label)
                    )
                csv_writer.writerow(headers)

                for r_idx in range(table.row_count):
                    row_display_data = table.get_row_at(r_idx)
                    row_values = [str(cell) for cell in row_display_data]
                    csv_writer.writerow(row_values)
                csv_data = output.getvalue()
                self.app.copy_to_clipboard(csv_data)
                self.schema_copy_feedback = Text.from_markup(
                    "[#A3BE8C]Table copied to clipboard as CSV![/]"
                )
                self.app.notify("Table copied as CSV.", title="Copy Success")
            except Exception as e:
                logger.error(f"Error copying table as CSV: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    "[#BF616A]Error copying table as CSV.[/]"
                )
                self.app.notify("Error copying table as CSV.", title="Copy Error", severity="error")

    def _run_analysis_task(
        self, uri: str, db_name: str, collection_name: str, sample_size: int
    ) -> Tuple[Dict | None, Dict | None, Dict | None, str | None]:
        try:
            collection_obj = SchemaAnalyser.get_collection(uri, db_name, collection_name)
            schema_data, field_stats_data = SchemaAnalyser.infer_schema_and_field_stats(
                collection_obj, sample_size
            )
            hierarchical_schema = SchemaAnalyser.schema_to_hierarchical(schema_data)
            return schema_data, field_stats_data, hierarchical_schema, None
        except (PyMongoConnectionFailure, PyMongoOperationFailure, ConnectionError) as e:
            logger.error(f"Schema analysis: DB error: {e}", exc_info=False)
            return None, None, None, f"DB Error: {e!s}"
        except Exception as e:
            logger.exception(f"Schema analysis task error: {e}")
            return None, None, None, f"Analysis Error: {e!s}"
