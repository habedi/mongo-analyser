import functools
import json
import logging
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

from mongo_analyser.core.analyser import SchemaAnalyser  # Assuming SchemaAnalyser is in core

logger = logging.getLogger(__name__)


class SchemaAnalysisView(Container):
    analysis_status = reactive(Text("Enter collection and click Analyze."))
    schema_copy_feedback = reactive(Text(""))

    def compose(self) -> ComposeResult:
        yield Label("Schema Analysis", classes="panel_title")
        yield Label("Collection Name:")
        yield Select(
            [],
            prompt="Connect to DB to see collections",
            id="schema_collection_select",
            allow_blank=True,  # allow_blank is true by default
        )
        yield Label("Sample Size (-1 for all):")
        yield Input(id="schema_sample_size_input", value="1000")
        yield Button("Analyze Schema", variant="primary", id="analyze_schema_button")
        yield Static(self.analysis_status, id="schema_status_label")

        yield Label("Field Schema & Statistics:", classes="panel_title_small")
        yield DataTable(id="schema_results_table", show_header=True, show_cursor=True)

        yield Label("Hierarchical Schema (JSON):", classes="panel_title_small")
        with VerticalScroll(classes="json_view_container"):  # Ensure this class is in app.tcss
            yield Markdown("```json\n{}\n```", id="schema_json_view")

        with Horizontal(classes="copy_button_container"):  # Ensure this class is in app.tcss
            yield Button("Copy Cell Value", id="copy_cell_button")
            yield Button("Copy Full JSON", id="copy_json_button")
        yield Static(self.schema_copy_feedback, id="schema_copy_feedback_label")

    def on_mount(self) -> None:
        self.update_collection_select()  # Call method to populate on mount

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#schema_collection_select", Select).focus()
        except NoMatches:
            logger.debug("SchemaAnalysisView: Could not focus default input.")

    def update_collection_select(self) -> None:  # Renamed from _update_collection_select
        try:
            select_widget = self.query_one("#schema_collection_select", Select)
            # collections is a List[str] from app.available_collections
            collections = self.app.available_collections
            current_active_collection = self.app.active_collection

            if collections:
                options = [(coll, coll) for coll in collections]
                current_value = select_widget.value
                select_widget.set_options(options)
                select_widget.disabled = False
                select_widget.prompt = "Select Collection"

                if current_active_collection and current_active_collection in collections:
                    select_widget.value = current_active_collection
                elif current_value != Select.BLANK and current_value in collections:
                    # Keep current selection if valid and no specific active_collection
                    pass  # Value is already set
                elif select_widget.value not in collections:  # Clear if invalid
                    select_widget.value = Select.BLANK
            else:
                select_widget.set_options([])
                select_widget.prompt = "Connect to DB to see collections"
                select_widget.disabled = True
                select_widget.value = Select.BLANK
        except NoMatches:
            logger.warning("SchemaAnalysisView: #schema_collection_select not found for update.")
        except Exception as e:
            logger.error(f"Error in update_collection_select: {e}", exc_info=True)

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "schema_collection_select":
            if event.value is not None and event.value != Select.BLANK:
                self.app.active_collection = str(event.value)
            else:  # Handles Select.BLANK
                self.app.active_collection = None

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
                if new_feedback.plain:  # Check if there's actual text
                    feedback_label.set_timer(2, feedback_label.update, Text(""))
            except NoMatches:
                logger.warning("Could not find #schema_copy_feedback_label.")

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
            table.clear(columns=True)  # Keep columns if you want to reuse them
            md_view.update("```json\n{}\n```")

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
                # _run_analysis_task needs to be defined within this class or imported
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
                    table.add_columns(
                        "Field", "Type", "Cardinality", "Missing (%)"
                    )  # Add columns if cleared
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
                else:  # No data but no error_msg implies empty result
                    self.analysis_status = Text.from_markup(
                        "[#D08770]Analysis completed with no data.[/]"
                    )
            except Exception as e:
                logger.error(f"Schema analysis handler error: {e}", exc_info=True)
                self.analysis_status = Text.from_markup(f"[#BF616A]Display Error: {e!s}[/]")

        elif event.button.id == "copy_json_button":
            # (Keep existing copy_json_button logic, ensure it uses self.app.copy_to_clipboard)
            try:
                md_view = self.query_one("#schema_json_view", Markdown)
                markdown_content = md_view.markdown
                # ... (rest of the logic to extract JSON)
                json_to_copy = "..."  # Placeholder for extracted JSON
                if markdown_content.startswith("```json\n") and markdown_content.endswith("\n```"):
                    json_to_copy = markdown_content[len("```json\n") : -len("\n```")]
                elif markdown_content.startswith(
                    "```json\n// Error"
                ) or markdown_content.startswith("```json\n{}"):
                    json_to_copy = (
                        "{}" if "Error" not in markdown_content else "// Error in schema data"
                    )
                else:
                    json_to_copy = markdown_content  # Fallback

                if json_to_copy:
                    self.app.copy_to_clipboard(json_to_copy)  # Use app's method
                    self.schema_copy_feedback = Text.from_markup(
                        "[#A3BE8C]Full JSON Schema Copied![/]"
                    )
                else:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No JSON content to copy.[/]"
                    )
            except Exception as e:
                logger.error(f"Error copying JSON: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup("[#BF616A]Error copying JSON.[/]")

        elif event.button.id == "copy_cell_button":
            # (Keep existing copy_cell_button logic, ensure it uses self.app.copy_to_clipboard)
            try:
                table = self.query_one("#schema_results_table", DataTable)
                if table.row_count == 0:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No data in table to copy.[/]"
                    )
                    return
                # ... (rest of the logic)
                cursor_row, cursor_col = table.cursor_coordinate
                if cursor_row < 0 or cursor_col < 0:
                    self.schema_copy_feedback = Text.from_markup(
                        "[#D08770]No cell selected in table.[/]"
                    )
                    return

                cell_value = table.get_cell_at(table.cursor_coordinate)
                self.app.copy_to_clipboard(str(cell_value))
                self.schema_copy_feedback = Text.from_markup(
                    f"[#A3BE8C]Cell ({cursor_row},{cursor_col}) value copied![/]"
                )
            except Exception as e:
                logger.error(f"Error copying cell value: {e}", exc_info=True)
                self.schema_copy_feedback = Text.from_markup(
                    "[#BF616A]Error copying cell value.[/]"
                )

    def _run_analysis_task(
        self, uri: str, db_name: str, collection_name: str, sample_size: int
    ) -> Tuple[Dict | None, Dict | None, Dict | None, str | None]:
        # This method can stay as is, since it's pure logic
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
