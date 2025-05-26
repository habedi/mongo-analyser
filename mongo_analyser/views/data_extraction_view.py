import functools
import json
import logging
from pathlib import Path
from typing import Dict  # Added Union

import pytz  # For timezone in extraction
from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container  # Removed Horizontal as it's not directly used
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Log, Select, Static
from textual.worker import Worker

from mongo_analyser.core.extractor import DataExtractor

logger = logging.getLogger(__name__)


class DataExtractionView(Container):
    extraction_status = reactive(Text("Enter details and click Start Extraction."))
    STATUS_PREFIX_Styled = Text.from_markup("[b #88C0D0]EXTRACT:[/] ")  # Keep this for log styling

    def _log_extraction_message(self, message_content: str):
        try:
            log_widget = self.query_one("#extraction_log", Log)
            lines_in_message = message_content.splitlines()
            if not lines_in_message and len(message_content) >= 0:  # Handle empty string
                lines_in_message = [message_content]
            elif (
                not lines_in_message and len(message_content) == 0
            ):  # Ensure one line for truly empty
                lines_in_message = [""]

            for line_part in lines_in_message:
                plain_prefix = self.STATUS_PREFIX_Styled.plain
                plain_line_part = Text(line_part).plain
                log_widget.write_line(f"{plain_prefix}{plain_line_part}")
        except NoMatches:
            logger.error("Extraction log widget not found.")

    def compose(self) -> ComposeResult:
        yield Label("Data Extraction", classes="panel_title")
        yield Label("Collection Name:")
        yield Select(
            [],
            prompt="Connect to DB to see collections",
            id="extract_collection_select",
            allow_blank=True,
        )
        yield Label("Path to Hierarchical Schema JSON:")
        yield Input(
            placeholder="/path/to/schema.json (from CLI or saved)", id="extract_schema_file"
        )
        yield Label("Output File Path (e.g., data/export.json.gz):")
        yield Input(id="extract_output_file", value="output.json.gz")
        yield Button("Start Extraction", variant="primary", id="start_extraction_button")
        yield Static(self.extraction_status, id="extraction_status_label")
        yield Label("Extraction Log:", classes="panel_title_small")
        yield Log(id="extraction_log", auto_scroll=True, highlight=True)

    def on_mount(self) -> None:
        self.update_collection_select()

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#extract_collection_select", Select).focus()
        except NoMatches:
            logger.debug("DataExtractionView: Could not focus default input.")

    def update_collection_select(self) -> None:  # Renamed from _update_collection_select
        try:
            select_widget = self.query_one("#extract_collection_select", Select)
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
                    pass  # Value already set and valid
                elif select_widget.value not in collections:  # Clear if invalid
                    select_widget.value = Select.BLANK
            else:
                select_widget.set_options([])
                select_widget.prompt = "Connect to DB to see collections"
                select_widget.disabled = True
                select_widget.value = Select.BLANK
        except NoMatches:
            logger.warning("DataExtractionView: #extract_collection_select not found for update.")
        except Exception as e:
            logger.error(
                f"Error in update_collection_select (DataExtractionView): {e}", exc_info=True
            )

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "extract_collection_select":
            if event.value is not None and event.value != Select.BLANK:
                self.app.active_collection = str(event.value)
            else:
                self.app.active_collection = None

    def watch_extraction_status(self, new_status: Text) -> None:
        if self.is_mounted:
            try:
                self.query_one("#extraction_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #extraction_status_label.")

    def _prepare_extraction_inputs(
        self,
    ) -> tuple[str | None, str | None, str | None, Dict | None, Path | None, Text | None]:
        uri = self.app.current_mongo_uri
        db_name = self.app.current_db_name
        if not uri or not db_name:
            return (
                None,
                None,
                None,
                None,
                None,
                Text.from_markup("[#BF616A]MongoDB not connected.[/]"),
            )

        collection_select = self.query_one("#extract_collection_select", Select)
        collection_name = (
            str(collection_select.value) if collection_select.value != Select.BLANK else None
        )

        schema_f_path_str = self.query_one("#extract_schema_file", Input).value.strip()
        output_f_path_str = self.query_one("#extract_output_file", Input).value.strip()

        if not collection_name:
            return (
                uri,
                db_name,
                None,
                None,
                None,
                Text.from_markup("[#BF616A]Collection name required.[/]"),
            )
        if not schema_f_path_str:
            return (
                uri,
                db_name,
                collection_name,
                None,
                None,
                Text.from_markup("[#BF616A]Schema file path required.[/]"),
            )
        if not output_f_path_str:
            return (
                uri,
                db_name,
                collection_name,
                None,
                None,
                Text.from_markup("[#BF616A]Output file path required.[/]"),
            )

        schema_file = Path(schema_f_path_str)
        output_file_p = Path(output_f_path_str)
        if not schema_file.is_file():
            return (
                uri,
                db_name,
                collection_name,
                None,
                output_file_p,
                Text.from_markup(f"[#BF616A]Schema file not found: {schema_file}[/]"),
            )
        try:
            with schema_file.open("r", encoding="utf-8") as f:
                schema_data = json.load(f)
        except json.JSONDecodeError:
            return (
                uri,
                db_name,
                collection_name,
                None,
                output_file_p,
                Text.from_markup(f"[#BF616A]Invalid JSON in schema: {schema_file}[/]"),
            )
        except IOError as e:
            return (
                uri,
                db_name,
                collection_name,
                None,
                output_file_p,
                Text.from_markup(f"[#BF616A]Could not read schema {schema_file}: {e}[/]"),
            )
        return uri, db_name, collection_name, schema_data, output_file_p, None

    def _run_extraction_task(
        self, uri: str, db_name: str, collection_name: str, schema_dict: Dict, output_file: Path
    ) -> str | None:  # Returns error message string or None for success
        try:
            # Use self.call_from_thread for UI updates from worker thread
            self.call_from_thread(
                self._log_extraction_message,
                f"Process started for {collection_name}. Output: {output_file}. This might take a while...",
            )
            DataExtractor.extract_data(
                mongo_uri=uri,
                db_name=db_name,
                collection_name=collection_name,
                schema=schema_dict,
                output_file=output_file,
                tz=pytz.utc,  # Example, could be configurable
                batch_size=1000,
                limit=-1,  # Example, could be configurable
            )
            return None  # Success
        except (PyMongoConnectionFailure, PyMongoOperationFailure, ConnectionError) as e:
            logger.error(f"Data extraction DB error: {e}", exc_info=False)
            return f"DB Error: {e!s}"
        except IOError as e:
            logger.error(f"Data extraction File I/O error: {e}", exc_info=True)
            return f"File I/O Error: {e!s}"
        except Exception as e:
            logger.exception(f"Data extraction task error: {e}")
            return f"Extraction Error: {e!s}"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start_extraction_button":
            self.extraction_status = Text.from_markup("[#EBCB8B]Preparing extraction...[/]")
            self._log_extraction_message("Preparation started...")  # Log to UI
            uri, db_name, coll, schema_dict, out_path, prep_err_text = (
                self._prepare_extraction_inputs()
            )

            if (
                prep_err_text
                or not uri
                or not db_name
                or not coll
                or not schema_dict
                or not out_path
            ):
                final_err_text = prep_err_text or Text.from_markup(
                    "[#BF616A]Critical error in input preparation.[/]"
                )
                self.extraction_status = final_err_text
                self._log_extraction_message(f"Preparation failed: {final_err_text.plain}")
                return

            self.extraction_status = Text.from_markup(
                f"[#EBCB8B]Starting extraction for '{coll}' to '{out_path}'...[/]"
            )
            self._log_extraction_message(
                f"Extraction for '{coll}' to '{out_path}' initiated by worker."
            )
            try:
                callable_with_args = functools.partial(
                    self._run_extraction_task, uri, db_name, coll, schema_dict, out_path
                )
                worker: Worker[str | None] = (
                    self.app.run_worker(  # Expects str (error) or None (success)
                        callable_with_args, thread=True, group="data_extraction"
                    )
                )
                error_msg_str = await worker.wait()
                if error_msg_str:
                    self.extraction_status = Text.from_markup(
                        f"[#BF616A]Extraction Failed: {error_msg_str}[/]"
                    )
                    self._log_extraction_message(f"FAILURE: {error_msg_str}")
                else:
                    success_msg = f"Extraction for '{coll}' completed to '{out_path}'."
                    self.extraction_status = Text.from_markup(f"[#A3BE8C]{success_msg}[/]")
                    self._log_extraction_message(f"SUCCESS: {success_msg}")
            except Exception as e:
                logger.error(f"Extraction handler error: {e}", exc_info=True)
                err_msg = f"Critical Error: {e!s}"
                self.extraction_status = Text.from_markup(f"[#BF616A]{err_msg}[/]")
                self._log_extraction_message(f"CRITICAL FAILURE: {e!s}")
