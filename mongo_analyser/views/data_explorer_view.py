import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    Static,
)
from textual.worker import Worker, WorkerCancelled

from mongo_analyser.core import DataExtractor
from mongo_analyser.dialogs import ErrorDialog

logger = logging.getLogger(__name__)


NO_DB_CONNECTION_TEXT_DE = Text.from_markup(
    "[#BF616A]MongoDB not connected. Please connect in the 'DB Connection' tab first.[/]"
)
NO_COLLECTION_SELECTED_TEXT_DE = Text.from_markup(
    "[#BF616A]No collection selected. Please select a collection from the dropdown.[/]"
)
UI_ERROR_INPUT_WIDGETS_TEXT_DE = Text.from_markup("[#BF616A]UI Error: Input widgets not found.[/]")


def _is_auth_error_from_op_failure_de(e: PyMongoOperationFailure) -> bool:
    if e.code == 13:
        return True
    error_msg_lower = str(e).lower()
    if "not authorized" in error_msg_lower or "unauthorized" in error_msg_lower:
        return True
    return False


class DataExplorerView(Container):
    sample_documents: reactive[List[Dict[str, Any]]] = reactive([])
    current_document_index: reactive[int] = reactive(0)
    status_message = reactive(Text("Select a collection and fetch documents."))
    feedback_message = reactive(Text(""))
    _feedback_timer_de: Optional[Any] = None

    def _get_default_sample_save_path(self) -> str:
        db_name = self.app.current_db_name
        collection_name = self.app.active_collection
        if db_name and collection_name:
            return f"output/{db_name}/{collection_name}_sample_docs.json"
        if collection_name:
            return f"output/{collection_name}_sample_docs.json"
        return "output/default_sample_docs.json"

    def on_mount(self) -> None:
        self._last_collections: List[str] = []
        self.update_collection_select()
        self._update_doc_nav_buttons_and_label()
        try:
            self.query_one("#data_fetch_loading_indicator", LoadingIndicator).display = False
        except NoMatches:
            logger.warning("DataExplorerView: #data_fetch_loading_indicator not found on mount.")

    def compose(self) -> ComposeResult:
        yield Label("Collection:")
        yield Select(
            [],
            prompt="Connect to DB to see collections",
            id="data_explorer_collection_select",
            allow_blank=True,
        )
        yield Label("Sample Size (Newest Docs):")
        yield Input(id="data_explorer_sample_size_input", value="10", placeholder="e.g., 10")
        yield Button("Fetch Sample Documents", id="fetch_documents_button", variant="primary")
        yield LoadingIndicator(id="data_fetch_loading_indicator")
        yield Static(self.status_message, id="data_explorer_status")
        with Horizontal(id="document_navigation"):
            yield Button("Previous", id="prev_doc_button", disabled=True)
            yield Label("Doc 0 of 0", id="doc_nav_label")
            yield Button("Next", id="next_doc_button", disabled=True)
        with VerticalScroll(id="document_display_area"):
            yield Markdown("```json\n{}\n```", id="document_json_view")
        with Horizontal(classes="action_button_group"):
            yield Button("Copy Current Doc", id="copy_current_doc_button")
            yield Button("Copy All Docs", id="copy_all_docs_button")
        yield Label("Save File Path:", classes="panel_title_small")
        yield Input(id="sample_docs_save_path_input", value=self._get_default_sample_save_path())
        yield Button("Save All Sampled Docs to File", id="save_sample_docs_button")
        yield Static(self.feedback_message, id="data_explorer_feedback_label")

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#data_explorer_collection_select", Select).focus()
        except NoMatches:
            logger.debug("DataExplorerView: Could not focus default select.")

    def update_collection_select(self) -> None:
        try:
            select_widget = self.query_one("#data_explorer_collection_select", Select)
            save_path_input = self.query_one("#sample_docs_save_path_input", Input)
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
                    save_path_input.value = self._get_path_for_collection_output(
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
            save_path_input.value = self._get_path_for_collection_output(final_selection_for_path)

            if needs_options_update or (
                select_widget.value == Select.BLANK and self.sample_documents
            ):
                self.sample_documents = []
                self.status_message = Text("Select a collection and fetch documents.")

        except NoMatches:
            logger.warning("DataExplorerView: Select or save path input not found for update.")
        except Exception as e:
            logger.error(f"Error in DataExplorerView.update_collection_select: {e}", exc_info=True)

    def _get_path_for_collection_output(self, collection_name: str | None) -> str:
        db_name = self.app.current_db_name
        if db_name and collection_name:
            return f"output/{db_name}/{collection_name}_sample_docs.json"
        if collection_name:
            return f"output/{collection_name}_sample_docs.json"
        return "output/default_sample_docs.json"

    @on(Select.Changed, "#data_explorer_collection_select")
    def on_collection_changed_de(self, event: Select.Changed) -> None:
        new_coll = str(event.value) if event.value != Select.BLANK else None
        if new_coll != self.app.active_collection:
            self.app.active_collection = new_coll

        try:
            save_path_input = self.query_one("#sample_docs_save_path_input", Input)
            save_path_input.value = self._get_path_for_collection_output(new_coll)
        except NoMatches:
            logger.warning("DataExplorerView: Save path input not found during collection change.")

        self.sample_documents = []
        self.status_message = Text("Collection changed. Fetch new sample documents.")
        self.feedback_message = Text("")

    @on(Button.Pressed, "#fetch_documents_button")
    async def fetch_documents_button_pressed(self) -> None:
        self.feedback_message = Text("")
        loader = self.query_one("#data_fetch_loading_indicator", LoadingIndicator)

        if not self.app.current_mongo_uri or not self.app.current_db_name:
            self.status_message = NO_DB_CONNECTION_TEXT_DE
            await self.app.push_screen(
                ErrorDialog("Connection Required", NO_DB_CONNECTION_TEXT_DE.plain)
            )
            return

        collection_name: Optional[str] = None
        try:
            coll_select = self.query_one("#data_explorer_collection_select", Select)
            if coll_select.value == Select.BLANK:
                self.status_message = NO_COLLECTION_SELECTED_TEXT_DE
                await self.app.push_screen(
                    ErrorDialog("Collection Required", NO_COLLECTION_SELECTED_TEXT_DE.plain)
                )
                return
            collection_name = str(coll_select.value)
        except NoMatches:
            self.status_message = UI_ERROR_INPUT_WIDGETS_TEXT_DE
            await self.app.push_screen(
                ErrorDialog("UI Error", UI_ERROR_INPUT_WIDGETS_TEXT_DE.plain)
            )
            return

        sample_size: int
        try:
            inp = self.query_one("#data_explorer_sample_size_input", Input)
            sample_size_str = inp.value.strip()
            if not sample_size_str:
                err_text = Text.from_markup("[#BF616A]Sample size cannot be empty.[/]")
                self.status_message = err_text
                await self.app.push_screen(ErrorDialog("Input Error", err_text.plain))
                return
            sample_size = int(sample_size_str)
            if sample_size <= 0:
                err_text = Text.from_markup("[#BF616A]Sample size must be a positive integer.[/]")
                self.status_message = err_text
                await self.app.push_screen(ErrorDialog("Input Error", err_text.plain))
                return
        except ValueError:
            err_text = Text.from_markup("[#BF616A]Invalid sample size. Must be an integer.[/]")
            self.status_message = err_text
            await self.app.push_screen(ErrorDialog("Input Error", err_text.plain))
            return
        except NoMatches:
            self.status_message = UI_ERROR_INPUT_WIDGETS_TEXT_DE
            await self.app.push_screen(
                ErrorDialog("UI Error", UI_ERROR_INPUT_WIDGETS_TEXT_DE.plain)
            )
            return

        uri = self.app.current_mongo_uri
        db_name = self.app.current_db_name

        self.status_message = Text(f"Fetching documents from '{collection_name}'…")
        self.sample_documents = []
        loader.display = True

        fetched_docs_result: Optional[List[Dict[str, Any]]] = None
        error_message_str: Optional[str] = None
        is_auth_issue = False

        try:
            worker: Worker[List[Dict[str, Any]]] = self.app.run_worker(
                partial(
                    DataExtractor.get_newest_documents, uri, db_name, collection_name, sample_size
                ),
                thread=True,
                group="doc_fetch",
            )
            fetched_docs_result = await worker.wait()

            if worker.is_cancelled:
                self.status_message = Text("Document fetching cancelled.")
            elif fetched_docs_result is not None:
                self.sample_documents = fetched_docs_result
                if not fetched_docs_result:
                    self.status_message = Text(f"No documents found in '{collection_name}'.")
                else:
                    self.status_message = Text(f"Fetched {len(fetched_docs_result)} documents.")

        except WorkerCancelled:
            self.status_message = Text("Document fetching cancelled during operation.")
        except PyMongoOperationFailure as e_op:
            logger.warning(
                f"DataExplorer: MongoDB operation failure fetching for '{collection_name}': {e_op}"
            )
            is_auth_issue = _is_auth_error_from_op_failure_de(e_op)
            err_details = e_op.details.get("errmsg", str(e_op)) if e_op.details else str(e_op)
            error_message_str = (
                f"Not authorized to read from collection '{collection_name}'."
                if is_auth_issue
                else f"Operation failed on '{collection_name}': {err_details}"
            )
        except (PyMongoConnectionFailure, ConnectionError) as e_conn:
            logger.error(
                f"DataExplorer: Connection error fetching for '{collection_name}': {e_conn}",
                exc_info=True,
            )
            error_message_str = f"Database Connection Error: {e_conn!s}"
        except Exception as e:
            logger.error(
                f"Unexpected error fetching documents for '{collection_name}': {e}", exc_info=True
            )
            error_message_str = f"Unexpected error: {e!s}"
        finally:
            if self.is_mounted:
                loader.display = False

        if error_message_str:
            self.status_message = Text.from_markup(f"[#BF616A]Error: {error_message_str[:100]}[/]")
            dialog_title = "Authorization Error" if is_auth_issue else "Fetch Error"
            await self.app.push_screen(ErrorDialog(dialog_title, error_message_str))

    def _update_document_view(self) -> None:
        try:
            md_view = self.query_one("#document_json_view", Markdown)
            if self.sample_documents and 0 <= self.current_document_index < len(
                self.sample_documents
            ):
                doc_to_display = self.sample_documents[self.current_document_index]
                doc_str = json.dumps(doc_to_display, indent=2, default=str)
                md_view.update(f"```json\n{doc_str}\n```")
            else:
                md_view.update("```json\n{}\n```")
        except NoMatches:
            logger.warning("DataExplorerView: Markdown view not found for update.")
        except IndexError:
            logger.warning("DataExplorerView: current_document_index out of bounds.")
            if self.is_mounted:
                try:
                    self.query_one("#document_json_view", Markdown).update("```json\n{}\n```")
                except NoMatches:
                    pass

    def _update_doc_nav_buttons_and_label(self) -> None:
        try:
            prev_button = self.query_one("#prev_doc_button", Button)
            next_button = self.query_one("#next_doc_button", Button)
            nav_label = self.query_one("#doc_nav_label", Label)
            total_docs = len(self.sample_documents)
            if total_docs > 0:
                nav_label.update(f"Doc {self.current_document_index + 1} of {total_docs}")
                prev_button.disabled = self.current_document_index <= 0
                next_button.disabled = self.current_document_index >= total_docs - 1
            else:
                nav_label.update("Doc 0 of 0")
                prev_button.disabled = True
                next_button.disabled = True
        except NoMatches:
            logger.warning("DataExplorerView: Navigation buttons or label not found.")

    @on(Button.Pressed, "#prev_doc_button")
    def previous_document_button_pressed(self) -> None:
        if self.current_document_index > 0:
            self.current_document_index -= 1
        self.feedback_message = Text("")

    @on(Button.Pressed, "#next_doc_button")
    def next_document_button_pressed(self) -> None:
        if self.current_document_index < len(self.sample_documents) - 1:
            self.current_document_index += 1
        self.feedback_message = Text("")

    @on(Button.Pressed, "#copy_current_doc_button")
    async def copy_current_doc_to_clipboard(self) -> None:
        if not self.sample_documents:
            self.feedback_message = Text.from_markup("[#D08770]No documents loaded to copy.[/]")
            self.app.notify("No documents to copy.", title="Copy Info", severity="warning")
            return
        if not (0 <= self.current_document_index < len(self.sample_documents)):
            self.feedback_message = Text.from_markup("[#BF616A]Invalid document index.[/]")
            self.app.notify("Invalid document index.", title="Copy Error", severity="error")
            return
        try:
            current_doc = self.sample_documents[self.current_document_index]
            doc_json_str = json.dumps(current_doc, indent=2, default=str)
            self.app.copy_to_clipboard(doc_json_str)
            self.feedback_message = Text.from_markup(
                "[#A3BE8C]Current document copied to clipboard![/]"
            )
            self.app.notify("Current document copied.", title="Copy Success")
        except Exception as e:
            logger.error(f"Error copying current document: {e}", exc_info=True)
            self.feedback_message = Text.from_markup(
                f"[#BF616A]Error copying document: {str(e)[:50]}[/]"
            )
            await self.app.push_screen(ErrorDialog("Copy Error", f"Could not copy document: {e!s}"))

    @on(Button.Pressed, "#copy_all_docs_button")
    async def copy_all_docs_to_clipboard(self) -> None:
        if not self.sample_documents:
            self.feedback_message = Text.from_markup("[#D08770]No documents loaded to copy.[/]")
            self.app.notify("No documents to copy.", title="Copy Info", severity="warning")
            return
        try:
            all_docs_json_str = json.dumps(self.sample_documents, indent=2, default=str)
            self.app.copy_to_clipboard(all_docs_json_str)
            self.feedback_message = Text.from_markup(
                f"[#A3BE8C]All {len(self.sample_documents)} sampled documents copied![/]"
            )
            self.app.notify(
                f"All {len(self.sample_documents)} documents copied.", title="Copy Success"
            )
        except Exception as e:
            logger.error(f"Error copying all documents: {e}", exc_info=True)
            self.feedback_message = Text.from_markup(
                f"[#BF616A]Error copying documents: {str(e)[:50]}[/]"
            )
            await self.app.push_screen(
                ErrorDialog("Copy Error", f"Could not copy documents: {e!s}")
            )

    @on(Button.Pressed, "#save_sample_docs_button")
    async def save_all_docs_to_file(self) -> None:
        try:
            save_path_input = self.query_one("#sample_docs_save_path_input", Input)
            save_path_str = save_path_input.value.strip()
        except NoMatches:
            self.app.notify("Save path input not found.", title="UI Error", severity="error")
            return
        if not save_path_str:
            self.feedback_message = Text.from_markup("[#BF616A]Save path cannot be empty.[/]")
            self.app.notify("Save path empty.", title="Save Error", severity="error")
            return
        if not self.sample_documents:
            self.feedback_message = Text.from_markup(
                "[#D08770]No documents to save. Fetch samples first.[/]"
            )
            self.app.notify("No documents to save.", title="Save Info", severity="warning")
            return
        save_path = Path(save_path_str)
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(self.sample_documents, f, indent=2, default=str)
            logger.info(f"Sampled documents have been saved to {save_path}")
            self.feedback_message = Text.from_markup(f"[#A3BE8C]Docs saved to {save_path.name}[/]")
            self.app.notify(f"Sampled documents saved to {save_path}", title="Save Success")
        except Exception as e:
            logger.error(f"Error saving documents to file {save_path}: {e}", exc_info=True)
            self.feedback_message = Text.from_markup(f"[#BF616A]Error saving: {str(e)[:50]}[/]")
            await self.app.push_screen(
                ErrorDialog("Save Error", f"Could not save documents: {e!s}")
            )

    def watch_status_message(self, new_status: Text) -> None:
        if self.is_mounted:
            try:
                self.query_one("#data_explorer_status", Static).update(new_status)
            except NoMatches:
                pass

    def watch_feedback_message(self, new_feedback: Text) -> None:
        if self.is_mounted:
            try:
                feedback_label = self.query_one("#data_explorer_feedback_label", Static)
                feedback_label.update(new_feedback)
                if self._feedback_timer_de is not None:
                    try:
                        self._feedback_timer_de.stop()
                    except AttributeError:
                        if hasattr(self._feedback_timer_de, "stop_no_wait"):
                            self._feedback_timer_de.stop_no_wait()
                    self._feedback_timer_de = None
                if new_feedback.plain:
                    self._feedback_timer_de = self.set_timer(
                        4, lambda: setattr(self, "feedback_message", Text(""))
                    )
            except NoMatches:
                pass
            except Exception as e:
                logger.error(f"Error in watch_feedback_message (DataExplorer): {e}", exc_info=True)

    def watch_sample_documents(
        self,
        old_docs: List[Dict[str, Any]],
        new_docs: List[Dict[str, Any]],
    ) -> None:
        if old_docs != new_docs or (not new_docs and self.current_document_index != 0):
            logger.debug(
                f"DataExplorerView: sample_documents changed. Old len: {len(old_docs)}, New len: {len(new_docs)}"
            )
            self.current_document_index = 0
            self._update_document_view()
            self._update_doc_nav_buttons_and_label()
            if self.is_mounted:
                self.feedback_message = Text("")

    def watch_current_document_index(self, old_idx: int, new_idx: int) -> None:
        if old_idx != new_idx and self.is_mounted:
            self._update_document_view()
            self._update_doc_nav_buttons_and_label()
            self.feedback_message = Text("")
