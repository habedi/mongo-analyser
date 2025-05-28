import json
import logging
from functools import partial
from typing import Any, Dict, List

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Markdown, Select, Static
from textual.worker import Worker, WorkerCancelled

from mongo_analyser.core.extractor import get_newest_documents
from mongo_analyser.dialogs import ErrorDialog

logger = logging.getLogger(__name__)


class DataExplorerView(Container):
    DEFAULT_CSS = """
    DataExplorerView {
        padding: 1;
        overflow: auto;
    }
    DataExplorerView > Label {
        margin-top: 1;
    }
    DataExplorerView > Select {
        width: 100%;
        margin-bottom: 1;
    }
    DataExplorerView > Input {
        width: 100%;
        margin-bottom: 1;
    }
    DataExplorerView > Button {
        width: 100%;
        margin-bottom: 1;
    }
    DataExplorerView #document_display_area {
        margin-top: 1;
        height: 25;
        border: round $primary-darken-1;
        background: $primary-background-darken-3;
        padding: 1;
        overflow: auto;
    }
    DataExplorerView #document_navigation {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    DataExplorerView #document_navigation Button {
        width: 1fr;
        max-width: 20;
        margin: 0 1;
    }
    DataExplorerView #doc_nav_label {
        width: auto;
        min-width: 10;
        padding: 0 1;
        text-align: center;
    }
    DataExplorerView #data_explorer_status {
        margin-top: 1;
        height: auto;
    }
    """

    sample_documents: reactive[List[Dict[str, Any]]] = reactive([])
    current_document_index: reactive[int] = reactive(0)
    status_message = reactive(Text("Select a collection and fetch documents."))

    def on_mount(self) -> None:
        self._last_collections: List[str] = []
        self.update_collection_select()
        self._update_doc_nav_buttons_and_label()

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
        yield Static(self.status_message, id="data_explorer_status")
        with Horizontal(id="document_navigation"):
            yield Button("Previous", id="prev_doc_button", disabled=True)
            yield Label("Doc 0 of 0", id="doc_nav_label")
            yield Button("Next", id="next_doc_button", disabled=True)
        with VerticalScroll(id="document_display_area"):
            yield Markdown("```json\n{}\n```", id="document_json_view")

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#data_explorer_collection_select", Select).focus()
        except NoMatches:
            logger.debug("DataExplorerView: Could not focus default select.")

    def update_collection_select(self) -> None:
        try:
            select_widget = self.query_one("#data_explorer_collection_select", Select)
            collections = self.app.available_collections
            if collections == self._last_collections:
                return
            self._last_collections = list(collections)
            if collections:
                options = [(c, c) for c in collections]
                select_widget.set_options(options)
                select_widget.disabled = False
                select_widget.prompt = "Select Collection"
                if (
                    self.app.active_collection in collections
                    and select_widget.value != self.app.active_collection
                ):
                    select_widget.value = self.app.active_collection
            else:
                select_widget.set_options([])
                select_widget.prompt = "Connect to DB to see collections"
                select_widget.disabled = True
                select_widget.value = Select.BLANK
            self.sample_documents = []
        except NoMatches:
            logger.warning(
                "DataExplorerView: #data_explorer_collection_select not found for update."
            )
        except Exception as e:
            logger.error(f"Error in DataExplorerView.update_collection_select: {e}", exc_info=True)

    @on(Select.Changed, "#data_explorer_collection_select")
    def on_collection_changed_de(self, event: Select.Changed) -> None:
        new_coll = str(event.value) if event.value != Select.BLANK else None
        if new_coll == self.app.active_collection:
            return
        self.app.active_collection = new_coll
        self.sample_documents = []
        self.status_message = Text("Collection changed. Fetch new documents.")

    @on(Button.Pressed, "#fetch_documents_button")
    async def fetch_documents_button_pressed(self) -> None:
        uri = self.app.current_mongo_uri
        db_name = self.app.current_db_name

        # find out which collection is selected…
        try:
            sel = self.query_one("#data_explorer_collection_select", Select)
            collection_name = None if sel.value == Select.BLANK else str(sel.value)
        except NoMatches:
            await self.app.push_screen(ErrorDialog("Error", "Collection select widget not found."))
            return

        if not uri or not db_name or not collection_name:
            self.status_message = Text.from_markup(
                "[#BF616A]MongoDB not connected or collection not selected.[/]"
            )
            await self.app.push_screen(
                ErrorDialog("Input Error", "Connect to DB and select a collection first.")
            )
            return

        # parse sample size…
        try:
            inp = self.query_one("#data_explorer_sample_size_input", Input)
            sample_size = int(inp.value)
            if sample_size <= 0:
                raise ValueError("non-positive")
        except Exception:
            await self.app.push_screen(ErrorDialog("Input Error", "Invalid sample size."))
            return

        self.status_message = Text(f"Fetching documents from '{collection_name}'…")
        # clear any previous docs
        self.sample_documents = []

        try:
            # **FIX**: wrap your target in partial() so run_worker only ever
            # sees one positional argument and your group kwarg appears once.
            worker: Worker[List[Dict[str, Any]]] = self.app.run_worker(
                partial(
                    get_newest_documents,
                    uri,
                    db_name,
                    collection_name,
                    sample_size,
                ),
                thread=True,
                group="doc_fetch",
            )
            fetched_docs = await worker.wait()

            if worker.is_cancelled:
                self.status_message = Text("Document fetching cancelled.")
                return

            self.sample_documents = fetched_docs
            if not fetched_docs:
                self.status_message = Text(f"No documents found in '{collection_name}'.")
            else:
                self.status_message = Text(f"Fetched {len(fetched_docs)} documents.")
        except WorkerCancelled:
            self.status_message = Text("Document fetching cancelled during operation.")
        except Exception as e:
            logger.error(f"Error fetching documents: {e}", exc_info=True)
            self.status_message = Text.from_markup(f"[#BF616A]Error: {e}[/]")
            await self.app.push_screen(ErrorDialog("Fetch Error", str(e)))

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
            logger.warning(
                "DataExplorerView: current_document_index out of bounds during view update."
            )
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
            if total_docs > 0 and 0 <= self.current_document_index < total_docs:
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

    @on(Button.Pressed, "#next_doc_button")
    def next_document_button_pressed(self) -> None:
        if self.current_document_index < len(self.sample_documents) - 1:
            self.current_document_index += 1

    def watch_status_message(self, new_status: Text) -> None:
        if self.is_mounted:
            try:
                self.query_one("#data_explorer_status", Static).update(new_status)
            except NoMatches:
                pass

    def watch_sample_documents(
        self,
        old_docs: List[Dict[str, Any]],
        new_docs: List[Dict[str, Any]],
    ) -> None:
        """Whenever sample_documents changes, reset to doc 0 and redraw everything."""
        if old_docs != new_docs:
            logger.debug(
                f"DataExplorerView: sample_documents changed. "
                f"Old len: {len(old_docs)}, New len: {len(new_docs)}"
            )
            # reset to first document
            self.current_document_index = 0

            # force a UI refresh of the JSON view and nav buttons
            self._update_document_view()
            self._update_doc_nav_buttons_and_label()

    def watch_current_document_index(self, old_idx: int, new_idx: int) -> None:
        if old_idx != new_idx and self.is_mounted:
            self._update_document_view()
            self._update_doc_nav_buttons_and_label()
