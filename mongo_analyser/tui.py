import asyncio
import functools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytz
from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Markdown,
    Select,
    Static,
    Tab,
    Tabs,
)
from textual.worker import Worker, WorkerFailed, WorkerState

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.core.analyser import SchemaAnalyser
from mongo_analyser.core.extractor import DataExtractor
from mongo_analyser.core.shared import redact_uri_password
from mongo_analyser.llm_chat import LiteLLMChat, LLMChat

logger = logging.getLogger(__name__)


class ChatView(Container):
    current_llm_worker: Worker | None = None
    llm_client_instance: LLMChat | None = None

    USER_LINE_BG_STYLE = "on dodger_blue1"
    AI_LINE_BG_STYLE = "on sea_green3"
    SYS_LINE_BG_STYLE = "on grey30"

    USER_PREFIX_Styled = Text.from_markup("[b white]USER:[/] ")
    AI_PREFIX_Styled = Text.from_markup("[b white]AI:[/] ")
    SYSTEM_PREFIX_Styled = Text.from_markup("[b bright_yellow]SYSTEM:[/] ")

    def _log_chat_message(self, line_bg_style: str, prefix_styled: Text, message_content: str):
        log_widget = self.query_one("#chat_log_widget", Log)
        lines_in_message = message_content.splitlines()
        if not lines_in_message and len(message_content) >= 0:
            lines_in_message = [message_content]

        for line_part in lines_in_message:
            final_markup_string = f"[{line_bg_style}]{prefix_styled.markup}{line_part}[/]"
            log_widget.write_line(final_markup_string)

    def on_mount(self) -> None:
        self.chat_history: list[dict[str, str]] = []
        self.llm_client_instance = None
        log_widget = self.query_one("#chat_log_widget", Log)
        log_widget.clear()
        self._log_chat_message(
            self.SYS_LINE_BG_STYLE,
            self.SYSTEM_PREFIX_Styled,
            "ChatView initialized. Default provider models loading...",
        )
        self._update_chat_status_line(status="Initializing...")

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        log_widget = self.query_one("#chat_log_widget", Log)
        log_widget.clear()
        self._log_chat_message(self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, status_message)
        self._update_chat_status_line(status="Idle", current_messages=0)

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat_interface_horizontal_layout"):
            with Vertical(id="chat_main_area", classes="chat_column_main"):
                yield Static(
                    "Provider: N/A | Model: N/A | Context: 0/0 | Status: Idle",
                    id="chat_status_line",
                    classes="chat_status",
                )
                with VerticalScroll(id="chat_log_scroll", classes="chat_log_container"):
                    yield Log(id="chat_log_widget", auto_scroll=True, highlight=True)
                with Horizontal(id="chat_input_bar", classes="chat_input_container"):
                    yield Input(
                        placeholder="Type a message...",
                        id="chat_message_input",
                        classes="chat_input_text",
                    )
                    yield Button(
                        "Send",
                        id="send_chat_message_button",
                        variant="primary",
                        classes="chat_button",
                    )
                    yield Button("Stop", id="stop_chat_message_button", classes="chat_button")
            with VerticalScroll(id="chat_config_panel", classes="chat_column_sidebar"):
                yield Label("Session Config", classes="panel_title")
                yield Label("Name:")
                yield Input(placeholder="New Chat X", id="chat_session_name")

                yield Label("Provider:")
                providers = [("Ollama", "ollama"), ("OpenAI", "openai"), ("Google", "google")]
                yield Select(
                    providers,
                    prompt="Select Provider",
                    id="llm_provider_select",
                    allow_blank=False,
                    value="ollama",
                )

                yield Label("Model:")
                yield Select(
                    [],
                    prompt="Select Model",
                    id="llm_model_select",
                    allow_blank=True,
                    value=Select.BLANK,
                )

                yield Label("Temperature:")
                yield Input(placeholder="0.7", id="llm_temperature", value="0.7")
                yield Label("Max Context Size / Output Tokens:")
                yield Input(placeholder="e.g., 2048", id="llm_max_context_size", value="2048")

                yield Label("Reasoning Effort: (UI Only)")
                effort_options = [
                    ("Low", "low"),
                    ("Medium", "medium"),
                    ("High", "high"),
                    ("Max", "max"),
                ]
                yield Select(
                    effort_options,
                    prompt="Select Effort",
                    id="llm_reasoning_effort",
                    allow_blank=False,
                    value="medium",
                )

                yield Label("Reasoning Budget: (UI Only)")
                yield Input(placeholder="e.g., 0, 100", id="llm_reasoning_budget", value="0")

                yield Button(
                    "New/Reset Session",
                    id="new_update_chat_session_button",
                    classes="config_button",
                )

    def _update_chat_status_line(
        self, status: str = "Idle", current_messages: int | None = None
    ) -> None:
        try:
            provider_select = self.query_one("#llm_provider_select", Select)
            model_select = self.query_one("#llm_model_select", Select)
            max_ctx_input = self.query_one("#llm_max_context_size", Input)
            chat_status_widget = self.query_one("#chat_status_line", Static)

            provider_val = provider_select.value
            model_val = model_select.value

            provider_display = (
                str(provider_val).capitalize()
                if provider_val not in (None, Select.BLANK)
                else "N/A"
            )
            model_display = str(model_val) if model_val not in (None, Select.BLANK) else "N/A"

            if self.llm_client_instance:
                provider_display = str(
                    getattr(self.llm_client_instance, "provider_hint", provider_display)
                ).capitalize()
                model_display = getattr(self.llm_client_instance, "raw_model_name", model_display)

            current_msg_count = (
                current_messages if current_messages is not None else len(self.chat_history) // 2
            )
            context_display = f"{current_msg_count} prompts / {max_ctx_input.value} tokens"

            chat_status_widget.update(
                f"Provider: {provider_display} | Model: {model_display} | Context: {context_display} | Status: {status}"
            )
        except NoMatches:
            logger.warning("Could not update chat status line, a widget was not found.")

    async def _load_models_for_provider(self, provider_value: str | object) -> None:
        if provider_value == Select.BLANK or provider_value is None:
            self._log_chat_message(
                self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, "Provider cleared."
            )
            return

        provider_value_str = str(provider_value)
        model_select_widget = self.query_one("#llm_model_select", Select)

        model_select_widget.set_options([])
        model_select_widget.value = Select.BLANK  # type: ignore
        model_select_widget.disabled = True
        original_prompt = "Select Model"
        model_select_widget.prompt = f"Loading models for {provider_value_str}..."
        self._update_chat_status_line(status=f"Loading {provider_value_str} models...")
        self._log_chat_message(
            self.SYS_LINE_BG_STYLE,
            self.SYSTEM_PREFIX_Styled,
            f"Fetching models for {provider_value_str}...",
        )

        listed_models: list[str] = []
        error_message: str | None = None

        client_config_for_list: Dict[str, Any] = {}

        try:
            callable_with_args = functools.partial(
                LiteLLMChat.list_models,
                provider=provider_value_str,
                client_config=client_config_for_list,
            )
            worker = self.app.run_worker(callable_with_args, thread=True)  # type: ignore
            listed_models = await worker.wait()
        except Exception as e:
            logger.error(f"Error listing models for '{provider_value_str}': {e}", exc_info=True)
            error_text = str(e)
            error_text = error_text[:37] + "..." if len(error_text) > 40 else error_text
            error_message = f"Failed to list models: {error_text}"

        model_select_widget.disabled = False
        new_status_for_line = "Ready"
        default_model_to_set = None

        if error_message:
            model_select_widget.prompt = error_message
            new_status_for_line = "Model list error"
            self._log_chat_message(self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, error_message)
        elif listed_models:
            model_options = [(name, name) for name in listed_models]
            model_select_widget.set_options(model_options)
            model_select_widget.prompt = original_prompt
            self._log_chat_message(
                self.SYS_LINE_BG_STYLE,
                self.SYSTEM_PREFIX_Styled,
                f"{len(listed_models)} models found for {provider_value_str}.",
            )

            if provider_value_str == "openai":
                preferred_models = ["gpt-4o-mini", "gpt-3.5-turbo"]
                for m in preferred_models:
                    if m in listed_models:
                        default_model_to_set = m
                        break
            elif provider_value_str == "google":
                preferred_models = [
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-pro-latest",
                    "gemini-pro",
                ]
                for m in preferred_models:
                    if m in listed_models:
                        default_model_to_set = m
                        break

            if default_model_to_set:
                model_select_widget.value = default_model_to_set  # type: ignore
            elif provider_value_str == "ollama":
                new_status_for_line = "Select Ollama model"
            else:
                new_status_for_line = "Select model"
        else:
            model_select_widget.prompt = "No models found."
            new_status_for_line = "No models"
            self._log_chat_message(
                self.SYS_LINE_BG_STYLE,
                self.SYSTEM_PREFIX_Styled,
                f"No models found for {provider_value_str}.",
            )

        if not default_model_to_set:
            self._update_chat_status_line(status=new_status_for_line)

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_provider_select":
            if self.llm_client_instance:
                self.llm_client_instance = None
                self._log_chat_message(
                    self.SYS_LINE_BG_STYLE,
                    self.SYSTEM_PREFIX_Styled,
                    "Provider changed. LLM client cleared.",
                )
            await self._load_models_for_provider(event.value)

        elif event.select.id == "llm_model_select":
            model_value = event.value
            if model_value is not None and model_value != Select.BLANK:
                self._update_chat_status_line(status=f"Initializing {model_value!s}...")
                self._log_chat_message(
                    self.SYS_LINE_BG_STYLE,
                    self.SYSTEM_PREFIX_Styled,
                    f"Setting up model: {model_value!s}...",
                )
                self.chat_history.clear()
                self.query_one("#chat_message_input", Input).value = ""
                client_created = self._create_and_set_llm_client()
                if client_created:
                    self._log_chat_message(
                        self.SYS_LINE_BG_STYLE,
                        self.SYSTEM_PREFIX_Styled,
                        "Session ready. LLM client configured.",
                    )
                    self._update_chat_status_line(status="Ready")
                else:
                    self._update_chat_status_line(status="Client Error")
                self.query_one("#chat_message_input", Input).focus()
            else:
                if self.llm_client_instance:
                    self.llm_client_instance = None
                    self._log_chat_message(
                        self.SYS_LINE_BG_STYLE,
                        self.SYSTEM_PREFIX_Styled,
                        "Model deselected. LLM client cleared.",
                    )
                self._update_chat_status_line(status="Select model")

    def _create_and_set_llm_client(self) -> bool:
        provider_select = self.query_one("#llm_provider_select", Select)
        model_select = self.query_one("#llm_model_select", Select)
        temp_input = self.query_one("#llm_temperature", Input)
        max_ctx_input = self.query_one("#llm_max_context_size", Input)

        provider_hint = str(provider_select.value)
        raw_model_name = str(model_select.value)

        def _sys_msg(message: str):
            self._log_chat_message(self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, message)

        if not provider_hint or provider_hint == str(Select.BLANK):
            _sys_msg("Please select a provider.")
            return False
        if not raw_model_name or raw_model_name == str(Select.BLANK):
            _sys_msg("Please select a model.")
            return False

        try:
            temperature = float(temp_input.value)
        except ValueError:
            _sys_msg("Temperature must be a valid number.")
            return False
        if not (0.0 <= temperature <= 2.0):
            _sys_msg("Temp must be 0.0-2.0.")
            return False
        try:
            max_c_val = int(max_ctx_input.value)
        except ValueError:
            _sys_msg("Max Context Size must be a valid int.")
            return False
        if not (max_c_val > 0):
            _sys_msg("Max Context Size must be positive.")
            return False

        client_constructor_kwargs: Dict[str, Any] = {
            "provider_hint": provider_hint,
            "temperature": temperature,
            "max_tokens": max_c_val,
        }
        client_constructor_kwargs = {
            k: v for k, v in client_constructor_kwargs.items() if v is not None
        }

        new_client: LLMChat | None = None
        try:
            new_client = LiteLLMChat(model_name=raw_model_name, **client_constructor_kwargs)
            if new_client:
                self.llm_client_instance = new_client
                logger.info(
                    f"LiteLLM client configured for actual model '{self.llm_client_instance.model_name}' (Provider hint: {provider_hint})"
                )
                self._update_chat_status_line(status="Ready")
                return True
            else:
                _sys_msg(f"Failed to create LiteLLM client for '{raw_model_name}'.")
                self.llm_client_instance = None
                return False
        except Exception as e:
            logger.error(
                f"Failed to create LiteLLM client for provider {provider_hint}, model {raw_model_name}: {e}",
                exc_info=True,
            )
            _sys_msg(f"Error creating LLM client: {e.__class__.__name__} - {str(e)[:50]}...")
            self.llm_client_instance = None
            return False

    async def _send_user_message(self) -> None:
        message_input = self.query_one("#chat_message_input", Input)
        if self.llm_client_instance is None:
            self._log_chat_message(
                self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, "LLM not configured."
            )
            return
        user_message = message_input.value.strip()
        if not user_message:
            return

        send_button = self.query_one("#send_chat_message_button", Button)
        stop_button = self.query_one("#stop_chat_message_button", Button)
        self._log_chat_message(self.USER_LINE_BG_STYLE, self.USER_PREFIX_Styled, user_message)
        history_for_llm = self.chat_history.copy()
        self.chat_history.append({"role": "user", "content": user_message})
        message_input.value = ""
        message_input.disabled = True
        send_button.disabled = True
        stop_button.disabled = False
        self._update_chat_status_line(
            status="Sending...", current_messages=len(self.chat_history) // 2
        )
        self._log_chat_message(self.AI_LINE_BG_STYLE, self.AI_PREFIX_Styled, "is thinking...")

        active_client = self.llm_client_instance

        def task_to_run_in_worker():
            return active_client.send_message(message=user_message, history=history_for_llm)

        self.current_llm_worker = self.app.run_worker(
            task_to_run_in_worker, thread=True, group="llm_call"
        )  # type: ignore

        try:
            response_text = await self.current_llm_worker.wait()
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.SUCCESS:
                self._log_chat_message(self.AI_LINE_BG_STYLE, self.AI_PREFIX_Styled, response_text)
                self.chat_history.append({"role": "assistant", "content": response_text})
        except WorkerFailed as e:
            error_to_log = e.error
            msg = (
                "LLM call cancelled."
                if isinstance(error_to_log, asyncio.CancelledError)
                else f"LLM Error: {error_to_log.__class__.__name__} - {str(error_to_log)[:50]}..."
            )
            logger.error(f"LLM WorkerFailed: {error_to_log}", exc_info=error_to_log)
            self._log_chat_message(self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, msg)
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
        except Exception as e:
            logger.error(f"LLM communication error: {e}", exc_info=True)
            self._log_chat_message(
                self.SYS_LINE_BG_STYLE,
                self.SYSTEM_PREFIX_Styled,
                f"Unexpected error: {e.__class__.__name__}",
            )
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
        finally:
            self.current_llm_worker = None
            message_input.disabled = False
            send_button.disabled = False
            stop_button.disabled = True
            self._update_chat_status_line(status="Idle")
            message_input.focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send_chat_message_button":
            await self._send_user_message()
        elif button_id == "stop_chat_message_button":
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
                self.current_llm_worker.cancel()
                self._log_chat_message(
                    self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, "Stopping LLM response..."
                )
                self.query_one("#stop_chat_message_button", Button).disabled = True
            else:
                self._log_chat_message(
                    self.SYS_LINE_BG_STYLE, self.SYSTEM_PREFIX_Styled, "No active LLM call."
                )
        elif button_id == "new_update_chat_session_button":
            self.chat_history.clear()
            self.query_one("#chat_message_input", Input).value = ""
            provider_select = self.query_one("#llm_provider_select", Select)
            model_select = self.query_one("#llm_model_select", Select)
            if provider_select.value == Select.BLANK or model_select.value == Select.BLANK:  # type: ignore
                self._log_chat_message(
                    self.SYS_LINE_BG_STYLE,
                    self.SYSTEM_PREFIX_Styled,
                    "Select provider and model first.",
                )
                self._update_chat_status_line(status="Needs config")
                return
            client_created = self._create_and_set_llm_client()
            if client_created:
                self._log_chat_message(
                    self.SYS_LINE_BG_STYLE,
                    self.SYSTEM_PREFIX_Styled,
                    "New session. LLM (re)configured.",
                )
                self._update_chat_status_line(status="Ready")
            else:
                self._update_chat_status_line(status="Client Error")
            self.query_one("#chat_message_input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_message_input":
            await self._send_user_message()


class DBConnectionView(Container):
    connection_status = reactive("[orange3]Not Connected[/orange3]")

    def compose(self) -> ComposeResult:
        yield Label("MongoDB Connection", classes="panel_title")
        yield Label("MongoDB URI:")
        yield Input(
            id="mongo_uri_input", value=os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        )
        yield Label("Database Name (for connection context):")
        yield Input(id="mongo_db_name_input", placeholder="e.g., my_database (optional if in URI)")
        yield Button("Connect & List Collections", variant="primary", id="connect_mongo_button")
        yield Static(self.connection_status, id="mongo_connection_status_label")
        yield Label("Collections:", classes="panel_title_small", id="collections_title_label")
        with VerticalScroll(id="collections_list_scroll", classes="collections_list_container"):
            yield Static("Connect to a database to see collections.", id="collections_placeholder")

    def watch_connection_status(self, new_status: str) -> None:
        if self.is_mounted:
            try:
                self.query_one("#mongo_connection_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #mongo_connection_status_label.")

    async def _connect_and_list_collections_task(
        self,
    ) -> tuple[bool, str, list[str], str | None, str | None]:
        uri = self.query_one("#mongo_uri_input", Input).value
        db_name_from_input = self.query_one("#mongo_db_name_input", Input).value or None
        core_db_manager.disconnect_mongo(core_db_manager.DEFAULT_ALIAS)
        is_active = core_db_manager.db_connection_active(uri=uri, db_name=db_name_from_input)
        if not is_active:
            return False, f"Connection Failed to {redact_uri_password(uri)}", [], None, None
        db_instance = core_db_manager.get_mongo_db()
        actual_db_name = db_instance.name
        collections = sorted(db_instance.list_collection_names())
        status_message = f"Connected to {redact_uri_password(uri)} (DB: {actual_db_name})"
        return True, status_message, collections, uri, actual_db_name

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect_mongo_button":
            collections_list_scroll = self.query_one("#collections_list_scroll")
            title_label = self.query_one("#collections_title_label", Label)
            self.connection_status = "Connecting..."  # type: ignore
            collections_list_scroll.remove_children()  # type: ignore
            collections_list_scroll.mount(Static("Working..."))
            title_label.visible = False
            try:
                worker: Worker[tuple[bool, str, list[str], str | None, str | None]] = (
                    self.app.run_worker(
                        # type: ignore
                        self._connect_and_list_collections_task,
                        thread=True,
                    )
                )
                (
                    success,
                    status_msg,
                    collections_data,
                    connected_uri,
                    connected_db_name,
                ) = await worker.wait()
                collections_list_scroll.remove_children()  # type: ignore
                if success and connected_uri and connected_db_name:
                    self.connection_status = f"[green]{status_msg}[/green]"  # type: ignore
                    self.app.current_mongo_uri = connected_uri  # type: ignore
                    self.app.current_db_name = connected_db_name  # type: ignore
                    if collections_data:
                        title_label.visible = True
                        for name in collections_data:
                            collections_list_scroll.mount(Label(name))
                    else:
                        title_label.visible = True
                        collections_list_scroll.mount(Static("No collections found."))
                else:
                    self.connection_status = f"[red]{status_msg}[/red]"  # type: ignore
                    self.app.current_mongo_uri = None
                    self.app.current_db_name = None  # type: ignore
                    collections_list_scroll.mount(Static("Failed."))
            except Exception as e:
                logger.error(f"DB Op Error: {e}", exc_info=True)
                collections_list_scroll.remove_children()  # type: ignore
                collections_list_scroll.mount(Static(f"Error: {str(e)[:100]}"))
                self.connection_status = "[red]Error connecting.[/red]"
                self.app.current_mongo_uri = None
                self.app.current_db_name = None  # type: ignore


class SchemaAnalysisView(Container):
    analysis_status = reactive("Enter collection and click Analyze.")

    def compose(self) -> ComposeResult:
        yield Label("Schema Analysis", classes="panel_title")
        yield Label("Collection Name:")
        yield Input(
            id="schema_collection_input", placeholder="e.g., users (must be connected to DB first)"
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

    def watch_analysis_status(self, new_status: str) -> None:
        if self.is_mounted:
            try:
                self.query_one("#schema_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #schema_status_label.")

    def _prepare_analysis_inputs(self) -> tuple[str | None, str | None, str, str, str | None]:
        uri = self.app.current_mongo_uri
        db_name = self.app.current_db_name  # type: ignore
        if not uri or not db_name:
            return None, None, "", "", "MongoDB not connected. Connect in 'DB Connection' tab."
        collection_name = self.query_one("#schema_collection_input", Input).value.strip()
        sample_size_str = self.query_one("#schema_sample_size_input", Input).value.strip()
        if not collection_name:
            return uri, db_name, "", sample_size_str, "Collection name cannot be empty."
        try:
            int(sample_size_str)
        except ValueError:
            return uri, db_name, collection_name, "", "Sample size must be an integer."
        return uri, db_name, collection_name, sample_size_str, None

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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze_schema_button":
            table = self.query_one("#schema_results_table", DataTable)
            md_view = self.query_one("#schema_json_view", Markdown)
            self.analysis_status = "Preparing analysis..."  # type: ignore
            table.clear(columns=True)
            md_view.update("```json\n{}\n```")
            uri, db_name, collection_name, sample_size_str, prep_error = (
                self._prepare_analysis_inputs()
            )
            if prep_error or not uri or not db_name:
                self.analysis_status = (
                    f"[red]Error: {prep_error or 'Missing DB connection details'}[/red]"  # type: ignore
                )
                return
            sample_size = int(sample_size_str)
            self.analysis_status = f"Analyzing '{collection_name}' (sample: {sample_size})..."  # type: ignore
            try:
                callable_with_args = functools.partial(
                    self._run_analysis_task, uri, db_name, collection_name, sample_size
                )
                worker = self.app.run_worker(callable_with_args, thread=True)  # type: ignore
                schema_data, field_stats_data, hierarchical_schema, error_msg = await worker.wait()  # type: ignore
                if error_msg:
                    self.analysis_status = f"[red]Error: {error_msg}[/red]"
                    return  # type: ignore
                if schema_data and field_stats_data and hierarchical_schema is not None:
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
                        table.add_row("No schema fields found or analyzed.")  # type: ignore
                    try:
                        json_output = json.dumps(hierarchical_schema, indent=4)
                        md_view.update(f"```json\n{json_output}\n```")
                        self.analysis_status = "Analysis complete."  # type: ignore
                    except TypeError as te:
                        logger.error(f"JSON serialization error: {te}", exc_info=True)
                        md_view.update("```json\n// Error: Schema non-serializable.\n{}\n```")
                        self.analysis_status = "Analysis complete (schema display error)."  # type: ignore
                else:
                    self.analysis_status = "[yellow]Analysis completed with no data.[/yellow]"  # type: ignore
            except Exception as e:
                logger.error(f"Schema analysis handler error: {e}", exc_info=True)
                self.analysis_status = f"[red]Display Error: {e!s}[/red]"  # type: ignore


class DataExtractionView(Container):
    extraction_status = reactive("Enter details and click Start Extraction.")
    SYS_LINE_BG_STYLE = "on dark_slate_gray2"
    STATUS_PREFIX_Styled = Text.from_markup("[b light_cyan]EXTRACT:[/] ")

    def _log_extraction_message(self, message_content: str):
        log_widget = self.query_one("#extraction_log", Log)
        lines_in_message = message_content.splitlines()
        if not lines_in_message and len(message_content) > 0:
            lines_in_message = [message_content]
        elif not lines_in_message and len(message_content) == 0:
            lines_in_message = [""]
        for line_part in lines_in_message:
            final_markup_string = (
                f"[{self.SYS_LINE_BG_STYLE}]{self.STATUS_PREFIX_Styled.markup}{line_part}[/]"
            )
            log_widget.write_line(final_markup_string)

    def compose(self) -> ComposeResult:
        yield Label("Data Extraction", classes="panel_title")
        yield Label("Collection Name:")
        yield Input(
            placeholder="e.g., users (requires active DB connection)", id="extract_collection_name"
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

    def watch_extraction_status(self, new_status: str) -> None:
        if self.is_mounted:
            try:
                self.query_one("#extraction_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #extraction_status_label.")

    def _prepare_extraction_inputs(
        self,
    ) -> tuple[str | None, str | None, str, Dict | None, Path | None, str | None]:
        uri = self.app.current_mongo_uri
        db_name = self.app.current_db_name  # type: ignore
        if not uri or not db_name:
            return None, None, "", None, None, "MongoDB not connected."
        collection_name = self.query_one("#extract_collection_name", Input).value.strip()
        schema_f_path_str = self.query_one("#extract_schema_file", Input).value.strip()
        output_f_path_str = self.query_one("#extract_output_file", Input).value.strip()

        if not collection_name:
            return uri, db_name, "", None, None, "Collection name required."
        if not schema_f_path_str:
            return uri, db_name, collection_name, None, None, "Schema file path required."
        if not output_f_path_str:
            return uri, db_name, collection_name, None, None, "Output file path required."

        schema_file = Path(schema_f_path_str)
        output_file_p = Path(output_f_path_str)
        if not schema_file.is_file():
            return (
                uri,
                db_name,
                collection_name,
                None,
                output_file_p,
                f"Schema file not found: {schema_file}",
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
                f"Invalid JSON in schema: {schema_file}",
            )
        except IOError as e:
            return (
                uri,
                db_name,
                collection_name,
                None,
                output_file_p,
                f"Could not read schema {schema_file}: {e}",
            )
        return uri, db_name, collection_name, schema_data, output_file_p, None

    def _run_extraction_task(
        self, uri: str, db_name: str, collection_name: str, schema_dict: Dict, output_file: Path
    ) -> str | None:
        try:
            self._log_extraction_message(
                f"Process started for {collection_name}. Output: {output_file}. This might take a while..."
            )
            DataExtractor.extract_data(
                mongo_uri=uri,
                db_name=db_name,
                collection_name=collection_name,
                schema=schema_dict,
                output_file=output_file,
                tz=pytz.utc,
                batch_size=1000,
                limit=-1,
            )
            return None
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
            self.extraction_status = "Preparing extraction..."  # type: ignore
            self._log_extraction_message("Preparation started...")
            uri, db_name, coll, schema_dict, out_path, prep_err = self._prepare_extraction_inputs()

            if prep_err or not uri or not db_name or not schema_dict or not out_path:
                final_err = prep_err or "Critical error in input preparation."
                self.extraction_status = f"[red]Error: {final_err}[/red]"  # type: ignore
                self._log_extraction_message(f"Preparation failed: {final_err}")
                return

            self.extraction_status = f"Starting extraction for '{coll}' to '{out_path}'..."  # type: ignore
            self._log_extraction_message(
                f"Extraction for '{coll}' to '{out_path}' initiated by worker."
            )
            try:
                callable_with_args = functools.partial(
                    self._run_extraction_task, uri, db_name, coll, schema_dict, out_path
                )
                worker = self.app.run_worker(callable_with_args, thread=True)  # type: ignore
                error_msg = await worker.wait()
                if error_msg:
                    self.extraction_status = f"[red]Extraction Failed: {error_msg}[/red]"  # type: ignore
                    self._log_extraction_message(f"FAILURE: {error_msg}")
                else:
                    success_msg = f"Extraction for '{coll}' completed to '{out_path}'."
                    self.extraction_status = f"[green]{success_msg}[/green]"  # type: ignore
                    self._log_extraction_message(f"SUCCESS: {success_msg}")
            except Exception as e:
                logger.error(f"Extraction handler error: {e}", exc_info=True)
                err_msg = f"Critical Error: {e!s}"
                self.extraction_status = f"[red]{err_msg}[/red]"
                self._log_extraction_message(f"CRITICAL FAILURE: {e!s}")  # type: ignore


class MongoAnalyserApp(App[None]):
    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, key_display=None, priority=True),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme", show=True),
    ]
    current_mongo_uri: reactive[str | None] = reactive(None)  # type: ignore
    current_db_name: reactive[str | None] = reactive(None)  # type: ignore

    def compose(self) -> ComposeResult:
        yield Header()
        yield Tabs(
            Tab(label="Chat", id="tab_chat"),
            Tab(label="DB Connection", id="tab_db_connection"),
            Tab(label="Schema Analysis", id="tab_schema_analysis"),
            Tab(label="Data Extraction", id="tab_data_extraction"),
        )
        with ContentSwitcher(initial="view_chat"):
            yield ChatView(id="view_chat")
            yield DBConnectionView(id="view_db_connection")
            yield SchemaAnalysisView(id="view_schema_analysis")
            yield DataExtractionView(id="view_data_extraction")
        yield Footer()

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        view_id: str | None = None
        if event.tab.id:
            try:
                suffix = event.tab.id.split("_", 1)[1]
                view_id = f"view_{suffix}"
            except IndexError:
                logger.warning(f"Could not parse suffix from tab ID: {event.tab.id}")
                return
        if not view_id:
            logger.warning(f"No view_id for tab {event.tab.id}")
            return
        try:
            switcher = self.query_one(ContentSwitcher)
        except NoMatches:
            logger.error("Critical: ContentSwitcher not found.")
            return
        try:
            switcher.current = view_id
        except NoMatches:
            logger.warning(f"View '{view_id}' not in ContentSwitcher.")
            return
        except Exception as e:
            logger.error(f"Error setting ContentSwitcher to '{view_id}': {e}", exc_info=True)
            return
        try:
            if view_id == "view_chat":
                self.query_one("#chat_message_input", Input).focus()
            elif view_id == "view_db_connection":
                self.query_one("#mongo_uri_input", Input).focus()
            elif view_id == "view_schema_analysis":
                self.query_one("#schema_collection_input", Input).focus()
            elif view_id == "view_data_extraction":
                self.query_one("#extract_collection_name", Input).focus()
        except NoMatches:
            logger.debug(f"No default focus target in '{view_id}'.")
        except Exception as e:
            logger.error(f"Error focusing in '{view_id}': {e}", exc_info=True)

    def action_toggle_theme(self) -> None:
        self.dark = not self.dark


def main_interactive_tui():
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        filename="mongo_analyser_tui.log",
        mode="a",
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    logger.info("Starting Mongo Analyser TUI (Textual)...")
    app = MongoAnalyserApp()
    app.run()


if __name__ == "__main__":
    main_interactive_tui()
