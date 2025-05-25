import asyncio
import functools
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import pytz
from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure
from rich.text import Text
from textual._path import CSSPathType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.driver import Driver
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
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

from mongo_analyser.core import db as core_db_manager
from mongo_analyser.core.analyser import SchemaAnalyser
from mongo_analyser.core.extractor import DataExtractor
from mongo_analyser.core.shared import redact_uri_password
from mongo_analyser.llm_chat import LiteLLMChat, LLMChat

logger = logging.getLogger(__name__)


def _format_bytes_tui(size_bytes: Any) -> str:
    if size_bytes is None or not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    try:
        i = int(math.floor(math.log(size_bytes, 1024)))
        if i >= len(size_name):
            i = len(size_name) - 1  # Cap at TB for TUI display
    except ValueError:  # math domain error for log(0)
        i = 0
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


class ChatView(Container):
    current_llm_worker: Worker | None = None
    llm_client_instance: LLMChat | None = None

    USER_PREFIX_Styled = Text.from_markup("[b #ECEFF4]USER:[/] ")
    AI_PREFIX_Styled = Text.from_markup("[b #ECEFF4]AI:[/] ")
    SYSTEM_PREFIX_Styled = Text.from_markup("[b #EBCB8B]SYSTEM:[/] ")

    def _log_chat_message(self, line_bg_style: str, prefix_styled: Text, message_content: str):
        log_widget = self.query_one("#chat_log_widget", Log)
        lines_in_message = message_content.splitlines()
        if not lines_in_message and len(message_content) >= 0:
            lines_in_message = [message_content]

        for line_part in lines_in_message:
            plain_prefix = prefix_styled.plain
            plain_line_part = Text(line_part).plain
            log_widget.write_line(f"{plain_prefix}{plain_line_part}")

    def on_mount(self) -> None:
        self.chat_history: list[dict[str, str]] = []
        self.llm_client_instance = None
        self._reset_chat_log_and_status("ChatView initialized. Select provider and model.")

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        try:
            log_widget = self.query_one("#chat_log_widget", Log)
            log_widget.clear()
        except NoMatches:
            logger.warning("Chat log widget not found for clearing.")
        self._log_chat_message("", self.SYSTEM_PREFIX_Styled, status_message)
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

                yield Label("API Key (Optional, uses env if blank):")
                yield Input(placeholder="sk-...", id="llm_api_key", password=True)
                yield Label("Base URL (Optional, for custom endpoints):")
                yield Input(placeholder="http://localhost:11434", id="llm_base_url")

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
        except Exception as e:
            logger.error(f"Error updating chat status line: {e}", exc_info=True)

    async def _load_models_for_provider(self, provider_value: str | object) -> None:
        default_select_prompt = "Select Model"

        if provider_value == Select.BLANK or provider_value is None:
            self._log_chat_message("", self.SYSTEM_PREFIX_Styled, "Provider cleared.")
            self.query_one("#llm_model_select", Select).set_options([])
            self.query_one("#llm_model_select", Select).value = Select.BLANK  # type: ignore
            self.query_one("#llm_model_select", Select).prompt = default_select_prompt
            self._update_chat_status_line(status="Provider cleared")
            return

        provider_value_str = str(provider_value)
        model_select_widget = self.query_one("#llm_model_select", Select)
        api_key_input = self.query_one("#llm_api_key", Input)
        base_url_input = self.query_one("#llm_base_url", Input)

        model_select_widget.set_options([])
        model_select_widget.value = Select.BLANK  # type: ignore
        model_select_widget.disabled = True
        model_select_widget.prompt = f"Loading models for {provider_value_str}..."
        self._update_chat_status_line(status=f"Loading {provider_value_str} models...")
        self._log_chat_message(
            "", self.SYSTEM_PREFIX_Styled, f"Fetching models for {provider_value_str}..."
        )

        listed_models: list[str] = []
        error_message: str | None = None

        client_config_for_list: Dict[str, Any] = {}
        if api_key_input.value:
            client_config_for_list["api_key"] = api_key_input.value
        if base_url_input.value:
            client_config_for_list["base_url"] = base_url_input.value

        try:
            callable_with_args = functools.partial(
                LiteLLMChat.list_models,
                provider=provider_value_str,
                client_config=client_config_for_list,
            )
            worker = self.app.run_worker(callable_with_args, thread=True,
                                         group="model_listing")  # type: ignore
            listed_models = await worker.wait()
        except WorkerCancelled:
            logger.warning(f"Model listing worker for '{provider_value_str}' was cancelled.")
            error_message = "Model loading cancelled or interrupted."
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
            self._log_chat_message("", self.SYSTEM_PREFIX_Styled, error_message)
        elif listed_models:
            model_options = [(name, name) for name in listed_models]
            model_select_widget.set_options(model_options)
            model_select_widget.prompt = default_select_prompt
            self._log_chat_message(
                "",
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
                "", self.SYSTEM_PREFIX_Styled, f"No models found for {provider_value_str}."
            )

        if not default_model_to_set:
            self._update_chat_status_line(status=new_status_for_line)

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_provider_select":
            if self.llm_client_instance:
                self.llm_client_instance = None
                self._log_chat_message(
                    "", self.SYSTEM_PREFIX_Styled, "Provider changed. LLM client cleared."
                )
            await self._load_models_for_provider(event.value)

        elif event.select.id == "llm_model_select":
            model_value = event.value
            if model_value is not None and model_value != Select.BLANK:
                self._reset_chat_log_and_status(
                    status_message=f"Setting up model: {model_value!s}. Session reset."
                )
                self._update_chat_status_line(status=f"Initializing {model_value!s}...")

                self.query_one("#chat_message_input", Input).value = ""

                client_created = self._create_and_set_llm_client()
                if client_created:
                    self._log_chat_message(
                        "", self.SYSTEM_PREFIX_Styled, "Session ready. LLM client configured."
                    )
                    self._update_chat_status_line(status="Ready")
                else:
                    self._log_chat_message(
                        "", self.SYSTEM_PREFIX_Styled, "Failed to configure LLM client."
                    )
                    self._update_chat_status_line(status="Client Error")
                self.query_one("#chat_message_input", Input).focus()
            else:
                if self.llm_client_instance:
                    self.llm_client_instance = None
                    self._log_chat_message(
                        "", self.SYSTEM_PREFIX_Styled, "Model deselected. LLM client cleared."
                    )
                self._update_chat_status_line(status="Select model")

    def _create_and_set_llm_client(self) -> bool:
        provider_select = self.query_one("#llm_provider_select", Select)
        model_select = self.query_one("#llm_model_select", Select)
        temp_input = self.query_one("#llm_temperature", Input)
        max_ctx_input = self.query_one("#llm_max_context_size", Input)
        api_key_input = self.query_one("#llm_api_key", Input)
        base_url_input = self.query_one("#llm_base_url", Input)

        provider_hint = str(provider_select.value)
        raw_model_name = str(model_select.value)

        def _sys_msg_plain(message: str):
            self._log_chat_message("", self.SYSTEM_PREFIX_Styled, message)

        if not provider_hint or provider_hint == str(Select.BLANK):
            _sys_msg_plain("Please select a provider.")
            return False
        if not raw_model_name or raw_model_name == str(Select.BLANK):
            _sys_msg_plain("Please select a model.")
            return False

        try:
            temperature = float(temp_input.value)
        except ValueError:
            _sys_msg_plain("Temperature must be a valid number.")
            return False
        if not (0.0 <= temperature <= 2.0):
            _sys_msg_plain("Temp must be 0.0-2.0.")
            return False

        try:
            max_c_val = int(max_ctx_input.value)
        except ValueError:
            _sys_msg_plain("Max Context Size must be a valid int.")
            return False
        if not (max_c_val > 0):
            _sys_msg_plain("Max Context Size must be positive.")
            return False

        client_constructor_kwargs: Dict[str, Any] = {
            "provider_hint": provider_hint,
            "temperature": temperature,
            "max_tokens": max_c_val,
        }
        if api_key_input.value:
            client_constructor_kwargs["api_key"] = api_key_input.value
        if base_url_input.value:
            client_constructor_kwargs["base_url"] = base_url_input.value

        client_constructor_kwargs = {
            k: v for k, v in client_constructor_kwargs.items() if v is not None
        }

        new_client: LLMChat | None = None
        try:
            new_client = LiteLLMChat(model_name=raw_model_name, **client_constructor_kwargs)
            if new_client:
                self.llm_client_instance = new_client
                logger.info(
                    f"LiteLLM client configured for actual model '{self.llm_client_instance.model_name}' (Provider hint: {provider_hint}, Raw: {raw_model_name})"
                )
                return True
            else:
                _sys_msg_plain(f"Failed to create LiteLLM client for '{raw_model_name}'.")
                self.llm_client_instance = None
                return False
        except Exception as e:
            logger.error(
                f"Failed to create LiteLLM client for provider {provider_hint}, model {raw_model_name}: {e}",
                exc_info=True,
            )
            _sys_msg_plain(f"Error creating LLM client: {e.__class__.__name__} - {str(e)[:50]}...")
            self.llm_client_instance = None
            return False

    async def _send_user_message(self) -> None:
        message_input = self.query_one("#chat_message_input", Input)
        if self.llm_client_instance is None:
            self._log_chat_message(
                "",
                self.SYSTEM_PREFIX_Styled,
                "LLM not configured. Please select provider/model and reset session.",
            )
            return
        user_message = message_input.value.strip()
        if not user_message:
            return

        send_button = self.query_one("#send_chat_message_button", Button)
        stop_button = self.query_one("#stop_chat_message_button", Button)

        self._log_chat_message("", self.USER_PREFIX_Styled, user_message)
        history_for_llm = self.chat_history.copy()
        self.chat_history.append({"role": "user", "content": user_message})

        message_input.value = ""
        message_input.disabled = True
        send_button.disabled = True
        stop_button.disabled = False
        self._update_chat_status_line(
            status="Sending...", current_messages=len(self.chat_history) // 2
        )

        self._log_chat_message("", self.AI_PREFIX_Styled, "is thinking...")

        active_client = self.llm_client_instance

        def task_to_run_in_worker():
            return active_client.send_message(message=user_message, history=history_for_llm)

        self.current_llm_worker = self.app.run_worker(
            task_to_run_in_worker, thread=True, group="llm_call"
        )  # type: ignore

        try:
            response_text = await self.current_llm_worker.wait()
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.SUCCESS:
                self._log_chat_message("", self.AI_PREFIX_Styled, response_text)
                self.chat_history.append({"role": "assistant", "content": response_text})
        except WorkerFailed as e:
            error_to_log = e.error
            msg = (
                "LLM call cancelled."
                if isinstance(error_to_log, asyncio.CancelledError)
                else f"LLM Error: {error_to_log.__class__.__name__} - {str(error_to_log)[:50]}..."
            )
            logger.error(f"LLM WorkerFailed: {error_to_log}", exc_info=error_to_log)
            self._log_chat_message("", self.SYSTEM_PREFIX_Styled, msg)
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
        except Exception as e:
            logger.error(f"LLM communication error: {e}", exc_info=True)
            self._log_chat_message(
                "", self.SYSTEM_PREFIX_Styled, f"Unexpected error: {e.__class__.__name__}"
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
                self._log_chat_message("", self.SYSTEM_PREFIX_Styled, "Stopping LLM response...")
            else:
                self._log_chat_message("", self.SYSTEM_PREFIX_Styled, "No active LLM call to stop.")

        elif button_id == "new_update_chat_session_button":
            provider_select = self.query_one("#llm_provider_select", Select)
            model_select = self.query_one("#llm_model_select", Select)

            if provider_select.value == Select.BLANK or model_select.value == Select.BLANK:  # type: ignore
                self._reset_chat_log_and_status(status_message="Select provider and model first.")
                self._update_chat_status_line(status="Needs config")
                return

            self._reset_chat_log_and_status(status_message="New session. LLM (re)configuring...")
            self.query_one("#chat_message_input", Input).value = ""

            client_created = self._create_and_set_llm_client()
            if client_created:
                self._log_chat_message("", self.SYSTEM_PREFIX_Styled, "LLM client (re)configured.")
                self._update_chat_status_line(status="Ready")
            else:
                self._log_chat_message(
                    "", self.SYSTEM_PREFIX_Styled, "Failed to (re)configure LLM client."
                )
                self._update_chat_status_line(status="Client Error")
            self.query_one("#chat_message_input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_message_input":
            await self._send_user_message()


class DBConnectionView(Container):
    connection_status = reactive("[#D08770]Not Connected[/]")

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
        # Replace VerticalScroll with DataTable for structured display
        yield DataTable(
            id="collections_data_table",
            show_header=True,
            show_cursor=True,
            classes="collections_list_container",
        )

    def on_mount(self) -> None:
        self.query_one("#collections_title_label", Label).visible = False
        table = self.query_one("#collections_data_table", DataTable)
        table.add_columns("Name", "Docs", "Avg Size", "Total Size", "Storage Size")
        table.visible = False  # Initially hide table

    def watch_connection_status(self, new_status: str) -> None:
        if self.is_mounted:
            try:
                self.query_one("#mongo_connection_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #mongo_connection_status_label.")

    async def _connect_and_list_collections_task(
        self,
    ) -> tuple[bool, str, List[Dict[str, Any]], str | None, str | None]:  # Return list of dicts
        uri = self.query_one("#mongo_uri_input", Input).value
        db_name_from_input = self.query_one("#mongo_db_name_input", Input).value.strip() or None

        collections_with_stats: List[Dict[str, Any]] = []
        self.app.available_collections = []  # Clear simple list first

        core_db_manager.disconnect_mongo(core_db_manager.DEFAULT_ALIAS)
        is_active = core_db_manager.db_connection_active(
            uri=uri, db_name=db_name_from_input, server_timeout_ms=3000
        )

        if not is_active:
            return False, f"Connection Failed to {redact_uri_password(uri)}", [], None, None

        db_instance = core_db_manager.get_mongo_db()
        actual_db_name = db_instance.name
        collection_names_list: List[str] = []
        try:
            fetched_names = sorted(db_instance.list_collection_names())
            for name in fetched_names:
                collection_names_list.append(name)  # For dropdowns
                try:
                    coll_stats = db_instance.command("collStats", name)
                    collections_with_stats.append(
                        {
                            "name": name,
                            "count": coll_stats.get("count", "N/A"),
                            "avgObjSize": coll_stats.get("avgObjSize", "N/A"),
                            "size": coll_stats.get("size", "N/A"),
                            "storageSize": coll_stats.get("storageSize", "N/A"),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not get stats for collection {name}: {e}")
                    collections_with_stats.append(
                        {  # Add with N/A stats if command fails
                            "name": name,
                            "count": "N/A",
                            "avgObjSize": "N/A",
                            "size": "N/A",
                            "storageSize": "N/A",
                        }
                    )
            self.app.available_collections = (
                collection_names_list  # Update app state with just names
            )
        except Exception as e:
            logger.error(f"Failed to list collections or their stats: {e}", exc_info=True)
            return (
                False,
                f"Connected, but failed to list collections/stats for {actual_db_name}: {e}",
                [],
                uri,
                actual_db_name,
            )

        status_message = f"Connected to {redact_uri_password(uri)} (DB: {actual_db_name})"
        return True, status_message, collections_with_stats, uri, actual_db_name

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect_mongo_button":
            collections_table = self.query_one("#collections_data_table", DataTable)
            title_label = self.query_one("#collections_title_label", Label)

            self.connection_status = "[#EBCB8B]Connecting...[/]"
            self.app.available_collections = []
            collections_table.clear()
            title_label.visible = False
            collections_table.visible = False

            try:
                worker: Worker[tuple[bool, str, List[Dict[str, Any]], str | None, str | None]] = (
                    self.app.run_worker(  # type: ignore
                        self._connect_and_list_collections_task, thread=True, group="db_connect"
                    )
                )
                (
                    success,
                    status_msg,
                    collections_stats_data,
                    connected_uri,
                    connected_db_name,
                ) = await worker.wait()

                if success and connected_uri and connected_db_name:
                    self.connection_status = f"[#A3BE8C]{status_msg}[/]"
                    self.app.current_mongo_uri = connected_uri  # type: ignore
                    self.app.current_db_name = connected_db_name  # type: ignore

                    if collections_stats_data:
                        title_label.visible = True
                        collections_table.visible = True
                        for coll_data in collections_stats_data:
                            collections_table.add_row(
                                coll_data["name"],
                                str(coll_data["count"]),
                                _format_bytes_tui(coll_data["avgObjSize"]),
                                _format_bytes_tui(coll_data["size"]),
                                _format_bytes_tui(coll_data["storageSize"]),
                            )
                    else:
                        title_label.visible = True
                        collections_table.visible = False  # Hide table if no data
                        collections_table.add_row(
                            "No collections found in DB.", "", "", "", ""
                        )  # Placeholder
                else:
                    self.connection_status = f"[#BF616A]{status_msg}[/]"
                    self.app.current_mongo_uri = None  # type: ignore
                    self.app.current_db_name = None  # type: ignore
                    self.app.available_collections = []
                    collections_table.clear()
                    collections_table.add_row(
                        "Failed to connect or list collections.", "", "", "", ""
                    )
                    title_label.visible = (
                        True  # Show title label even on failure to indicate attempt
                    )
                    collections_table.visible = True  # Show table with failure message
            except Exception as e:
                logger.error(f"DB Connection Op Error: {e}", exc_info=True)
                collections_table.clear()
                collections_table.add_row(f"Error: {str(e)[:100]}", "", "", "", "")
                self.connection_status = "[#BF616A]Error during connection process.[/]"
                self.app.current_mongo_uri = None  # type: ignore
                self.app.current_db_name = None  # type: ignore
                self.app.available_collections = []
                title_label.visible = True
                collections_table.visible = True


class SchemaAnalysisView(Container):
    analysis_status = reactive("Enter collection and click Analyze.")
    schema_copy_feedback = reactive("")

    def compose(self) -> ComposeResult:
        yield Label("Schema Analysis", classes="panel_title")
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

        with Horizontal(classes="copy_button_container"):
            yield Button("Copy Cell Value", id="copy_cell_button")
            yield Button("Copy Full JSON", id="copy_json_button")
        yield Static(self.schema_copy_feedback, id="schema_copy_feedback_label")

    def on_mount(self) -> None:
        self._update_collection_select()

    def _update_collection_select(self) -> None:
        try:
            select_widget = self.query_one("#schema_collection_select", Select)
            collections = self.app.available_collections  # This is List[str]
            current_active_collection = self.app.active_collection

            if collections:
                options = [(coll, coll) for coll in collections]
                current_value = select_widget.value
                select_widget.set_options(options)
                select_widget.disabled = False
                select_widget.prompt = "Select Collection"
                if current_active_collection and current_active_collection in collections:
                    select_widget.value = current_active_collection  # type: ignore
                elif current_value != Select.BLANK and current_value in collections:
                    select_widget.value = current_value  # type: ignore
                elif select_widget.value not in collections:
                    select_widget.value = Select.BLANK  # type: ignore
            else:
                select_widget.set_options([])
                select_widget.prompt = "Connect to DB to see collections"
                select_widget.disabled = True
                select_widget.value = Select.BLANK  # type: ignore
        except NoMatches:
            logger.warning("SchemaAnalysisView: #schema_collection_select not found for update.")

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "schema_collection_select":
            if event.value is not None and event.value != Select.BLANK:
                self.app.active_collection = str(event.value)
            else:
                if event.value == Select.BLANK:
                    self.app.active_collection = None

    def watch_analysis_status(self, new_status: str) -> None:
        if self.is_mounted:
            try:
                self.query_one("#schema_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #schema_status_label.")

    def watch_schema_copy_feedback(self, new_feedback: str) -> None:
        if self.is_mounted:
            try:
                feedback_label = self.query_one("#schema_copy_feedback_label", Static)
                feedback_label.update(new_feedback)
                if new_feedback:
                    feedback_label.set_timer(2, feedback_label.update, "")
            except NoMatches:
                logger.warning("Could not find #schema_copy_feedback_label.")

    def _prepare_analysis_inputs(
        self,
    ) -> tuple[str | None, str | None, str | None, str, str | None]:
        uri = self.app.current_mongo_uri  # type: ignore
        db_name = self.app.current_db_name  # type: ignore
        if not uri or not db_name:
            return None, None, None, "", "MongoDB not connected. Connect in 'DB Connection' tab."

        collection_select = self.query_one("#schema_collection_select", Select)
        collection_name = (
            str(collection_select.value) if collection_select.value != Select.BLANK else None
        )

        sample_size_str = self.query_one("#schema_sample_size_input", Input).value.strip()

        if not collection_name:
            return uri, db_name, None, sample_size_str, "Please select a collection."
        try:
            int(sample_size_str)
        except ValueError:
            return uri, db_name, collection_name, "", "Sample size must be an integer."
        return uri, db_name, collection_name, sample_size_str, None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze_schema_button":
            table = self.query_one("#schema_results_table", DataTable)
            md_view = self.query_one("#schema_json_view", Markdown)

            self.analysis_status = "[#EBCB8B]Preparing analysis...[/]"
            self.schema_copy_feedback = ""
            table.clear(columns=True)
            md_view.update("```json\n{}\n```")

            uri, db_name, collection_name, sample_size_str, prep_error = (
                self._prepare_analysis_inputs()
            )
            if prep_error or not uri or not db_name or not collection_name:
                self.analysis_status = (
                    f"[#BF616A]Error: {prep_error or 'Missing DB connection or collection'}[/]"
                )
                return

            sample_size = int(sample_size_str)
            self.analysis_status = (
                f"[#EBCB8B]Analyzing '{collection_name}' (sample: {sample_size})...[/]"
            )

            try:
                callable_with_args = functools.partial(
                    self._run_analysis_task, uri, db_name, collection_name, sample_size
                )
                worker = self.app.run_worker(
                    callable_with_args, thread=True, group="schema_analysis"
                )  # type: ignore
                schema_data, field_stats_data, hierarchical_schema, error_msg = await worker.wait()  # type: ignore

                if error_msg:
                    self.analysis_status = f"[#BF616A]Error: {error_msg}[/]"
                    return

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
                        self.analysis_status = "[#A3BE8C]Analysis complete.[/]"
                    except TypeError as te:
                        logger.error(f"JSON serialization error: {te}", exc_info=True)
                        md_view.update("```json\n// Error: Schema non-serializable.\n{}\n```")
                        self.analysis_status = (
                            "[#D08770]Analysis complete (schema display error).[/]"
                        )
                else:
                    self.analysis_status = "[#D08770]Analysis completed with no data.[/]"
            except Exception as e:
                logger.error(f"Schema analysis handler error: {e}", exc_info=True)
                self.analysis_status = f"[#BF616A]Display Error: {e!s}[/]"

        elif event.button.id == "copy_json_button":
            try:
                md_view = self.query_one("#schema_json_view", Markdown)
                markdown_content = md_view.markdown
                if markdown_content.startswith("```json\n") and markdown_content.endswith("\n```"):
                    json_to_copy = markdown_content[len("```json\n"): -len("\n```")]
                elif markdown_content.startswith(
                    "```json\n// Error"
                ) or markdown_content.startswith("```json\n{}"):
                    json_to_copy = (
                        "{}" if "Error" not in markdown_content else "// Error in schema data"
                    )
                else:
                    json_to_copy = markdown_content

                if json_to_copy:
                    self.app.copy_to_clipboard(json_to_copy)
                    self.schema_copy_feedback = "Full JSON Schema Copied!"
                else:
                    self.schema_copy_feedback = "No JSON content to copy."
            except NoMatches:
                self.schema_copy_feedback = "Error: JSON view not found."
            except Exception as e:
                logger.error(f"Error copying JSON: {e}", exc_info=True)
                self.schema_copy_feedback = "Error copying JSON."

        elif event.button.id == "copy_cell_button":
            try:
                table = self.query_one("#schema_results_table", DataTable)
                if table.row_count == 0:
                    self.schema_copy_feedback = "No data in table to copy."
                    return

                cursor_row, cursor_col = table.cursor_coordinate
                if cursor_row < 0 or cursor_col < 0:
                    self.schema_copy_feedback = "No cell selected in table."
                    return

                cell_value = table.get_cell_at(table.cursor_coordinate)
                self.app.copy_to_clipboard(str(cell_value))
                self.schema_copy_feedback = f"Cell ({cursor_row},{cursor_col}) value copied!"
            except NoMatches:
                self.schema_copy_feedback = "Error: Table not found."
            except Exception as e:
                logger.error(f"Error copying cell value: {e}", exc_info=True)
                self.schema_copy_feedback = "Error copying cell value."

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


class DataExtractionView(Container):
    extraction_status = reactive("Enter details and click Start Extraction.")
    STATUS_PREFIX_Styled = Text.from_markup("[b #88C0D0]EXTRACT:[/] ")

    def _log_extraction_message(self, message_content: str):
        try:
            log_widget = self.query_one("#extraction_log", Log)
            lines_in_message = message_content.splitlines()
            if not lines_in_message and len(message_content) >= 0:
                lines_in_message = [message_content]
            elif not lines_in_message and len(message_content) == 0:
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
        self._update_collection_select()

    def _update_collection_select(self) -> None:
        try:
            select_widget = self.query_one("#extract_collection_select", Select)
            collections = self.app.available_collections  # This is List[str]
            current_active_collection = self.app.active_collection

            if collections:
                options = [(coll, coll) for coll in collections]
                current_value = select_widget.value
                select_widget.set_options(options)
                select_widget.disabled = False
                select_widget.prompt = "Select Collection"
                if current_active_collection and current_active_collection in collections:
                    select_widget.value = current_active_collection  # type: ignore
                elif current_value != Select.BLANK and current_value in collections:
                    select_widget.value = current_value  # type: ignore
                elif select_widget.value not in collections:
                    select_widget.value = Select.BLANK  # type: ignore
            else:
                select_widget.set_options([])
                select_widget.prompt = "Connect to DB to see collections"
                select_widget.disabled = True
                select_widget.value = Select.BLANK  # type: ignore
        except NoMatches:
            logger.warning("DataExtractionView: #extract_collection_select not found for update.")

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "extract_collection_select":
            if event.value is not None and event.value != Select.BLANK:
                self.app.active_collection = str(event.value)
            else:
                if event.value == Select.BLANK:
                    self.app.active_collection = None

    def watch_extraction_status(self, new_status: str) -> None:
        if self.is_mounted:
            try:
                self.query_one("#extraction_status_label", Static).update(new_status)
            except NoMatches:
                logger.warning("Could not find #extraction_status_label.")

    def _prepare_extraction_inputs(
        self,
    ) -> tuple[str | None, str | None, str | None, Dict | None, Path | None, str | None]:
        uri = self.app.current_mongo_uri  # type: ignore
        db_name = self.app.current_db_name  # type: ignore
        if not uri or not db_name:
            return None, None, None, None, None, "MongoDB not connected."

        collection_select = self.query_one("#extract_collection_select", Select)
        collection_name = (
            str(collection_select.value) if collection_select.value != Select.BLANK else None
        )

        schema_f_path_str = self.query_one("#extract_schema_file", Input).value.strip()
        output_f_path_str = self.query_one("#extract_output_file", Input).value.strip()

        if not collection_name:
            return uri, db_name, None, None, None, "Collection name required."
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
            self.extraction_status = "[#EBCB8B]Preparing extraction...[/]"
            self._log_extraction_message("Preparation started...")
            uri, db_name, coll, schema_dict, out_path, prep_err = self._prepare_extraction_inputs()

            if prep_err or not uri or not db_name or not coll or not schema_dict or not out_path:
                final_err = prep_err or "Critical error in input preparation."
                self.extraction_status = f"[#BF616A]Error: {final_err}[/]"
                self._log_extraction_message(f"Preparation failed: {final_err}")
                return

            self.extraction_status = (
                f"[#EBCB8B]Starting extraction for '{coll}' to '{out_path}'...[/]"
            )
            self._log_extraction_message(
                f"Extraction for '{coll}' to '{out_path}' initiated by worker."
            )
            try:
                callable_with_args = functools.partial(
                    self._run_extraction_task, uri, db_name, coll, schema_dict, out_path
                )
                worker = self.app.run_worker(
                    callable_with_args, thread=True, group="data_extraction"
                )  # type: ignore
                error_msg = await worker.wait()
                if error_msg:
                    self.extraction_status = f"[#BF616A]Extraction Failed: {error_msg}[/]"
                    self._log_extraction_message(f"FAILURE: {error_msg}")
                else:
                    success_msg = f"Extraction for '{coll}' completed to '{out_path}'."
                    self.extraction_status = f"[#A3BE8C]{success_msg}[/]"
                    self._log_extraction_message(f"SUCCESS: {success_msg}")
            except Exception as e:
                logger.error(f"Extraction handler error: {e}", exc_info=True)
                err_msg = f"Critical Error: {e!s}"
                self.extraction_status = f"[#BF616A]{err_msg}[/]"
                self._log_extraction_message(f"CRITICAL FAILURE: {e!s}")


class MongoAnalyserApp(App[None]):
    TITLE = "Mongo Analyser TUI"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, key_display=None, priority=True),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme (Textual Default)", show=True),
    ]
    current_mongo_uri: reactive[str | None] = reactive(None)
    current_db_name: reactive[str | None] = reactive(None)
    available_collections: reactive[List[str]] = reactive([])
    active_collection: reactive[str | None] = reactive(None)

    def __init__(
        self,
        driver_class: Type[Driver] | None = None,
        css_path: CSSPathType | None = None,
        watch_css: bool = False,
        ansi_color: bool = False,
    ):
        super().__init__(driver_class, css_path, watch_css, ansi_color)
        self.dark = None

    def watch_available_collections(
        self, old_collections: List[str], new_collections: List[str]
    ) -> None:
        logger.debug(
            f"App: available_collections changed from {old_collections} to: {new_collections}"
        )
        try:
            schema_view = self.query_one(SchemaAnalysisView)
            if schema_view.is_mounted:
                schema_view._update_collection_select()
        except NoMatches:
            pass
        except Exception as e:
            logger.error(f"Error updating schema view collections: {e}", exc_info=True)

        try:
            extract_view = self.query_one(DataExtractionView)
            if extract_view.is_mounted:
                extract_view._update_collection_select()
        except NoMatches:
            pass
        except Exception as e:
            logger.error(f"Error updating extraction view collections: {e}", exc_info=True)

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
        if event.tab and event.tab.id:
            try:
                suffix = event.tab.id.split("_", 1)[1]
                view_id = f"view_{suffix}"
            except IndexError:
                logger.warning(f"Could not parse suffix from tab ID: {event.tab.id}")
                return

        if not view_id:
            logger.warning(
                f"No view_id could be determined for tab {event.tab.id if event.tab else 'None'}"
            )
            return

        try:
            switcher = self.query_one(ContentSwitcher)
        except NoMatches:
            logger.error("Critical: ContentSwitcher not found.")
            return

        try:
            switcher.current = view_id
            if view_id == "view_schema_analysis":
                schema_view = self.query_one(SchemaAnalysisView)
                if schema_view.is_mounted:
                    schema_view._update_collection_select()
            elif view_id == "view_data_extraction":
                extract_view = self.query_one(DataExtractionView)
                if extract_view.is_mounted:
                    extract_view._update_collection_select()

        except NoMatches:
            logger.warning(
                f"View '{view_id}' not found in ContentSwitcher. Available: {switcher.children}"
            )
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
                self.query_one("#schema_collection_select", Select).focus()
            elif view_id == "view_data_extraction":
                self.query_one("#extract_collection_select", Select).focus()
        except NoMatches:
            logger.debug(f"No default focus target (primary input) found in view '{view_id}'.")
        except Exception as e:
            logger.error(f"Error focusing input in view '{view_id}': {e}", exc_info=True)

    def action_toggle_theme(self) -> None:
        if hasattr(self, "dark"):
            self.dark = not self.dark
            logger.info(f"Textual dark mode toggled to: {self.dark}")
        else:
            logger.warning(
                "The 'dark' attribute for theme toggling is not available on the App object. "
                "This might be due to the Textual version. Theme toggling via Ctrl+T may not function "
                "as expected with Textual's built-in mechanism."
            )


def main_interactive_tui():
    log_file_path = "mongo_analyser_tui.log"
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "DEBUG").upper(),
        filename=log_file_path,
        filemode="a",
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

    logger.info(f"Starting Mongo Analyser TUI. Logging to {Path(log_file_path).resolve()}")
    try:
        app = MongoAnalyserApp()
        app.run()
    except Exception as e:
        logger.critical("MongoAnalyserApp failed to run.", exc_info=True)
        print(f"A critical error occurred: {e}. Check logs at {Path(log_file_path).resolve()}")
    finally:
        logger.info("Mongo Analyser TUI finished.")
        core_db_manager.disconnect_all_mongo()


if __name__ == "__main__":
    main_interactive_tui()
