import functools
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    LoadingIndicator,
    Markdown,
    Static,
)
from textual.worker import Worker, WorkerCancelled, WorkerState

import mongo_analyser.core.db as core_db_manager
from mongo_analyser.core import DataExtractor, SchemaAnalyser
from mongo_analyser.dialogs import ErrorDialog
from mongo_analyser.llm_chat import (
    GoogleChat,
    LLMChat,
    OllamaChat,
    OpenAIChat,
)
from mongo_analyser.widgets import ChatMessageList, ChatMessageWidget, LLMConfigPanel

logger = logging.getLogger(__name__)


class ChatView(Container):
    current_llm_worker: Worker | None = None
    llm_client_instance: LLMChat | None = None

    ROLE_USER = "user"
    ROLE_AI = "assistant"
    ROLE_SYSTEM = "system"

    PROVIDER_CLASSES: Dict[str, Type[LLMChat]] = {
        "ollama": OllamaChat,
        "openai": OpenAIChat,
        "google": GoogleChat,
    }

    CONTEXT_BLOCK_START_MARKER = "--- START APP-PROVIDED CONTEXT ---\n"
    CONTEXT_BLOCK_END_MARKER = "\n--- END APP-PROVIDED CONTEXT ---\n\n"
    SCHEMA_SECTION_KEY = "schema"
    METADATA_SECTION_KEY = "metadata"
    SAMPLEDOCS_SECTION_KEY = "sample_docs"

    SECTION_TITLE_TEMPLATES = {
        SCHEMA_SECTION_KEY: "[SCHEMA FOR COLLECTION: '{collection_name}']\n",
        METADATA_SECTION_KEY: "[FIELD STATISTICS FOR COLLECTION: '{collection_name}']\n",
        SAMPLEDOCS_SECTION_KEY: "[SAMPLE DOCUMENTS FOR COLLECTION: '{collection_name}'"
        " ({num_docs} docs)]\n",
    }
    DEFAULT_SAMPLE_DOCS_COUNT = 3

    _current_ai_message_widget: Optional[ChatMessageWidget] = None
    _full_response_content: str = ""

    def __init__(
        self,
        *children: Widget,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        disabled: bool = False,
    ):
        super().__init__(
            *children,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.chat_history: List[Dict[str, str]] = []
        self.active_contexts: Dict[str, Dict[str, Any]] = {}

    def _log_chat_message(self, role: str, message_content: str) -> None:
        try:
            chat_list_widget = self.query_one("#chat_log_widget", ChatMessageList)
            chat_list_widget.add_message(role, message_content)
            chat_list_widget.scroll_end(animate=False)
        except NoMatches:
            logger.warning("Chat log widget (#chat_log_widget) not found for logging message.")
        except Exception as e:
            logger.error(f"Error logging chat message: {e}", exc_info=True)

    def _update_chat_status_line(
        self, status: str = "Idle", current_messages: int | None = None
    ) -> None:
        try:
            panel = self.query_one(LLMConfigPanel)
            chat_status_widget = self.query_one("#chat_status_line", Static)
            prov = panel.provider or ""
            mdl = panel.model or ""
            provider_display = prov.capitalize() if prov else "N/A"
            model_display = mdl if mdl else "N/A"
            max_hist = panel.max_history_messages
            if max_hist == -1:
                max_hist_display = "None"
            elif max_hist == 0:
                max_hist_display = "All"
            else:
                max_hist_display = str(max_hist)
            actual_hist_len = len(self._get_effective_history_for_llm())
            history_info = f"History: {actual_hist_len} (max: {max_hist_display})"
            chat_status_widget.update(
                f"Provider: {provider_display} | Model: {model_display} | {history_info} | Status: {status}"
            )
        except NoMatches:
            logger.warning("ChatView: Could not update chat status line (widget not found).")
        except Exception as e:
            logger.error(f"Error updating chat status line: {e}", exc_info=True)

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        self.chat_history.clear()
        self.active_contexts.clear()
        try:
            log_widget = self.query_one("#chat_log_widget", ChatMessageList)
            log_widget.clear_messages()
        except NoMatches:
            logger.warning("Chat log widget not found for clearing during reset.")

        self._apply_contexts_to_input_field()
        self._log_chat_message(self.ROLE_SYSTEM, status_message)
        self._update_chat_status_line(status="Idle", current_messages=0)

    def on_mount(self) -> None:
        self.chat_history: List[Dict[str, str]] = []
        self.active_contexts: Dict[str, Dict[str, Any]] = {}
        self.llm_client_instance = None
        self._reset_chat_log_and_status("ChatView initialized. Configure LLM in sidebar.")
        self.focus_default_widget()
        try:
            self.query_one("#chat_model_loading_indicator", LoadingIndicator).display = False
            self.query_one("#active_context_indicator_label", Static).display = False
        except NoMatches:
            logger.warning(
                "ChatView: Initial display setup for indicators failed (widget not found)."
            )

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat_interface_horizontal_layout"):
            with Vertical(id="chat_main_area", classes="chat_column_main"):
                yield Static(
                    "Provider: N/A | Model: N/A | History: 0 (max: N/A) | Status: Idle",
                    id="chat_status_line",
                    classes="chat_status",
                )
                yield LoadingIndicator(id="chat_model_loading_indicator")
                yield ChatMessageList(id="chat_log_widget")
                with Horizontal(classes="chat_action_buttons"):
                    yield Button(
                        "Prepend Collection Schema",
                        id="prepend_schema_button",
                        classes="context_button",
                    )
                    yield Button(
                        "Prepend Collection Metadata",
                        id="prepend_metadata_button",
                        classes="context_button",
                    )
                    yield Button(
                        "Prepend Sample Docs",
                        id="prepend_sample_docs_button",
                        classes="context_button",
                    )
                yield Static("", id="active_context_indicator_label", classes="context_indicator")
                with Horizontal(id="chat_input_bar", classes="chat_input_container"):
                    yield Input(placeholder="Type a message...", id="chat_message_input")
                    yield Button(
                        "Send",
                        id="send_chat_message_button",
                        variant="primary",
                        classes="chat_button",
                    )
                    yield Button(
                        "Stop", id="stop_chat_message_button", classes="chat_button", disabled=True
                    )
            yield LLMConfigPanel(id="chat_llm_config_panel", classes="chat_column_sidebar")

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#chat_message_input", Input).focus()
        except NoMatches:
            logger.debug("ChatView: Could not focus default input (#chat_message_input).")

    @on(LLMConfigPanel.ProviderChanged)
    async def handle_provider_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ProviderChanged
    ) -> None:
        logger.info(f"ChatView: ProviderChanged event with provider: {event.provider}")
        self.llm_client_instance = None
        self._log_chat_message(self.ROLE_SYSTEM, "Provider changed. LLM client reset.")
        if event.provider:
            try:
                panel = self.query_one(LLMConfigPanel)
                if panel.provider != event.provider:
                    panel.provider = event.provider
            except NoMatches:
                pass
            self.app.call_later(self._load_models_for_provider, event.provider)
        else:
            try:
                panel = self.query_one(LLMConfigPanel)
                panel.update_models_list([], "Select Provider First")
                panel.model = None
            except NoMatches:
                logger.error(
                    "ChatView: LLMConfigPanel not found when handling null provider change."
                )
            self._update_chat_status_line(status="Provider cleared")

    async def _load_models_for_provider(self, provider_value: str) -> None:
        loader = self.query_one("#chat_model_loading_indicator", LoadingIndicator)
        try:
            panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found when loading models.")
            self._log_chat_message(self.ROLE_SYSTEM, "Internal error: LLM config panel missing.")
            self._update_chat_status_line(status="Panel Error")
            if loader.is_mounted:
                loader.display = False
            return

        llm_class = self.PROVIDER_CLASSES.get(provider_value)
        if not llm_class:
            panel.update_models_list([], f"Unknown provider: {provider_value}")
            if panel.model is not None:
                panel.model = None
            self._log_chat_message(
                self.ROLE_SYSTEM, f"Cannot load models for unknown provider: {provider_value}"
            )
            if loader.is_mounted:
                loader.display = False
            return

        panel.set_model_select_loading(True, f"Loading models for {provider_value}...")
        self._update_chat_status_line(status=f"Loading {provider_value} models...")
        self._log_chat_message(self.ROLE_SYSTEM, f"Fetching models for {provider_value}...")
        if loader.is_mounted:
            loader.display = True

        listed: List[str] = []
        error: Optional[str] = None

        client_cfg_from_panel = panel.get_llm_config()
        client_cfg_from_panel["provider_hint"] = provider_value

        try:
            worker: Worker[List[str]] = self.app.run_worker(
                functools.partial(llm_class.list_models, client_config=client_cfg_from_panel),
                thread=True,
                group="model_listing",
            )
            listed = await worker.wait()
            if worker.is_cancelled:
                error = "Model loading cancelled by worker."
        except WorkerCancelled:
            error = "Model loading cancelled."
        except Exception as e:
            logger.error(f"ChatView: Error listing models for {provider_value}: {e}", exc_info=True)
            error = f"Failed to list models: {e.__class__.__name__}: {str(e)[:60]}"

        if loader.is_mounted:
            loader.display = False
        panel.set_model_select_loading(False)

        current_status_after_listing = "Models loaded"
        if error:
            panel.update_models_list([], error)
            self._log_chat_message(self.ROLE_SYSTEM, error)
            current_status_after_listing = "Model list error"
            if panel.model is not None:
                panel.model = None
        else:
            options = [(m, m) for m in listed]
            prompt_if_empty = "No models found for this provider." if not listed else "Select Model"
            panel.update_models_list(options, prompt_if_empty)
            current_status_after_listing = "Models loaded" if listed else "No models found"

            selected_model_for_panel: Optional[str] = None
            if listed:
                default_configured_model_key = f"llm_default_model_{provider_value}"
                default_configured_model = self.app.config_manager.get_setting(
                    default_configured_model_key
                )

                if default_configured_model and default_configured_model in listed:
                    selected_model_for_panel = default_configured_model
                    logger.info(
                        f"Using configured default model for {provider_value}: {selected_model_for_panel}"
                    )
                else:
                    if default_configured_model:
                        logger.warning(
                            f"Configured default model '{default_configured_model}' for {provider_value} not found in listed models. Using first available."
                        )
                    selected_model_for_panel = listed[0]
                    logger.info(
                        f"Using first available model for {provider_value}: {selected_model_for_panel}"
                    )

            if panel.model != selected_model_for_panel:
                panel.model = selected_model_for_panel
            elif selected_model_for_panel is None and panel.model is not None:
                panel.model = None
            elif selected_model_for_panel is not None and panel.model is None:
                panel.model = selected_model_for_panel

        self._update_chat_status_line(status=current_status_after_listing)
        if panel.model is None and (error or not listed):
            await self.handle_model_change_from_llm_config_panel(LLMConfigPanel.ModelChanged(None))
        elif panel.model is not None:
            if (
                self.llm_client_instance is None
                or self.llm_client_instance.model_name != panel.model
            ):
                await self.handle_model_change_from_llm_config_panel(
                    LLMConfigPanel.ModelChanged(panel.model)
                )

    @on(LLMConfigPanel.ModelChanged)
    async def handle_model_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ModelChanged
    ) -> None:
        model_value = event.model
        logger.info(
            f"ChatView: handle_model_change_from_llm_config_panel received ModelChanged"
            f" event with model: {model_value}"
        )
        self.active_contexts.clear()
        if model_value:
            self._reset_chat_log_and_status(f"Model set to: {model_value}. Session reset.")
            try:
                self.query_one("#chat_message_input", Input).value = ""
            except NoMatches:
                pass
            self._apply_contexts_to_input_field()
            if self._create_and_set_llm_client():
                self._log_chat_message(self.ROLE_SYSTEM, "Session ready. LLM client configured.")
                self._update_chat_status_line(status="Ready")
            else:
                self._log_chat_message(
                    self.ROLE_SYSTEM,
                    "Client configuration error for selected model. Check logs for details.",
                )
                self._update_chat_status_line(status="Client Error")
            self.focus_default_widget()
        else:
            self.llm_client_instance = None
            self._reset_chat_log_and_status("No model selected or available.")
            self._log_chat_message(
                self.ROLE_SYSTEM, "Model deselected or unavailable. LLM client cleared."
            )
            self._update_chat_status_line(status="Select model")
            self._apply_contexts_to_input_field()

    def _create_and_set_llm_client(self) -> bool:
        try:
            panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found during client creation.")
            self._log_chat_message(self.ROLE_SYSTEM, "LLM Configuration panel not found.")
            return False
        cfg = panel.get_llm_config()
        provider = cfg.get("provider_hint")
        model_name = cfg.get("model_name")

        if not provider or not model_name:
            self._log_chat_message(
                self.ROLE_SYSTEM, "Provider or model not properly selected for client creation."
            )
            return False
        llm_class = self.PROVIDER_CLASSES.get(provider)
        if not llm_class:
            self._log_chat_message(
                self.ROLE_SYSTEM, f"Unknown provider '{provider}' for client creation."
            )
            return False

        client_kwargs = {"model_name": model_name}
        temperature = cfg.get(
            "temperature", self.app.config_manager.get_setting("llm_default_temperature")
        )

        if provider == "ollama":
            client_kwargs.update(
                {
                    k: v
                    for k, v in cfg.items()
                    if k
                    not in ["provider_hint", "model_name", "temperature", "max_history_messages"]
                }
            )
            client_kwargs.setdefault("options", {})
            if temperature is not None:
                client_kwargs["options"]["temperature"] = temperature

        elif provider == "openai":
            client_kwargs.update(
                {
                    k: v
                    for k, v in cfg.items()
                    if k not in ["provider_hint", "model_name", "max_history_messages"]
                }
            )
            if temperature is not None:
                client_kwargs["temperature"] = temperature

        elif provider == "google":
            client_kwargs.update(
                {
                    k: v
                    for k, v in cfg.items()
                    if k
                    not in ["provider_hint", "model_name", "temperature", "max_history_messages"]
                }
            )
            if temperature is not None:
                client_kwargs["generation_config"] = {"temperature": temperature}

        logger.debug(
            f"ChatView: Instantiating LLM class '{llm_class.__name__}' "
            f"with effective kwargs: {client_kwargs}"
        )
        try:
            new_client = llm_class(**client_kwargs)
            self.llm_client_instance = new_client
            logger.info(f"ChatView: LLM client successfully created for {provider}:{model_name}.")
            return True
        except Exception as e:
            logger.error(
                f"ChatView: Failed to create LLM client for {provider}:{model_name}."
                f" KWARGS: {client_kwargs}. Error: {e}",
                exc_info=True,
            )
            self._log_chat_message(
                self.ROLE_SYSTEM,
                f"Error creating LLM client for '{model_name}':"
                f" {e.__class__.__name__} - {str(e)[:100]}. See console log for full traceback.",
            )
            self.llm_client_instance = None
            return False

    def _get_effective_history_for_llm(self) -> List[Dict[str, str]]:
        hist = [m for m in self.chat_history if m["role"] in {self.ROLE_USER, self.ROLE_AI}]
        try:
            panel = self.query_one(LLMConfigPanel)
            max_hist = panel.max_history_messages
            if max_hist == -1:
                return []
            if max_hist is not None and max_hist == 0:
                return hist
            if max_hist is not None and 0 < max_hist < len(hist):
                return hist[-max_hist:]
        except NoMatches:
            logger.warning("LLMConfigPanel not found in _get_effective_history_for_llm")
        return hist

    def _prepare_for_stream(self) -> None:
        self._full_response_content = ""
        try:
            chat_list_widget = self.query_one("#chat_log_widget", ChatMessageList)
            self._current_ai_message_widget = ChatMessageWidget(self.ROLE_AI, "●")
            chat_list_widget.mount(self._current_ai_message_widget)
            chat_list_widget.scroll_end(animate=False)
        except NoMatches:
            logger.error("ChatView: Could not find #chat_log_widget to prepare for stream.")
            self._current_ai_message_widget = None

    def _handle_stream_chunk(self, chunk: str) -> None:
        if self._current_ai_message_widget and self._current_ai_message_widget.is_mounted:
            self._full_response_content += chunk
            cursor_char = " ●" if len(self._full_response_content) % 10 < 5 else "●"
            markdown_widget = self._current_ai_message_widget.query_one(Markdown)
            markdown_widget.update(self._full_response_content + cursor_char)
            try:
                self.query_one("#chat_log_widget", ChatMessageList).scroll_end(animate=False)
            except NoMatches:
                pass

    def _handle_stream_end(self, cancelled: bool = False) -> None:
        final_text = self._full_response_content.strip()

        if self.current_llm_worker and self.current_llm_worker.state == WorkerState.CANCELLED:
            cancelled = True

        if cancelled and not final_text.endswith("[Stopped by user]"):
            final_text += "\n[Stopped by user]"

        if self._current_ai_message_widget and self._current_ai_message_widget.is_mounted:
            self._current_ai_message_widget.query_one(Markdown).update(final_text)

        self.chat_history.append({"role": self.ROLE_AI, "content": final_text})
        self._cleanup_after_send()

    def _handle_stream_error(self, error: Exception) -> None:
        logger.error(f"Error in streaming worker task: {error}", exc_info=True)
        err_msg = f"LLM Error: {error!s}"
        if self._current_ai_message_widget and self._current_ai_message_widget.is_mounted:
            self._current_ai_message_widget.query_one(Markdown).update(err_msg)
        else:
            self._log_chat_message(self.ROLE_SYSTEM, err_msg)

        self.chat_history.append({"role": self.ROLE_SYSTEM, "content": err_msg})
        self._cleanup_after_send()

    def _cleanup_after_send(self) -> None:
        if self.is_mounted:
            try:
                input_widget = self.query_one("#chat_message_input", Input)
                send_btn = self.query_one("#send_chat_message_button", Button)
                stop_btn = self.query_one("#stop_chat_message_button", Button)

                input_widget.disabled = False
                send_btn.disabled = False
                stop_btn.disabled = True
                self._update_chat_status_line(status="Ready")
                input_widget.focus()
            except NoMatches:
                logger.warning("ChatView: UI elements not found in _cleanup_after_send.")

        self._current_ai_message_widget = None
        self.current_llm_worker = None

    def _streaming_worker_task(
        self, client: LLMChat, message: str, history: List[Dict[str, str]]
    ) -> None:
        """The actual function run by the worker to stream messages."""
        try:
            for chunk in client.stream_message(message=message, history=history):
                if self.current_llm_worker and self.current_llm_worker.is_cancelled:
                    self.app.call_from_thread(self._handle_stream_end, cancelled=True)
                    return
                self.app.call_from_thread(self._handle_stream_chunk, chunk)

            self.app.call_from_thread(self._handle_stream_end, cancelled=False)
        except Exception as e:
            logger.error(f"Exception in _streaming_worker_task: {e}", exc_info=True)
            self.app.call_from_thread(self._handle_stream_error, e)

    async def _send_user_message(self) -> None:
        try:
            input_widget = self.query_one("#chat_message_input", Input)
            send_btn = self.query_one("#send_chat_message_button", Button)
            stop_btn = self.query_one("#stop_chat_message_button", Button)
        except NoMatches:
            await self.app.push_screen(ErrorDialog("UI Error", "Chat UI elements missing."))
            return

        if not self.llm_client_instance:
            await self.app.push_screen(
                ErrorDialog("LLM Error", "LLM not configured. Please select a provider and model.")
            )
            return

        user_typed_message = input_widget.value.strip()
        if not user_typed_message:
            self.app.notify("Cannot send an empty message.", title="Input Error", severity="error")
            return

        self._log_chat_message(self.ROLE_USER, user_typed_message)
        self.chat_history.append({"role": self.ROLE_USER, "content": user_typed_message})

        history_for_llm = self._get_effective_history_for_llm()
        context_block_content = self._build_context_block_string()
        message_for_llm = context_block_content + user_typed_message

        input_widget.value = ""
        self.active_contexts.clear()
        self._apply_contexts_to_input_field()

        input_widget.disabled = True
        send_btn.disabled = True
        stop_btn.disabled = False
        self._update_chat_status_line(status="Receiving...")

        self._prepare_for_stream()

        client = self.llm_client_instance

        task = functools.partial(
            self._streaming_worker_task,
            client=client,
            message=message_for_llm,
            history=history_for_llm,
        )

        if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
            self.current_llm_worker.cancel()

        self.current_llm_worker = self.app.run_worker(
            task, thread=True, group="llm_call_stream", exclusive=True
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "send_chat_message_button":
            await self._send_user_message()
        elif bid == "stop_chat_message_button":
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
                logger.info("Stop button pressed. Cancelling LLM worker.")
                self.current_llm_worker.cancel()

                self._log_chat_message(self.ROLE_SYSTEM, "Attempting to stop LLM response...")
                try:
                    self.query_one("#stop_chat_message_button", Button).disabled = True
                except NoMatches:
                    pass

        elif bid == "prepend_schema_button":
            await self._handle_prepend_context(self.SCHEMA_SECTION_KEY, event.button)
        elif bid == "prepend_metadata_button":
            await self._handle_prepend_context(self.METADATA_SECTION_KEY, event.button)
        elif bid == "prepend_sample_docs_button":
            await self._handle_prepend_context(self.SAMPLEDOCS_SECTION_KEY, event.button)

    async def _handle_prepend_context(self, context_type_key: str, button: Button) -> None:
        coll = self.app.active_collection
        if not coll:
            self.app.notify(
                "No active collection selected to prepend context.",
                title="Context Error",
                severity="warning",
            )
            return

        content_str: Optional[str] = None
        num_docs_for_title: Optional[int] = None
        original_button_label = button.label
        button.label = Text("Loading...", style="italic yellow")
        button.disabled = True

        try:
            if context_type_key == self.SCHEMA_SECTION_KEY:
                content_str = await self._get_active_collection_hierarchical_schema()
            elif context_type_key == self.METADATA_SECTION_KEY:
                content_str = await self._get_active_collection_field_stats()
            elif context_type_key == self.SAMPLEDOCS_SECTION_KEY:
                num_docs_for_title = self.DEFAULT_SAMPLE_DOCS_COUNT
                content_str = await self._get_active_collection_sample_docs(
                    num_docs=num_docs_for_title
                )

            if content_str:
                self._set_context_for_input(
                    context_type_key, coll, content_str, num_docs=num_docs_for_title
                )
                self._log_chat_message(
                    self.ROLE_SYSTEM,
                    f"{context_type_key.replace('_', ' ').capitalize()}"
                    f" for collection '{coll}' prepared for input.",
                )
                self.app.notify(
                    f"{context_type_key.replace('_', ' ').capitalize()}"
                    f" for '{coll}' added/updated for next message.",
                    title="Context Added",
                )
                button.label = Text(
                    f"{context_type_key.replace('_', ' ').capitalize()} Added ✓",
                    style="italic green",
                )
            else:
                self.app.notify(
                    f"Could not retrieve {context_type_key.replace('_', ' ')} for '{coll}'.",
                    title="Context Error",
                    severity="error",
                )
                button.label = original_button_label

        except Exception as e:
            logger.error(
                f"Error handling prepend context for {context_type_key}: {e}", exc_info=True
            )
            self.app.notify(f"Error: {e}", title="Context Preparation Error", severity="error")
            button.label = original_button_label
        finally:

            def revert_button():
                if button.is_mounted and button.label != original_button_label:
                    if "Added ✓" in button.label.plain:
                        self.set_timer(1.5, lambda: setattr(button, "label", original_button_label))
                    else:
                        button.label = original_button_label
                button.disabled = False

            if "Loading..." in button.label.plain:
                button.label = original_button_label
                button.disabled = False
            else:
                self.set_timer(2.5, revert_button)

    async def _fetch_schema_and_stats_if_needed(
        self, collection_name: str
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        if not (self.app.current_mongo_uri and self.app.current_db_name):
            return None, None

        cached_results = self.app.current_schema_analysis_results
        if (
            isinstance(cached_results, dict)
            and cached_results.get("collection_name") == collection_name
            and "hierarchical_schema" in cached_results
            and "field_stats" in cached_results
        ):
            logger.info(f"Using cached schema and stats for '{collection_name}'.")
            return cached_results["hierarchical_schema"], cached_results["field_stats"]

        logger.info(f"No cached results for '{collection_name}' or cache mismatch. Fetching live.")
        self.app.notify(f"Fetching schema/stats for '{collection_name}'...", title="Data Fetch")

        try:
            if not core_db_manager.db_connection_active(
                uri=self.app.current_mongo_uri,
                db_name=self.app.current_db_name,
                server_timeout_ms=3000,
            ):
                raise ConnectionError("DB connection lost or could not be re-established.")

            pymongo_collection = SchemaAnalyser.get_collection(
                self.app.current_mongo_uri, self.app.current_db_name, collection_name
            )

            analysis_task = functools.partial(
                SchemaAnalyser.infer_schema_and_field_stats,
                collection=pymongo_collection,
                sample_size=100,
            )
            worker: Worker[Tuple[Dict, Dict]] = self.app.run_worker(
                analysis_task, thread=True, group="chat_context_fetch"
            )
            flat_schema_data, field_stats_data = await worker.wait()

            if worker.is_cancelled:
                self.app.notify(
                    f"Schema/stats fetch for '{collection_name}' cancelled.",
                    title="Fetch Cancelled",
                )
                return None, None

            if not flat_schema_data and not field_stats_data:
                self.app.notify(
                    f"No schema/stats data returned for '{collection_name}'.", title="Fetch Error"
                )
                return None, None

            hierarchical_schema = SchemaAnalyser.schema_to_hierarchical(flat_schema_data or {})

            self.app.current_schema_analysis_results = {
                "flat_schema": flat_schema_data,
                "field_stats": field_stats_data,
                "hierarchical_schema": hierarchical_schema,
                "collection_name": collection_name,
            }
            return hierarchical_schema, field_stats_data

        except Exception as e:
            logger.error(f"Error fetching schema/stats for injection: {e}", exc_info=True)
            await self.app.push_screen(
                ErrorDialog(
                    "Data Fetch Error", f"Could not fetch data for '{collection_name}': {e!s}"
                )
            )
            return None, None

    async def _get_active_collection_hierarchical_schema(self) -> Optional[str]:
        coll = self.app.active_collection
        if not coll:
            return None
        hierarchical_schema, _ = await self._fetch_schema_and_stats_if_needed(coll)
        if hierarchical_schema is None:
            return None
        try:
            return json.dumps(hierarchical_schema, indent=2, default=str)
        except TypeError:
            logger.error(f"Hierarchical schema for '{coll}' is not JSON serializable.")
            return str(hierarchical_schema)

    async def _get_active_collection_field_stats(self) -> Optional[str]:
        coll = self.app.active_collection
        if not coll:
            return None
        _, field_stats = await self._fetch_schema_and_stats_if_needed(coll)
        if field_stats is None:
            return None
        try:
            return json.dumps(field_stats, indent=2, default=str)
        except TypeError:
            logger.error(f"Field stats for '{coll}' are not JSON serializable.")
            return str(field_stats)

    async def _get_active_collection_sample_docs(self, num_docs: int) -> Optional[str]:
        coll = self.app.active_collection
        if not (self.app.current_mongo_uri and self.app.current_db_name and coll):
            return None

        self.app.notify(f"Fetching {num_docs} sample docs for '{coll}'...", title="Data Fetch")
        try:
            if not core_db_manager.db_connection_active(
                uri=self.app.current_mongo_uri,
                db_name=self.app.current_db_name,
                server_timeout_ms=3000,
            ):
                raise ConnectionError("DB connection lost for sample doc fetching.")

            fetch_task = functools.partial(
                DataExtractor.get_newest_documents,
                self.app.current_mongo_uri,
                self.app.current_db_name,
                coll,
                num_docs,
            )
            worker: Worker[List[Dict]] = self.app.run_worker(
                fetch_task, thread=True, group="chat_sample_docs_fetch"
            )
            documents = await worker.wait()

            if worker.is_cancelled:
                self.app.notify(
                    f"Sample doc fetch for '{coll}' cancelled.", title="Fetch Cancelled"
                )
                return None

            if not documents:
                self.app.notify(f"No sample documents found for '{coll}'.", title="Data Fetch")
                return "[]"

            return json.dumps(documents, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error fetching sample documents for injection: {e}", exc_info=True)
            await self.app.push_screen(
                ErrorDialog(
                    "Sample Docs Fetch Error", f"Could not fetch samples for '{coll}': {e!s}"
                )
            )
            return None

    def _set_context_for_input(
        self,
        context_type_key: str,
        collection_name: str,
        content: str,
        num_docs: Optional[int] = None,
    ) -> None:
        self.active_contexts[context_type_key] = {
            "collection": collection_name,
            "content": content,
            "num_docs": num_docs,
        }
        self._apply_contexts_to_input_field()

    def _build_context_block_string(self) -> str:
        if not self.active_contexts:
            return ""

        block_content_parts = []
        ordered_keys = [
            self.SCHEMA_SECTION_KEY,
            self.METADATA_SECTION_KEY,
            self.SAMPLEDOCS_SECTION_KEY,
        ]

        for key in ordered_keys:
            if key in self.active_contexts:
                context_item = self.active_contexts[key]
                title_template = self.SECTION_TITLE_TEMPLATES[key]
                title_args = {"collection_name": context_item["collection"]}
                if key == self.SAMPLEDOCS_SECTION_KEY and context_item.get("num_docs") is not None:
                    title_args["num_docs"] = context_item["num_docs"]
                section_title = title_template.format(**title_args)
                block_content_parts.append(
                    f"{section_title}```json\n{context_item['content']}\n```\n"
                )
        if not block_content_parts:
            return ""
        return (
            self.CONTEXT_BLOCK_START_MARKER
            + "\n".join(block_content_parts)
            + self.CONTEXT_BLOCK_END_MARKER
        )

    def _update_active_context_indicator(self) -> None:
        try:
            indicator_label = self.query_one("#active_context_indicator_label", Static)
            if not self.active_contexts:
                indicator_label.update("")
                indicator_label.display = False
                return

            active_items = []
            ordered_keys = [
                self.SCHEMA_SECTION_KEY,
                self.METADATA_SECTION_KEY,
                self.SAMPLEDOCS_SECTION_KEY,
            ]
            for key in ordered_keys:
                if key in self.active_contexts:
                    item = self.active_contexts[key]
                    coll_name = item["collection"]
                    key_display = key.replace("_", " ").capitalize()
                    if key == self.SAMPLEDOCS_SECTION_KEY and item.get("num_docs") is not None:
                        active_items.append(f"{key_display} ({item['num_docs']}) for '{coll_name}'")
                    else:
                        active_items.append(f"{key_display} for '{coll_name}'")

            if active_items:
                indicator_text = "Prepending: " + "; ".join(active_items)
                indicator_label.update(Text(indicator_text, style="italic dim"))
                indicator_label.display = True
            else:
                indicator_label.update("")
                indicator_label.display = False
        except NoMatches:
            logger.warning("Active context indicator label not found.")

    def _apply_contexts_to_input_field(self) -> None:
        self._update_active_context_indicator()
        try:
            input_widget = self.query_one("#chat_message_input", Input)
            if self.app.focused is input_widget:
                input_widget.action_end()
        except NoMatches:
            logger.warning("Chat input widget not found in _apply_contexts_to_input_field.")

    @on(LLMConfigPanel.NewSessionRequested)
    async def handle_new_session_requested_from_llm_config_panel(
        self,
    ) -> None:
        logger.info("ChatView: NewSessionRequested event received.")
        self.active_contexts.clear()
        try:
            panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            await self.app.push_screen(ErrorDialog("UI Error", "LLM Config panel not found."))
            return
        current_model_on_panel = panel.model
        if not panel.provider or not current_model_on_panel:
            self._reset_chat_log_and_status("Select provider and model first.")
            await self.app.push_screen(
                ErrorDialog(
                    "Configuration Incomplete",
                    "Please select a provider and model before starting a new session.",
                )
            )
            return
        self._reset_chat_log_and_status("New session started. LLM (re)configuring...")
        try:
            self.query_one("#chat_message_input", Input).value = ""
        except NoMatches:
            pass
        self._apply_contexts_to_input_field()
        if self._create_and_set_llm_client():
            self._log_chat_message(self.ROLE_SYSTEM, "LLM client (re)configured for new session.")
            self._update_chat_status_line(status="Ready")
        else:
            self._log_chat_message(
                self.ROLE_SYSTEM, "Client Error during new session setup. Check logs."
            )
            self._update_chat_status_line(status="Client Error")
        self.focus_default_widget()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_message_input":
            try:
                send_button = self.query_one("#send_chat_message_button", Button)
                if not send_button.disabled:
                    await self._send_user_message()
            except NoMatches:
                logger.error("Send button not found on input submission.")
