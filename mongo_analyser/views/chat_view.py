import functools
import json
import logging
from typing import Dict, List, Optional, Tuple, Type

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
    Static,
)
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

import mongo_analyser.core.db as core_db_manager
from mongo_analyser.core.analyser import SchemaAnalyser
from mongo_analyser.dialogs import ErrorDialog
from mongo_analyser.llm_chat import (
    GoogleChat,
    LLMChat,
    OllamaChat,
    OpenAIChat,
)
from mongo_analyser.widgets import ChatMessageList, LLMConfigPanel

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

    SCHEMA_INJECT_START_MARKER_BASE = "CONTEXT: The MongoDB schema for the collection '"
    SCHEMA_INJECT_END_MARKER = "\n```\n\nBased on this schema, "

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

    def _log_chat_message(self, role: str, message_content: str) -> None:
        """Logs a message to the chat display."""
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
        """Updates the status line above the chat log."""
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
            logger.debug(
                f"Status line updated: Provider: {provider_display}, Model: {model_display}, Status: {status}"
            )
        except NoMatches:
            logger.warning("ChatView: Could not update chat status line (widget not found).")
        except Exception as e:
            logger.error(f"Error updating chat status line: {e}", exc_info=True)

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        """Clears chat history and resets the chat log display."""
        self.chat_history.clear()
        try:
            log_widget = self.query_one("#chat_log_widget", ChatMessageList)
            log_widget.clear_messages()
        except NoMatches:
            logger.warning("Chat log widget not found for clearing during reset.")
        self._log_chat_message(self.ROLE_SYSTEM, status_message)
        self._update_chat_status_line(status="Idle", current_messages=0)

    def on_mount(self) -> None:
        logger.info("ChatView: on_mount CALLED.")
        self.chat_history: List[Dict[str, str]] = []
        self.llm_client_instance = None
        self._reset_chat_log_and_status("ChatView initialized. Configure LLM in sidebar.")
        self.focus_default_widget()
        try:
            self.query_one("#chat_model_loading_indicator", LoadingIndicator).display = False
        except NoMatches:
            logger.warning("ChatView: #chat_model_loading_indicator not found on mount")

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
                    yield Button("Inject Active Schema", id="inject_schema_button")
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
        logger.debug(f"ChatView: Starting _load_models_for_provider for '{provider_value}'")
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
            else:
                await self.handle_model_change_from_llm_config_panel(
                    LLMConfigPanel.ModelChanged(None)
                )
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
        if panel.provider != provider_value:
            panel.provider = provider_value
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

        determined_model_for_panel: Optional[str] = None
        current_status_after_listing = "Models loaded"
        if error:
            panel.update_models_list([], error)
            self._log_chat_message(self.ROLE_SYSTEM, error)
            current_status_after_listing = "Model list error"
            if panel.model is not None:
                panel.model = None
            else:
                determined_model_for_panel = None
        else:
            options = [(m, m) for m in listed]
            prompt_if_empty = "No models found for this provider." if not listed else "Select Model"
            panel.update_models_list(options, prompt_if_empty)
            current_status_after_listing = "Models loaded" if listed else "No models found"
            if listed:
                default: Optional[str] = None
                if provider_value == "ollama":
                    preferred_ollama = ["gemma3:4b", "qwen3:8b"]
                    for p_base in preferred_ollama:
                        if p_base in listed:
                            default = p_base
                            break
                        if default is None:
                            for model_with_tag in listed:
                                if model_with_tag.startswith(p_base + ":"):
                                    default = model_with_tag
                                    break
                        if default:
                            break
                elif provider_value == "openai":
                    preferred_openai = ["gpt-4.1-nano", "gpt-4.1-mini"]
                    for p in preferred_openai:
                        if p in listed:
                            default = p
                            break
                elif provider_value == "google":
                    preferred_google = ["gemini-2.0-flash", "gemini-1.5-flash"]
                    for p in preferred_google:
                        if p in listed:
                            default = p
                            break
                if default is None and listed:
                    default = listed[0]
                determined_model_for_panel = default
                if panel.model != determined_model_for_panel:
                    panel.model = determined_model_for_panel
            else:
                if panel.model is not None:
                    panel.model = None
                else:
                    determined_model_for_panel = None

        self._update_chat_status_line(status=current_status_after_listing)
        logger.info(
            f"ChatView: _load_models_for_provider determined model for panel: '{panel.model}'. Directly processing this model."
        )
        await self.handle_model_change_from_llm_config_panel(
            LLMConfigPanel.ModelChanged(panel.model)
        )

    @on(LLMConfigPanel.ModelChanged)
    async def handle_model_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ModelChanged
    ) -> None:
        model_value = event.model
        logger.info(
            f"ChatView: handle_model_change_from_llm_config_panel received ModelChanged event with model: {model_value}"
        )
        if model_value:
            self._reset_chat_log_and_status(f"Model set to: {model_value}. Session reset.")
            try:
                self.query_one("#chat_message_input", Input).value = ""
            except NoMatches:
                pass
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

    def _create_and_set_llm_client(self) -> bool:
        logger.debug("ChatView: Attempting _create_and_set_llm_client")
        try:
            panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found during client creation.")
            self._log_chat_message(self.ROLE_SYSTEM, "LLM Configuration panel not found.")
            return False
        cfg = panel.get_llm_config()
        provider = cfg.get("provider_hint")
        model_name = cfg.get("model_name")
        logger.debug(
            f"ChatView: _create_and_set_llm_client using provider='{provider}', model_name='{model_name}' from panel config."
        )
        if not provider or not model_name:
            self._log_chat_message(
                self.ROLE_SYSTEM, "Provider or model not properly selected for client creation."
            )
            logger.warning(
                f"ChatView: Client creation skipped. Provider='{provider}', Model='{model_name}'"
            )
            return False
        llm_class = self.PROVIDER_CLASSES.get(provider)
        if not llm_class:
            self._log_chat_message(
                self.ROLE_SYSTEM, f"Unknown provider '{provider}' for client creation."
            )
            logger.error(f"ChatView: Unknown provider '{provider}' found during client creation.")
            return False
        client_kwargs = {"model_name": model_name}
        temperature = cfg.get("temperature")
        if provider == "ollama":
            client_kwargs["options"] = {}
            if temperature is not None:
                client_kwargs["options"]["temperature"] = temperature
        elif provider == "openai":
            if temperature is not None:
                client_kwargs["temperature"] = temperature
        elif provider == "google":
            if temperature is not None:
                client_kwargs["generation_config"] = {"temperature": temperature}
        logger.debug(
            f"ChatView: Instantiating LLM class '{llm_class.__name__}' with kwargs: {client_kwargs}"
        )
        try:
            new_client = llm_class(**client_kwargs)
            self.llm_client_instance = new_client
            logger.info(f"ChatView: LLM client successfully created for {provider}:{model_name}.")
            return True
        except Exception as e:
            logger.error(
                f"ChatView: Failed to create LLM client for {provider}:{model_name}. KWARGS: {client_kwargs}. Error: {e}",
                exc_info=True,
            )
            self._log_chat_message(
                self.ROLE_SYSTEM,
                f"Error creating LLM client for '{model_name}': {e.__class__.__name__} - {str(e)[:100]}. See console log for full traceback.",
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
            if max_hist is not None and max_hist > 0 and len(hist) > max_hist:
                return hist[-max_hist:]
        except NoMatches:
            logger.warning("LLMConfigPanel not found in _get_effective_history_for_llm")
        return hist

    async def _send_user_message(self) -> None:
        logger.debug(
            f"ChatView: _send_user_message called. LLM client is {'SET' if self.llm_client_instance else 'None'}."
        )
        try:
            input_widget = self.query_one("#chat_message_input", Input)
        except NoMatches:
            await self.app.push_screen(ErrorDialog("UI Error", "Chat input field not found."))
            return
        if not self.llm_client_instance:
            logger.warning("ChatView: Send attempt while llm_client_instance is None.")
            await self.app.push_screen(
                ErrorDialog("LLM Error", "LLM not configured. Please select a provider and model.")
            )
            return
        user_text = input_widget.value.strip()
        if not user_text:
            return
        send_btn = self.query_one("#send_chat_message_button", Button)
        stop_btn = self.query_one("#stop_chat_message_button", Button)
        self._log_chat_message(self.ROLE_USER, user_text)
        self.chat_history.append({"role": self.ROLE_USER, "content": user_text})
        history_for_llm = self._get_effective_history_for_llm()
        if (
            history_for_llm
            and history_for_llm[-1]["content"] == user_text
            and history_for_llm[-1]["role"] == self.ROLE_USER
        ):
            history_for_llm = history_for_llm[:-1]
        input_widget.value = ""
        input_widget.disabled = True
        send_btn.disabled = True
        stop_btn.disabled = False
        self._update_chat_status_line(status="Sending...")
        client = self.llm_client_instance
        task = functools.partial(client.send_message, message=user_text, history=history_for_llm)
        if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
            self.current_llm_worker.cancel()
        self.current_llm_worker = self.app.run_worker(task, thread=True, group="llm_call")
        try:
            ai_response = await self.current_llm_worker.wait()
            if self.current_llm_worker.state == WorkerState.SUCCESS:
                clean_response = ai_response.strip() if isinstance(ai_response, str) else ""
                self._log_chat_message(self.ROLE_AI, clean_response)
                self.chat_history.append({"role": self.ROLE_AI, "content": clean_response})
        except WorkerFailed as wf:
            err_msg = str(wf.error) if wf.error else "LLM call failed with no specific error."
            self._log_chat_message(self.ROLE_SYSTEM, f"LLM Error: {err_msg}")
            logger.error(f"LLM WorkerFailed: {wf.error}", exc_info=True)
        except WorkerCancelled:
            self._log_chat_message(self.ROLE_SYSTEM, "LLM call stopped by user.")
        except Exception as e:
            logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
            self._log_chat_message(self.ROLE_SYSTEM, f"Unexpected error: {e!s}")
        finally:
            self.current_llm_worker = None
            if self.is_mounted:
                try:
                    input_widget.disabled = False
                    send_btn.disabled = False
                    stop_btn.disabled = True
                    self._update_chat_status_line(status="Ready")
                    input_widget.focus()
                except Exception as e_finally:
                    logger.error(
                        f"Error in _send_user_message finally block: {e_finally}", exc_info=True
                    )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "send_chat_message_button":
            await self._send_user_message()
        elif bid == "stop_chat_message_button":
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
                self.current_llm_worker.cancel()
                self._log_chat_message(self.ROLE_SYSTEM, "Attempting to stop LLM response...")
        elif bid == "inject_schema_button":
            await self._inject_schema_into_input()

    async def _get_active_collection_schema_for_injection(self) -> Optional[str]:
        coll = self.app.active_collection
        if not (self.app.current_mongo_uri and self.app.current_db_name and coll):
            self.app.notify(
                "No active DB connection or collection selected to inject schema.",
                title="Schema Injection Info",
                severity="warning",
            )
            return None
        cached_schema_info = self.app.current_schema_analysis_results
        schema_to_inject: Optional[Dict] = None
        if (
            isinstance(cached_schema_info, dict)
            and cached_schema_info.get("collection_name") == coll
            and "hierarchical_schema" in cached_schema_info
        ):
            schema_to_inject = cached_schema_info["hierarchical_schema"]
            logger.info(f"Using cached schema for '{coll}' for injection.")
        else:
            logger.info(
                f"No cached schema for '{coll}' or cache mismatch. Fetching live schema for injection."
            )
            self.app.notify(f"Fetching schema for '{coll}' to inject...", title="Schema Injection")
            try:
                if not core_db_manager.db_connection_active(
                    uri=self.app.current_mongo_uri,
                    db_name=self.app.current_db_name,
                    force_reconnect=False,
                    server_timeout_ms=2000,
                ):
                    if not core_db_manager.db_connection_active(
                        uri=self.app.current_mongo_uri,
                        db_name=self.app.current_db_name,
                        force_reconnect=True,
                        server_timeout_ms=3000,
                    ):
                        raise ConnectionError(
                            "DB connection lost or could not be re-established before fetching schema."
                        )
                pymongo_collection = SchemaAnalyser.get_collection(
                    self.app.current_mongo_uri, self.app.current_db_name, coll
                )
                analysis_task = functools.partial(
                    SchemaAnalyser.infer_schema_and_field_stats,
                    collection=pymongo_collection,
                    sample_size=100,
                )
                worker: Worker[Tuple[Dict, Dict]] = self.app.run_worker(
                    analysis_task, thread=True, group="schema_injection_fetch"
                )
                schema_data, _ = await worker.wait()
                if worker.is_cancelled:
                    self.app.notify(
                        f"Schema fetch for '{coll}' cancelled.",
                        title="Schema Injection",
                        severity="warning",
                    )
                    return None
                if not schema_data:
                    self.app.notify(
                        f"No schema data returned for '{coll}'. Cannot inject.",
                        title="Schema Injection Error",
                        severity="error",
                    )
                    return None
                schema_to_inject = SchemaAnalyser.schema_to_hierarchical(schema_data)
            except ConnectionError as ce:
                logger.error(
                    f"DB Connection error fetching schema for injection: {ce}", exc_info=True
                )
                await self.app.push_screen(
                    ErrorDialog("DB Connection Error", f"Could not connect to fetch schema: {ce}")
                )
                return None
            except WorkerCancelled:
                self.app.notify(
                    f"Schema fetch for '{coll}' was cancelled by worker.",
                    title="Schema Injection",
                    severity="warning",
                )
                return None
            except Exception as e:
                logger.error(f"Error fetching schema for injection: {e}", exc_info=True)
                await self.app.push_screen(
                    ErrorDialog("Schema Fetch Error", f"Could not fetch schema for '{coll}': {e!s}")
                )
                return None
        if schema_to_inject is None:
            self.app.notify(
                f"Could not obtain schema for '{coll}'.", title="Schema Injection", severity="error"
            )
            return None
        try:
            return json.dumps(schema_to_inject, indent=2, default=str)
        except TypeError:
            logger.error(
                f"Schema for '{coll}' is not JSON serializable for injection, even with default=str."
            )
            return str(schema_to_inject)

    async def _inject_schema_into_input(self) -> None:
        coll = self.app.active_collection
        if not coll:
            self.app.notify(
                "No active collection selected to inject schema.",
                title="Schema Injection Info",
                severity="warning",
            )
            return

        schema_str = await self._get_active_collection_schema_for_injection()
        if schema_str is None:
            return

        try:
            inp = self.query_one("#chat_message_input", Input)
            current_val = inp.value
            new_schema_prefix_block = (
                f"{self.SCHEMA_INJECT_START_MARKER_BASE}{coll}' is as follows:\n"
                f"```json\n{schema_str}{self.SCHEMA_INJECT_END_MARKER}"
            )
            text_after_any_previous_schema = current_val
            if current_val.startswith(self.SCHEMA_INJECT_START_MARKER_BASE):
                end_of_context_marker_for_stripping = "\n\nBased on this schema, "
                try:
                    idx_after_old_context = current_val.index(
                        end_of_context_marker_for_stripping
                    ) + len(end_of_context_marker_for_stripping)
                    text_after_any_previous_schema = current_val[idx_after_old_context:]
                    logger.debug(
                        f"Found existing schema context. User text after it: '{text_after_any_previous_schema}'"
                    )
                except ValueError:
                    logger.debug(
                        "Existing schema context start marker found, but end marker is missing/altered."
                    )
                    if current_val.startswith(self.SCHEMA_INJECT_START_MARKER_BASE):
                        text_after_any_previous_schema = ""
                        self._log_chat_message(
                            self.ROLE_SYSTEM,
                            "Existing schema context was altered. Replacing with new schema.",
                        )
            inp.value = new_schema_prefix_block + text_after_any_previous_schema.lstrip()

            button = self.query_one("#inject_schema_button", Button)
            original_label = button.label
            button.label = Text("Schema Injected ✓", style="italic green")
            button.disabled = True

            def revert_button_state():
                if button.is_mounted:
                    button.label = original_label
                    button.disabled = False

            self.set_timer(2.5, revert_button_state)

            inp.focus()
            inp.action_end()
            self._log_chat_message(
                self.ROLE_SYSTEM,
                f"Schema for collection '{coll}' has been prepared and added to your message input. You can now ask questions about it or use it in your query.",
            )
            self.app.notify(
                f"Schema for '{coll}' injected/updated in input.", title="Schema Injected"
            )
        except NoMatches:
            logger.error("Chat input widget or inject button not found for schema injection.")
            self.app.notify("Chat input field not found.", title="UI Error", severity="error")

    @on(LLMConfigPanel.NewSessionRequested)
    async def handle_new_session_requested_from_llm_config_panel(
        self, event: LLMConfigPanel.NewSessionRequested
    ) -> None:
        logger.info("ChatView: NewSessionRequested event received.")
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
