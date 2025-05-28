# mongo_analyser/views/chat_view.py

import functools
import json
import logging
from typing import Dict, List, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Input, Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

import mongo_analyser.core.db as core_db_manager
from mongo_analyser.core.analyser import SchemaAnalyser
from mongo_analyser.dialogs import ErrorDialog
from mongo_analyser.llm_chat.wrapper import LiteLLMChat
from mongo_analyser.widgets import ChatMessageList, LLMConfigPanel

logger = logging.getLogger(__name__)


class ChatView(Container):
    current_llm_worker: Worker | None = None
    llm_client_instance: LiteLLMChat | None = None

    ROLE_USER = "user"
    ROLE_AI = "assistant"
    ROLE_SYSTEM = "system"

    def __init__(
        self,
        *children: Widget,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        disabled: bool = False,
        markup: bool = True,
    ):
        super().__init__(
            *children,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            markup=markup,
        )
        self.chat_history: List[Dict[str, str]] = []

    def on_mount(self) -> None:
        logger.info("ChatView: on_mount CALLED.")
        self.chat_history: List[Dict[str, str]] = []
        self.llm_client_instance = None
        self._reset_chat_log_and_status("ChatView initialized. Configure LLM in sidebar.")
        self.focus_default_widget()

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        self.chat_history.clear()
        try:
            log_widget = self.query_one("#chat_log_widget", ChatMessageList)
            log_widget.clear_messages()
        except NoMatches:
            logger.warning("Chat log widget not found for clearing during reset.")
        self._log_chat_message(self.ROLE_SYSTEM, status_message)
        self._update_chat_status_line(status="Idle", current_messages=0)

    def _log_chat_message(self, role: str, message_content: str) -> None:
        try:
            chat_list_widget = self.query_one("#chat_log_widget", ChatMessageList)
            chat_list_widget.add_message(role, message_content)
        except NoMatches:
            logger.warning("Chat log widget (#chat_log_widget) not found for logging message.")
        except Exception as e:
            logger.error(f"Error logging chat message: {e}", exc_info=True)

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat_interface_horizontal_layout"):
            with Vertical(id="chat_main_area", classes="chat_column_main"):
                yield Static(
                    "Provider: N/A | Model: N/A | History: 0 (max: N/A) | Status: Idle",
                    id="chat_status_line",
                    classes="chat_status",
                )
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

    def _update_chat_status_line(
        self, status: str = "Idle", current_messages: int | None = None
    ) -> None:
        try:
            panel = self.query_one(LLMConfigPanel)
            chat_status = self.query_one("#chat_status_line", Static)

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

            chat_status.update(
                f"Provider: {provider_display} | Model: {model_display} | {history_info} | Status: {status}"
            )
        except NoMatches:
            logger.warning("ChatView: Could not update chat status line.")
        except Exception as e:
            logger.error(f"Error updating chat status line: {e}", exc_info=True)

    @on(LLMConfigPanel.ProviderChanged)
    async def handle_provider_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ProviderChanged
    ) -> None:
        logger.info(f"ChatView: ProviderChanged event with provider: {event.provider}")
        self.llm_client_instance = None
        self._log_chat_message(self.ROLE_SYSTEM, "Provider changed. LLM client reset.")
        self.app.call_later(self._load_models_for_provider, event.provider)

    async def _load_models_for_provider(self, provider_value: Optional[str]) -> None:
        try:
            panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found when loading models.")
            return

        panel.set_model_select_loading(True, f"Loading models for {provider_value}...")
        self._update_chat_status_line(status=f"Loading {provider_value} models...")
        self._log_chat_message(self.ROLE_SYSTEM, f"Fetching models for {provider_value}...")

        listed: List[str] = []
        error: Optional[str] = None
        cfg = panel.get_llm_config()

        try:
            worker: Worker[List[str]] = self.app.run_worker(
                functools.partial(
                    LiteLLMChat.list_models, provider=provider_value, client_config=cfg
                ),
                thread=True,
                group="model_listing",
            )
            listed = await worker.wait()
            if worker.is_cancelled:
                error = "Model loading cancelled."
        except Exception as e:
            logger.error(f"ChatView: Error listing models: {e}", exc_info=True)
            error = f"Failed to list models: {e.__class__.__name__}: {str(e)[:60]}"

        panel.set_model_select_loading(False)

        if error:
            panel.update_models_list([], error)
            panel.model = None
            self._log_chat_message(self.ROLE_SYSTEM, error)
            self._update_chat_status_line(status="Model list error")
            return

        options = [(m, m) for m in listed]
        panel.update_models_list(options, "Select Model")

        # pick sensible defaults per‐provider
        default: Optional[str] = None
        if provider_value == "ollama":
            for cand in ["llama2", "mistral"]:
                if cand in listed:
                    default = cand
                    break
        elif provider_value == "openai":
            for cand in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]:
                if cand in listed:
                    default = cand
                    break
        elif provider_value in ("google", "google_vertex"):
            for cand in ["gemini-1.5-pro", "gemini-1.5-flash-latest"]:
                if cand in listed:
                    default = cand
                    break

        if default is None and listed:
            default = listed[0]
        if default:
            panel.model = default

        self._update_chat_status_line(status="Models loaded")

    @on(LLMConfigPanel.ModelChanged)
    async def handle_model_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ModelChanged
    ) -> None:
        model_value = event.model
        logger.info(f"ChatView: ModelChanged event with model: {model_value}")
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
                self._log_chat_message(self.ROLE_SYSTEM, "Client Error")
                self._update_chat_status_line(status="Client Error")
            self.focus_default_widget()
        else:
            self.llm_client_instance = None
            self._log_chat_message(self.ROLE_SYSTEM, "Model deselected. LLM client cleared.")
            self._update_chat_status_line(status="Select model")

    def _create_and_set_llm_client(self) -> bool:
        try:
            panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found.")
            self._log_chat_message(self.ROLE_SYSTEM, "LLM Configuration panel not found.")
            return False

        cfg = panel.get_llm_config()
        provider = cfg.get("provider_hint")
        model = cfg.get("model_name")
        if not provider or not model:
            self._log_chat_message(self.ROLE_SYSTEM, "Provider or model not selected.")
            return False

        extra = cfg.copy()
        extra.pop("provider_hint", None)
        extra.pop("model_name", None)

        try:
            client = LiteLLMChat(model_name=model, provider_hint=provider, **extra)
            self.llm_client_instance = client
            logger.info(f"ChatView: LLM client created for {provider}:{model}")
            return True
        except Exception as e:
            logger.error(f"ChatView: Failed to create LLM client: {e}", exc_info=True)
            self._log_chat_message(
                self.ROLE_SYSTEM, f"Error creating LLM client: {e.__class__.__name__}"
            )
            return False

    def _get_effective_history_for_llm(self) -> List[Dict[str, str]]:
        hist = [m for m in self.chat_history if m["role"] in {self.ROLE_USER, self.ROLE_AI}]
        try:
            panel = self.query_one(LLMConfigPanel)
            max_hist = panel.max_history_messages
            if max_hist == -1:
                return []
            if max_hist > 0 and len(hist) > max_hist:
                return hist[-max_hist:]
        except NoMatches:
            pass
        return hist

    async def _send_user_message(self) -> None:
        try:
            input_widget = self.query_one("#chat_message_input", Input)
        except NoMatches:
            await self.app.push_screen(ErrorDialog("UI Error", "Chat input field not found."))
            return

        if not self.llm_client_instance:
            await self.app.push_screen(ErrorDialog("LLM Error", "LLM not configured."))
            return

        user_text = input_widget.value.strip()
        if not user_text:
            return

        send_btn = self.query_one("#send_chat_message_button", Button)
        stop_btn = self.query_one("#stop_chat_message_button", Button)

        self._log_chat_message(self.ROLE_USER, user_text)
        self.chat_history.append({"role": self.ROLE_USER, "content": user_text})

        history = self._get_effective_history_for_llm()
        if history and history[-1]["role"] == self.ROLE_USER:
            history = history[:-1]

        input_widget.value = ""
        input_widget.disabled = True
        send_btn.disabled = True
        stop_btn.disabled = False
        self._update_chat_status_line(status="Sending...")

        client = self.llm_client_instance
        task = functools.partial(client.send_message, message=user_text, history=history)

        if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
            self.current_llm_worker.cancel()

        self.current_llm_worker = self.app.run_worker(task, thread=True, group="llm_call")

        try:
            ai_response = await self.current_llm_worker.wait()
            if self.current_llm_worker.state == WorkerState.SUCCESS:
                self._log_chat_message(self.ROLE_AI, ai_response)
                self.chat_history.append({"role": self.ROLE_AI, "content": ai_response})
        except WorkerFailed as wf:
            err = wf.error
            self._log_chat_message(self.ROLE_SYSTEM, f"LLM Error: {err}")
            logger.error(f"LLM WorkerFailed: {err}", exc_info=True)
        except WorkerCancelled:
            self._log_chat_message(self.ROLE_SYSTEM, "LLM call stopped by user.")
        except Exception as e:
            logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
            self._log_chat_message(self.ROLE_SYSTEM, f"Unexpected error: {e}")
        finally:
            self.current_llm_worker = None
            if self.is_mounted:
                input_widget.disabled = False
                send_btn.disabled = False
                stop_btn.disabled = True
                self._update_chat_status_line(status="Ready")
                input_widget.focus()

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
            self.app.notify("No active DB connection or collection.", severity="warning")
            return None

        cached = getattr(self.app, "current_schema_analysis_results", {}) or {}
        if cached.get("collection_name") == coll and "hierarchical_schema" in cached:
            schema = cached["hierarchical_schema"]
        else:
            try:
                if not core_db_manager.db_connection_active(
                    uri=self.app.current_mongo_uri,
                    db_name=self.app.current_db_name,
                ):
                    raise ConnectionError("DB connection lost before fetching schema.")
                col = SchemaAnalyser.get_collection(
                    self.app.current_mongo_uri,
                    self.app.current_db_name,
                    coll,
                )
                schema_data, _ = SchemaAnalyser.infer_schema_and_field_stats(col, sample_size=100)
                schema = SchemaAnalyser.schema_to_hierarchical(schema_data)
            except Exception as e:
                logger.error(f"Error fetching schema for injection: {e}", exc_info=True)
                await self.app.push_screen(ErrorDialog("Schema Fetch Error", str(e)))
                return None

        try:
            return json.dumps(schema, indent=2, default=str)
        except TypeError:
            return str(schema)

    async def _inject_schema_into_input(self) -> None:
        coll = self.app.active_collection
        if not coll:
            self.app.notify("No active collection selected.", severity="warning")
            return

        schema_str = await self._get_active_collection_schema_for_injection()
        if not schema_str:
            return

        try:
            inp = self.query_one("#chat_message_input", Input)
            prefix = f"CONTEXT: MongoDB schema for '{coll}':\n```json\n{schema_str}\n```\n\n"
            inp.value = prefix + inp.value
            inp.focus()
            self._log_chat_message(
                self.ROLE_SYSTEM,
                f"Schema for '{coll}' injected into input.",
            )
        except NoMatches:
            logger.error("Chat input widget not found for schema injection.")

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

        if not panel.provider or not panel.model:
            self._reset_chat_log_and_status("Select provider and model first.")
            self._update_chat_status_line(status="Needs config")
            await self.app.push_screen(
                ErrorDialog(
                    "Configuration Incomplete",
                    "Please select a provider and model.",
                )
            )
            return

        self._reset_chat_log_and_status("New session started. LLM (re)configuring...")
        try:
            self.query_one("#chat_message_input", Input).value = ""
        except NoMatches:
            pass

        if self._create_and_set_llm_client():
            self._log_chat_message(
                self.ROLE_SYSTEM,
                "LLM client (re)configured for new session.",
            )
            self._update_chat_status_line(status="Ready")
        else:
            self._log_chat_message(self.ROLE_SYSTEM, "Client Error")
            self._update_chat_status_line(status="Client Error")

        self.focus_default_widget()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_message_input":
            await self._send_user_message()
