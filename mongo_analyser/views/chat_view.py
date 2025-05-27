import asyncio
import functools
import json
import logging
from typing import Dict, List, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Select, Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

import mongo_analyser.core.db as core_db_manager

# For schema access (simplified, ideally app state or service)
from mongo_analyser.core.analyser import SchemaAnalyser
from mongo_analyser.dialogs import ErrorDialog
from mongo_analyser.llm_chat import LiteLLMChat, LLMChat
from mongo_analyser.widgets import ChatMessageList, LLMConfigPanel

logger = logging.getLogger(__name__)


class ChatView(Container):
    current_llm_worker: Worker | None = None
    llm_client_instance: LLMChat | None = None

    ROLE_USER = "user"
    ROLE_AI = "assistant"  # Consistent with OpenAI's 'assistant' role
    ROLE_SYSTEM = "system"

    def on_mount(self) -> None:
        logger.info("ChatView: on_mount CALLED.")
        self.chat_history: List[Dict[str, str]] = []  # Stores {role: str, content: str}
        self.llm_client_instance = None  # Initialize LLM client as None
        self._reset_chat_log_and_status("ChatView initialized. Configure LLM in sidebar.")
        self.focus_default_widget()

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        self.chat_history = []
        try:
            log_widget = self.query_one("#chat_log_widget", ChatMessageList)
            log_widget.clear_messages()
        except NoMatches:
            logger.warning("Chat log widget not found for clearing during reset.")

        self._log_chat_message(self.ROLE_SYSTEM, status_message)
        self._update_chat_status_line(status="Idle", current_messages=0)

    def _log_chat_message(self, role: str, message_content: str):
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
                    classes="chat_status",  # Ensure this class is styled
                )
                yield ChatMessageList(id="chat_log_widget")

                with Horizontal(classes="chat_action_buttons"):
                    yield Button("Inject Active Schema", id="inject_schema_button")
                    # yield Button("Inject Stats Summary", id="inject_stats_button") # Placeholder for now

                with Horizontal(id="chat_input_bar", classes="chat_input_container"):
                    yield Input(
                        placeholder="Type a message or inject schema...",
                        id="chat_message_input",
                        # classes="chat_input_text", # Redundant if styled by ID
                    )
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
            llm_config_panel = self.query_one(LLMConfigPanel)
            chat_status_widget = self.query_one("#chat_status_line", Static)

            provider_val = llm_config_panel.provider
            model_val = llm_config_panel.model

            max_tokens_display = "N/A"  # This is output max_tokens, not context window
            if self.llm_client_instance and hasattr(self.llm_client_instance, "max_tokens"):
                max_tokens_display = str(self.llm_client_instance.max_tokens)
            elif (
                "max_tokens" in llm_config_panel.get_llm_config()
            ):  # Fallback to config panel if client not ready
                max_tokens_display = str(llm_config_panel.get_llm_config().get("max_tokens", "N/A"))

            provider_display = str(provider_val).capitalize() if provider_val else "N/A"
            model_display = str(model_val) if model_val else "N/A"

            if self.llm_client_instance:  # Use more precise names from client if available
                provider_display = str(
                    getattr(self.llm_client_instance, "provider_hint", provider_display)
                ).capitalize()
                model_display = getattr(self.llm_client_instance, "raw_model_name", model_display)

            # History display
            max_hist_val = llm_config_panel.max_history_messages
            max_hist_display = "N/A"
            if max_hist_val is not None:
                if max_hist_val == 0:
                    max_hist_display = "All"
                elif max_hist_val == -1:
                    max_hist_display = "None"
                else:
                    max_hist_display = str(max_hist_val)

            actual_history_len = len(self._get_effective_history_for_llm())  # Number of turns sent

            history_info = f"History: {actual_history_len} (max: {max_hist_display})"

            chat_status_widget.update(
                f"Provider: {provider_display} | Model: {model_display} | {history_info} | Status: {status}"
            )
        except NoMatches:
            logger.warning(
                "ChatView: Could not update chat status line, a required widget was not found."
            )
        except Exception as e:  # Catch any other error during update
            logger.error(f"Error updating chat status line: {e}", exc_info=True)

    @on(LLMConfigPanel.ProviderChanged)
    async def handle_provider_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ProviderChanged
    ) -> None:
        logger.info(f"ChatView: ProviderChanged event with provider: {event.provider}")
        if self.llm_client_instance:  # Clear old client if provider changes
            self.llm_client_instance = None
            logger.info("ChatView: LLM client instance cleared due to provider change.")
            self._log_chat_message(self.ROLE_SYSTEM, "Provider changed. LLM client reset.")

        # Defer the model loading to ensure LLMConfigPanel is fully queryable by the time _load_models_for_provider runs
        # self.app.call_later can schedule a coroutine.
        self.app.call_later(self._load_models_for_provider, event.provider)

    async def _load_models_for_provider(self, provider_value: Optional[str]) -> None:
        logger.info(
            f"ChatView: _load_models_for_provider called (potentially deferred) with provider_value: {provider_value}"
        )
        try:
            llm_config_panel = self.query_one(
                LLMConfigPanel
            )  # This should now succeed due to call_later
        except NoMatches:
            logger.error(
                "ChatView: LLMConfigPanel not found even after deferral. Cannot load models."
            )
            # Optionally, notify user or log a system message in chat
            self._log_chat_message(
                self.ROLE_SYSTEM, "Error: LLM Configuration panel not found. Cannot load models."
            )
            return

        if provider_value is None or provider_value == Select.BLANK:
            logger.info("ChatView: Provider is None or blank. Clearing models list.")
            if llm_config_panel.is_mounted:
                llm_config_panel.update_models_list([], "Select Provider First")
                llm_config_panel.model = None
                self._update_chat_status_line(status="Provider cleared")
            return

        provider_value_str = str(provider_value)
        logger.info(f"ChatView: Attempting to load models for provider: {provider_value_str}")

        if llm_config_panel.is_mounted:
            llm_config_panel.set_model_select_loading(
                True, f"Loading models for {provider_value_str}..."
            )
            self._update_chat_status_line(status=f"Loading {provider_value_str} models...")
            self._log_chat_message(self.ROLE_SYSTEM, f"Fetching models for {provider_value_str}...")

        listed_models: List[str] = []
        error_message: str | None = None

        client_config_for_list = (
            llm_config_panel.get_llm_config() if llm_config_panel.is_mounted else {}
        )
        logger.debug(f"ChatView: Client config for listing models: {client_config_for_list}")

        try:
            callable_with_args = functools.partial(
                LiteLLMChat.list_models,
                provider=provider_value_str,
                client_config=client_config_for_list,
            )
            logger.debug(
                f"ChatView: Preparing worker for LiteLLMChat.list_models with provider '{provider_value_str}'."
            )
            worker: Worker[List[str]] = self.app.run_worker(
                callable_with_args, thread=True, group="model_listing"
            )
            listed_models = await worker.wait()

            if worker.is_cancelled:
                logger.warning(
                    f"ChatView: Model listing worker for '{provider_value_str}' was cancelled."
                )
                error_message = "Model loading cancelled."
            else:
                logger.info(
                    f"ChatView: Worker finished. LiteLLMChat.list_models returned {len(listed_models)} models for provider '{provider_value_str}'."
                )

        except WorkerCancelled:
            logger.warning(
                f"ChatView: Model listing task for '{provider_value_str}' was cancelled during await."
            )
            error_message = "Model loading cancelled."
        except Exception as e:
            logger.error(
                f"ChatView: Error from worker during model listing for '{provider_value_str}': {e}",
                exc_info=True,
            )
            error_message = f"Failed to list models: {str(e)[:50]}..."

        if not llm_config_panel.is_mounted:
            return

        llm_config_panel.set_model_select_loading(False)
        default_model_to_set: Optional[str] = None

        if error_message:
            llm_config_panel.update_models_list([], error_message)
            llm_config_panel.model = None
            self._log_chat_message(self.ROLE_SYSTEM, error_message)
            self._update_chat_status_line(status="Model list error")
        elif listed_models:
            model_options = [(name, name) for name in listed_models]
            llm_config_panel.update_models_list(model_options, "Select Model")
            logger.info(
                f"ChatView: {len(listed_models)} models found for {provider_value_str}. Models: {listed_models[:5]}..."
            )

            if provider_value_str == "openai":
                preferred_models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
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

            if not default_model_to_set and listed_models:
                default_model_to_set = listed_models[0]
                logger.info(
                    f"ChatView: No preferred default for '{provider_value_str}', defaulting to first model: {default_model_to_set}"
                )

            if default_model_to_set:
                llm_config_panel.model = default_model_to_set
            else:
                llm_config_panel.update_models_list([], "No models found.")
                llm_config_panel.model = None
                self._log_chat_message(
                    self.ROLE_SYSTEM,
                    f"No models available for {provider_value_str} after filtering.",
                )
                self._update_chat_status_line(status="No models")
        else:
            llm_config_panel.update_models_list([], "No models found for this provider.")
            llm_config_panel.model = None
            self._log_chat_message(self.ROLE_SYSTEM, f"No models found for {provider_value_str}.")
            self._update_chat_status_line(status="No models")

    @on(LLMConfigPanel.ModelChanged)
    async def handle_model_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ModelChanged
    ) -> None:
        logger.info(f"ChatView: ModelChanged event with model: {event.model}")
        model_value = event.model

        if model_value and model_value != Select.BLANK:
            self._reset_chat_log_and_status(
                status_message=f"Model set to: {model_value}. Session reset."
            )
            try:
                self.query_one("#chat_message_input", Input).value = ""
            except NoMatches:
                pass

            client_created = self._create_and_set_llm_client()
            if client_created:
                self._log_chat_message(self.ROLE_SYSTEM, "Session ready. LLM client configured.")
                self._update_chat_status_line(status="Ready")
            else:
                self._log_chat_message(
                    self.ROLE_SYSTEM, "Failed to configure LLM client for the new model."
                )
                self._update_chat_status_line(status="Client Error")

            try:
                self.query_one("#chat_message_input", Input).focus()
            except NoMatches:
                pass
        else:
            if self.llm_client_instance:
                self.llm_client_instance = None
                logger.info("ChatView: LLM client instance cleared due to model deselection.")
                self._log_chat_message(self.ROLE_SYSTEM, "Model deselected. LLM client cleared.")
            self._update_chat_status_line(status="Select model")

    def _create_and_set_llm_client(self) -> bool:
        try:
            llm_config_panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found when trying to create LLM client.")
            self._log_chat_message(self.ROLE_SYSTEM, "LLM Configuration panel not found.")
            return False

        config = llm_config_panel.get_llm_config()
        raw_model_name = config.get("model_name")
        provider_hint = config.get("provider_hint")

        if not provider_hint:
            self._log_chat_message(
                self.ROLE_SYSTEM, "Provider not selected. Cannot create LLM client."
            )
            return False
        if not raw_model_name or raw_model_name == Select.BLANK:
            self._log_chat_message(
                self.ROLE_SYSTEM, "Model not selected. Cannot create LLM client."
            )
            return False

        logger.info(
            f"ChatView: Creating LiteLLMChat client with raw_model_name='{raw_model_name}', provider_hint='{provider_hint}', full_config={config}"
        )
        try:
            new_client = LiteLLMChat(
                model_name=str(raw_model_name), provider_hint=str(provider_hint), **config
            )
            self.llm_client_instance = new_client
            logger.info(
                f"ChatView: LiteLLM client configured. Effective model for API: '{self.llm_client_instance.model_name}', "
                f"Provider hint used: '{provider_hint}', Raw model name from panel: '{raw_model_name}'"
            )
            return True
        except Exception as e:
            logger.error(
                f"ChatView: Failed to create LiteLLM client for provider '{provider_hint}', raw model '{raw_model_name}': {e}",
                exc_info=True,
            )
            self._log_chat_message(
                self.ROLE_SYSTEM,
                f"Error creating LLM client: {e.__class__.__name__} - {str(e)[:60]}...",
            )
            self.llm_client_instance = None
            return False

    def _get_effective_history_for_llm(self) -> List[Dict[str, str]]:
        llm_formatted_history: List[Dict[str, str]] = []
        for msg in self.chat_history:
            if msg["role"] == self.ROLE_USER or msg["role"] == self.ROLE_AI:
                llm_formatted_history.append({"role": msg["role"], "content": msg["content"]})

        try:
            llm_config_panel = self.query_one(LLMConfigPanel)
            max_hist = llm_config_panel.max_history_messages

            if max_hist is not None:
                if max_hist == -1:
                    return []
                if max_hist == 0:
                    return llm_formatted_history
                if max_hist > 0 and len(llm_formatted_history) > max_hist:
                    return llm_formatted_history[-max_hist:]
        except NoMatches:
            logger.warning("LLMConfigPanel not found for history truncation, using full history.")

        return llm_formatted_history

    async def _send_user_message(self) -> None:
        try:
            message_input_widget = self.query_one("#chat_message_input", Input)
        except NoMatches:
            logger.error("ChatView: #chat_message_input not found to send message.")
            await self.app.push_screen(ErrorDialog("UI Error", "Chat input field not found."))
            return

        if self.llm_client_instance is None:
            self._log_chat_message(
                self.ROLE_SYSTEM,
                "LLM not configured. Please select provider/model and start a new session via the config panel.",
            )
            await self.app.push_screen(ErrorDialog("LLM Error", "LLM not configured. Use sidebar."))
            return

        user_message_content = message_input_widget.value.strip()
        if not user_message_content:
            return

        try:
            send_button = self.query_one("#send_chat_message_button", Button)
            stop_button = self.query_one("#stop_chat_message_button", Button)
        except NoMatches:
            logger.error("ChatView: Send or Stop button not found.")
            await self.app.push_screen(ErrorDialog("UI Error", "Send/Stop buttons not found."))
            return

        self._log_chat_message(self.ROLE_USER, user_message_content)
        self.chat_history.append({"role": self.ROLE_USER, "content": user_message_content})

        history_for_llm_call = self._get_effective_history_for_llm()
        if (
            history_for_llm_call
            and history_for_llm_call[-1]["role"] == self.ROLE_USER
            and history_for_llm_call[-1]["content"] == user_message_content
        ):
            history_for_llm_call = history_for_llm_call[:-1]

        message_input_widget.value = ""
        message_input_widget.disabled = True
        send_button.disabled = True
        stop_button.disabled = False
        self._update_chat_status_line(status="Sending...", current_messages=len(self.chat_history))

        active_llm_client = self.llm_client_instance

        def task_to_run_in_worker():
            logger.debug(
                f"ChatView: Worker starting send_message to model {active_llm_client.model_name}"
            )
            return active_llm_client.send_message(
                message=user_message_content, history=history_for_llm_call
            )

        if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
            logger.info("ChatView: Cancelling previous running LLM worker before starting new one.")
            self.current_llm_worker.cancel()

        self.current_llm_worker = self.app.run_worker(
            task_to_run_in_worker, thread=True, group="llm_call"
        )

        try:
            response_text = await self.current_llm_worker.wait()

            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.SUCCESS:
                self._log_chat_message(self.ROLE_AI, response_text)
                self.chat_history.append({"role": self.ROLE_AI, "content": response_text})

        except WorkerFailed as e_failed:
            error_to_log = e_failed.error
            logger.error(f"ChatView: LLM WorkerFailed: {error_to_log}", exc_info=error_to_log)
            if (
                isinstance(error_to_log, asyncio.CancelledError)
                and self.current_llm_worker
                and self.current_llm_worker.is_cancelled
            ):
                msg_to_log = "LLM call stopped by user."
            else:
                msg_to_log = (
                    f"LLM Error: {error_to_log.__class__.__name__} - {str(error_to_log)[:60]}..."
                )
            self._log_chat_message(self.ROLE_SYSTEM, msg_to_log)
            if (
                self.chat_history
                and self.chat_history[-1]["role"] == self.ROLE_USER
                and self.chat_history[-1]["content"] == user_message_content
            ):
                self.chat_history.pop()

        except WorkerCancelled:
            logger.info("ChatView: LLM call explicitly cancelled by user (WorkerCancelled).")
            self._log_chat_message(self.ROLE_SYSTEM, "LLM call stopped by user.")
            if (
                self.chat_history
                and self.chat_history[-1]["role"] == self.ROLE_USER
                and self.chat_history[-1]["content"] == user_message_content
            ):
                self.chat_history.pop()

        except Exception as e_unexpected:
            logger.error(
                f"ChatView: Unexpected error during LLM communication: {e_unexpected}",
                exc_info=True,
            )
            self._log_chat_message(
                self.ROLE_SYSTEM, f"Unexpected error: {e_unexpected.__class__.__name__}"
            )
            if (
                self.chat_history
                and self.chat_history[-1]["role"] == self.ROLE_USER
                and self.chat_history[-1]["content"] == user_message_content
            ):
                self.chat_history.pop()
        finally:
            self.current_llm_worker = None
            if self.is_mounted:
                try:
                    message_input_widget.disabled = False
                    send_button.disabled = False
                    stop_button.disabled = True
                    self._update_chat_status_line(status="Ready")
                    message_input_widget.focus()
                except Exception as e_fin:
                    logger.warning(
                        f"ChatView: Error in finally block of _send_user_message: {e_fin}"
                    )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send_chat_message_button":
            await self._send_user_message()
        elif button_id == "stop_chat_message_button":
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
                logger.info("ChatView: Stop button pressed, cancelling LLM worker.")
                self.current_llm_worker.cancel()
                self._log_chat_message(self.ROLE_SYSTEM, "Attempting to stop LLM response...")
            else:
                self._log_chat_message(self.ROLE_SYSTEM, "No active LLM call to stop.")
                try:
                    self.query_one("#stop_chat_message_button", Button).disabled = True
                except NoMatches:
                    pass

        elif button_id == "inject_schema_button":
            await self._inject_schema_into_input()

    async def _get_active_collection_schema_for_injection(self) -> Optional[str]:
        active_collection_name = self.app.active_collection
        if (
            not self.app.current_mongo_uri
            or not self.app.current_db_name
            or not active_collection_name
        ):
            self.app.notify("No active DB connection or collection selected.", severity="warning")
            return None

        if (
            self.app.current_schema_analysis_results
            and self.app.current_schema_analysis_results.get("collection_name")
            == active_collection_name
            and "hierarchical_schema" in self.app.current_schema_analysis_results
        ):
            logger.info(f"Using cached schema for '{active_collection_name}' for injection.")
            schema_to_inject = self.app.current_schema_analysis_results["hierarchical_schema"]
            try:
                return json.dumps(schema_to_inject, indent=2, default=str)
            except TypeError:
                return str(schema_to_inject)

        self.app.notify(
            f"Fetching schema for '{active_collection_name}' to inject...", severity="information"
        )
        try:
            if not core_db_manager.db_connection_active(
                uri=self.app.current_mongo_uri, db_name=self.app.current_db_name
            ):
                raise ConnectionError("DB connection lost before fetching schema for injection.")

            collection_obj = SchemaAnalyser.get_collection(
                self.app.current_mongo_uri, self.app.current_db_name, active_collection_name
            )
            schema_data, _ = SchemaAnalyser.infer_schema_and_field_stats(
                collection_obj, sample_size=100
            )
            if not schema_data:
                self.app.notify(
                    f"Could not generate schema for '{active_collection_name}'.", severity="warning"
                )
                return None

            hierarchical = SchemaAnalyser.schema_to_hierarchical(schema_data)
            return json.dumps(hierarchical, indent=2, default=str)
        except Exception as e:
            logger.error(
                f"Error fetching schema for injection for '{active_collection_name}': {e}",
                exc_info=True,
            )
            await self.app.push_screen(
                ErrorDialog(
                    "Schema Fetch Error",
                    f"Could not get schema for {active_collection_name}: {e!s}",
                )
            )
            return None

    async def _inject_schema_into_input(self) -> None:
        active_collection_name = self.app.active_collection
        if not active_collection_name:
            self.app.notify("No active collection selected to inject schema.", severity="warning")
            return

        schema_json_str = await self._get_active_collection_schema_for_injection()

        if schema_json_str:
            try:
                chat_input_widget = self.query_one("#chat_message_input", Input)
                current_text = chat_input_widget.value

                max_schema_len_for_prompt = 3000
                truncated_schema_str = schema_json_str
                if len(schema_json_str) > max_schema_len_for_prompt:
                    truncated_schema_str = (
                        schema_json_str[:max_schema_len_for_prompt] + "\n... (schema truncated) ..."
                    )
                    self.app.notify(
                        "Schema was long and has been truncated for the prompt.",
                        severity="information",
                        timeout=5.0,
                    )

                new_text = (
                    f"CONTEXT: The following is the MongoDB schema for the collection '{active_collection_name}':\n"
                    f"```json\n{truncated_schema_str}\n```\n\n"
                    f"Based on this schema, {current_text}"
                )
                chat_input_widget.value = new_text
                chat_input_widget.focus()
                self._log_chat_message(
                    self.ROLE_SYSTEM,
                    f"Schema for '{active_collection_name}' pre-filled into input. You can now ask questions about it.",
                )
            except NoMatches:
                logger.error("Chat input widget not found for schema injection.")
        else:
            self._log_chat_message(
                self.ROLE_SYSTEM,
                f"Failed to retrieve or inject schema for '{active_collection_name}'.",
            )

    @on(LLMConfigPanel.NewSessionRequested)
    async def handle_new_session_requested_from_llm_config_panel(
        self, event: LLMConfigPanel.NewSessionRequested
    ) -> None:
        logger.info("ChatView: NewSessionRequested event received.")
        try:
            llm_config_panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found for new session request.")
            await self.app.push_screen(ErrorDialog("UI Error", "LLM Config panel not found."))
            return

        if (
            llm_config_panel.provider is None
            or llm_config_panel.model is None
            or llm_config_panel.model == Select.BLANK
        ):
            self._reset_chat_log_and_status(
                status_message="Select provider and model first for a new session."
            )
            self._update_chat_status_line(status="Needs config")
            await self.app.push_screen(
                ErrorDialog(
                    "Configuration Incomplete",
                    "Please select a provider and model in the sidebar before starting a new session.",
                )
            )
            return

        self._reset_chat_log_and_status(
            status_message="New session started. LLM (re)configuring..."
        )
        try:
            self.query_one("#chat_message_input", Input).value = ""
        except NoMatches:
            pass

        client_created = self._create_and_set_llm_client()
        if client_created:
            self._log_chat_message(self.ROLE_SYSTEM, "LLM client (re)configured for new session.")
            self._update_chat_status_line(status="Ready")
        else:
            self._log_chat_message(
                self.ROLE_SYSTEM, "Failed to (re)configure LLM client for new session."
            )
            self._update_chat_status_line(status="Client Error")

        try:
            self.query_one("#chat_message_input", Input).focus()
        except NoMatches:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_message_input":
            await self._send_user_message()
