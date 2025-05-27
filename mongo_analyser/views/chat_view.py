# mongo_analyser/views/chat_view.py
import asyncio
import functools
import logging
from typing import Dict, List, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Select, Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

from mongo_analyser.llm_chat import LiteLLMChat, LLMChat
from mongo_analyser.widgets import ChatMessageList, LLMConfigPanel

logger = logging.getLogger(__name__)


class ChatView(Container):
    current_llm_worker: Worker | None = None
    llm_client_instance: LLMChat | None = None

    ROLE_USER = "user"
    ROLE_AI = "assistant"
    ROLE_SYSTEM = "system"

    # Removed custom __init__ method.
    # Initialization of chat_history and llm_client_instance is handled in on_mount.

    def _log_chat_message(self, role: str, message_content: str):
        try:
            chat_list_widget = self.query_one("#chat_log_widget", ChatMessageList)
            chat_list_widget.add_message(role, message_content)
        except NoMatches:
            logger.warning("Chat log widget (#chat_log_widget) not found for logging message.")
        except Exception as e:
            logger.error(f"Error logging chat message: {e}", exc_info=True)

    def on_mount(self) -> None:
        logger.info("ChatView: on_mount CALLED.")
        self.chat_history: List[Dict[str, str]] = []
        self.llm_client_instance = None
        self._reset_chat_log_and_status("ChatView initialized. Configure LLM in sidebar.")

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        self.chat_history = []
        try:
            log_widget = self.query_one("#chat_log_widget", ChatMessageList)
            log_widget.clear_messages()
        except NoMatches:
            logger.warning("Chat log widget not found for clearing during reset.")
        self._log_chat_message(self.ROLE_SYSTEM, status_message)
        self._update_chat_status_line(status="Idle", current_messages=0)

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat_interface_horizontal_layout"):
            with Vertical(id="chat_main_area", classes="chat_column_main"):
                yield Static(
                    "Provider: N/A | Model: N/A | Context: 0/0 | Status: Idle",
                    id="chat_status_line",
                    classes="chat_status",
                )
                yield ChatMessageList(id="chat_log_widget")

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

            max_tokens_display = "N/A"
            if self.llm_client_instance and hasattr(self.llm_client_instance, "max_tokens"):
                max_tokens_display = str(self.llm_client_instance.max_tokens)
            elif "max_tokens" in llm_config_panel.get_llm_config():
                max_tokens_display = str(llm_config_panel.get_llm_config().get("max_tokens", "N/A"))

            provider_display = str(provider_val).capitalize() if provider_val else "N/A"
            model_display = str(model_val) if model_val else "N/A"

            if self.llm_client_instance:
                provider_display = str(
                    getattr(self.llm_client_instance, "provider_hint", provider_display)
                ).capitalize()
                model_display = getattr(self.llm_client_instance, "raw_model_name", model_display)

            current_msg_count = (
                current_messages if current_messages is not None else len(self.chat_history)
            )
            context_display = f"{current_msg_count} turns / {max_tokens_display} tokens"

            chat_status_widget.update(
                f"Provider: {provider_display} | Model: {model_display} | Context: {context_display} | Status: {status}"
            )
        except NoMatches:
            logger.warning(
                "Could not update chat status line, a required widget (LLMConfigPanel or #chat_status_line) was not found."
            )
        except Exception as e:
            logger.error(f"Error updating chat status line: {e}", exc_info=True)

    @on(LLMConfigPanel.ProviderChanged)
    async def handle_provider_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ProviderChanged
    ) -> None:
        logger.info(
            f"ChatView: handle_provider_change_from_llm_config_panel CALLED with provider: {event.provider}"
        )
        if self.llm_client_instance:
            self.llm_client_instance = None
            logger.info("ChatView: LLM client instance cleared due to provider change.")
            self._log_chat_message(self.ROLE_SYSTEM, "Provider changed. LLM client reset.")
        await self._load_models_for_provider(event.provider)

    async def _load_models_for_provider(self, provider_value: Optional[str]) -> None:
        logger.info(
            f"ChatView: _load_models_for_provider called with provider_value: {provider_value}"
        )
        try:
            llm_config_panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("LLMConfigPanel not found in ChatView. This should not happen.")
            return

        if provider_value is None or provider_value == Select.BLANK:
            logger.info("ChatView: Provider is None or blank. Clearing models.")
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
            logger.info(
                f"ChatView: Worker finished. LiteLLMChat.list_models returned {len(listed_models)} models for provider '{provider_value_str}'. Models: {listed_models}"
            )
        except WorkerCancelled:
            logger.warning(
                f"ChatView: Model listing worker for '{provider_value_str}' was cancelled."
            )
            error_message = "Model loading cancelled."
        except Exception as e:
            logger.error(
                f"ChatView: Error from worker during model listing for '{provider_value_str}': {e}",
                exc_info=True,
            )
            error_message = f"Failed to list models: {str(e)[:37]}..."

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
            logger.info(f"ChatView: {len(listed_models)} models found for {provider_value_str}.")

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

            if not default_model_to_set and listed_models:
                default_model_to_set = listed_models[0]
                logger.info(
                    f"ChatView: No preferred default found for '{provider_value_str}', defaulting to first model: {default_model_to_set}"
                )

            if default_model_to_set:
                llm_config_panel.model = default_model_to_set
            else:
                llm_config_panel.update_models_list([], "No models found.")
                llm_config_panel.model = None
                self._log_chat_message(
                    self.ROLE_SYSTEM, f"No models available for {provider_value_str}."
                )
                self._update_chat_status_line(status="No models")
        else:
            llm_config_panel.update_models_list([], "No models found.")
            llm_config_panel.model = None
            self._log_chat_message(self.ROLE_SYSTEM, f"No models found for {provider_value_str}.")
            self._update_chat_status_line(status="No models")

    @on(LLMConfigPanel.ModelChanged)
    async def handle_model_change_from_llm_config_panel(
        self, event: LLMConfigPanel.ModelChanged
    ) -> None:
        logger.info(
            f"ChatView: handle_model_change_from_llm_config_panel CALLED with model: {event.model}"
        )
        model_value = event.model
        if model_value and model_value != Select.BLANK:
            self._reset_chat_log_and_status(
                status_message=f"Model set to: {model_value}. Session reset."
            )
            try:
                self.query_one("#chat_message_input", Input).value = ""
            except NoMatches:
                logger.warning("ChatView: #chat_message_input not found to clear value.")

            client_created = self._create_and_set_llm_client()
            if client_created:
                self._log_chat_message(self.ROLE_SYSTEM, "Session ready. LLM client configured.")
                self._update_chat_status_line(status="Ready")
            else:
                self._log_chat_message(self.ROLE_SYSTEM, "Failed to configure LLM client.")
                self._update_chat_status_line(status="Client Error")

            try:
                self.query_one("#chat_message_input", Input).focus()
            except NoMatches:
                logger.warning("ChatView: #chat_message_input not found to focus.")
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
                self.ROLE_SYSTEM, "Please select a provider in the config panel."
            )
            return False
        if not raw_model_name or raw_model_name == Select.BLANK:
            self._log_chat_message(self.ROLE_SYSTEM, "Please select a model in the config panel.")
            return False

        logger.info(
            f"ChatView: Creating LiteLLMChat client with model_name='{raw_model_name}', config={config}"
        )
        try:
            client_kwargs = {
                k: v for k, v in config.items() if k not in ["model_name", "provider_hint"]
            }
            new_client = LiteLLMChat(
                model_name=str(raw_model_name), provider_hint=str(provider_hint), **client_kwargs
            )
            self.llm_client_instance = new_client
            logger.info(
                f"ChatView: LiteLLM client configured. Effective model for API: '{self.llm_client_instance.model_name}', Provider hint used: '{provider_hint}', Raw model name from panel: '{raw_model_name}'"
            )
            return True
        except Exception as e:
            logger.error(
                f"ChatView: Failed to create LiteLLM client for provider '{provider_hint}', raw model '{raw_model_name}': {e}",
                exc_info=True,
            )
            self._log_chat_message(
                self.ROLE_SYSTEM,
                f"Error creating LLM client: {e.__class__.__name__} - {str(e)[:50]}...",
            )
            self.llm_client_instance = None
            return False

    async def _send_user_message(self) -> None:
        try:
            message_input = self.query_one("#chat_message_input", Input)
        except NoMatches:
            logger.error("ChatView: #chat_message_input not found to send message.")
            return

        if self.llm_client_instance is None:
            self._log_chat_message(
                self.ROLE_SYSTEM,
                "LLM not configured. Please select provider/model and reset/start session.",
            )
            return
        user_message = message_input.value.strip()
        if not user_message:
            return

        try:
            send_button = self.query_one("#send_chat_message_button", Button)
            stop_button = self.query_one("#stop_chat_message_button", Button)
        except NoMatches:
            logger.error("ChatView: Send or Stop button not found.")
            return

        self._log_chat_message(self.ROLE_USER, user_message)
        history_for_llm = self.chat_history.copy()
        self.chat_history.append({"role": "user", "content": user_message})

        message_input.value = ""
        message_input.disabled = True
        send_button.disabled = True
        stop_button.disabled = False
        self._update_chat_status_line(status="Sending...", current_messages=len(self.chat_history))

        active_client = self.llm_client_instance

        def task_to_run_in_worker():
            logger.debug(
                f"ChatView: Worker starting send_message to model {active_client.model_name}"
            )
            return active_client.send_message(message=user_message, history=history_for_llm)

        if self.current_llm_worker and self.current_llm_worker.state == WorkerState.RUNNING:
            logger.info("ChatView: Cancelling previous running LLM worker.")
            self.current_llm_worker.cancel()

        self.current_llm_worker = self.app.run_worker(
            task_to_run_in_worker, thread=True, group="llm_call"
        )

        try:
            response_text = await self.current_llm_worker.wait()
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.SUCCESS:
                self._log_chat_message(self.ROLE_AI, response_text)
                self.chat_history.append({"role": "assistant", "content": response_text})
        except WorkerFailed as e:
            error_to_log = e.error
            logger.error(f"ChatView: LLM WorkerFailed: {error_to_log}", exc_info=error_to_log)
            if (
                isinstance(error_to_log, asyncio.CancelledError)
                and self.current_llm_worker
                and self.current_llm_worker.is_cancelled
            ):
                msg = "LLM call stopped by user."
            else:
                msg = f"LLM Error: {error_to_log.__class__.__name__} - {str(error_to_log)[:50]}..."
            self._log_chat_message(self.ROLE_SYSTEM, msg)
            if (
                self.chat_history
                and self.chat_history[-1]["role"] == "user"
                and self.chat_history[-1]["content"] == user_message
            ):
                self.chat_history.pop()
        except Exception as e:
            logger.error(f"ChatView: Unexpected error during LLM communication: {e}", exc_info=True)
            self._log_chat_message(self.ROLE_SYSTEM, f"Unexpected error: {e.__class__.__name__}")
            if (
                self.chat_history
                and self.chat_history[-1]["role"] == "user"
                and self.chat_history[-1]["content"] == user_message
            ):
                self.chat_history.pop()
        finally:
            self.current_llm_worker = None
            if self.is_mounted:
                try:
                    message_input.disabled = False
                    send_button.disabled = False
                    stop_button.disabled = True
                    self._update_chat_status_line(status="Ready")
                    message_input.focus()
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

    @on(LLMConfigPanel.NewSessionRequested)
    async def handle_new_session_requested_from_llm_config_panel(
        self, event: LLMConfigPanel.NewSessionRequested
    ) -> None:
        logger.info("ChatView: handle_new_session_requested_from_llm_config_panel CALLED.")
        try:
            llm_config_panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            logger.error("ChatView: LLMConfigPanel not found for new session request.")
            return

        if (
            llm_config_panel.provider is None
            or llm_config_panel.model is None
            or llm_config_panel.model == Select.BLANK
        ):
            self._reset_chat_log_and_status(
                status_message="Select provider and model first for new session."
            )
            self._update_chat_status_line(status="Needs config")
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
