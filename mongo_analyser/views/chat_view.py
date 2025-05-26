import asyncio
import functools
import logging
from typing import Any, Dict

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Log, Select, Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

from mongo_analyser.llm_chat import LiteLLMChat, LLMChat
from mongo_analyser.widgets import LLMConfigPanel  # Updated imports

logger = logging.getLogger(__name__)


class ChatView(Container):
    current_llm_worker: Worker | None = None
    llm_client_instance: LLMChat | None = None

    USER_PREFIX_Styled = Text.from_markup("[b #ECEFF4]USER:[/] ")
    AI_PREFIX_Styled = Text.from_markup("[b #ECEFF4]AI:[/] ")
    SYSTEM_PREFIX_Styled = Text.from_markup("[b #EBCB8B]SYSTEM:[/] ")

    def _log_chat_message(self, prefix_styled: Text, message_content: str):
        # This would ideally use ChatMessageList and ChatMessageWidget
        try:
            log_widget = self.query_one("#chat_log_widget", Log)  # Keep for now, or replace
            lines_in_message = message_content.splitlines()
            if not lines_in_message and len(message_content) >= 0:
                lines_in_message = [message_content]

            for line_part in lines_in_message:
                plain_prefix = prefix_styled.plain
                plain_line_part = Text(line_part).plain
                log_widget.write_line(f"{plain_prefix}{plain_line_part}")
        except NoMatches:
            logger.warning("Chat log widget not found for logging message.")

    def on_mount(self) -> None:
        self.chat_history: list[dict[str, str]] = []
        self.llm_client_instance = None
        self._reset_chat_log_and_status("ChatView initialized. Configure LLM in sidebar.")

    def _reset_chat_log_and_status(self, status_message: str = "New session started.") -> None:
        try:
            log_widget = self.query_one("#chat_log_widget", Log)  # Or ChatMessageList
            log_widget.clear()
        except NoMatches:
            logger.warning("Chat log widget not found for clearing.")
        self._log_chat_message(self.SYSTEM_PREFIX_Styled, status_message)
        self._update_chat_status_line(status="Idle", current_messages=0)

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat_interface_horizontal_layout"):
            with Vertical(id="chat_main_area", classes="chat_column_main"):
                yield Static(
                    "Provider: N/A | Model: N/A | Context: 0/0 | Status: Idle",
                    id="chat_status_line",
                    classes="chat_status",
                )
                # Replace Log with ChatMessageList for richer display
                # yield ChatMessageList(id="chat_message_list_widget")
                with VerticalScroll(
                    id="chat_log_scroll", classes="chat_log_container"
                ):  # Temporary
                    yield Log(id="chat_log_widget", auto_scroll=True, highlight=True)  # Temporary

                with Horizontal(id="chat_input_bar", classes="chat_input_container"):
                    # Replace Input with ChatInput
                    # yield ChatInput(id="chat_user_input", placeholder="Type a message...")
                    yield Input(  # Temporary
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
            # Use the new LLMConfigPanel widget
            yield LLMConfigPanel(id="chat_llm_config_panel", classes="chat_column_sidebar")

    def focus_default_widget(self) -> None:
        try:
            self.query_one("#chat_message_input", Input).focus()
        except NoMatches:
            logger.debug("ChatView: Could not focus default input.")

    def _update_chat_status_line(
        self, status: str = "Idle", current_messages: int | None = None
    ) -> None:
        try:
            llm_config_panel = self.query_one(LLMConfigPanel)
            chat_status_widget = self.query_one("#chat_status_line", Static)

            provider_val = llm_config_panel.provider
            model_val = llm_config_panel.model
            max_ctx_val = llm_config_panel.max_context_size

            provider_display = str(provider_val).capitalize() if provider_val else "N/A"
            model_display = str(model_val) if model_val else "N/A"

            if self.llm_client_instance:
                provider_display = str(
                    getattr(self.llm_client_instance, "provider_hint", provider_display)
                ).capitalize()
                model_display = getattr(self.llm_client_instance, "raw_model_name", model_display)

            current_msg_count = (
                current_messages if current_messages is not None else len(self.chat_history) // 2
            )
            context_display = f"{current_msg_count} prompts / {max_ctx_val} tokens"

            chat_status_widget.update(
                f"Provider: {provider_display} | Model: {model_display} | Context: {context_display} | Status: {status}"
            )
        except NoMatches:
            logger.warning("Could not update chat status line, a widget was not found.")
        except Exception as e:
            logger.error(f"Error updating chat status line: {e}", exc_info=True)

    async def _load_models_for_provider(self, provider_value: str | object) -> None:
        llm_config_panel = self.query_one(LLMConfigPanel)
        if not llm_config_panel:
            return

        if provider_value == Select.BLANK or provider_value is None:
            self._log_chat_message(self.SYSTEM_PREFIX_Styled, "Provider cleared.")
            llm_config_panel.update_models_list([], "Select Model")
            self._update_chat_status_line(status="Provider cleared")
            return

        provider_value_str = str(provider_value)
        llm_config_panel.set_model_select_loading(
            True, f"Loading models for {provider_value_str}..."
        )
        self._update_chat_status_line(status=f"Loading {provider_value_str} models...")
        self._log_chat_message(
            self.SYSTEM_PREFIX_Styled, f"Fetching models for {provider_value_str}..."
        )

        listed_models: list[str] = []
        error_message: str | None = None
        client_config_for_list = llm_config_panel.get_api_config()

        try:
            callable_with_args = functools.partial(
                LiteLLMChat.list_models,
                provider=provider_value_str,
                client_config=client_config_for_list,
            )
            worker = self.app.run_worker(callable_with_args, thread=True, group="model_listing")
            listed_models = await worker.wait()
        except WorkerCancelled:
            logger.warning(f"Model listing worker for '{provider_value_str}' was cancelled.")
            error_message = "Model loading cancelled."
        except Exception as e:
            logger.error(f"Error listing models for '{provider_value_str}': {e}", exc_info=True)
            error_text = str(e)[:37] + "..." if len(str(e)) > 40 else str(e)
            error_message = f"Failed to list models: {error_text}"

        llm_config_panel.set_model_select_loading(False)
        new_status_for_line = "Ready"
        default_model_to_set = None

        if error_message:
            llm_config_panel.update_models_list([], error_message)
            new_status_for_line = "Model list error"
            self._log_chat_message(self.SYSTEM_PREFIX_Styled, error_message)
        elif listed_models:
            model_options = [(name, name) for name in listed_models]
            llm_config_panel.update_models_list(model_options, "Select Model")  # Pass options
            self._log_chat_message(
                self.SYSTEM_PREFIX_Styled,
                f"{len(listed_models)} models found for {provider_value_str}.",
            )
            # Default model selection logic (remains the same)
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
                llm_config_panel.model = default_model_to_set  # Set model on panel
            elif provider_value_str == "ollama":
                new_status_for_line = "Select Ollama model"
            else:
                new_status_for_line = "Select model"
        else:
            llm_config_panel.update_models_list([], "No models found.")
            new_status_for_line = "No models"
            self._log_chat_message(
                self.SYSTEM_PREFIX_Styled, f"No models found for {provider_value_str}."
            )

        if not default_model_to_set:  # Update status if no default was set
            self._update_chat_status_line(status=new_status_for_line)

    async def on_llm_config_panel_provider_changed(
        self, event: LLMConfigPanel.ProviderChanged
    ) -> None:
        # Event from LLMConfigPanel when provider select changes
        if self.llm_client_instance:
            self.llm_client_instance = None
            self._log_chat_message(
                self.SYSTEM_PREFIX_Styled, "Provider changed. LLM client cleared."
            )
        await self._load_models_for_provider(event.provider)

    async def on_llm_config_panel_model_changed(self, event: LLMConfigPanel.ModelChanged) -> None:
        # Event from LLMConfigPanel when model select changes
        model_value = event.model
        if model_value:
            self._reset_chat_log_and_status(
                status_message=f"Setting up model: {model_value}. Session reset."
            )
            self._update_chat_status_line(status=f"Initializing {model_value}...")
            self.query_one("#chat_message_input", Input).value = ""  # Or ChatInput

            client_created = self._create_and_set_llm_client()
            if client_created:
                self._log_chat_message(
                    self.SYSTEM_PREFIX_Styled, "Session ready. LLM client configured."
                )
                self._update_chat_status_line(status="Ready")
            else:
                self._log_chat_message(self.SYSTEM_PREFIX_Styled, "Failed to configure LLM client.")
                self._update_chat_status_line(status="Client Error")
            self.query_one("#chat_message_input", Input).focus()  # Or ChatInput
        else:
            if self.llm_client_instance:
                self.llm_client_instance = None
                self._log_chat_message(
                    self.SYSTEM_PREFIX_Styled, "Model deselected. LLM client cleared."
                )
            self._update_chat_status_line(status="Select model")

    def _create_and_set_llm_client(self) -> bool:
        try:
            llm_config_panel = self.query_one(LLMConfigPanel)
        except NoMatches:
            self._log_chat_message(self.SYSTEM_PREFIX_Styled, "LLM Configuration panel not found.")
            return False

        provider_hint = llm_config_panel.provider
        raw_model_name = llm_config_panel.model
        temperature = llm_config_panel.temperature
        max_c_val = llm_config_panel.max_context_size
        api_key_val = llm_config_panel.api_key
        base_url_val = llm_config_panel.base_url

        def _sys_msg_plain(message: str):
            self._log_chat_message(self.SYSTEM_PREFIX_Styled, message)

        if not provider_hint:
            _sys_msg_plain("Please select a provider in the config panel.")
            return False
        if not raw_model_name:
            _sys_msg_plain("Please select a model in the config panel.")
            return False
        if temperature is None:  # Assuming panel returns None if invalid
            _sys_msg_plain("Temperature must be a valid number.")
            return False
        if max_c_val is None or not (max_c_val > 0):  # Assuming panel returns None or validates
            _sys_msg_plain("Max Context Size must be a positive valid integer.")
            return False

        client_constructor_kwargs: Dict[str, Any] = {
            "provider_hint": provider_hint,
            "temperature": temperature,
            "max_tokens": max_c_val,
        }
        if api_key_val:
            client_constructor_kwargs["api_key"] = api_key_val
        if base_url_val:
            client_constructor_kwargs["base_url"] = base_url_val

        # Ensure no None values are passed if LiteLLMChat expects them to be absent
        client_constructor_kwargs = {
            k: v for k, v in client_constructor_kwargs.items() if v is not None
        }

        new_client: LLMChat | None = None
        try:
            new_client = LiteLLMChat(model_name=raw_model_name, **client_constructor_kwargs)
            self.llm_client_instance = new_client
            logger.info(
                f"LiteLLM client configured for actual model '{self.llm_client_instance.model_name}' (Provider hint: {provider_hint}, Raw: {raw_model_name})"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to create LiteLLM client for provider {provider_hint}, model {raw_model_name}: {e}",
                exc_info=True,
            )
            _sys_msg_plain(f"Error creating LLM client: {e.__class__.__name__} - {str(e)[:50]}...")
            self.llm_client_instance = None
            return False

    async def _send_user_message(self) -> None:
        # This method remains largely the same, just query the new ChatInput if used
        message_input = self.query_one("#chat_message_input", Input)  # Or ChatInput
        if self.llm_client_instance is None:
            self._log_chat_message(
                self.SYSTEM_PREFIX_Styled,
                "LLM not configured. Please select provider/model and reset session.",
            )
            return
        user_message = message_input.value.strip()
        if not user_message:
            return

        send_button = self.query_one("#send_chat_message_button", Button)
        stop_button = self.query_one("#stop_chat_message_button", Button)

        self._log_chat_message(self.USER_PREFIX_Styled, user_message)
        history_for_llm = self.chat_history.copy()
        self.chat_history.append({"role": "user", "content": user_message})

        message_input.value = ""
        message_input.disabled = True
        send_button.disabled = True
        stop_button.disabled = False
        self._update_chat_status_line(
            status="Sending...", current_messages=len(self.chat_history) // 2
        )
        self._log_chat_message(self.AI_PREFIX_Styled, "is thinking...")
        active_client = self.llm_client_instance

        def task_to_run_in_worker():
            return active_client.send_message(message=user_message, history=history_for_llm)

        self.current_llm_worker = self.app.run_worker(
            task_to_run_in_worker, thread=True, group="llm_call"
        )

        try:
            response_text = await self.current_llm_worker.wait()
            if self.current_llm_worker and self.current_llm_worker.state == WorkerState.SUCCESS:
                self._log_chat_message(self.AI_PREFIX_Styled, response_text)
                self.chat_history.append({"role": "assistant", "content": response_text})
        except WorkerFailed as e:  # Error handling remains similar
            error_to_log = e.error
            msg = (
                "LLM call cancelled."
                if isinstance(error_to_log, asyncio.CancelledError)
                else f"LLM Error: {error_to_log.__class__.__name__} - {str(error_to_log)[:50]}..."
            )
            logger.error(f"LLM WorkerFailed: {error_to_log}", exc_info=error_to_log)
            self._log_chat_message(self.SYSTEM_PREFIX_Styled, msg)
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()  # Rollback user message on failure
        except Exception as e:
            logger.error(f"LLM communication error: {e}", exc_info=True)
            self._log_chat_message(
                self.SYSTEM_PREFIX_Styled, f"Unexpected error: {e.__class__.__name__}"
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
                self._log_chat_message(self.SYSTEM_PREFIX_Styled, "Stopping LLM response...")
            else:
                self._log_chat_message(self.SYSTEM_PREFIX_Styled, "No active LLM call to stop.")
        # Button "new_update_chat_session_button" is now inside LLMConfigPanel
        # So, ChatView needs to listen to an event from LLMConfigPanel instead.

    async def on_llm_config_panel_new_session_requested(
        self, event: LLMConfigPanel.NewSessionRequested
    ) -> None:
        # This event is posted by LLMConfigPanel when its "New/Reset Session" button is pressed.
        llm_config_panel = self.query_one(LLMConfigPanel)
        if llm_config_panel.provider is None or llm_config_panel.model is None:
            self._reset_chat_log_and_status(status_message="Select provider and model first.")
            self._update_chat_status_line(status="Needs config")
            return

        self._reset_chat_log_and_status(status_message="New session. LLM (re)configuring...")
        self.query_one("#chat_message_input", Input).value = ""  # Or ChatInput

        client_created = self._create_and_set_llm_client()
        if client_created:
            self._log_chat_message(self.SYSTEM_PREFIX_Styled, "LLM client (re)configured.")
            self._update_chat_status_line(status="Ready")
        else:
            self._log_chat_message(self.SYSTEM_PREFIX_Styled, "Failed to (re)configure LLM client.")
            self._update_chat_status_line(status="Client Error")
        self.query_one("#chat_message_input", Input).focus()  # Or ChatInput

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        # Or ChatInput.Submitted
        if event.input.id == "chat_message_input":  # Or ChatInput's ID
            await self._send_user_message()
