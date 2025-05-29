# mongo_analyser/widgets/llm_config_panel.py
import logging
from typing import Any, Dict, List, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Select

logger = logging.getLogger(__name__)


class LLMConfigPanel(VerticalScroll):
    DEFAULT_CSS = """
    LLMConfigPanel {
        border: round $primary-darken-1;
        background: $primary-background-darken-3;
        padding: 0 1;
        overflow-y: auto;
    }
    LLMConfigPanel > Label { margin-top: 1; }
    LLMConfigPanel > Input, LLMConfigPanel > Select { width: 100%; }
    LLMConfigPanel > Button { width: 100%; margin-top: 1; }
    """

    class ProviderChanged(Message):
        def __init__(self, provider: Optional[str]):
            self.provider = provider
            super().__init__()

    class ModelChanged(Message):
        def __init__(self, model: Optional[str]):
            self.model = model
            super().__init__()

    class NewSessionRequested(Message):
        pass

    provider: reactive[Optional[str]] = reactive(None)
    model: reactive[Optional[str]] = reactive(None)
    temperature: reactive[Optional[float]] = reactive(0.7)
    max_history_messages: reactive[Optional[int]] = reactive(20)

    def compose(self) -> ComposeResult:
        yield Label("Session Config", classes="panel_title")

        yield Label("Provider:")
        yield Select(
            [("Ollama", "ollama"), ("OpenAI", "openai"), ("Google", "google")],
            prompt="Select Provider",
            id="llm_config_provider_select",
            allow_blank=False,
            value="ollama",
        )

        yield Label("Model:")
        yield Select(
            [],
            prompt="Select Provider First",
            id="llm_config_model_select",
            allow_blank=True,
            value=Select.BLANK,
        )

        yield Label("Temperature (e.g., 0.7):")
        yield Input(placeholder="0.7", id="llm_config_temperature", value="0.7")

        yield Label("Max History (0=all, -1=none):")
        yield Input(placeholder="20", id="llm_config_max_history", value="20")

        yield Button(
            "New Chat Session",
            id="llm_config_new_session_button",
            variant="primary",
        )

    def on_mount(self) -> None:
        self.provider = None  # Initialize to ensure watch_provider fires if initial value is same
        select = self.query_one("#llm_config_provider_select", Select)
        # Programmatically set value to trigger on_select_changed and load initial models
        select.value = "ollama"

        ms = self.query_one("#llm_config_model_select", Select)
        ms.disabled = True
        ms.value = Select.BLANK  # Ensure it's blank initially

        self._update_temperature()
        self._update_max_history()

        logger.info("LLMConfigPanel: on_mount complete; loading default provider models.")

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_config_provider_select":
            new_provider_value = str(event.value)
            # Check if the provider actually changed to avoid redundant operations
            if self.provider != new_provider_value:
                self.provider = new_provider_value  # This will trigger watch_provider
                # Clear & disable model list while loading
                try:
                    model_select = self.query_one("#llm_config_model_select", Select)
                    model_select.set_options([])  # Clear existing options
                    model_select.disabled = True
                    model_select.prompt = "Loading models…"
                    model_select.value = Select.BLANK  # Reset value
                    self.model = None  # Clear reactive model
                except NoMatches:
                    logger.warning("LLMConfigPanel: Model select not found during provider change.")
                # Post message to ChatView to initiate model loading
                self.post_message(self.ProviderChanged(self.provider))

        elif event.select.id == "llm_config_model_select":
            new_model_value = str(event.value) if event.value != Select.BLANK else None
            if self.model != new_model_value:
                self.model = new_model_value  # This will trigger watch_model
                self.post_message(self.ModelChanged(self.model))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "llm_config_temperature":
            self._update_temperature()
        elif event.input.id == "llm_config_max_history":
            self._update_max_history()

    def _update_temperature(self) -> None:
        try:
            temp_input = self.query_one("#llm_config_temperature", Input)
            self.temperature = float(temp_input.value)
        except ValueError:
            self.temperature = 0.7  # Default value on error
            # Optionally, reset input value to default if desired
            # temp_input.value = "0.7"
        except NoMatches:
            logger.error("LLMConfigPanel: Temperature input not found.")
            self.temperature = 0.7

    def _update_max_history(self) -> None:
        try:
            history_input = self.query_one("#llm_config_max_history", Input)
            self.max_history_messages = int(history_input.value)
        except ValueError:
            self.max_history_messages = 20  # Default value on error
            # Optionally, reset input value
            # history_input.value = "20"
        except NoMatches:
            logger.error("LLMConfigPanel: Max history input not found.")
            self.max_history_messages = 20

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm_config_new_session_button":
            self.post_message(self.NewSessionRequested())

    def update_models_list(self, models: List[Tuple[str, str]], prompt_text_if_empty: str) -> None:
        if not self.is_mounted:
            return
        try:
            sel = self.query_one("#llm_config_model_select", Select)
            current_selected_model = sel.value

            sel.set_options(models)  # This clears old options and adds new ones

            if models:
                sel.disabled = False
                sel.prompt = "Select Model"  # Default prompt when models are available
                # Try to re-select the previously selected model if it's in the new list
                if current_selected_model in [m_val for _, m_val in models]:
                    sel.value = current_selected_model
                else:
                    # If previous selection is not in new list, or no previous selection, set to blank
                    # Or, you could default to the first model: sel.value = models[0][1]
                    sel.value = Select.BLANK
            else:
                sel.disabled = True
                sel.prompt = (
                    prompt_text_if_empty  # e.g., "No models available", "Error loading models"
                )
                sel.value = Select.BLANK
                if self.model is not None:  # If a model was reactively set, clear it
                    self.model = None
        except NoMatches:
            logger.error("LLMConfigPanel: model select not found in update_models_list")

    def set_model_select_loading(
        self, loading: bool, loading_text: str = "Loading models..."
    ) -> None:
        if not self.is_mounted:
            return
        try:
            sel = self.query_one("#llm_config_model_select", Select)
            sel.disabled = loading  # Disable while loading, enable after (if models were found)
            if loading:
                sel.prompt = loading_text
            else:
                # This part relies on update_models_list to have set the correct state
                # If sel.disabled is False, it means update_models_list found models.
                # If sel.disabled is True, update_models_list set a specific prompt (e.g., "No models").
                if not sel.disabled:  # Models are available
                    sel.prompt = "Select Model"
                # If sel.disabled is True, its prompt was already set by update_models_list, so we don't change it here.
        except NoMatches:
            logger.warning("LLMConfigPanel: Model select not found in set_model_select_loading.")

    def watch_model(self, old_model: Optional[str], new_model: Optional[str]) -> None:
        # This watcher ensures the Select widget reflects the reactive `model` value.
        # This is useful if `self.model` is changed programmatically elsewhere.
        if not self.is_mounted:
            return
        try:
            sel = self.query_one("#llm_config_model_select", Select)
            # Only update the widget's value if it's different from the reactive value
            # to prevent potential update loops or unnecessary on_select_changed events.
            if sel.value != (new_model or Select.BLANK):
                sel.value = new_model or Select.BLANK
        except NoMatches:
            logger.warning("LLMConfigPanel: Model select not found in watch_model.")

    def get_llm_config(self) -> Dict[str, Any]:
        cfg = {
            "provider_hint": self.provider,
            "model_name": self.model,
            "temperature": self.temperature,
            "max_history_messages": self.max_history_messages,
        }
        # Filter out None values, as some LLM libraries might not like None for optional params
        return {k: v for k, v in cfg.items() if v is not None}
