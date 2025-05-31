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
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_HISTORY = 20

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
    temperature: reactive[Optional[float]] = reactive(DEFAULT_TEMPERATURE)
    max_history_messages: reactive[Optional[int]] = reactive(DEFAULT_MAX_HISTORY)

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

        yield Label(f"Temperature (default: {self.DEFAULT_TEMPERATURE}):")
        yield Input(
            placeholder=str(self.DEFAULT_TEMPERATURE),
            id="llm_config_temperature",
            value=str(self.DEFAULT_TEMPERATURE),
            tooltip="Controls randomness (0.0-1.0). Higher values (e.g., 0.9) for more creative and"
            " diverse responses, lower (e.g., 0.2) for more deterministic/focused ones."
            f" Default: {self.DEFAULT_TEMPERATURE}",
        )

        yield Label(f"Max History (0=all, -1=none, default: {self.DEFAULT_MAX_HISTORY}):")
        yield Input(
            placeholder=str(self.DEFAULT_MAX_HISTORY),
            id="llm_config_max_history",
            value=str(self.DEFAULT_MAX_HISTORY),
            tooltip="Number of recent conversation turns (user+AI message pairs) to include as context."
            " '0' includes all available history. '-1' includes no history (current message only)."
            f" Default: {self.DEFAULT_MAX_HISTORY}",
        )

        yield Button("New Chat Session", id="llm_config_new_session_button", variant="primary")

    def on_mount(self) -> None:
        self.provider = None
        select = self.query_one("#llm_config_provider_select", Select)
        select.value = "ollama"
        ms = self.query_one("#llm_config_model_select", Select)
        ms.disabled = True
        ms.value = Select.BLANK
        self._update_temperature()
        self._update_max_history()
        logger.info("LLMConfigPanel: on_mount complete; loading default provider models.")

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_config_provider_select":
            new_provider_value = str(event.value)
            if self.provider != new_provider_value:
                self.provider = new_provider_value
                try:
                    model_select = self.query_one("#llm_config_model_select", Select)
                    model_select.set_options([])
                    model_select.disabled = True
                    model_select.prompt = "Loading models…"
                    model_select.value = Select.BLANK
                    self.model = None
                except NoMatches:
                    logger.warning("LLMConfigPanel: Model select not found during provider change.")
                self.post_message(self.ProviderChanged(self.provider))
        elif event.select.id == "llm_config_model_select":
            new_model_value = str(event.value) if event.value != Select.BLANK else None
            if self.model != new_model_value:
                self.model = new_model_value
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
            self.temperature = self.DEFAULT_TEMPERATURE
        except NoMatches:
            logger.error("LLMConfigPanel: Temperature input not found.")
            self.temperature = self.DEFAULT_TEMPERATURE

    def _update_max_history(self) -> None:
        try:
            history_input = self.query_one("#llm_config_max_history", Input)
            self.max_history_messages = int(history_input.value)
        except ValueError:
            self.max_history_messages = self.DEFAULT_MAX_HISTORY
        except NoMatches:
            logger.error("LLMConfigPanel: Max history input not found.")
            self.max_history_messages = self.DEFAULT_MAX_HISTORY

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm_config_new_session_button":
            self.post_message(self.NewSessionRequested())

    def update_models_list(self, models: List[Tuple[str, str]], prompt_text_if_empty: str) -> None:
        if not self.is_mounted:
            return
        try:
            sel = self.query_one("#llm_config_model_select", Select)
            current_selected_model = sel.value
            sel.set_options(models)
            if models:
                sel.disabled = False
                sel.prompt = "Select Model"
                if current_selected_model in [m_val for _, m_val in models]:
                    sel.value = current_selected_model
                else:
                    sel.value = Select.BLANK
            else:
                sel.disabled = True
                sel.prompt = prompt_text_if_empty
                sel.value = Select.BLANK
                if self.model is not None:
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
            sel.disabled = loading
            if loading:
                sel.prompt = loading_text
            else:
                if not sel.disabled:
                    sel.prompt = "Select Model"
        except NoMatches:
            logger.warning("LLMConfigPanel: Model select not found in set_model_select_loading.")

    def watch_model(self, new_model: Optional[str]) -> None:
        if not self.is_mounted:
            return
        try:
            sel = self.query_one("#llm_config_model_select", Select)
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
        return {k: v for k, v in cfg.items() if v is not None}
