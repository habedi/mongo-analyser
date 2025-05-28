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
        # start with Ollama selected, immediately fire off a ProviderChanged
        self.provider = None
        select = self.query_one("#llm_config_provider_select", Select)
        # programmatically set to trigger our on_select_changed
        select.value = "ollama"

        # disable model list until loaded
        ms = self.query_one("#llm_config_model_select", Select)
        ms.disabled = True
        ms.value = Select.BLANK

        # init temperature & history
        self._update_temperature()
        self._update_max_history()

        logger.info("LLMConfigPanel: on_mount complete; loading default provider models.")

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_config_provider_select":
            newp = str(event.value)
            if self.provider != newp:
                self.provider = newp
                # clear & disable model list while loading
                try:
                    ms = self.query_one("#llm_config_model_select", Select)
                    ms.set_options([])
                    ms.disabled = True
                    ms.prompt = "Loading models…"
                except NoMatches:
                    pass
                self.post_message(self.ProviderChanged(self.provider))

        elif event.select.id == "llm_config_model_select":
            newm = str(event.value) if event.value != Select.BLANK else None
            if self.model != newm:
                self.model = newm
                self.post_message(self.ModelChanged(self.model))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "llm_config_temperature":
            self._update_temperature()
        elif event.input.id == "llm_config_max_history":
            self._update_max_history()

    def _update_temperature(self) -> None:
        try:
            self.temperature = float(self.query_one("#llm_config_temperature", Input).value)
        except Exception:
            self.temperature = 0.7

    def _update_max_history(self) -> None:
        try:
            self.max_history_messages = int(self.query_one("#llm_config_max_history", Input).value)
        except Exception:
            self.max_history_messages = 20

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm_config_new_session_button":
            self.post_message(self.NewSessionRequested())

    def update_models_list(self, models: List[Tuple[str, str]], prompt_text: str) -> None:
        if not self.is_mounted:
            return
        try:
            sel = self.query_one("#llm_config_model_select", Select)
            current = sel.value
            sel.set_options(models)
            if models:
                sel.disabled = False
                sel.prompt = "Select Model"
                sel.value = current if current in [m for _, m in models] else Select.BLANK
            else:
                sel.disabled = True
                sel.prompt = prompt_text
                sel.value = Select.BLANK
                self.model = None
        except NoMatches:
            logger.error("LLMConfigPanel: model select not found in update_models_list")

    def set_model_select_loading(
        self, loading: bool, loading_text: str = "Loading models..."
    ) -> None:
        try:
            sel = self.query_one("#llm_config_model_select", Select)
            sel.disabled = loading
            sel.prompt = (
                loading_text if loading else ("Select Model" if sel._options else "No models")
            )
        except NoMatches:
            pass

    def watch_model(self, old: Optional[str], new: Optional[str]) -> None:
        if not self.is_mounted:
            return
        try:
            sel = self.query_one("#llm_config_model_select", Select)
            sel.value = new or Select.BLANK
        except NoMatches:
            pass

    def get_llm_config(self) -> Dict[str, Any]:
        cfg = {
            "provider_hint": self.provider,
            "model_name": self.model,
            "temperature": self.temperature,
            "max_history_messages": self.max_history_messages,
        }
        return {k: v for k, v in cfg.items() if v is not None}
