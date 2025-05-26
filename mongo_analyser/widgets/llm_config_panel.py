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
        /* Styles for this panel, e.g., border, background, padding */
        border: round $primary-darken-1; /* Example */
        background: $primary-background-darken-3; /* Example */
        padding: 0 1;
    }
    LLMConfigPanel > Label { margin-top: 1; }
    LLMConfigPanel > Input, LLMConfigPanel > Select { width: 100%; }
    LLMConfigPanel > Button { width: 100%; margin-top: 2; }
    """

    # --- Custom Events ---
    class ProviderChanged(Message):
        def __init__(self, provider: Optional[str]):
            self.provider = provider
            super().__init__()

    class ModelChanged(Message):
        def __init__(self, model: Optional[str]):
            self.model = model
            super().__init__()

    class NewSessionRequested(Message):
        pass  # No data needed, just a signal

    # --- Reactive properties for easy access by parent view ---
    provider: reactive[Optional[str]] = reactive(None)
    model: reactive[Optional[str]] = reactive(None)
    temperature: reactive[Optional[float]] = reactive(0.7)
    max_context_size: reactive[Optional[int]] = reactive(2048)
    api_key: reactive[Optional[str]] = reactive(None)
    base_url: reactive[Optional[str]] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Label("Session Config", classes="panel_title")  # Use your app's class
        yield Label("Name:")
        yield Input(placeholder="New Chat X", id="llm_config_session_name")  # Prefixed IDs

        yield Label("Provider:")
        providers = [("Ollama", "ollama"), ("OpenAI", "openai"), ("Google", "google")]
        yield Select(
            providers,
            prompt="Select Provider",
            id="llm_config_provider_select",
            allow_blank=False,
            value="ollama",
        )

        yield Label("Model:")
        yield Select(
            [],
            prompt="Select Model",
            id="llm_config_model_select",
            allow_blank=True,
            value=Select.BLANK,
        )

        yield Label("Temperature:")
        yield Input(placeholder="0.7", id="llm_config_temperature", value="0.7")
        yield Label("Max Context Size / Output Tokens:")
        yield Input(placeholder="e.g., 2048", id="llm_config_max_context_size", value="2048")

        yield Label("API Key (Optional, uses env if blank):")
        yield Input(placeholder="sk-...", id="llm_config_api_key", password=True)
        yield Label("Base URL (Optional, for custom endpoints):")
        yield Input(placeholder="http://localhost:11434", id="llm_config_base_url")

        yield Button(
            "New/Reset Session",
            id="llm_config_new_session_button",
            classes="config_button",  # Use your app's class
        )

    def on_mount(self) -> None:
        # Initialize reactive properties from widget values
        self.provider = self.query_one("#llm_config_provider_select", Select).value
        self.model = self.query_one("#llm_config_model_select", Select).value
        self._update_temperature()
        self._update_max_context_size()
        self._update_api_key()
        self._update_base_url()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_config_provider_select":
            self.provider = str(event.value) if event.value != Select.BLANK else None
            self.model = None  # Reset model when provider changes
            try:
                model_select = self.query_one("#llm_config_model_select", Select)
                model_select.value = Select.BLANK
                model_select.set_options([])  # Clear old models
            except NoMatches:
                pass
            self.post_message(self.ProviderChanged(self.provider))
        elif event.select.id == "llm_config_model_select":
            self.model = str(event.value) if event.value != Select.BLANK else None
            self.post_message(self.ModelChanged(self.model))

    async def on_input_changed(self, event: Input.Changed) -> None:
        # Using changed instead of submitted for more immediate reactive updates
        if event.input.id == "llm_config_temperature":
            self._update_temperature()
        elif event.input.id == "llm_config_max_context_size":
            self._update_max_context_size()
        elif event.input.id == "llm_config_api_key":
            self._update_api_key()
        elif event.input.id == "llm_config_base_url":
            self._update_base_url()

    def _update_temperature(self) -> None:
        try:
            val_str = self.query_one("#llm_config_temperature", Input).value
            self.temperature = float(val_str)
        except (ValueError, NoMatches):
            self.temperature = None  # Or keep old value, or default

    def _update_max_context_size(self) -> None:
        try:
            val_str = self.query_one("#llm_config_max_context_size", Input).value
            val_int = int(val_str)
            self.max_context_size = val_int if val_int > 0 else None
        except (ValueError, NoMatches):
            self.max_context_size = None

    def _update_api_key(self) -> None:
        try:
            self.api_key = self.query_one("#llm_config_api_key", Input).value or None
        except NoMatches:
            pass

    def _update_base_url(self) -> None:
        try:
            self.base_url = self.query_one("#llm_config_base_url", Input).value or None
        except NoMatches:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm_config_new_session_button":
            self.post_message(self.NewSessionRequested())

    def update_models_list(self, models: List[Tuple[str, str]], prompt_text: str) -> None:
        try:
            model_select = self.query_one("#llm_config_model_select", Select)
            model_select.set_options(models)
            model_select.prompt = prompt_text
            if not models:  # If no models, ensure value is blank
                model_select.value = Select.BLANK
                self.model = None
        except NoMatches:
            logger.error("LLMConfigPanel: Could not find model_select to update.")

    def set_model_select_loading(self, loading: bool, loading_text: str = "Loading...") -> None:
        try:
            model_select = self.query_one("#llm_config_model_select", Select)
            model_select.disabled = loading
            if loading:
                model_select.prompt = loading_text
            else:
                # Prompt will be reset by update_models_list or if it was an error
                pass
        except NoMatches:
            logger.error("LLMConfigPanel: Could not find model_select to set loading state.")

    def get_api_config(self) -> Dict[str, Any]:
        """Returns a dict with api_key and base_url if they are set."""
        config = {}
        if self.api_key:
            config["api_key"] = self.api_key
        if self.base_url:
            config["base_url"] = self.base_url
        return config
