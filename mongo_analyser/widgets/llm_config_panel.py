import logging
from typing import Any, Dict, List, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Select, TextArea

logger = logging.getLogger(__name__)


class LLMConfigPanel(VerticalScroll):
    DEFAULT_CSS = """
    LLMConfigPanel {
        border: round $primary-darken-1;
        background: $primary-background-darken-3;
        padding: 0 1;
    }
    LLMConfigPanel > Label { margin-top: 1; }
    LLMConfigPanel > Input, LLMConfigPanel > Select { width: 100%; }
    LLMConfigPanel > TextArea { width: 100%; height: 5; margin-bottom: 1; }
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
    system_prompt: reactive[Optional[str]] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Label("Session Config", classes="panel_title")

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
            prompt="Select Provider First",
            id="llm_config_model_select",
            allow_blank=True,
            value=Select.BLANK,
        )

        yield Label("System Prompt (Optional):")
        yield TextArea(id="llm_config_system_prompt", text="Be based.")  # Changed default text

        yield Label("Temperature:")
        yield Input(placeholder="0.7", id="llm_config_temperature", value="0.7")

        yield Button(
            "New Session",
            id="llm_config_new_session_button",
            classes="config_button",
        )

    def on_mount(self) -> None:
        self.provider = self.query_one("#llm_config_provider_select", Select).value
        self.model = None

        model_select = self.query_one("#llm_config_model_select", Select)
        model_select.value = Select.BLANK
        model_select.disabled = True

        self._update_system_prompt()
        self._update_temperature()

        if self.provider:
            self.post_message(self.ProviderChanged(self.provider))

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_config_provider_select":
            logger.info(f"LLMConfigPanel: Provider Select changed. Selected value: {event.value}")
            self.provider = str(event.value) if event.value != Select.BLANK else None
            try:
                model_select = self.query_one("#llm_config_model_select", Select)
                model_select.set_options([])
                model_select.value = Select.BLANK
                model_select.disabled = True
                model_select.prompt = "Loading models..."
            except NoMatches:
                logger.warning(
                    "LLMConfigPanel: Model select widget not found while clearing for provider change."
                )
            logger.info(
                f"LLMConfigPanel: Posting ProviderChanged message with provider: {self.provider}"
            )
            self.post_message(self.ProviderChanged(self.provider))
        elif event.select.id == "llm_config_model_select":
            logger.info(
                f"LLMConfigPanel: Model Select MANUALLY changed. Selected value: {event.value}"
            )
            if event.value != Select.BLANK:
                self.model = str(event.value)
                logger.info(
                    f"LLMConfigPanel: Posting ModelChanged message with model: {self.model}"
                )
                self.post_message(self.ModelChanged(self.model))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "llm_config_temperature":
            self._update_temperature()

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "llm_config_system_prompt":
            self._update_system_prompt()

    def _update_system_prompt(self) -> None:
        try:
            self.system_prompt = self.query_one("#llm_config_system_prompt", TextArea).text or None
        except NoMatches:
            self.system_prompt = None

    def _update_temperature(self) -> None:
        try:
            val_str = self.query_one("#llm_config_temperature", Input).value
            self.temperature = float(val_str)
        except (ValueError, NoMatches):
            self.temperature = 0.7

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm_config_new_session_button":
            logger.info("LLMConfigPanel: New Session button pressed.")
            self.post_message(self.NewSessionRequested())

    def update_models_list(self, models: List[Tuple[str, str]], prompt_text: str) -> None:
        logger.debug(
            f"LLMConfigPanel: update_models_list called with {len(models)} models, prompt: '{prompt_text}'"
        )
        try:
            model_select = self.query_one("#llm_config_model_select", Select)
            model_select.set_options(models)

            if models:
                model_select.disabled = False
                model_select.prompt = "Select Model"
            else:
                model_select.disabled = True
                model_select.prompt = prompt_text
                model_select.value = Select.BLANK
        except NoMatches:
            logger.error("LLMConfigPanel: Could not find model_select to update model list.")

    def watch_model(self, old_model: Optional[str], new_model: Optional[str]) -> None:
        logger.debug(
            f"LLMConfigPanel: watch_model triggered. Old: {old_model}, New model: {new_model}"
        )
        try:
            model_select = self.query_one("#llm_config_model_select", Select)

            if new_model is None or new_model == Select.BLANK:
                if model_select.value != Select.BLANK:
                    model_select.value = Select.BLANK
                    logger.debug(
                        "LLMConfigPanel: watch_model - Select widget value set to BLANK as new_model is None/BLANK."
                    )
                return

            current_options = model_select._options
            has_options = bool(current_options)
            model_is_in_options = any(opt[1] == new_model for opt in current_options)

            if model_is_in_options:
                if model_select.value != new_model:
                    model_select.value = new_model
                    logger.debug(
                        f"LLMConfigPanel: watch_model - Select widget value programmatically set to '{new_model}'."
                    )
            elif has_options:
                logger.warning(
                    f"LLMConfigPanel.watch_model: '{new_model}' is not in current options. Current options: {current_options}. Value remains '{model_select.value}'. Awaiting ChatView to set a valid model from new options if they changed."
                )
            elif not has_options:
                if model_select.value != Select.BLANK:
                    model_select.value = Select.BLANK
                    logger.debug(
                        "LLMConfigPanel: watch_model - Select widget has no options, value ensured to be BLANK."
                    )
        except NoMatches:
            logger.debug(
                "LLMConfigPanel: model_select not found in watch_model (likely during setup or if panel is not fully mounted)."
            )
        except Exception as e:
            logger.error(f"LLMConfigPanel: Unexpected error in watch_model: {e}", exc_info=True)

    def set_model_select_loading(self, loading: bool, loading_text: str = "Loading...") -> None:
        try:
            model_select = self.query_one("#llm_config_model_select", Select)
            model_select.disabled = loading
            if loading:
                model_select.prompt = loading_text
            else:
                if not model_select._options and model_select.value == Select.BLANK:
                    model_select.prompt = "No models available"
                elif model_select.value != Select.BLANK:
                    model_select.prompt = "Select Model"
        except NoMatches:
            logger.error("LLMConfigPanel: Could not find model_select to set loading state.")

    def get_llm_config(self) -> Dict[str, Any]:
        config = {
            "provider_hint": self.provider,
            "model_name": self.model,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
        }
        return {k: v for k, v in config.items() if v is not None}
