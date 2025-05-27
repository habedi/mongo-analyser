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
        overflow-y: auto;
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
    max_history_messages: reactive[Optional[int]] = reactive(20)

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
        yield TextArea(id="llm_config_system_prompt", text="You are a helpful assistant.")

        yield Label("Temperature (e.g., 0.7):")
        yield Input(placeholder="0.7", id="llm_config_temperature", value="0.7")

        yield Label("Max History Messages (0=all, -1=none):")
        yield Input(placeholder="20", id="llm_config_max_history", value="20")

        yield Button(
            "New Chat Session",
            id="llm_config_new_session_button",
            classes="config_button",
            variant="primary",
        )

    def on_mount(self) -> None:
        # The Select widget for provider has a default value ("ollama").
        # Its 'Changed' event will fire shortly after mount, handled by on_select_changed.
        # self.provider will be set there.

        self.model = None  # Model starts as None

        model_select = self.query_one("#llm_config_model_select", Select)
        model_select.value = Select.BLANK
        model_select.disabled = True  # Disabled until provider is chosen and models load

        self._update_system_prompt()
        self._update_temperature()
        self._update_max_history()

        logger.info(
            "LLMConfigPanel.on_mount: Initialization complete. Awaiting Select.Changed for initial provider."
        )

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm_config_provider_select":
            new_provider_value = str(event.value) if event.value != Select.BLANK else None
            logger.info(
                f"LLMConfigPanel: Provider Select UI changed/initialized. Selected value: {new_provider_value}. Current reactive self.provider: {self.provider}"
            )

            # This handler will be called for the Select's initial value and subsequent user changes.
            # Update reactive self.provider and post ProviderChanged only if the value is actually different
            # from the current reactive state.
            if self.provider != new_provider_value:
                self.provider = new_provider_value  # Update reactive property

                try:
                    model_select = self.query_one("#llm_config_model_select", Select)
                    model_select.set_options([])
                    model_select.value = Select.BLANK
                    self.model = None  # Reset model when provider changes
                    model_select.disabled = True
                    model_select.prompt = (
                        "Loading models..." if self.provider else "Select Provider First"
                    )
                except NoMatches:
                    logger.warning(
                        "LLMConfigPanel: Model select widget not found while clearing for provider change."
                    )

                logger.info(
                    f"LLMConfigPanel: Posting ProviderChanged message with new provider: {self.provider}"
                )
                self.post_message(
                    self.ProviderChanged(self.provider)
                )  # Post based on the new self.provider
            else:
                # This case means the event fired for a value that self.provider already holds.
                # This can happen if the widget re-fires for its current value under some circumstances.
                logger.debug(
                    f"LLMConfigPanel: Provider Select event for '{new_provider_value}', but self.provider is already this value. No new ProviderChanged posted from this handler."
                )

        elif event.select.id == "llm_config_model_select":
            logger.info(f"LLMConfigPanel: Model Select UI changed. Selected value: {event.value}")
            new_model_value = str(event.value) if event.value != Select.BLANK else None
            if self.model != new_model_value:
                self.model = new_model_value  # Update reactive property

                logger.info(
                    f"LLMConfigPanel: Posting ModelChanged message with new model: {self.model}"
                )
                self.post_message(self.ModelChanged(self.model))  # Post based on new self.model
            else:
                logger.debug(
                    f"LLMConfigPanel: Model Select event for '{new_model_value}', but self.model is already this value. No new ModelChanged posted."
                )

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "llm_config_temperature":
            self._update_temperature()
        elif event.input.id == "llm_config_max_history":
            self._update_max_history()

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

    def _update_max_history(self) -> None:
        try:
            val_str = self.query_one("#llm_config_max_history", Input).value
            self.max_history_messages = int(val_str)
        except (ValueError, NoMatches):
            self.max_history_messages = 20

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "llm_config_new_session_button":
            logger.info("LLMConfigPanel: New Session button pressed.")
            self.post_message(self.NewSessionRequested())

    def update_models_list(self, models: List[Tuple[str, str]], prompt_text: str) -> None:
        if not self.is_mounted:
            logger.warning(
                "LLMConfigPanel.update_models_list called when panel not mounted. Aborting."
            )
            return
        logger.debug(
            f"LLMConfigPanel: update_models_list called with {len(models)} models, prompt: '{prompt_text}'"
        )
        try:
            model_select = self.query_one("#llm_config_model_select", Select)
            current_selected_value_in_widget = model_select.value

            model_select.set_options(models)

            if models:
                model_select.disabled = False
                model_select.prompt = "Select Model"
                if current_selected_value_in_widget != Select.BLANK and any(
                    opt[1] == current_selected_value_in_widget for opt in models
                ):
                    if model_select.value != current_selected_value_in_widget:
                        model_select.value = current_selected_value_in_widget
                elif model_select.value != Select.BLANK:
                    model_select.value = Select.BLANK
                    if self.model is not None:
                        logger.debug(
                            f"LLMConfigPanel.update_models_list: Clearing reactive self.model as widget value '{current_selected_value_in_widget}' is no longer valid."
                        )
            else:
                model_select.disabled = True
                model_select.prompt = prompt_text
                if model_select.value != Select.BLANK:
                    model_select.value = Select.BLANK
                if self.model is not None:
                    self.model = None
        except NoMatches:
            logger.error(
                "LLMConfigPanel: Critical - Could not find #llm_config_model_select in update_models_list."
            )

    def watch_model(self, old_model: Optional[str], new_model: Optional[str]) -> None:
        logger.debug(
            f"LLMConfigPanel: watch_model triggered. Old: {old_model}, New model: {new_model}"
        )
        if not self.is_mounted:
            return

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
            model_is_in_options = any(opt[1] == new_model for opt in current_options)

            if model_is_in_options:
                if model_select.value != new_model:
                    model_select.value = new_model
                    logger.debug(
                        f"LLMConfigPanel: watch_model - Select widget value programmatically set to '{new_model}'."
                    )
            elif current_options:
                logger.warning(
                    f"LLMConfigPanel.watch_model: Reactive model '{new_model}' is not in current UI options. "
                    f"UI options: {[opt[1] for opt in current_options]}. UI value remains '{model_select.value}'. "
                )
            elif not current_options:
                if model_select.value != Select.BLANK:
                    model_select.value = Select.BLANK
                    logger.debug(
                        "LLMConfigPanel: watch_model - Select widget has no options, value ensured to be BLANK."
                    )

        except NoMatches:
            logger.debug(
                "LLMConfigPanel: model_select not found in watch_model (likely during setup)."
            )
        except Exception as e:
            logger.error(f"LLMConfigPanel: Unexpected error in watch_model: {e}", exc_info=True)

    def set_model_select_loading(
        self, loading: bool, loading_text: str = "Loading models..."
    ) -> None:
        if not self.is_mounted:
            logger.warning(
                "LLMConfigPanel.set_model_select_loading called when panel not mounted. Aborting."
            )
            return
        try:
            model_select = self.query_one("#llm_config_model_select", Select)
            model_select.disabled = loading
            if loading:
                model_select.prompt = loading_text
            else:
                if not model_select._options and model_select.value == Select.BLANK:
                    model_select.prompt = "No models available"
                elif model_select.value == Select.BLANK and model_select._options:
                    model_select.prompt = "Select Model"
        except NoMatches:
            logger.error(
                "LLMConfigPanel: Critical - Could not find #llm_config_model_select in set_model_select_loading."
            )

    def get_llm_config(self) -> Dict[str, Any]:
        config = {
            "provider_hint": self.provider,
            "model_name": self.model,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "max_history_messages": self.max_history_messages,
        }
        return {k: v for k, v in config.items() if v is not None}
