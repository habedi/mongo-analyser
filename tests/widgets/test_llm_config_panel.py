import pytest
from textual.css.query import NoMatches
from textual.widgets import Select

from mongo_analyser.widgets.llm_config_panel import LLMConfigPanel


class DummySelect:
    def __init__(self):
        self.options_set = None
        self.disabled = False
        self.prompt = ""
        self.value = None

    def set_options(self, options):
        self.options_set = options


class DummyInput:
    def __init__(self, initial):
        self.value = initial


@pytest.fixture
def panel(monkeypatch):
    p = LLMConfigPanel()

    monkeypatch.setattr(type(p), "is_mounted", True)
    return p


def test_on_mount_initial_state(monkeypatch, panel):
    dummy_provider_select = DummySelect()
    dummy_model_select = DummySelect()
    dummy_temp_input = DummyInput("0.42")
    dummy_hist_input = DummyInput("7")

    def fake_query_one(selector, widget_type=None):
        if selector == "#llm_config_provider_select":
            return dummy_provider_select
        if selector == "#llm_config_model_select":
            return dummy_model_select
        if selector == "#llm_config_temperature":
            return dummy_temp_input
        if selector == "#llm_config_max_history":
            return dummy_hist_input
        raise NoMatches()

    monkeypatch.setattr(panel, "query_one", fake_query_one)

    assert panel.provider is None
    assert panel.model is None
    assert panel.temperature == panel.DEFAULT_TEMPERATURE
    assert panel.max_history_messages == panel.DEFAULT_MAX_HISTORY

    panel.on_mount()

    assert dummy_provider_select.value == "ollama"

    assert dummy_model_select.disabled is True

    assert panel.temperature == 0.42
    assert panel.max_history_messages == 7


@pytest.mark.parametrize(
    "input_val, expected_temp, expected_hist",
    [
        ("0.7", 0.7, LLMConfigPanel.DEFAULT_MAX_HISTORY),
        ("not_a_number", LLMConfigPanel.DEFAULT_TEMPERATURE, LLMConfigPanel.DEFAULT_MAX_HISTORY),
        ("5", 5.0, 5),
    ],
)
def test_update_temperature_and_history(monkeypatch, panel, input_val, expected_temp,
                                        expected_hist):
    dummy_input_t = DummyInput(input_val)

    def fake_query_temp(selector, t=None):
        if selector == "#llm_config_temperature":
            return dummy_input_t
        raise NoMatches()

    monkeypatch.setattr(panel, "query_one", fake_query_temp)

    panel._update_temperature()
    assert panel.temperature == expected_temp

    dummy_input_h = DummyInput(input_val)

    def fake_query_hist(selector, t=None):
        if selector == "#llm_config_max_history":
            return dummy_input_h
        raise NoMatches()

    monkeypatch.setattr(panel, "query_one", fake_query_hist)

    panel._update_max_history()
    assert panel.max_history_messages == expected_hist


def test_update_models_list(monkeypatch, panel):
    dummy_model_select = DummySelect()
    panel.model = "keepme"

    monkeypatch.setattr(panel, "query_one", lambda *_args, **_kwargs: dummy_model_select)

    new_models = [("Model A", "A"), ("Model B", "B")]
    panel.update_models_list(new_models, prompt_text_if_empty="none")
    assert dummy_model_select.options_set == new_models
    assert dummy_model_select.disabled is False
    assert dummy_model_select.prompt == "Select Model"

    assert dummy_model_select.value == Select.BLANK

    panel.model = "B"
    panel.update_models_list(new_models, prompt_text_if_empty="none")
    assert dummy_model_select.value == "B"

    panel.model = "was_nonnull"
    panel.update_models_list([], prompt_text_if_empty="No models")
    assert dummy_model_select.disabled is True
    assert dummy_model_select.prompt == "No models"
    assert dummy_model_select.value == Select.BLANK

    assert panel.model is None


def test_set_model_select_loading(monkeypatch, panel):
    dummy_model_select = DummySelect()
    monkeypatch.setattr(panel, "query_one", lambda *_args, **_kwargs: dummy_model_select)

    panel.set_model_select_loading(True, loading_text="Busy…")
    assert dummy_model_select.disabled is True
    assert dummy_model_select.prompt == "Busy…"

    panel.set_model_select_loading(False, loading_text="Busy…")
    assert dummy_model_select.disabled is False
    assert dummy_model_select.prompt == "Select Model"


def test_watch_model(monkeypatch, panel):
    dummy_model_select = DummySelect()
    dummy_model_select.value = "old"
    monkeypatch.setattr(panel, "query_one", lambda *_args, **_kwargs: dummy_model_select)

    panel.watch_model("new_model")
    assert dummy_model_select.value == "new_model"

    panel.watch_model(None)
    assert dummy_model_select.value == Select.BLANK


def test_get_llm_config_defaults_and_changes(panel):
    expected = {
        "temperature": panel.DEFAULT_TEMPERATURE,
        "max_history_messages": panel.DEFAULT_MAX_HISTORY,
    }
    assert panel.get_llm_config() == expected

    panel.provider = "openai"
    panel.model = "gpt-test"
    panel.temperature = 0.123
    panel.max_history_messages = 5

    expected2 = {
        "provider_hint": "openai",
        "model_name": "gpt-test",
        "temperature": 0.123,
        "max_history_messages": 5,
    }
    assert panel.get_llm_config() == expected2
