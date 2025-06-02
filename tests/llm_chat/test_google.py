import pytest

import mongo_analyser.llm_chat.google as google_mod
from mongo_analyser.llm_chat.google import GoogleChat


class DummyGenAI:
    class GenerativeModel:
        def __init__(self, name, generation_config=None, safety_settings=None, **kwargs):
            self.name = name
            self.generated = []

        def generate_content(self, contents, stream=False):
            class FakePart:
                def __init__(self, text):
                    self.text = text

            class FakeResponse:
                def __init__(self, parts):
                    self.parts = parts

            return FakeResponse([FakePart("OK:" + contents[-1]["parts"][0]["text"])])

    @staticmethod
    def configure(api_key, **kwargs):
        pass

    @staticmethod
    def list_models():
        class M:
            def __init__(self, name, methods):
                self.name = name
                self.supported_generation_methods = methods

        return [
            M("good-model-1", ["generateContent"]),
            M("bad-model-exp", ["generateContent"]),
            M("another-model", ["notSupported"]),
        ]


@pytest.fixture(autouse=True)
def patch_genai(monkeypatch):
    monkeypatch.setattr(google_mod, "genai", DummyGenAI)
    yield


def test_constructor_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GoogleChat("any-model")

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")
    chat = GoogleChat("any-model")
    assert chat._effective_api_key == "dummy_key"
    assert chat.model_name == "any-model"


def test_format_history_and_send_message(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")
    chat = GoogleChat("my-model")

    res = chat.send_message("hello", history=[{"role": "assistant", "content": ""}])
    assert res == "OK:hello"

    h = [{"role": "assistant", "content": "hi"}]
    res2 = chat.send_message("bye", history=h)
    assert res2 == "OK:bye"


def test_list_models_filters_and_sorts(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")

    models = GoogleChat.list_models({"api_key": "dummy_key"})

    assert "good-model-1" in models
    assert "bad-model-exp" not in models
