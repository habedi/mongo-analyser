import pytest

import mongo_analyser.llm_chat.ollama as ollama_mod
from mongo_analyser.llm_chat.ollama import OllamaChat


class DummyOllamaClient:
    def __init__(self, **kwargs):
        self.params = kwargs

    def list(self):
        return {"models": [{"model": "m1"}, {"name": "m2"}, {"model": "granite-embedding:latest"}]}

    def chat(self, model, messages, stream=False, options=None, keep_alive=None):
        if stream:
            return iter([
                {"message": {"content": "chunk1"}, "done": False},
                {"message": {"content": "chunk2"}, "done": False},
                {"done": True},
            ])

        return {"message": {"content": "reply:" + messages[-1]["content"]}}


class DummyResponseError(Exception):
    def __init__(self, status_code, error):
        self.status_code = status_code
        self.error = error


@pytest.fixture(autouse=True)
def patch_ollama(monkeypatch):
    dummy_module = type("module", (), {
        "Client": DummyOllamaClient,
        "ResponseError": DummyResponseError,
    })
    monkeypatch.setattr(ollama_mod, "ollama", dummy_module)
    yield


def test_constructor_and_send(monkeypatch):
    chat = OllamaChat("modelX", host="http://fake", timeout=5, options={"opt1": 1})
    resp = chat.send_message("msg", history=[{"role": "user", "content": "hi"}])
    assert resp == "reply:msg"


def test_stream_message_yields_chunks(monkeypatch):
    chat = OllamaChat("modelX")
    chunks = list(chat.stream_message("abc", history=[]))
    assert chunks == ["chunk1", "chunk2"]


def test_list_models_filters_blocklist(monkeypatch):
    models = OllamaChat.list_models({"host": "http://fake", "timeout": 5})

    assert "granite-embedding:latest" not in models
    assert set(models) == {"m1", "m2"}
