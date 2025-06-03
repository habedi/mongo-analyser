import pytest

import mongo_analyser.llm_chat.openai as openai_mod
from mongo_analyser.llm_chat.openai import OpenAIChat


class DummyAPIError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(message)


class DummyAuthError(Exception):
    pass


class DummyOpenAIClient:
    def __init__(self, **kwargs):
        self.base_url = kwargs.get("base_url", "http://fake-openai")

        class Completions:
            def __init__(self, client):
                self._client = client

            def create(self, **params):
                class Choice:
                    class Message:
                        def __init__(self, content):
                            self.content = content

                    def __init__(self, text):
                        self.message = Choice.Message(text)

                last_user_content = params["messages"][-1]["content"]
                return type("Resp", (), {"choices": [Choice("resp:" + last_user_content)]})

        chat_holder = type("ChatHolder", (), {})()
        chat_holder.completions = Completions(self)
        self.chat = chat_holder


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    fake_openai = type(
        "module",
        (),
        {
            "OpenAI": DummyOpenAIClient,
            "APIError": DummyAPIError,
            "AuthenticationError": DummyAuthError,
        },
    )

    monkeypatch.setattr(openai_mod, "openai", fake_openai)
    yield


def test_constructor_without_key_warns_but_succeeds(monkeypatch, caplog):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    caplog.set_level("WARNING", logger="mongo_analyser.llm_chat.openai")

    chat = OpenAIChat("some-model")
    assert "OpenAI API key not provided" in caplog.text

    assert chat.client_config.get("api_key") is None

    assert hasattr(chat.client, "base_url")


def test_send_message_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    chat = OpenAIChat("test-model")
    reply = chat.send_message("hello", history=[{"role": "user", "content": "prior"}])
    assert reply == "resp:hello"


def test_list_models_without_key_returns_empty(monkeypatch, caplog):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    caplog.set_level("WARNING", logger="mongo_analyser.llm_chat.openai")
    models = OpenAIChat.list_models({"base_url": "https://api.openai.com/v1"})
    assert models == []
    assert "Cannot list OpenAI models" in caplog.text
