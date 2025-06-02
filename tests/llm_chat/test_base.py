from mongo_analyser.llm_chat.base import LLMChat


class DummyChat(LLMChat):
    """A trivial subclass that doesn’t actually call any external client."""

    def _initialize_client(self, **kwargs):
        return object()

    def send_message(self, message: str, history=None) -> str:
        return f"echo:{message}"

    def stream_message(self, message: str, history=None):
        yield f"streamed:{message}"

    @staticmethod
    def list_models(client_config=None):
        return ["modelA", "modelB"]


def test_format_history_skips_malformed_and_formats_properly():
    dummy = DummyChat("unused_model")

    history = [
        {"role": "user", "content": "hello"},
        {"role": "ai", "content": "hi there"},
        {"role": "assistant", "content": ""},
        {"role": None, "content": "oops"},
        {"role": "user", "content": None},
    ]
    formatted = dummy.format_history(history)
    assert formatted == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_get_available_models_returns_list_models_output():
    dummy = DummyChat("unused_model")
    models = dummy.get_available_models()
    assert models == ["modelA", "modelB"]


def test_send_and_stream_message_from_dummy():
    dummy = DummyChat("unused_model")
    assert dummy.send_message("test", history=[]) == "echo:test"
    stream = list(dummy.stream_message("abc", history=[]))
    assert stream == ["streamed:abc"]
