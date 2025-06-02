from mongo_analyser.widgets.chat_input import ChatInput


class DummyInput:
    def __init__(self):
        self.value = ""
        self.focused = False

    def focus(self, scroll_visible=True):
        self.focused = True


def test_value_getter_and_setter(monkeypatch):
    widget = ChatInput()
    dummy = DummyInput()

    monkeypatch.setattr(widget, "query_one", lambda *_args, **_kwargs: dummy)

    assert widget.value == ""

    widget.value = "hello"
    assert dummy.value == "hello"

    widget.clear()
    assert dummy.value == ""


def test_focus(monkeypatch):
    widget = ChatInput()
    dummy = DummyInput()
    monkeypatch.setattr(widget, "query_one", lambda *_args, **_kwargs: dummy)

    widget.focus()
    assert dummy.focused is True
