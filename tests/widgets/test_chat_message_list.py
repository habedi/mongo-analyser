from mongo_analyser.widgets.chat_message_list import ChatMessageList
from mongo_analyser.widgets.chat_message_widget import ChatMessageWidget


class DummyMsgWidget:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class DummyContainer:
    def __init__(self):
        self.mounted = None
        self.scrolled = False

    def mount(self, widget):
        self.mounted = widget

    def scroll_end(self, animate=True):
        self.scrolled = True


def test_add_message_calls_mount_and_scroll(monkeypatch):
    """
    When add_message(...) is called, ChatMessageList.mount(...) and
    scroll_end(...) should be triggered exactly once.
    """
    dummy_container = DummyContainer()
    lst = ChatMessageList()

    monkeypatch.setattr(lst, "mount", dummy_container.mount)
    monkeypatch.setattr(lst, "scroll_end", dummy_container.scroll_end)

    lst.add_message("user", "hello world")

    assert isinstance(dummy_container.mounted, ChatMessageWidget)

    assert dummy_container.scrolled is True


def test_clear_messages_calls_remove_on_single(monkeypatch):
    """
    clear_messages() invokes query(...).remove(), so query(...) must return
    exactly one object that has remove(). We simulate that here.
    """
    lst = ChatMessageList()

    dummy = DummyMsgWidget()

    monkeypatch.setattr(lst, "query", lambda *_args, **_kwargs: dummy)

    lst.clear_messages()
    assert dummy.removed is True
