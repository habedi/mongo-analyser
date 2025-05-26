# mongo_analyser/widgets/chat_input.py
from textual.app import ComposeResult
from textual.widgets import Input, Static


class ChatInput(Static):  # Or whatever base class you decided, e.g., Container
    DEFAULT_CSS = """
    ChatInput {
        layout: horizontal;
        height: auto;
        width: 100%;
    }
    ChatInput > Input {
        width: 1fr;
        min-height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type your message here...", id="chat_internal_input")

    @property
    def value(self) -> str:
        try:
            return self.query_one(Input).value
        except:  # Add specific exception if possible, e.g., NoMatches
            return ""

    @value.setter
    def value(self, new_value: str) -> None:
        try:
            self.query_one(Input).value = new_value
        except:  # Add specific exception
            pass

    def clear(self) -> None:
        self.value = ""

    def focus(self, scroll_visible: bool = True) -> None:
        try:
            self.query_one(Input).focus(scroll_visible)
        except:  # Add specific exception
            pass
