from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Input, Static


class ChatInput(Static):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type your message here...", id="chat_internal_input")

    @property
    def value(self) -> str:
        try:
            return self.query_one(Input).value
        except NoMatches:
            return ""

    @value.setter
    def value(self, new_value: str) -> None:
        try:
            self.query_one(Input).value = new_value
        except NoMatches:
            pass

    def clear(self) -> None:
        self.value = ""

    def focus(self, scroll_visible: bool = True) -> None:
        try:
            self.query_one(Input).focus(scroll_visible)
        except NoMatches:
            pass
