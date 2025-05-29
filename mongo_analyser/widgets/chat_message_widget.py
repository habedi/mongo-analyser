from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Markdown, Static


class ChatMessageWidget(Vertical):
    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        role_text_str = self.role.upper()
        role_text = Text(role_text_str, no_wrap=True)

        yield Static(role_text, classes=f"role_{self.role.lower()}")
        yield Markdown(self.content)
