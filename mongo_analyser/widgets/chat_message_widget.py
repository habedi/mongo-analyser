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
        role_text = Text(f"{self.role.upper()}:")
        yield Static(role_text, classes=f"role_{self.role.lower()}")
        yield Markdown(self.content)
