from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Markdown, Static


class ChatMessageWidget(Vertical):
    DEFAULT_CSS = """
    ChatMessageWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: round $primary-background-darken-2;
    }
    ChatMessageWidget .role_user { color: $success; }
    ChatMessageWidget .role_assistant { color: $secondary; }
    ChatMessageWidget .role_system { color: $warning; }
    ChatMessageWidget > Markdown { background: transparent; margin-top: 0; }
    """

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        role_text = Text(f"{self.role.upper()}:")
        yield Static(role_text, classes=f"role_{self.role.lower()}")
        yield Markdown(self.content)
