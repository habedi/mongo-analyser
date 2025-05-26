from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Markdown, Static  # Use Markdown for content


class ChatMessageWidget(Vertical):  # Vertical container for role + message
    DEFAULT_CSS = """
    ChatMessageWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: round $primary-background-darken-2; /* Example border */
    }
    ChatMessageWidget .role_user { color: $success; } /* Example */
    ChatMessageWidget .role_assistant { color: $secondary; } /* Example */
    ChatMessageWidget .role_system { color: $warning; } /* Example */
    ChatMessageWidget > Markdown { background: transparent; margin-top: 0; }
    """

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        role_display = Text(f"{self.role.upper()}:", classes=f"role_{self.role.lower()}")
        yield Static(role_display)
        yield Markdown(self.content)  # Render content as Markdown
