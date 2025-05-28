from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ErrorDialog(ModalScreen[None]):
    DEFAULT_CSS = """
    ErrorDialog {
        align: center middle;
    }
    ErrorDialog > Vertical {
        background: $panel-darken-2; /* Standard Textual variable for a darker panel */
        color: $text;               /* Standard Textual variable for text */
        width: auto;
        min-width: 40;
        max-width: 80%;
        height: auto;
        padding: 1 2;
        border: thick $error;       /* Standard Textual variable for error color (often red) */
    }
    ErrorDialog Static { /* Static is the base for Label */
        margin-bottom: 1;
        text-align: center;
    }
    ErrorDialog Label {
        margin-top:1;
        margin-bottom: 1;
    }
    ErrorDialog Center {
        margin-top: 1;
        height: auto;
    }
    """
    BINDINGS = [Binding("escape", "dismiss", show=False)]

    def __init__(self, title: str, message: str):
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical() as v_layout:
            v_layout.border_title = self._title
            # The title is now part of the border_title of the Vertical container
            yield Label(self._message, classes="dialog_message_error")
            with Center():
                yield Button(
                    "OK", variant="error", id="ok_button"
                )  # variant="error" will use $error color

    def on_mount(self) -> None:
        self.query_one("#ok_button", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok_button":
            self.dismiss()
