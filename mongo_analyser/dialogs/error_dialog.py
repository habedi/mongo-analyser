from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ErrorDialog(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", show=False)]

    def __init__(self, title: str, message: str):
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical() as v_layout:
            v_layout.border_title = self._title

            yield Label(self._message, classes="dialog_message_error")
            with Center():
                yield Button("OK", variant="error", id="ok_button")

    def on_mount(self) -> None:
        self.query_one("#ok_button", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok_button":
            self.dismiss()
