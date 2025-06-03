from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmDialog(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", show=False)]

    def __init__(self, title: str, message: str, yes_label: str = "Yes", no_label: str = "No"):
        super().__init__()
        self._title = title
        self._message = message
        self._yes_label = yes_label
        self._no_label = no_label

    def compose(self) -> ComposeResult:
        with Vertical() as v_layout:
            v_layout.border_title = self._title
            yield Label(self._message)
            with Horizontal():
                yield Button(self._yes_label, variant="primary", id="yes_button")
                yield Button(self._no_label, id="no_button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes_button":
            self.dismiss(True)
        elif event.button.id == "no_button":
            self.dismiss(False)
