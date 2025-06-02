from textual.widgets import Button

from mongo_analyser.dialogs.confirm_dialog import ConfirmDialog


class DummyConfirmDialog(ConfirmDialog):
    """
    Subclass ConfirmDialog so that dismiss() just records the value
    instead of trying to close a real ModalScreen.
    """

    def __init__(self, title: str, message: str, yes_label: str = "Yes", no_label: str = "No"):
        super().__init__(title, message, yes_label, no_label)
        self.dismissed_value = None

    def dismiss(self, value: bool):
        self.dismissed_value = value


def test_constructor_stores_title_message_and_labels():
    dlg = ConfirmDialog("My Title", "My Message", yes_label="OK", no_label="Cancel")

    assert dlg._title == "My Title"
    assert dlg._message == "My Message"
    assert dlg._yes_label == "OK"
    assert dlg._no_label == "Cancel"


def test_on_button_pressed_yes_sets_dismiss_true():
    dlg = DummyConfirmDialog("T", "M", yes_label="Proceed", no_label="Abort")

    yes_btn = Button("Proceed", id="yes_button")
    event = Button.Pressed(yes_btn)
    dlg.on_button_pressed(event)
    assert dlg.dismissed_value is True


def test_on_button_pressed_no_sets_dismiss_false():
    dlg = DummyConfirmDialog("T", "M", yes_label="Proceed", no_label="Abort")
    no_btn = Button("Abort", id="no_button")
    event = Button.Pressed(no_btn)
    dlg.on_button_pressed(event)
    assert dlg.dismissed_value is False
