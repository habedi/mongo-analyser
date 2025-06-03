from textual.widgets import Button

from mongo_analyser.dialogs.error_dialog import ErrorDialog


class DummyErrorDialog(ErrorDialog):
    """
    Subclass ErrorDialog so that dismiss() just records that it was called.
    """

    def __init__(self, title: str, message: str):
        super().__init__(title, message)
        self.was_dismissed = False

    def dismiss(self):
        self.was_dismissed = True


def test_constructor_stores_title_and_message():
    dlg = ErrorDialog("Err Title", "Err Message")
    assert dlg._title == "Err Title"
    assert dlg._message == "Err Message"


def test_on_button_pressed_ok_sets_dismiss():
    dlg = DummyErrorDialog("E", "M")
    ok_btn = Button("OK", id="ok_button")
    event = Button.Pressed(ok_btn)
    dlg.on_button_pressed(event)
    assert dlg.was_dismissed is True
