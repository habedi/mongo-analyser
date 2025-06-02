import pytest
from textual.widgets import Static, Markdown

from mongo_analyser.widgets.chat_message_widget import ChatMessageWidget


@pytest.mark.parametrize(
    "role, content",
    [
        ("user", "hello **world**"),
        ("assistant", "hi there"),
    ],
)
def test_compose_generates_two_children_with_correct_classes(role, content):
    """
    ChatMessageWidget.compose() should yield exactly two children:
      1. A Static widget with classes "message-role" and "role_<role>"
      2. A Markdown widget with class "message-content-box"
    """
    widget = ChatMessageWidget(role, content)
    children = list(widget.compose())

    assert len(children) == 2

    first, second = children

    assert isinstance(first, Static), f"First child should be Static, got {type(first)}"

    expected_class1 = "message-role"
    expected_class2 = f"role_{role.lower()}"
    assert expected_class1 in first.classes
    assert expected_class2 in first.classes

    assert isinstance(second, Markdown), f"Second child should be Markdown, got {type(second)}"

    assert "message-content-box" in second.classes
