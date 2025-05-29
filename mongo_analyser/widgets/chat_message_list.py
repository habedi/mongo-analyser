from textual.containers import VerticalScroll

from .chat_message_widget import ChatMessageWidget


class ChatMessageList(VerticalScroll):
    DEFAULT_CSS = """
    ChatMessageList {
        width: 100%;
        height: 1fr; /* Fill available space */
        background: $boost; /* Example */
    }
    """

    def add_message(self, role: str, content: str) -> None:
        message_widget = ChatMessageWidget(role, content)
        self.mount(message_widget)
        self.scroll_end(animate=True)

    def clear_messages(self) -> None:
        self.query(ChatMessageWidget).remove()
