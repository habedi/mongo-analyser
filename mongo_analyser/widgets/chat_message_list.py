from textual.containers import VerticalScroll

from .chat_message_widget import ChatMessageWidget


class ChatMessageList(VerticalScroll):
    def add_message(self, role: str, content: str) -> None:
        message_widget = ChatMessageWidget(role, content)
        self.mount(message_widget)
        self.scroll_end(animate=True)

    def clear_messages(self) -> None:
        self.query(ChatMessageWidget).remove()
