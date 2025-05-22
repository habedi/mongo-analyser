import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

logger = logging.getLogger(__name__)


class LLMChat(ABC):

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        logger.info(f"Initializing {self.__class__.__name__} with model: {self.model_name}")
        self.client = self._initialize_client(**kwargs)

    @abstractmethod
    def _initialize_client(self, **kwargs: Any) -> Any:
        """Initializes and returns the API client for the LLM service."""
        pass

    @abstractmethod
    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        """Sends a message and returns the complete response."""
        pass

    @abstractmethod
    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        """Sends a message and streams the response chunks."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Lists available models from the provider."""
        pass

    def format_history(self, history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Formats the chat history into the structure expected by the LLM API.
        The default format is a list of dictionaries, each with 'role' and 'content'.
        Example: [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi!'}]
        """
        return history or []
