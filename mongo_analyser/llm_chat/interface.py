import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class LLMChat(ABC):
    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.client_config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with model: {self.model_name}")
        self.client = self._initialize_client(**self.client_config)

    @abstractmethod
    def _initialize_client(self, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        pass

    @abstractmethod
    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        pass

    @staticmethod
    @abstractmethod
    def list_models(client_config: Optional[Dict[str, Any]] = None) -> List[str]:
        pass

    def get_available_models(self) -> List[str]:
        return self.__class__.list_models(client_config=self.client_config)

    def format_history(self, history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        formatted_history = []
        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content")
                if role and content:
                    if role == "ai":
                        role = "assistant"
                    formatted_history.append({"role": role, "content": content})
                else:
                    logger.warning(f"Skipping malformed message in history: {msg}")
        return formatted_history
