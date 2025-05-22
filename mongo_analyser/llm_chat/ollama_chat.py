import logging
from typing import Any, Dict, Iterator, List

import ollama

from .base_chat import LLMChat

logger = logging.getLogger(__name__)


class OllamaChat(LLMChat):
    def __init__(self, model_name: str, host: str = None, timeout: int = 30, **kwargs: Any):
        """
        Initializes the Ollama chat client.
        Args:
            model_name: The name of the Ollama model to use.
            host: The URL of the Ollama host. Defaults to None, which uses the client's default.
                  You can set OLLAMA_HOST environment variable.
            timeout: Timeout for API requests in seconds.
            **kwargs: Additional options for the ollama client.
        """
        self.host = host
        self.timeout = timeout
        self._client_options = kwargs
        super().__init__(model_name)  # Calls _initialize_client

    def _initialize_client(self, **kwargs: Any) -> ollama.Client:
        client_params = {"timeout": self.timeout}
        if self.host:
            client_params["host"] = self.host
        client_params.update(self._client_options)

        try:
            client = ollama.Client(**client_params)
            # Test connection by listing local models (lightweight check)
            client.list()
            logger.info(f"Ollama client initialized. Host: {client_params.get('host', 'default')}")
            return client
        except Exception as e:  # Broad exception for connection/setup issues
            logger.error(f"Failed to initialize or connect to Ollama: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        formatted_history = self.format_history(history)
        messages = formatted_history + [{"role": "user", "content": message}]

        try:
            logger.debug(
                f"Sending message to Ollama model {self.model_name}. History length: {len(formatted_history)}"
            )
            response = self.client.chat(model=self.model_name, messages=messages, stream=False)
            assistant_response = response.get("message", {}).get("content", "")
            logger.debug(f"Received response from Ollama model {self.model_name}")
            return assistant_response
        except ollama.ResponseError as e:
            logger.error(
                f"Ollama API ResponseError for model {self.model_name}: {e.status_code} - {e.error}",
                exc_info=True,
            )
            return f"Error: Ollama API error ({e.status_code}) - {e.error}"
        except Exception as e:
            logger.error(
                f"Error communicating with Ollama model {self.model_name}: {e}", exc_info=True
            )
            return "Error: Could not get a response from Ollama."

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        formatted_history = self.format_history(history)
        messages = formatted_history + [{"role": "user", "content": message}]

        try:
            logger.debug(
                f"Streaming message to Ollama model {self.model_name}. History length: {len(formatted_history)}"
            )
            stream = self.client.chat(model=self.model_name, messages=messages, stream=True)
            for chunk in stream:
                if not chunk.get("done"):
                    content_chunk = chunk.get("message", {}).get("content", "")
                    if content_chunk:
                        yield content_chunk
                else:  # Last message with full context, stats, etc.
                    logger.debug(f"Ollama stream finished for model {self.model_name}")
                    # Optionally process chunk.get('total_duration'), etc.
                    break
        except ollama.ResponseError as e:
            logger.error(
                f"Ollama API ResponseError during stream for model {self.model_name}: {e.status_code} - {e.error}",
                exc_info=True,
            )
            yield f"Error: Ollama API error ({e.status_code}) - {e.error}"
        except Exception as e:
            logger.error(f"Error streaming from Ollama model {self.model_name}: {e}", exc_info=True)
            yield "Error: Could not stream response from Ollama."

    def get_available_models(self) -> List[str]:
        try:
            logger.debug("Fetching available Ollama models.")
            models_data = self.client.list()
            available_models = [
                model.get("model") for model in models_data.get("models", []) if model.get("model")
            ]
            logger.info(f"Found Ollama models: {available_models}")
            return available_models
        except Exception as e:  # Broad exception for connection/setup issues
            logger.error(f"Error fetching Ollama models: {e}", exc_info=True)
            return []

    # format_history can use the base class implementation if it's suitable
    # or be overridden here if Ollama client library expects a different format.
    # The official ollama library's chat method expects a list of dicts
    # like [{'role': 'user', 'content': '...'}], which matches the base.
