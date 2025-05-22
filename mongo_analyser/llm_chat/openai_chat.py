import logging
import os
from typing import List, Dict, Any, Iterator

import openai

from .base_chat import LLMChat

logger = logging.getLogger(__name__)


class OpenAIChat(LLMChat):

    def __init__(self, model_name: str, api_key: str = None, timeout: float = 30.0,
                 max_retries: int = 2, **kwargs: Any):
        """
        Initializes the OpenAI chat client.
        Args:
            model_name: The name of the OpenAI model to use (e.g., "gpt-3.5-turbo").
            api_key: OpenAI API key. If None, attempts to use OPENAI_API_KEY environment variable.
            timeout: Timeout for API requests in seconds.
            max_retries: Maximum number of retries for API calls.
            **kwargs: Additional options for the OpenAI client (e.g., base_url for Azure OpenAI).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error(
                "OpenAI API key not provided or found in OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key is required.")

        self._client_options = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": max_retries,
            **kwargs
        }
        super().__init__(model_name)  # Calls _initialize_client

    def _initialize_client(self, **kwargs: Any) -> openai.OpenAI:
        try:
            # kwargs here are from the constructor, self._client_options already incorporates them.
            client = openai.OpenAI(**self._client_options)
            # A lightweight call to check if the client is functional, e.g., listing models.
            # This might incur a small cost or have rate limits, use judiciously or rely on first actual call.
            # client.models.list(limit=1) # Example check
            logger.info(f"OpenAI client initialized for model {self.model_name}.")
            return client
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI AuthenticationError: {e}", exc_info=True)
            raise PermissionError(f"OpenAI authentication failed: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        formatted_history = self.format_history(history)
        messages = formatted_history + [{"role": "user", "content": message}]

        try:
            logger.debug(
                f"Sending message to OpenAI model {self.model_name}. History length: {len(formatted_history)}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
            )
            assistant_response = response.choices[0].message.content
            logger.debug(
                f"Received response from OpenAI model {self.model_name}. Tokens used: {response.usage}")
            return assistant_response.strip() if assistant_response else ""
        except openai.APIError as e:
            logger.error(
                f"OpenAI APIError for model {self.model_name}: {e.__class__.__name__} - {e}",
                exc_info=True)
            return f"Error: OpenAI API error ({e.__class__.__name__}) - {e}"
        except Exception as e:
            logger.error(f"Error communicating with OpenAI model {self.model_name}: {e}",
                         exc_info=True)
            return "Error: Could not get a response from OpenAI."

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        formatted_history = self.format_history(history)
        messages = formatted_history + [{"role": "user", "content": message}]

        try:
            logger.debug(
                f"Streaming message to OpenAI model {self.model_name}. History length: {len(formatted_history)}")
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk is not None:
                    yield content_chunk
            logger.debug(f"OpenAI stream finished for model {self.model_name}")
        except openai.APIError as e:
            logger.error(
                f"OpenAI APIError during stream for model {self.model_name}: {e.__class__.__name__} - {e}",
                exc_info=True)
            yield f"Error: OpenAI API error ({e.__class__.__name__}) - {e}"
        except Exception as e:  # Catch more specific exceptions
            logger.error(f"Error streaming from OpenAI model {self.model_name}: {e}", exc_info=True)
            yield "Error: Could not stream response from OpenAI."

    def get_available_models(self) -> List[str]:
        try:
            logger.debug("Fetching available OpenAI models.")
            response = self.client.models.list()
            # You might want to filter these further, e.g., for specific capabilities or families
            available_models = [model.id for model in response.data]
            logger.info(f"Found {len(available_models)} OpenAI models.")
            return available_models
        except openai.APIError as e:
            logger.error(f"OpenAI APIError fetching models: {e.__class__.__name__} - {e}",
                         exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}", exc_info=True)
            return []

    # format_history uses the base class implementation, which is compatible
    # with OpenAI's expected format: list of {"role": ..., "content": ...}
