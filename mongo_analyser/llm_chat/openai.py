import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import openai

from .base import LLMChat

logger = logging.getLogger(__name__)

OPENAI_MODEL_BLOCKLIST = [
    "babbage-002",
    "dall-e-2",
    "dall-e-3",
    "davinci-002",
    "gpt-3.5-turbo-instruct-0914",
    "code-davinci-002",
    "code-cushman-001",
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-002",
    "text-davinci-003",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "tts-1",
    "tts-1-1106",
    "tts-1-hd",
    "tts-1-hd-1106",
    "whisper-1",
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
    "gpt-4o-mini-search-preview",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-tts",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-transcribe",
    "gpt-image-1",
    "omni-moderation-2024-09-26",
    "omni-moderation-latest",
    "chatgpt-4o-latest",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    "gpt-4-0125-preview",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
]


class OpenAIChat(LLMChat):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        self._effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._timeout = timeout
        self._max_retries = max_retries

        self._client_options = kwargs.pop("client_options", {})
        self._completion_kwargs = kwargs

        if not self._effective_api_key and not self._base_url:
            if not self._effective_api_key and (
                not self._base_url or "api.openai.com" in self._base_url
            ):
                logger.warning(
                    "OpenAI API key not provided. Client might fail if targeting api.openai.com."
                )

        init_kwargs_for_super = {
            "api_key": self._effective_api_key,
            "base_url": self._base_url,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
            "client_options": self._client_options,
            **self._completion_kwargs,
        }
        super().__init__(
            model_name, **{k: v for k, v in init_kwargs_for_super.items() if v is not None}
        )

    def _initialize_client(self, **kwargs: Any) -> openai.OpenAI:
        client_init_params = {
            "api_key": kwargs.get("api_key", self._effective_api_key),
            "base_url": kwargs.get("base_url", self._base_url),
            "timeout": kwargs.get("timeout", self._timeout),
            "max_retries": kwargs.get("max_retries", self._max_retries),
            **kwargs.get("client_options", self._client_options),
        }
        client_init_params = {k: v for k, v in client_init_params.items() if v is not None}

        try:
            client = openai.OpenAI(**client_init_params)
            logger.info(
                f"OpenAI client initialized for model {self.model_name}. Endpoint: {client.base_url}"
            )
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

        completion_params_from_config = {
            k: v
            for k, v in self.client_config.items()
            if k
            not in ["api_key", "base_url", "timeout", "max_retries", "client_options", "model_name"]
        }

        try:
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                **completion_params_from_config,
            }
            response = self.client.chat.completions.create(**completion_params)

            assistant_response = ""
            if response.choices and response.choices[0].message:
                assistant_response = response.choices[0].message.content or ""
            return assistant_response.strip()
        except openai.APIError as e:
            logger.error(f"OpenAI APIError: {e.__class__.__name__} - {e}", exc_info=True)
            return f"Error: OpenAI API error ({e.message if hasattr(e, 'message') else e})"
        except Exception as e:
            logger.error(f"Error communicating with OpenAI: {e}", exc_info=True)
            return f"Error: Could not get response. {e.__class__.__name__}"

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        formatted_history = self.format_history(history)
        messages = formatted_history + [{"role": "user", "content": message}]

        completion_params_from_config = {
            k: v
            for k, v in self.client_config.items()
            if k
            not in ["api_key", "base_url", "timeout", "max_retries", "client_options", "model_name"]
        }
        try:
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                **completion_params_from_config,
            }
            stream = self.client.chat.completions.create(**completion_params)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    content_chunk = chunk.choices[0].delta.content
                    if content_chunk is not None:
                        yield content_chunk
        except openai.APIError as e:
            logger.error(f"OpenAI APIError during stream: {e}", exc_info=True)
            yield f"Error: OpenAI API error ({e.message if hasattr(e, 'message') else e})"
        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}", exc_info=True)
            yield f"Error: Could not stream response. {e.__class__.__name__}"

    @staticmethod
    def list_models(client_config: Optional[Dict[str, Any]] = None) -> List[str]:
        cfg = client_config or {}
        static_client_params = {
            "api_key": cfg.get("api_key", os.getenv("OPENAI_API_KEY")),
            "base_url": cfg.get("base_url", os.getenv("OPENAI_BASE_URL")),
            "timeout": cfg.get("timeout", 30.0),
            "max_retries": cfg.get("max_retries", 2),
            **(cfg.get("client_options", {})),
        }
        static_client_params = {k: v for k, v in static_client_params.items() if v is not None}

        effective_base_url = static_client_params.get("base_url", "https://api.openai.com/v1")

        if not static_client_params.get("api_key") and "api.openai.com" in effective_base_url:
            logger.warning("Cannot list OpenAI models from api.openai.com without API key.")
            return []
        try:
            temp_client = openai.OpenAI(**static_client_params)
            response = temp_client.models.list()

            all_models = sorted([model.id for model in response.data if model.id])

            filtered_models = [
                model_id
                for model_id in all_models
                if model_id not in OPENAI_MODEL_BLOCKLIST
                and not any(
                    blocked_prefix in model_id for blocked_prefix in ["codex-", "o1-", "o3-", "o4-"]
                )
            ]

            logger.info(
                f"Found {len(all_models)} OpenAI models, displaying {len(filtered_models)}"
                f" after filtering."
            )
            return filtered_models
        except openai.APIError as e:
            logger.error(f"OpenAI APIError fetching models: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}", exc_info=True)
            return []
