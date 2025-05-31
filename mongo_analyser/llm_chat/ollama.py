import logging
from typing import Any, Dict, Iterator, List, Optional

import ollama

from .base import LLMChat

logger = logging.getLogger(__name__)

OLLAMA_MODEL_BLOCKLIST = [
    "granite-embedding:latest",
    "nomic-embed-text:latest",
]


class OllamaChat(LLMChat):
    def __init__(self, model_name: str, host: str = None, timeout: int = 60, **kwargs: Any):
        self._host = host
        self._timeout = timeout

        self._client_options_from_constructor = kwargs.pop("options", {})
        if not isinstance(self._client_options_from_constructor, dict):
            logger.warning(
                f"Invalid 'options' provided to OllamaChat constructor"
                f" (expected dict, got {type(self._client_options_from_constructor)}). Resetting to empty dict."
            )
            self._client_options_from_constructor = {}

        self._keep_alive = kwargs.pop("keep_alive", "5m")
        self._client_init_kwargs = kwargs

        init_kwargs_for_super = {
            "host": self._host,
            "timeout": self._timeout,
            "options": self._client_options_from_constructor.copy(),
            "keep_alive": self._keep_alive,
            **self._client_init_kwargs,
        }
        super().__init__(model_name, **init_kwargs_for_super)

    def _initialize_client(self, **kwargs: Any) -> ollama.Client:
        host_to_use = kwargs.get("host", self._host)
        timeout_to_use = kwargs.get("timeout", self._timeout)

        additional_client_params = {
            k: v
            for k, v in kwargs.items()
            if k not in ["host", "timeout", "options", "keep_alive", "model_name"] and v is not None
        }

        client_params: Dict[str, Any] = {"timeout": timeout_to_use}
        if host_to_use:
            client_params["host"] = host_to_use

        client_params.update(additional_client_params)

        try:
            client = ollama.Client(**client_params)
            client.list()
            logger.info(f"Ollama client initialized. Effective params: {client_params}")
            return client
        except Exception as e:
            logger.error(
                f"Failed to initialize or connect to Ollama with params {client_params}: {e}",
                exc_info=True,
            )
            raise ConnectionError(
                f"Failed to connect to Ollama at {client_params.get('host', 'default')}: {e}"
            )

    def _get_effective_options(self) -> Dict[str, Any]:
        effective_opts = self._client_options_from_constructor.copy()

        if isinstance(self.client_config.get("options"), dict):
            effective_opts.update(self.client_config["options"])

        config_temperature = self.client_config.get("temperature")
        if config_temperature is not None:
            effective_opts["temperature"] = config_temperature

        return effective_opts

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        formatted_history = self.format_history(history)
        messages = formatted_history + [{"role": "user", "content": message}]

        effective_options = self._get_effective_options()

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options=effective_options if effective_options else None,
                keep_alive=self.client_config.get("keep_alive", self._keep_alive),
            )
            assistant_response = response.get("message", {}).get("content", "")
            return assistant_response
        except ollama.ResponseError as e:
            logger.error(
                f"Ollama API ResponseError for model {self.model_name}:"
                f" {e.status_code} - {e.error}",
                exc_info=True,
            )
            return f"Error: Ollama API error ({e.status_code}) - {e.error}"
        except Exception as e:
            logger.error(
                f"Error communicating with Ollama model {self.model_name}: {e}", exc_info=True
            )
            return f"Error: Could not get response from Ollama. {e.__class__.__name__}: {e}"

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        formatted_history = self.format_history(history)
        messages = formatted_history + [{"role": "user", "content": message}]

        effective_options = self._get_effective_options()

        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options=effective_options if effective_options else None,
                keep_alive=self.client_config.get("keep_alive", self._keep_alive),
            )
            for chunk in stream:
                if not chunk.get("done", False):
                    content_chunk = chunk.get("message", {}).get("content", "")
                    if content_chunk:
                        yield content_chunk
                else:
                    break
        except ollama.ResponseError as e:
            logger.error(
                f"Ollama API ResponseError during stream for model {self.model_name}:"
                f" {e.status_code} - {e.error}",
                exc_info=True,
            )
            yield f"Error: Ollama API error ({e.status_code}) - {e.error}"
        except Exception as e:
            logger.error(f"Error streaming from Ollama model {self.model_name}: {e}", exc_info=True)
            yield f"Error: Could not stream response. {e.__class__.__name__}: {e}"

    @staticmethod
    def list_models(client_config: Optional[Dict[str, Any]] = None) -> List[str]:
        cfg = client_config or {}

        client_args_for_listing: Dict[str, Any] = {}

        host = cfg.get("host")
        if host is not None:
            client_args_for_listing["host"] = host

        timeout = cfg.get("timeout")
        if timeout is not None:
            client_args_for_listing["timeout"] = timeout
        else:
            client_args_for_listing["timeout"] = 30

        try:
            logger.debug(
                f"Attempting to list Ollama models with client args: {client_args_for_listing}"
            )
            temp_client = ollama.Client(**client_args_for_listing)
            models_data = temp_client.list()

            all_models = [
                model_info.get("model", model_info.get("name"))
                for model_info in models_data.get("models", [])
                if model_info.get("model") or model_info.get("name")
            ]
            all_models = sorted(list(set(m for m in all_models if m)))

            filtered_models = [
                model_name for model_name in all_models if model_name not in OLLAMA_MODEL_BLOCKLIST
            ]

            logger.info(
                f"Found {len(all_models)} Ollama models, displaying {len(filtered_models)}"
                f" after filtering."
            )
            if not all_models:
                logger.warning(
                    f"Ollama client at {client_args_for_listing.get('host', 'default')}"
                    f" reported no models. Raw response: {models_data}"
                )

            return filtered_models
        except Exception as e:
            logger.error(
                f"Error fetching Ollama models with client_args {client_args_for_listing}: {e}",
                exc_info=True,
            )
            return []
