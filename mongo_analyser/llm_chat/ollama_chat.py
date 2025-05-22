import logging
from typing import Any, Dict, Iterator, List, Optional

import ollama

from .base_chat import LLMChat

logger = logging.getLogger(__name__)

OLLAMA_MODEL_BLOCKLIST = [
    "granite-embedding:latest",  # Example embedding model
    "nomic-embed-text:latest",  # Example embedding model
    # Add other models you wish to hide from the selection list
]


class OllamaChat(LLMChat):
    def __init__(self, model_name: str, host: str = None, timeout: int = 30, **kwargs: Any):
        self._host = host
        self._timeout = timeout
        # Ensure 'options' is a dict, even if not provided in kwargs initially
        self._client_options = kwargs.pop("options", {})
        if not isinstance(
            self._client_options, dict
        ):  # Safeguard if options was passed but not as dict
            logger.warning(
                f"Invalid 'options' provided to OllamaChat (expected dict, got {type(self._client_options)}). Resetting to empty dict."
            )
            self._client_options = {}

        self._keep_alive = kwargs.pop("keep_alive", "5m")
        self._client_init_kwargs = kwargs  # Remaining kwargs

        init_kwargs_for_super = {
            "host": self._host,
            "timeout": self._timeout,
            "options": self._client_options.copy(),  # Pass a copy to avoid modification issues
            "keep_alive": self._keep_alive,
            **self._client_init_kwargs,
        }
        super().__init__(model_name, **init_kwargs_for_super)

    def _initialize_client(self, **kwargs: Any) -> ollama.Client:
        host = kwargs.get("host", self._host)
        timeout = kwargs.get("timeout", self._timeout)
        client_params_from_config = {
            k: v
            for k, v in kwargs.items()
            if k not in ["host", "timeout", "options", "keep_alive", "model_name"]
        }

        client_params: Dict[str, Any] = {"timeout": timeout}
        if host:
            client_params["host"] = host
        client_params.update(client_params_from_config)

        try:
            client = ollama.Client(**client_params)
            # Test connection by listing models (or a more lightweight ping if available)
            client.list()
            logger.info(
                f"Ollama client initialized. Host: {client_params.get('host', 'default')}, Timeout: {timeout}s"
            )
            return client
        except Exception as e:
            logger.error(f"Failed to initialize or connect to Ollama: {e}", exc_info=True)
            raise ConnectionError(
                f"Failed to connect to Ollama at {client_params.get('host', 'default')}: {e}"
            )

    def _get_effective_options(self) -> Dict[str, Any]:
        """Merges instance options with globally configured options from client_config."""
        # Start with a copy of the options set during __init__ (potentially from TUI)
        effective_opts = self._client_options.copy()
        # Override/add with any options from the broader client_config if they exist
        # This allows for dynamic updates if client_config were to be changed, though less common here.
        if isinstance(self.client_config.get("options"), dict):
            effective_opts.update(self.client_config["options"])
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
                options=effective_options,
                keep_alive=self.client_config.get("keep_alive", self._keep_alive),
            )
            assistant_response = response.get("message", {}).get("content", "")
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
                options=effective_options,
                keep_alive=self.client_config.get("keep_alive", self._keep_alive),
            )
            for chunk in stream:
                if not chunk.get("done", False):
                    content_chunk = chunk.get("message", {}).get("content", "")
                    if content_chunk:
                        yield content_chunk
                else:
                    # Optionally handle final summary/stats if present in 'chunk' when done
                    break
        except ollama.ResponseError as e:
            logger.error(
                f"Ollama API ResponseError during stream for model {self.model_name}: {e.status_code} - {e.error}",
                exc_info=True,
            )
            yield f"Error: Ollama API error ({e.status_code}) - {e.error}"
        except Exception as e:
            logger.error(f"Error streaming from Ollama model {self.model_name}: {e}", exc_info=True)
            yield f"Error: Could not stream response. {e.__class__.__name__}: {e}"

    @staticmethod
    def list_models(client_config: Optional[Dict[str, Any]] = None) -> List[str]:
        cfg = client_config or {}
        host = cfg.get("host")
        timeout = cfg.get("timeout", 30)

        client_params_from_config = {
            k: v
            for k, v in cfg.items()
            if k not in ["host", "timeout", "options", "keep_alive", "model_name"]
        }

        client_list_params: Dict[str, Any] = {"timeout": timeout}
        if host:
            client_list_params["host"] = host
        client_list_params.update(client_params_from_config)

        try:
            temp_client = ollama.Client(**client_list_params)
            models_data = temp_client.list()
            # The API returns 'model' in recent versions, but 'name' was used in older ones.
            # Let's try 'model' first, then fall back to 'name' for wider compatibility.
            all_models = [
                model_info.get("model", model_info.get("name"))
                for model_info in models_data.get("models", [])
                if model_info.get("model") or model_info.get("name")
            ]
            all_models = sorted(
                list(set(m for m in all_models if m))
            )  # Deduplicate and remove None/empty

            filtered_models = [
                model_name for model_name in all_models if model_name not in OLLAMA_MODEL_BLOCKLIST
            ]

            logger.info(
                f"Found {len(all_models)} Ollama models, displaying {len(filtered_models)} after filtering."
            )
            return filtered_models
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}", exc_info=True)
            return []
