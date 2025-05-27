import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional

import litellm

from .interface import LLMChat

logger = logging.getLogger(__name__)

MODEL_BLOCKLISTS = {
    "openai": [
        "babbage",
        "davinci",
        "curie",
        "ada",
        "babbage-002",
        "dall-e-2",
        "dall-e-3",
        "davinci-002",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-instruct-0914",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "tts-1",
        "tts-1-hd",
        "whisper-1",
    ],
    "google": [
        "models/text-bison-001",
        "models/chat-bison-001",
        "models/embedding-gecko-001",
        "models/embedding-001",
        "models/aqa",
    ],
    "google_suffix_blocklist": [
        "-exp",
        "-preview",
        "-tuning",
        "-thinking",
        "-tts",
        "-experimental",
        "-transcribe",
    ],
    "ollama": [
        "nomic-embed-text",
        "mxbai-embed-large",
        "snowflake-arctic-embed",
        "granite-embedding",
    ],
}


def _ends_with_hyphen_numbers(model_name: str) -> bool:
    name_part = model_name.split(":")[0]
    match = re.search(r"-\d+$", name_part)
    return bool(match)


class LiteLLMChat(LLMChat):
    def __init__(self, model_name: str, provider_hint: Optional[str] = None, **kwargs: Any):
        self.raw_model_name = model_name
        self.provider_hint = (
            provider_hint.lower() if provider_hint else self._guess_provider(model_name)
        )
        self.config_params = (
            kwargs  # Includes temperature, max_tokens, api_key, base_url, system_prompt
        )

        fq_model_name = self.raw_model_name
        if self.provider_hint == "ollama" and not self.raw_model_name.startswith("ollama/"):
            fq_model_name = f"ollama/{self.raw_model_name}"
        elif (
            self.provider_hint == "google"
            and not self.raw_model_name.startswith("gemini/")
            and "/" not in self.raw_model_name
        ):
            fq_model_name = f"gemini/{self.raw_model_name}"
        elif (
            self.provider_hint
            and self.provider_hint not in ["ollama", "google"]
            and "/" not in self.raw_model_name
        ):
            if self.provider_hint in [
                "openai",
                "azure",
                "anthropic",
                "mistral",
                "cohere",
            ]:
                fq_model_name = f"{self.provider_hint}/{self.raw_model_name}"

        super().__init__(fq_model_name, **kwargs)

    def _guess_provider(self, model_name: str) -> Optional[str]:
        model_lower = model_name.lower()
        if model_lower.startswith("gpt-") or "openai/" in model_lower:
            return "openai"
        if (
            model_lower.startswith("gemini/")
            or "gemini" in model_lower
            or "google/" in model_lower
            or model_lower.startswith("models/")
        ):
            return "google"
        if model_lower.startswith("ollama/"):
            return "ollama"
        if "claude" in model_lower or "anthropic/" in model_lower:
            return "anthropic"
        if "mistral" in model_lower and not model_lower.startswith("ollama/"):
            if "mistral/" in model_lower:
                return "mistral"
        if "azure/" in model_lower:
            return "azure"
        return None

    def _initialize_client(self, **kwargs: Any) -> Any:
        self.api_key = self.config_params.get("api_key")
        self.base_url = self.config_params.get("base_url")  # For LiteLLM, this is often api_base
        self.temperature = float(self.config_params.get("temperature", 0.7))
        self.max_tokens = int(self.config_params.get("max_tokens", 2048))
        self.system_prompt = self.config_params.get("system_prompt")

        if self.provider_hint == "openai" and self.api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.api_key
            logger.debug("Set OPENAI_API_KEY from config panel for this session.")
        if self.provider_hint == "google" and self.api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = self.api_key
            logger.debug("Set GOOGLE_API_KEY from config panel for this session.")

        logger.info(
            f"LiteLLMChat configured for model '{self.model_name}' (Provider hint: {self.provider_hint}, Raw name from panel: {self.raw_model_name})"
        )
        return (
            None  # LiteLLM doesn't require a persistent client object in the same way some SDKs do
        )

    def _prepare_messages_payload(
        self, message: str, history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        formatted_history = self.format_history(history)  # user/assistant turns
        messages_payload = []
        if self.system_prompt:
            messages_payload.append({"role": "system", "content": self.system_prompt})
        messages_payload.extend(formatted_history)
        messages_payload.append({"role": "user", "content": message})
        return messages_payload

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        messages_payload = self._prepare_messages_payload(message, history)

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages_payload,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.base_url:  # LiteLLM uses 'api_base' for custom endpoints
            call_kwargs["api_base"] = self.base_url

        # Add any other relevant kwargs from self.config_params not already handled
        for k, v in self.config_params.items():
            if (
                k
                not in [
                    "model_name",
                    "provider_hint",
                    "api_key",
                    "base_url",
                    "temperature",
                    "max_tokens",
                    "system_prompt",
                ]
                and k not in call_kwargs
            ):
                call_kwargs[k] = v

        try:
            logger.debug(
                f"Calling LiteLLM completion for {self.model_name} with relevant kwargs: { {k: v for k, v in call_kwargs.items() if k != 'messages'} }"
            )
            response = litellm.completion(**call_kwargs)
            assistant_response = ""
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                assistant_response = response.choices[0].message.content
            return assistant_response.strip()
        except Exception as e:
            logger.error(
                f"LiteLLM completion error for model {self.model_name}: {e}", exc_info=True
            )
            return f"Error from LLM ({self.model_name}): {e.__class__.__name__} - {e!s}"

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        messages_payload = self._prepare_messages_payload(message, history)

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages_payload,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.base_url:
            call_kwargs["api_base"] = self.base_url

        for k, v in self.config_params.items():
            if (
                k
                not in [
                    "model_name",
                    "provider_hint",
                    "api_key",
                    "base_url",
                    "temperature",
                    "max_tokens",
                    "system_prompt",
                    "stream",
                ]
                and k not in call_kwargs
            ):
                call_kwargs[k] = v
        try:
            logger.debug(
                f"Calling LiteLLM streaming completion for {self.model_name} with relevant kwargs: { {k: v for k, v in call_kwargs.items() if k != 'messages'} }"
            )
            response_stream = litellm.completion(**call_kwargs)
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    yield content_chunk
        except Exception as e:
            logger.error(f"LiteLLM streaming error for model {self.model_name}: {e}", exc_info=True)
            yield f"Error streaming from LLM ({self.model_name}): {e.__class__.__name__} - {e!s}"

    @staticmethod
    def list_models(
        provider: Optional[str] = None, client_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        logger.info(f"LiteLLMChat.list_models called for provider: {provider}")
        cfg = client_config or {}
        all_litellm_models_available = litellm.model_list
        logger.debug(
            f"LiteLLM.model_list contains {len(all_litellm_models_available)} initial entries."
        )

        ollama_timeout = cfg.get("timeout", 30)  # Default timeout for Ollama client list
        common_suffix_blocklist = MODEL_BLOCKLISTS.get("google_suffix_blocklist", [])

        if provider and provider.lower() == "ollama":
            logger.info("Attempting to list Ollama models.")
            # Use base_url from client_config if provided for Ollama listing
            ollama_base_url_from_config = cfg.get("base_url")
            ollama_host_env = os.getenv("OLLAMA_HOST")
            litellm_ollama_base_url_env = os.getenv("LITELLM_OLLAMA_BASE_URL")

            ollama_base_url_to_use = (
                ollama_base_url_from_config or ollama_host_env or litellm_ollama_base_url_env
            )

            if not ollama_base_url_to_use:
                ollama_base_url_to_use = "http://localhost:11434"
                logger.info(
                    f"No Ollama host/base_url configured or found in env, defaulting to: {ollama_base_url_to_use} for dynamic listing."
                )

            if ollama_base_url_to_use:
                logger.info(
                    f"Attempting to dynamically list Ollama models from: {ollama_base_url_to_use} with timeout {ollama_timeout}s"
                )
                try:
                    import ollama as ollama_client_lib

                    client_params = {"host": ollama_base_url_to_use, "timeout": ollama_timeout}
                    client = ollama_client_lib.Client(**client_params)
                    models_info = client.list()  # This is a blocking call
                    logger.debug(f"Ollama dynamic list raw response: {models_info}")

                    if not models_info or "models" not in models_info:
                        logger.warning(
                            f"Dynamic Ollama listing from {ollama_base_url_to_use} returned no 'models' key or unexpected format."
                        )
                        ollama_models_raw = []
                    else:
                        ollama_models_raw = []
                        for model_data in models_info.get("models", []):
                            model_name = model_data.get("model", model_data.get("name"))
                            if model_name:
                                ollama_models_raw.append(model_name)

                        ollama_models_raw = sorted(list(set(ollama_models_raw)))
                        logger.info(
                            f"Dynamically listed {len(ollama_models_raw)} unique models from Ollama host {ollama_base_url_to_use} before filtering."
                        )
                        logger.debug(f"Raw dynamic Ollama models: {ollama_models_raw}")

                    specific_ollama_blocklist = MODEL_BLOCKLISTS.get("ollama", [])
                    filtered_ollama_models = []
                    for m_name in ollama_models_raw:
                        passes_blocklist = not any(
                            blocked_name in m_name for blocked_name in specific_ollama_blocklist
                        )
                        passes_hyphen_check = m_name.count("-") < 3
                        passes_num_suffix_check = not _ends_with_hyphen_numbers(m_name)
                        passes_common_suffix = not any(
                            suffix in m_name for suffix in common_suffix_blocklist
                        )

                        if (
                            passes_blocklist
                            and passes_hyphen_check
                            and passes_num_suffix_check
                            and passes_common_suffix
                        ):
                            filtered_ollama_models.append(m_name)
                        else:
                            logger.debug(
                                f"Ollama dynamic model '{m_name}' filtered out. Blocklist: {not passes_blocklist}, Hyphen: {not passes_hyphen_check}, NumSuffix: {not passes_num_suffix_check}, CommonSuffix: {not passes_common_suffix}"
                            )

                    logger.info(
                        f"Dynamically found {len(ollama_models_raw)} models, filtered to {len(filtered_ollama_models)} Ollama models."
                    )
                    return filtered_ollama_models
                except ImportError:
                    logger.warning(
                        "Python 'ollama' package not installed. Falling back to static list for Ollama models. "
                        "Install with: pip install ollama"
                    )
                except (
                    Exception
                ) as e:  # Catch broader exceptions from client.list() like connection errors
                    logger.error(
                        f"Failed to dynamically list Ollama models from {ollama_base_url_to_use}: {e}. Falling back to static list.",
                        exc_info=False,
                    )

            logger.info("Executing Ollama static list fallback.")
            ollama_specific_blocklist = MODEL_BLOCKLISTS.get("ollama", [])
            ollama_models_from_static_list = []
            for m_fq in all_litellm_models_available:
                if m_fq.startswith("ollama/"):
                    model_name_part = m_fq.split("/", 1)[1]
                    logger.debug(
                        f"Considering static Ollama model: {m_fq} (name part: {model_name_part})"
                    )

                    passes_blocklist = not any(
                        blocked_name in model_name_part
                        for blocked_name in ollama_specific_blocklist
                    )
                    passes_hyphen_check = model_name_part.count("-") < 3
                    passes_num_suffix_check = not _ends_with_hyphen_numbers(model_name_part)
                    passes_common_suffix = not any(
                        suffix in model_name_part for suffix in common_suffix_blocklist
                    )

                    if (
                        passes_blocklist
                        and passes_hyphen_check
                        and passes_num_suffix_check
                        and passes_common_suffix
                    ):
                        ollama_models_from_static_list.append(model_name_part)
                    else:
                        logger.debug(
                            f"Static Ollama model '{model_name_part}' from '{m_fq}' filtered out. Blocklist: {not passes_blocklist}, Hyphen: {not passes_hyphen_check}, NumSuffix: {not passes_num_suffix_check}, CommonSuffix: {not passes_common_suffix}"
                        )

            logger.info(
                f"Using static list fallback: Found {len(ollama_models_from_static_list)} 'ollama/' prefixed models from litellm.model_list after all filters."
            )
            return sorted(list(set(ollama_models_from_static_list)))

        processed_models = []
        if provider:
            provider_lower = provider.lower()
            logger.info(f"Listing models for non-Ollama provider: {provider_lower}")
            specific_provider_blocklist = MODEL_BLOCKLISTS.get(provider_lower, [])

            for model_id_fq in all_litellm_models_available:
                logger.debug(f"Considering model from litellm.model_list: {model_id_fq}")
                try:
                    parts = model_id_fq.split("/", 1)
                    model_actual_provider = ""
                    model_base_name = model_id_fq

                    if len(parts) == 2:
                        model_actual_provider = parts[0].lower()
                        model_base_name = parts[1]
                    elif provider_lower == "openai" and (
                        model_id_fq.startswith("gpt-") or model_id_fq.startswith("ft:gpt-")
                    ):
                        model_actual_provider = "openai"
                    elif provider_lower == "google" and (
                        "gemini" in model_id_fq or model_id_fq.startswith("models/")
                    ):
                        model_actual_provider = "google"

                    if model_actual_provider == provider_lower:
                        logger.debug(
                            f"Matched provider for model '{model_id_fq}'. Base name: '{model_base_name}'. Applying filters."
                        )
                        passes_blocklist = model_base_name not in specific_provider_blocklist
                        passes_common_suffix = not any(
                            suffix in model_base_name for suffix in common_suffix_blocklist
                        )
                        passes_hyphen_check = model_base_name.count("-") < 3
                        passes_num_suffix_check = not _ends_with_hyphen_numbers(model_base_name)

                        if (
                            passes_blocklist
                            and passes_common_suffix
                            and passes_hyphen_check
                            and passes_num_suffix_check
                        ):
                            processed_models.append(model_base_name)
                        else:
                            logger.debug(
                                f"Model '{model_base_name}' for provider '{provider_lower}' filtered out. Blocklist: {not passes_blocklist}, CommonSuffix: {not passes_common_suffix}, Hyphen: {not passes_hyphen_check}, NumSuffix: {not passes_num_suffix_check}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Could not process model_id '{model_id_fq}' for provider '{provider_lower}': {e}"
                    )
        else:
            logger.warning(
                "LiteLLMChat.list_models called without provider (and not Ollama), returning empty list."
            )
            return []

        final_list = sorted(list(set(processed_models)))
        if provider and provider.lower() != "ollama":
            logger.info(
                f"LiteLLM: Prepared {len(final_list)} models for provider '{provider}' (non-Ollama path) for TUI after all filters. Final list: {final_list}"
            )
        return final_list
