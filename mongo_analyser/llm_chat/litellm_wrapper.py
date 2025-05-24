import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import litellm

from .base import LLMChat

logger = logging.getLogger(__name__)

# Centralized blocklists (examples, expand as needed from your original files)
MODEL_BLOCKLISTS = {
    "openai": [
        "babbage",
        "davinci",
        "curie",
        "ada",  # Generic older models
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
        # Add specific gpt-4 vision previews, etc., if desired
    ],
    "google": [  # These are often partial names or old names from your list
        "models/text-bison-001",
        "models/chat-bison-001",
        "models/embedding-gecko-001",
        "models/embedding-001",
        "models/aqa",
        # Add specific full model IDs from your previous google.py blocklist
        # Example: "gemini/gemini-1.0-pro-001" (if that's the LiteLLM name)
    ],
    "google_suffix_blocklist": [
        "-exp",
        "-preview",
        "-tuning",
        "-thinking",
        "-tts",
        "-experimental",
    ],
    "ollama": ["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed"],
    # common embedding models
}


class LiteLLMChat(LLMChat):
    def __init__(self, model_name: str, provider_hint: Optional[str] = None, **kwargs: Any):
        self.raw_model_name = model_name
        self.provider_hint = (
            provider_hint.lower() if provider_hint else self._guess_provider(model_name)
        )
        self.config_params = kwargs

        fq_model_name = self.raw_model_name
        if self.provider_hint == "ollama" and not self.raw_model_name.startswith("ollama/"):
            fq_model_name = f"ollama/{self.raw_model_name}"
        elif self.provider_hint == "google" and not self.raw_model_name.startswith("gemini/"):
            # Check if it's a known Gemini model that doesn't need prefix for litellm
            # For instance, "gemini-pro" might be globally recognized by litellm
            # However, being explicit with "gemini/" is safer for many cases.
            # If raw_model_name is like "gemini-1.5-pro-latest", litellm usually handles it.
            # If it's just "gemini-pro", prefixing helps.
            if "/" not in self.raw_model_name:  # Avoid gemini/gemini/pro
                fq_model_name = f"gemini/{self.raw_model_name}"

        super().__init__(fq_model_name, **kwargs)

    def _guess_provider(self, model_name: str) -> Optional[str]:
        model_lower = model_name.lower()
        if model_lower.startswith("gpt-"):
            return "openai"
        if model_lower.startswith("gemini/") or "gemini" in model_lower:
            return "google"
        if model_lower.startswith("ollama/"):
            return "ollama"
        # Add more heuristics if needed
        return None

    def _initialize_client(self, **kwargs: Any) -> Any:
        self.api_key = self.config_params.get("api_key")
        self.base_url = self.config_params.get("base_url")
        self.temperature = float(self.config_params.get("temperature", 0.7))
        self.max_tokens = int(self.config_params.get("max_tokens", 2048))

        # For debugging LiteLLM calls
        # litellm.set_verbose = True
        # litellm.set_verbose_api_keys=True

        # Set global API keys if provided, otherwise LiteLLM uses environment variables
        # Note: This global setting might not be ideal if multiple instances need different keys
        # It's often better to pass api_key and api_base to each litellm.completion call.
        if self.provider_hint == "openai" and self.api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.provider_hint == "google" and self.api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = self.api_key  # Or specific key for Vertex, etc.

        logger.info(
            f"LiteLLMChat configured for model '{self.model_name}' (Provider hint: {self.provider_hint})"
        )
        return None

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        formatted_history = self.format_history(history)
        messages_payload = formatted_history + [{"role": "user", "content": message}]

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages_payload,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Pass API key and base URL per call if available from instance config
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.base_url:
            call_kwargs["api_base"] = self.base_url

        # Allow overriding other litellm params from config_params
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
                ]
                and k not in call_kwargs
            ):
                call_kwargs[k] = v

        try:
            logger.debug(
                f"Calling LiteLLM completion for {self.model_name} with kwargs: { {k: v for k, v in call_kwargs.items() if k != 'messages'} }"
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
        formatted_history = self.format_history(history)
        messages_payload = formatted_history + [{"role": "user", "content": message}]

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
                    "stream",
                ]
                and k not in call_kwargs
            ):
                call_kwargs[k] = v

        try:
            logger.debug(
                f"Calling LiteLLM streaming completion for {self.model_name} with kwargs: { {k: v for k, v in call_kwargs.items() if k != 'messages'} }"
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
        # client_config could hold api_key/base_url for dynamic listing attempts
        cfg = client_config or {}
        all_litellm_models_available = litellm.model_list  # This is litellm.utils.model_list

        # Try dynamic Ollama listing if provider is ollama
        if provider and provider.lower() == "ollama":
            try:
                # LiteLLM can use OLLAMA_HOST env var, or you can pass base_url for a specific call
                # litellm.get_ollama_models() is a utility that might exist or can be built
                # For simplicity, using litellm.completion to list models from a ollama proxy
                # A more direct approach:
                ollama_base_url = cfg.get(
                    "base_url", os.getenv("OLLAMA_HOST", os.getenv("LITELLM_OLLAMA_BASE_URL"))
                )
                if ollama_base_url:  # if a host is configured
                    logger.info(
                        f"Attempting to dynamically list Ollama models from: {ollama_base_url}"
                    )
                    # This requires ollama python package or direct http requests
                    try:
                        import ollama as ollama_client_lib

                        client = ollama_client_lib.Client(host=ollama_base_url)
                        models_info = client.list()
                        ollama_models = sorted([m["name"] for m in models_info.get("models", [])])
                        # Remove common embedding models
                        blocklist = MODEL_BLOCKLISTS.get("ollama", [])
                        filtered_ollama_models = [
                            m
                            for m in ollama_models
                            if not any(blocked in m for blocked in blocklist)
                        ]
                        logger.info(
                            f"Dynamically found {len(filtered_ollama_models)} Ollama models."
                        )
                        return filtered_ollama_models
                    except ImportError:
                        logger.warning(
                            "Python 'ollama' package not installed. Falling back to static list for Ollama models."
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to dynamically list Ollama models: {e}. Falling back to static list."
                        )
                # Fallback for Ollama if dynamic listing fails or no host given:
                ollama_models_from_static_list = [
                    m.split("/", 1)[1]
                    for m in all_litellm_models_available
                    if m.startswith("ollama/")
                    and m.split("/", 1)[1] not in MODEL_BLOCKLISTS.get("ollama", [])
                ]
                return sorted(list(set(ollama_models_from_static_list)))

            except Exception as e:
                logger.error(f"Error in Ollama model listing logic: {e}")
                # Fall through to generic filtering if dynamic fails badly

        processed_models = []
        if provider:
            provider_lower = provider.lower()
            blocklist = MODEL_BLOCKLISTS.get(provider_lower, [])
            suffix_blocklist = MODEL_BLOCKLISTS.get(
                f"{provider_lower}_suffix_blocklist", []
            )  # Adjusted key name

            for (
                model_id_fq
            ) in all_litellm_models_available:  # FQ = Fully Qualified, e.g. "gemini/gemini-pro"
                try:
                    # Determine the provider and base name from the FQ name
                    # LiteLLM's model_info can be slow if called for every model.
                    # Heuristic based on prefix is faster for a static list.

                    current_model_provider = ""
                    display_name = model_id_fq

                    if model_id_fq.startswith("gpt-") or model_id_fq.startswith("ft:gpt-"):
                        current_model_provider = "openai"
                    elif model_id_fq.startswith("gemini/"):
                        current_model_provider = "google"
                        display_name = model_id_fq.split("/", 1)[1]
                    # Add other known prefixes: "anthropic/", "mistral/", "azure/", etc.

                    if current_model_provider == provider_lower:
                        if display_name not in blocklist and not any(
                            suffix in display_name for suffix in suffix_blocklist
                        ):
                            processed_models.append(display_name)
                except Exception as e:
                    logger.debug(
                        f"Could not process model_id '{model_id_fq}' for provider '{provider_lower}': {e}"
                    )
        else:  # List all, but this is generally too broad for the TUI
            logger.warning(
                "LiteLLMChat.list_models called without provider hint, may return many models."
            )
            # Return a subset or curated list if no provider is given, or empty as per original
            return []

        final_list = sorted(list(set(processed_models)))  # Deduplicate and sort
        logger.info(
            f"LiteLLM: Prepared {len(final_list)} models for provider '{provider}' for TUI."
        )
        return final_list
