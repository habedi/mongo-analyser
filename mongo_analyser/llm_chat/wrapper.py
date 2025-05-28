import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional

import litellm  # Ensure litellm is installed

from .interface import LLMChat

logger = logging.getLogger(__name__)

# More refined blocklists, focusing on non-chat models
MODEL_BLOCKLISTS = {
    "openai": [
        r"babbage.*",
        r"davinci.*",
        r"curie.*",
        r"ada.*",  # Older completion models
        r"dall-e.*",  # Image models
        r"text-embedding.*",
        r"*-embedding-.*",  # Embedding models
        r"tts-.*",  # Text-to-speech
        r"whisper-.*",  # Speech-to-text
        r"gpt-3\.5-turbo-instruct.*",  # Instruct models (usually not for chat UI)
        r"text-moderation-.*",  # Moderation models
    ],
    "google": [  # For Gemini API (gemini/*)
        r"models/embedding-.*",
        r"models/aqa",  # Specific non-chat models
        r".*-tts",
        r".*-transcribe",
        r".*-vision",  # General non-chat suffixes
        # Models like text-bison, chat-bison are for Vertex AI, not directly listed by gemini/* usually
    ],
    "google_vertex": [  # If you were to list Vertex AI models separately
        r"text-bison.*",
        r"chat-bison.*",
        r"embedding-gecko.*",
    ],
    "ollama": [  # Models typically used for embeddings or other non-chat tasks
        r".*embed.*",
        r"nomic-embed-text",
        r"mxbai-embed-large",
        r"snowflake-arctic-embed",
        r"all-minilm",
        r"bge-.*",
        r"distiluse-base-multilingual-cased",
        r"e5-.*",
        r"gemma:.*-text",  # some gemma are instruct
        r"sentence-transformers/.*",
        r"vision",
        r"llava",  # Vision models
    ],
    "common_suffixes_to_avoid": [  # General suffixes that often indicate non-chat models
        "-instruct",
        "-code",
        "-edit",
        "-vision",
        "-embed",
        "-tts",
        "-stt",
        "-preview",
        "-experimental",  # Often less stable or for specific uses
    ],
}


def _is_model_blocked(model_name: str, provider: Optional[str]) -> bool:
    """Checks if a model name matches any regex pattern in the blocklists."""
    block_patterns: List[str] = []
    if provider:
        block_patterns.extend(MODEL_BLOCKLISTS.get(provider.lower(), []))

    # Add common suffixes if not already covered by provider-specific list
    # (or apply them universally)
    # block_patterns.extend(MODEL_BLOCKLISTS.get("common_suffixes_to_avoid", []))

    for pattern in block_patterns:
        try:
            if re.search(pattern, model_name, re.IGNORECASE):
                logger.debug(
                    f"Model '{model_name}' blocked by pattern '{pattern}' for provider '{provider}'."
                )
                return True
        except re.error as e:
            logger.warning(f"Regex error in blocklist pattern '{pattern}': {e}")
    return False


class LiteLLMChat(LLMChat):
    def __init__(self, model_name: str, provider_hint: Optional[str] = None, **kwargs: Any):
        self.raw_model_name = model_name  # The name from the config panel (e.g., "llama3")
        self.provider_hint = (
            provider_hint.lower() if provider_hint else self._guess_provider(model_name)
        )

        # kwargs from config panel: api_key, base_url, temperature, system_prompt, max_history_messages etc.
        self.config_params = kwargs

        # Construct the fully qualified model name for LiteLLM
        # Examples: "ollama/llama2", "gpt-3.5-turbo", "gemini/gemini-pro"
        fq_model_name = self.raw_model_name
        if self.provider_hint == "ollama" and not self.raw_model_name.startswith("ollama/"):
            fq_model_name = f"ollama/{self.raw_model_name}"
        elif (
            self.provider_hint == "google"
            and not self.raw_model_name.startswith("gemini/")
            and "/" not in self.raw_model_name
        ):
            # For Google, LiteLLM often expects "gemini/model-name"
            fq_model_name = f"gemini/{self.raw_model_name}"
        # For OpenAI, model names like "gpt-3.5-turbo" are usually used directly without a prefix with LiteLLM,
        # unless it's a specific variant like "openai/gpt-4o-mini".
        # LiteLLM handles "gpt-3.5-turbo" as an OpenAI model by default.
        # If provider_hint is "openai" and model_name doesn't have "openai/", it's usually fine.

        # Other providers might require prefixes like "anthropic/claude-2"
        # This logic can be expanded based on LiteLLM's conventions for different providers.

        super().__init__(fq_model_name, **kwargs)  # Pass fq_model_name to parent

    def _guess_provider(self, model_name: str) -> Optional[str]:
        model_lower = model_name.lower()
        if model_lower.startswith("gpt-") or "openai/" in model_lower or "gpt-4" in model_lower:
            return "openai"
        if "gemini" in model_lower or "google/" in model_lower or model_lower.startswith("models/"):
            return "google"  # models/ for Vertex
        if model_lower.startswith("ollama/"):
            return "ollama"
        if "claude" in model_lower or "anthropic/" in model_lower:
            return "anthropic"
        if "mistral" in model_lower and not model_lower.startswith("ollama/"):
            if (
                "mistral/" in model_lower
                or "open-mistral" in model_lower
                or "mixtral" in model_lower
            ):
                return (
                    "mistral"  # or specific provider like "together_ai/Mixtral-8x7B-Instruct-v0.1"
                )
        if "azure/" in model_lower:
            return "azure"  # For Azure OpenAI
        # Add more guesses based on common model name patterns
        return None  # Cannot guess

    def _initialize_client(self, **kwargs: Any) -> Any:
        # LiteLLM doesn't require a persistent client object.
        # We store config params here for use in send_message/stream_message.
        self.api_key = self.config_params.get("api_key")
        self.base_url = self.config_params.get("base_url")  # For LiteLLM, this is 'api_base'
        self.temperature = float(self.config_params.get("temperature", 0.7))
        self.max_tokens = int(self.config_params.get("max_tokens", 2048))  # Output tokens
        self.system_prompt = self.config_params.get("system_prompt")
        # max_history_messages is handled by ChatView, not directly by LiteLLM call.

        # Set environment variables for API keys if provided and not already set
        # LiteLLM often picks up keys from env vars.
        if self.provider_hint == "openai" and self.api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.api_key
            logger.debug("Set OPENAI_API_KEY from config for this session.")
        if self.provider_hint == "google" and self.api_key and not os.getenv("GOOGLE_API_KEY"):
            # For Gemini API, it's often GOOGLE_API_KEY
            os.environ["GOOGLE_API_KEY"] = self.api_key
            logger.debug("Set GOOGLE_API_KEY from config for this session.")
        # Add for other providers as needed (e.g., ANTHROPIC_API_KEY)

        logger.info(
            f"LiteLLMChat configured for effective model '{self.model_name}' "
            f"(Provider hint: {self.provider_hint}, Raw name from panel: {self.raw_model_name}, "
            f"Temp: {self.temperature}, MaxOutputTokens: {self.max_tokens})"
        )
        return None  # No client object needed

    def _prepare_messages_payload(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        # History is already filtered by ChatView
    ) -> List[Dict[str, str]]:
        # History should be user/assistant turns. System prompt is handled separately.
        messages_payload: List[Dict[str, str]] = []
        if self.system_prompt:  # Add system prompt if provided
            messages_payload.append({"role": "system", "content": self.system_prompt})

        if history:  # Add formatted history (user/assistant turns)
            messages_payload.extend(history)

        messages_payload.append({"role": "user", "content": message})  # Add current user message
        return messages_payload

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        messages_payload = self._prepare_messages_payload(message, history)

        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,  # The fully qualified model name
            "messages": messages_payload,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,  # Max output tokens
        }
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.base_url:
            call_kwargs["api_base"] = self.base_url  # LiteLLM uses 'api_base'

        # Add any other relevant kwargs from self.config_params not explicitly handled
        # This allows passing custom params supported by LiteLLM or specific models.
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
                    "max_history_messages",
                ]
                and k not in call_kwargs
            ):
                call_kwargs[k] = v

        try:
            # Redact messages for logging if too long or sensitive
            logged_kwargs = {
                k: (v if k != "messages" else f"<{len(v)} messages>")
                for k, v in call_kwargs.items()
            }
            logger.debug(
                f"Calling LiteLLM completion for {self.model_name} with kwargs: {logged_kwargs}"
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
            # Raise a more specific error or return a formatted error message
            return (
                f"Error from LLM ({self.raw_model_name}): {e.__class__.__name__} - {str(e)[:100]}"
            )

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        # Streaming implementation would be similar to send_message but with stream=True
        # and yielding content from chunks. This is not used by the current ChatView.
        # If needed, implement similarly to send_message.
        logger.warning(
            "stream_message is called but not fully utilized by ChatView's current send mechanism."
        )
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
        # ... add other config_params ...
        try:
            response_stream = litellm.completion(**call_kwargs)
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LiteLLM streaming error for model {self.model_name}: {e}", exc_info=True)
            yield f"Error streaming from LLM ({self.raw_model_name}): {e.__class__.__name__} - {str(e)[:100]}"

    @staticmethod
    def list_models(
        provider: Optional[str] = None, client_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        logger.info(f"LiteLLMChat.list_models called for provider: {provider}")
        cfg = client_config or {}  # Config from panel (API keys, base_url for Ollama)

        final_model_list: List[str] = []

        if provider and provider.lower() == "ollama":
            logger.info("Attempting to list Ollama models dynamically.")
            ollama_base_url = (
                cfg.get("base_url")
                or os.getenv("OLLAMA_HOST")
                or os.getenv("LITELLM_OLLAMA_BASE_URL")
                or "http://localhost:11434"
            )
            ollama_timeout = cfg.get("timeout", 10)  # Shorter timeout for listing

            try:
                import ollama as ollama_client_lib  # Requires 'pip install ollama'

                client = ollama_client_lib.Client(host=ollama_base_url, timeout=ollama_timeout)
                models_info = client.list()  # This is a blocking call

                if models_info and "models" in models_info:
                    raw_ollama_models = [
                        m.get("model", m.get("name"))
                        for m in models_info["models"]
                        if m.get("model", m.get("name"))
                    ]
                    raw_ollama_models = sorted(list(set(raw_ollama_models)))  # Unique, sorted
                    logger.info(
                        f"Dynamically listed {len(raw_ollama_models)} unique models from Ollama host {ollama_base_url} before filtering."
                    )

                    for (
                        model_name_fq
                    ) in raw_ollama_models:  # model_name_fq is like 'llama2:latest' or 'mistral:7b'
                        base_model_name = model_name_fq.split(":")[
                            0
                        ]  # Get 'llama2' from 'llama2:latest'
                        if not _is_model_blocked(
                            base_model_name, "ollama"
                        ) and not _is_model_blocked(
                            model_name_fq, "ollama"
                        ):  # Check both base and FQ
                            final_model_list.append(base_model_name)  # Add the base name
                        else:
                            logger.debug(
                                f"Ollama dynamic model '{model_name_fq}' (base: {base_model_name}) filtered out by blocklist."
                            )
                    final_model_list = sorted(list(set(final_model_list)))  # Unique base names
                else:
                    logger.warning(
                        f"Dynamic Ollama listing from {ollama_base_url} returned no 'models' or unexpected format."
                    )
            except ImportError:
                logger.warning(
                    "Python 'ollama' package not installed. Cannot dynamically list Ollama models. Install with: pip install ollama"
                )
            except Exception as e:
                logger.error(
                    f"Failed to dynamically list Ollama models from {ollama_base_url}: {e}. Check Ollama server.",
                    exc_info=False,
                )

            if not final_model_list:  # Fallback to static list if dynamic fails or returns empty
                logger.info(
                    "Falling back to static list for Ollama models from litellm.model_list."
                )
                for m_fq in litellm.model_list:
                    if m_fq.startswith("ollama/"):
                        model_name_part = m_fq.split("/", 1)[1]
                        if not _is_model_blocked(model_name_part, "ollama"):
                            final_model_list.append(model_name_part)
                final_model_list = sorted(list(set(final_model_list)))
            logger.info(f"Prepared {len(final_model_list)} Ollama models for TUI.")
            return final_model_list

        elif provider:  # For other providers (OpenAI, Google Gemini, etc.)
            logger.info(f"Listing models for non-Ollama provider: {provider}")
            # LiteLLM's model_list contains fully qualified names like "openai/gpt-3.5-turbo" or just "gpt-3.5-turbo"
            # We need to filter based on the provider hint and then extract the base model name.

            # Set API key for the provider if in config, as some listing might need it
            provider_env_key_map = {
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",  # For Gemini
                "anthropic": "ANTHROPIC_API_KEY",
            }
            api_key_from_config = cfg.get("api_key")
            original_env_val = None
            env_key_to_set = provider_env_key_map.get(provider.lower())

            if env_key_to_set and api_key_from_config and not os.getenv(env_key_to_set):
                original_env_val = os.getenv(env_key_to_set)  # Should be None if not set
                os.environ[env_key_to_set] = api_key_from_config
                logger.debug(f"Temporarily set {env_key_to_set} for model listing.")

            try:
                # Some providers might have their own listing functions via litellm, or use litellm.get_model_list()
                # For simplicity, we'll iterate litellm.model_list and filter.
                # litellm.get_model_list(provider=provider) might be an option for some.

                for model_id_fq in (
                    litellm.model_list
                ):  # e.g., "gpt-3.5-turbo", "openai/gpt-4o", "gemini/gemini-pro"
                    model_provider_guess = ""
                    model_base_name = model_id_fq

                    if "/" in model_id_fq:
                        parts = model_id_fq.split("/", 1)
                        model_provider_guess = parts[0].lower()
                        model_base_name = parts[1]
                    else:  # No prefix, try to guess based on name structure
                        if model_id_fq.startswith("gpt-"):
                            model_provider_guess = "openai"
                        elif "gemini" in model_id_fq:
                            model_provider_guess = "google"
                        # Add more guesses if needed

                    if (
                        model_provider_guess == provider.lower()
                        or (
                            (provider.lower() == "openai" and model_provider_guess == "")
                            and model_id_fq.startswith("gpt-")
                        )
                        or (
                            (provider.lower() == "google" and model_provider_guess == "")
                            and "gemini" in model_id_fq
                        )
                    ):  # Match provider
                        if not _is_model_blocked(
                            model_base_name, provider
                        ) and not _is_model_blocked(
                            model_id_fq, provider
                        ):  # Check both base and FQ
                            final_model_list.append(model_base_name)  # Add the base name
                        else:
                            logger.debug(
                                f"Model '{model_id_fq}' (base: {model_base_name}) for provider '{provider}' filtered out by blocklist."
                            )
            finally:  # Restore env var if we changed it
                if (
                    env_key_to_set
                    and api_key_from_config
                    and os.getenv(env_key_to_set) == api_key_from_config
                ):
                    if original_env_val is None:
                        del os.environ[env_key_to_set]
                        logger.debug(f"Cleared temporarily set {env_key_to_set}.")
                    else:  # This case should not happen if we only set if not os.getenv()
                        os.environ[env_key_to_set] = original_env_val
                        logger.debug(f"Restored {env_key_to_set}.")

            final_model_list = sorted(list(set(final_model_list)))
            logger.info(
                f"Prepared {len(final_model_list)} models for provider '{provider}' for TUI."
            )
            return final_model_list

        else:  # No provider specified, and not Ollama
            logger.warning(
                "LiteLLMChat.list_models called without a specific provider (and not Ollama). Returning empty list."
            )
            return []
