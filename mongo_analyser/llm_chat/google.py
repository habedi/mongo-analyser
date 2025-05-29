import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySettingDict,
)

from .base import LLMChat

logger = logging.getLogger(__name__)

GOOGLE_MODEL_BLOCKLIST = [
    "models/gemini-1.0-pro-001",
    "models/text-bison-001",
    "models/chat-bison-001",
    "models/embedding-gecko-001",
    "models/embedding-001",
    "models/aqa",
    "models/gemini-1.5-flash-001-tuning",
    "models/gemini-1.5-flash-8b-exp-0827",
    "models/gemini-1.5-flash-8b-exp-0924",
    "models/gemini-2.5-pro-exp-03-25",
    "models/gemini-2.5-pro-preview-03-25",
    "models/gemini-2.5-flash-preview-04-17",
    "models/gemini-2.5-flash-preview-05-20",
    "models/gemini-2.5-flash-preview-04-17-thinking",
    "models/gemini-2.5-pro-preview-05-06",
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-flash-lite-preview-02-05",
    "models/gemini-2.0-flash-lite-preview",
    "models/gemini-2.0-pro-exp",
    "models/gemini-2.0-pro-exp-02-05",
    "models/gemini-exp-1206",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    "models/gemini-2.0-flash-thinking-exp",
    "models/gemini-2.0-flash-thinking-exp-1219",
    "models/gemini-2.5-flash-preview-tts",
    "models/gemini-2.5-pro-preview-tts",
    "models/learnlm-2.0-flash-experimental",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro-002",
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-flash-002",
    "models/gemini-1.5-flash-8b-001",
    "models/gemini-1.0-pro-vision-latest",
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-flash-8b-latest",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-pro-latest",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-pro-vision",
    "models/gemma-3-12b-it",
    "models/gemma-3-1b-it",
    "models/gemma-3-27b-it",
    "models/gemma-3-4b-it",
    "models/gemma-3n-e4b-it",
]

GOOGLE_MODEL_SUFFIX_BLOCKLIST = [
    "-exp",
    "-preview",
    "-tuning",
    "-thinking",
    "-tts",
    "-experimental",
]


class GoogleChat(LLMChat):
    def __init__(self, model_name: str, api_key: str = None, **kwargs: Any):
        self._effective_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self._effective_api_key:
            logger.error(
                "Google API key not provided or found in GOOGLE_API_KEY environment variable."
            )
            raise ValueError("Google API key is required for GoogleChat.")

        self._model_kwargs = kwargs.pop("model_kwargs", {})
        generation_config_input = kwargs.pop("generation_config", {})
        safety_settings_input = kwargs.pop("safety_settings", None)

        self._generation_config: Optional[GenerationConfig] = None
        if generation_config_input and isinstance(generation_config_input, dict):
            self._generation_config = GenerationConfig(**generation_config_input)

        self._safety_settings: Optional[List[SafetySettingDict]] = None
        if safety_settings_input and isinstance(safety_settings_input, list):
            processed_settings: List[SafetySettingDict] = []
            for s_config_dict in safety_settings_input:
                if "category" in s_config_dict and "threshold" in s_config_dict:
                    try:
                        category_str = str(s_config_dict["category"]).upper()
                        threshold_str = str(s_config_dict["threshold"]).upper()
                        if not hasattr(HarmCategory, category_str):
                            continue
                        if not hasattr(HarmBlockThreshold, threshold_str):
                            continue
                        category_enum = HarmCategory[category_str]
                        threshold_enum = HarmBlockThreshold[threshold_str]
                        processed_settings.append(
                            {"category": category_enum, "threshold": threshold_enum}
                        )
                    except Exception as e:
                        logger.error(
                            f"Error processing safety setting: {s_config_dict}. Exception: {e}",
                            exc_info=True,
                        )
            if processed_settings:
                self._safety_settings = processed_settings

        init_kwargs_for_super = {
            "api_key": self._effective_api_key,
            **kwargs,
            "_model_kwargs": self._model_kwargs,
            "_generation_config_input": generation_config_input,
            "_safety_settings_input": safety_settings_input,
        }

        super().__init__(model_name, **init_kwargs_for_super)

    def _initialize_client(self, **kwargs: Any) -> genai.GenerativeModel:
        current_api_key = kwargs.get("api_key", self._effective_api_key)
        model_init_kwargs = kwargs.get("_model_kwargs", self._model_kwargs)

        try:
            genai.configure(api_key=current_api_key, **kwargs.get("client_options", {}))
            model = genai.GenerativeModel(
                self.model_name,
                generation_config=self._generation_config,
                safety_settings=self._safety_settings,
                **model_init_kwargs,
            )
            logger.info(f"Google GenAI client initialized for model {self.model_name}.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}", exc_info=True)
            if "API key not valid" in str(e) or "permission" in str(e).lower():
                raise PermissionError(f"Google API key is invalid or missing permissions: {e}")
            raise ConnectionError(f"Failed to configure or initialize Google GenAI client: {e}")

    def format_history(self, history: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        formatted_history = []
        if history:
            for message in history:
                role = message.get("role")
                content = message.get("content")
                if not content:
                    continue
                api_role = "model" if role == "assistant" else "user"
                formatted_history.append({"role": api_role, "parts": [{"text": content}]})
        return formatted_history

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        google_formatted_history = self.format_history(history)
        contents_for_api = google_formatted_history + [
            {"role": "user", "parts": [{"text": message}]}
        ]
        try:
            response = self.client.generate_content(contents_for_api, stream=False)
            assistant_response = (
                "".join(part.text for part in response.parts if hasattr(part, "text"))
                if response.parts
                else (response.text if hasattr(response, "text") else "")
            )
            if not assistant_response:
                if (
                    hasattr(response, "prompt_feedback")
                    and response.prompt_feedback
                    and response.prompt_feedback.block_reason
                ):
                    return (
                        f"Error: Prompt blocked by Google. Reason:"
                        f" {response.prompt_feedback.block_reason.name}."
                    )
                if (
                    hasattr(response, "candidates")
                    and response.candidates
                    and hasattr(response.candidates[0], "finish_reason")
                ):
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason and finish_reason.name not in [
                        "STOP",
                        "UNSPECIFIED",
                        "MAX_TOKENS",
                    ]:
                        return f"Error: Response generation stopped. Reason: {finish_reason.name}."
            return assistant_response
        except Exception as e:
            logger.error(
                f"Error communicating with Google model {self.model_name}: {e}", exc_info=True
            )
            return f"Error: Could not get response from Google AI. {e.__class__.__name__}: {e}"

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        google_formatted_history = self.format_history(history)
        contents_for_api = google_formatted_history + [
            {"role": "user", "parts": [{"text": message}]}
        ]
        try:
            response_stream = self.client.generate_content(contents_for_api, stream=True)
            for chunk in response_stream:
                if (
                    hasattr(chunk, "prompt_feedback")
                    and chunk.prompt_feedback
                    and chunk.prompt_feedback.block_reason
                ):
                    yield (
                        f"Error: Prompt blocked by Google. Reason:"
                        f" {chunk.prompt_feedback.block_reason.name}."
                    )
                    return
                if chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, "text") and part.text:
                            yield part.text
                elif hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
                if (
                    hasattr(chunk, "candidates")
                    and chunk.candidates
                    and hasattr(chunk.candidates[0], "finish_reason")
                ):
                    finish_reason = chunk.candidates[0].finish_reason
                    if finish_reason and finish_reason.name not in [
                        "STOP",
                        "UNSPECIFIED",
                        "MAX_TOKENS",
                    ]:
                        if finish_reason.name == "SAFETY":
                            yield "Error: Streaming stopped due to safety filters."
                            return
        except Exception as e:
            logger.error(f"Error streaming from Google model {self.model_name}: {e}", exc_info=True)
            yield f"Error: Could not stream response. {e.__class__.__name__}: {e}"

    @staticmethod
    def list_models(client_config: Optional[Dict[str, Any]] = None) -> List[str]:
        cfg = client_config or {}
        api_key_to_use = cfg.get("api_key", os.getenv("GOOGLE_API_KEY"))
        if not api_key_to_use:
            logger.warning(
                "Cannot list Google models without API key (via client_config or GOOGLE_API_KEY)."
            )
            return []
        try:
            genai.configure(api_key=api_key_to_use, **cfg.get("client_options", {}))

            all_models_from_api = [
                m.name
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]

            filtered_models = []
            for model_name in all_models_from_api:
                if model_name in GOOGLE_MODEL_BLOCKLIST:
                    continue
                if any(suffix in model_name for suffix in GOOGLE_MODEL_SUFFIX_BLOCKLIST):
                    continue

                if "-latest" not in model_name and f"{model_name}-latest" in all_models_from_api:
                    pass

                filtered_models.append(model_name)

            filtered_models.sort()

            logger.info(
                f"Found {len(all_models_from_api)} Google models supporting generateContent,"
                f" displaying {len(filtered_models)} after filtering."
            )
            return filtered_models
        except Exception as e:
            logger.error(f"Error fetching Google models: {e}", exc_info=True)
            return []
