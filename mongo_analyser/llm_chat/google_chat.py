import logging
import os
from typing import List, Dict, Any, Iterator

import google.generativeai as genai

from .base_chat import LLMChat

logger = logging.getLogger(__name__)


class GoogleChat(LLMChat):

    def __init__(self, model_name: str, api_key: str = None, **kwargs: Any):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error(
                "Google API key not provided or found in GOOGLE_API_KEY environment variable.")
            raise ValueError("Google API key is required.")

        self._model_kwargs = kwargs.pop('model_kwargs', {})
        self._generation_config_kwargs = kwargs.pop('generation_config', {})
        self._safety_settings_kwargs = kwargs.pop('safety_settings', None)

        super().__init__(model_name, **kwargs)

    def _initialize_client(self, **kwargs: Any) -> genai.GenerativeModel:
        try:
            genai.configure(api_key=self.api_key, **kwargs)
            model = genai.GenerativeModel(self.model_name, **self._model_kwargs)
            logger.info(f"Google GenAI client initialized for model {self.model_name}.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}", exc_info=True)
            if "API key not valid" in str(e) or "permission" in str(e).lower():
                raise PermissionError(f"Google API key is invalid or missing permissions: {e}")
            raise ConnectionError(f"Failed to configure or initialize Google GenAI client: {e}")

    def format_history(self, history: List[Dict[str, str]] = None) -> List[
        Dict[str, Any]]:  # Return type changed
        """
        Formats chat history for the Google Generative AI API.
        Google expects roles 'user' and 'model', and content under a 'parts' key.
        """
        formatted_history = []
        if history:
            for message in history:
                role = message.get("role")
                content = message.get("content")
                if not content:  # Skip empty content messages
                    continue

                api_role = ""
                if role == "assistant":
                    api_role = "model"
                elif role == "user":
                    api_role = "user"
                else:
                    logger.warning(f"Unknown role '{role}' in history, skipping.")
                    continue

                # Ensure content is wrapped in 'parts': [{'text': ...}]
                formatted_history.append({"role": api_role, "parts": [{"text": content}]})
        return formatted_history

    def send_message(self, message: str, history: List[Dict[str, str]] = None) -> str:
        # For send_message, we convert generic history to Google's format for start_chat
        # start_chat can take history with role 'user'/'model' and content directly,
        # but it's safer to provide it in the full Content object structure if it's complex.
        # The genai.GenerativeModel.start_chat method is quite flexible.
        # We will use the newly corrected format_history here too for consistency,
        # as start_chat also accepts a list of Content objects.
        google_formatted_history = self.format_history(history)

        chat_session = self.client.start_chat(history=google_formatted_history)

        try:
            logger.debug(
                f"Sending message to Google model {self.model_name}. History length: {len(google_formatted_history)}")
            response = chat_session.send_message(
                message,  # send_message takes a simple string for the new message
                generation_config=genai.types.GenerationConfig(
                    **self._generation_config_kwargs) if self._generation_config_kwargs else None,
                safety_settings=self._safety_settings_kwargs
            )
            assistant_response = "".join(part.text for part in response.parts if
                                         hasattr(part, 'text')) if response.parts else response.text
            logger.debug(f"Received response from Google model {self.model_name}.")
            return assistant_response
        except Exception as e:
            logger.error(f"Error communicating with Google model {self.model_name}: {e}",
                         exc_info=True)
            if isinstance(e, genai.types.BlockedPromptException):
                return "Error: The prompt was blocked by Google's safety filters."
            if isinstance(e, genai.types.StopCandidateException):
                return "Error: Response generation stopped unexpectedly."
            return f"Error: Could not get a response from Google AI. {e.__class__.__name__}: {e}"

    def stream_message(self, message: str, history: List[Dict[str, str]] = None) -> Iterator[str]:
        # Use the corrected format_history
        google_formatted_history = self.format_history(history)

        # The new user message also needs to be in the {"role": "user", "parts": [{"text": message}]} structure
        current_user_message_for_api = {"role": "user", "parts": [{"text": message}]}
        contents_for_api = google_formatted_history + [current_user_message_for_api]

        try:
            logger.debug(
                f"Streaming message to Google model {self.model_name}. Full contents for API: {contents_for_api}")
            response_stream = self.client.generate_content(
                contents_for_api,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    **self._generation_config_kwargs) if self._generation_config_kwargs else None,
                safety_settings=self._safety_settings_kwargs
            )
            for chunk in response_stream:
                if chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            yield part.text
                elif hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
            logger.debug(f"Google stream finished for model {self.model_name}")
        except Exception as e:
            logger.error(f"Error streaming from Google model {self.model_name}: {e}", exc_info=True)
            if isinstance(e, genai.types.BlockedPromptException):
                yield "Error: The prompt was blocked by Google's safety filters during streaming."
            elif isinstance(e, genai.types.StopCandidateException):
                yield "Error: Streaming stopped unexpectedly due to safety or other reasons."
            else:
                yield f"Error: Could not stream response from Google AI. {e.__class__.__name__}: {e}"

    def get_available_models(self) -> List[str]:
        try:
            logger.debug("Fetching available Google GenAI models.")
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            logger.info(f"Found {len(available_models)} Google models supporting generateContent.")
            return available_models
        except Exception as e:
            logger.error(f"Error fetching Google models: {e}", exc_info=True)
            return []
