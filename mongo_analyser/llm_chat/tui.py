import logging
import sys
from typing import Type, List, Dict

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from .base_chat import LLMChat
from .ollama_chat import OllamaChat
from .openai_chat import OpenAIChat

# Configure logging for this module
logger = logging.getLogger(__name__)

# Basic styling for the prompt (optional)
prompt_style = Style.from_dict({
    'prompt': '#ansiblue bold',
    'user-input': '',  # Default text color
})


def select_llm_provider() -> Type[LLMChat] | None:
    """Prompts the user to select an LLM provider."""
    print("\nSelect LLM Provider:")
    print("1. Ollama")
    print("2. OpenAI")
    print("0. Cancel")

    while True:
        try:
            choice = input("Enter choice: ").strip()
            if choice == '1':
                return OllamaChat
            elif choice == '2':
                return OpenAIChat
            elif choice == '0':
                return None
            else:
                print("Invalid choice. Please enter 1, 2, or 0.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_ollama_config() -> Dict:
    """Gets configuration specific to Ollama."""
    config = {}
    host_input = input(
        "Enter Ollama host (leave blank for default, e.g., http://localhost:11434 or OLLAMA_HOST env var): ").strip()
    if host_input:
        config['host'] = host_input

    timeout_input = input("Enter timeout for Ollama requests in seconds (default: 30): ").strip()
    if timeout_input:
        try:
            config['timeout'] = int(timeout_input)
        except ValueError:
            print("Invalid timeout value, using default.")
    return config


def get_openai_config() -> Dict:
    """Gets configuration specific to OpenAI."""
    config = {}
    # API key is primarily handled by environment variable OPENAI_API_KEY
    # or can be passed if LLMChat constructor is modified further.
    # For now, we assume it's in the environment or the OpenAIChat class handles prompting if truly necessary.
    print("OpenAI API key should be set via the OPENAI_API_KEY environment variable.")

    timeout_input = input("Enter timeout for OpenAI requests in seconds (default: 30.0): ").strip()
    if timeout_input:
        try:
            config['timeout'] = float(timeout_input)
        except ValueError:
            print("Invalid timeout value, using default.")

    max_retries_input = input("Enter maximum retries for OpenAI requests (default: 2): ").strip()
    if max_retries_input:
        try:
            config['max_retries'] = int(max_retries_input)
        except ValueError:
            print("Invalid retries value, using default.")
    return config


def select_model(llm_instance: LLMChat) -> str | None:
    """Fetches and allows user to select a model from the provider."""
    print("\nFetching available models...")
    try:
        models = llm_instance.get_available_models()
        if not models:
            print("No models found for this provider or an error occurred.")
            print("You might need to pull a model first (e.g., 'ollama pull llama3').")
            manual_model = input("Enter model name manually (or press Enter to cancel): ").strip()
            return manual_model if manual_model else None

        print("Available models:")
        for i, model_name in enumerate(models):
            print(f"{i + 1}. {model_name}")
        print("0. Cancel / Enter manually")

        while True:
            try:
                choice_input = input(
                    f"Select model (1-{len(models)}, or 0 to cancel/enter manually): ").strip()
                model_choice_idx = int(choice_input)

                if model_choice_idx == 0:
                    manual_model = input(
                        "Enter model name manually (or press Enter to cancel): ").strip()
                    return manual_model if manual_model else None
                elif 1 <= model_choice_idx <= len(models):
                    return models[model_choice_idx - 1]
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(models)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    except Exception as e:
        logger.error(f"Could not retrieve models for {llm_instance.__class__.__name__}: {e}",
                     exc_info=True)
        print(f"Error fetching models: {e}")
        manual_model = input(
            "Could not fetch models. Enter model name manually (or press Enter to cancel): ").strip()
        return manual_model if manual_model else None


def run_chat_session(chat_instance: LLMChat):
    """Runs the interactive chat session with the selected LLM."""
    print(f"\nChatting with {chat_instance.model_name} via {chat_instance.__class__.__name__}.")
    print("Type '/exit' or '/quit' to end the session.")
    print("Type '/history' to view current session history.")
    print("---")

    # Create a unique history file per model if desired, or a general one
    history_file = f".chat_history_{chat_instance.__class__.__name__}_{chat_instance.model_name}.txt"
    history_file = history_file.replace(":", "_")  # Sanitize for filename
    session = PromptSession(history=FileHistory(history_file),
                            auto_suggest=AutoSuggestFromHistory())

    chat_history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = session.prompt("You: ", style=prompt_style).strip()

            if user_input.lower() in ['/exit', '/quit']:
                print("Exiting chat session.")
                break

            if user_input.lower() == '/history':
                if not chat_history:
                    print("Chat history is empty.")
                else:
                    print("\n--- Chat History ---")
                    for entry in chat_history:
                        print(f"{entry['role'].capitalize()}: {entry['content']}")
                    print("--- End of History ---")
                continue

            if not user_input:  # Handle empty input
                continue

            # Add user message to history *before* sending, so LLM has context
            # but pass history *without* current input for the API call for some models
            current_turn_user_message = {"role": "user", "content": user_input}

            api_history = chat_history.copy()  # History up to the previous turn

            print(f"{chat_instance.model_name}: ", end="", flush=True)

            full_response = ""
            try:
                for chunk in chat_instance.stream_message(user_input, history=api_history):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # Newline after streaming is done
            except Exception as e:
                logger.error(f"Error during streaming with {chat_instance.model_name}: {e}",
                             exc_info=True)
                print(f"\nError during streaming: {e}")
                # Optionally, decide if you want to skip adding this failed turn to history
                continue

            chat_history.append(current_turn_user_message)  # Add user message for this turn
            chat_history.append(
                {"role": "assistant", "content": full_response})  # Add assistant response

        except KeyboardInterrupt:
            print("\nExiting chat session (KeyboardInterrupt).")
            break
        except EOFError:  # Handle Ctrl+D
            print("\nExiting chat session (EOF).")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in chat session: {e}", exc_info=True)
            print(f"\nAn unexpected error occurred: {e}")
            break


def main_tui():
    """Main function to drive the TUI for LLM chat selection and interaction."""
    print("Welcome to the LLM Chat Interface!")

    llm_provider_class = select_llm_provider()
    if not llm_provider_class:
        print("No provider selected. Exiting.")
        return

    config_params = {}
    temp_instance_for_listing = None

    try:
        if llm_provider_class == OllamaChat:
            config_params = get_ollama_config()
            # Ollama client for listing doesn't strictly need a real model name if host is set
            temp_instance_for_listing = llm_provider_class(model_name="dummy-for-listing",
                                                           **config_params)
        elif llm_provider_class == OpenAIChat:
            config_params = get_openai_config()
            # OpenAI client for listing needs to be initialized (which checks API key)
            # A common, often available model can be used as a placeholder if required by __init__
            temp_instance_for_listing = llm_provider_class(model_name="gpt-3.5-turbo",
                                                           **config_params)
        else:
            print(f"Provider {llm_provider_class.__name__} setup not fully implemented in TUI.")
            return
    except (ValueError, ConnectionError, PermissionError) as e:
        logger.error(f"Configuration or Connection Error for {llm_provider_class.__name__}: {e}",
                     exc_info=True)
        print(f"Error: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error initializing client for model listing: {e}", exc_info=True)
        print(f"An unexpected error occurred during provider setup: {e}")
        return

    selected_model_name = select_model(temp_instance_for_listing)
    if not selected_model_name:
        print("No model selected. Exiting.")
        return

    try:
        # Re-initialize with the chosen model and potentially updated/specific config
        chat_instance = llm_provider_class(model_name=selected_model_name, **config_params)
        run_chat_session(chat_instance)
    except (ValueError, ConnectionError, PermissionError) as e:
        logger.error(f"Error initializing chat with model {selected_model_name}: {e}",
                     exc_info=True)
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error initializing chat instance: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # This allows you to run the TUI directly for testing:
    # python -m mongo_analyser.llm_chat.tui

    # Basic logging setup for direct execution
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbose output from clients
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout for TUI visibility
        ]
    )
    main_tui()
