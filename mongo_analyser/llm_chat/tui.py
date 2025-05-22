import logging
import sys
from typing import Dict, List, Type

from prompt_toolkit import PromptSession, print_formatted_text, HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from .base_chat import LLMChat
from .google_chat import GoogleChat
from .ollama_chat import OllamaChat
from .openai_chat import OpenAIChat

# Configure logging for this module
logger = logging.getLogger(__name__)

# Enhanced styling for the TUI
tui_style = Style.from_dict(
    {
        "prompt": "ansicyan bold",  # For "Enter choice:", etc.
        "user-prompt": "ansiblue bold",  # Specifically for "You: "
        "user-input": "",  # For user's typed text (controlled by PromptSession)
        "menu-item-index": "ansigreen",  # For "1.", "2."
        "menu-item-text": "ansigreen",  # For "Ollama", "OpenAI"
        "info": "ansigray",  # For informational messages
        "error": "ansired bold",  # For error messages
        "ai-name": "ansimagenta bold",  # For "ModelName:"
        "ai-response": "ansiyellow",  # For the AI's text
        "history-role-user": "ansiblue bold",
        "history-role-assistant": "ansimagenta bold",
        "history-content": "",  # Default color for history content
        "welcome": "ansibrightblue bold underline",
        "status": "ansibrightblack",  # For "Fetching models..."
        "cancel": "ansiyellow",
        "command": "ansibrightcyan",  # For '/exit', '/history'
    }
)


def print_html(text: str):
    """Helper function to print HTML formatted text with the TUI style."""
    print_formatted_text(HTML(text), style=tui_style)


def select_llm_provider() -> Type[LLMChat] | None:
    """Prompts the user to select an LLM provider with colors."""
    print_html("\n<menu-prompt>Select LLM Provider:</menu-prompt>")
    print_html("<menu-item-index>1.</menu-item-index> <menu-item-text>Ollama</menu-item-text>")
    print_html("<menu-item-index>2.</menu-item-index> <menu-item-text>OpenAI</menu-item-text>")
    print_html("<menu-item-index>3.</menu-item-index> <menu-item-text>Google</menu-item-text>")
    print_html("<menu-item-index>0.</menu-item-index> <cancel>Cancel</cancel>")

    while True:
        try:
            choice = input_styled("Enter choice: ").strip()
            if choice == "1":
                return OllamaChat
            elif choice == "2":
                return OpenAIChat
            elif choice == "3":
                return GoogleChat
            elif choice == "0":
                return None
            else:
                print_html("<error>Invalid choice. Please enter 1, 2, 3, or 0.</error>")
        except ValueError:  # Should not happen with input_styled returning str
            print_html("<error>Invalid input. Please enter a number.</error>")
        except (KeyboardInterrupt, EOFError):
            print_html("\n<info>Selection cancelled.</info>")
            return None


def input_styled(message: str) -> str:
    """Gets input from the user with a styled prompt."""
    # PromptSession is not strictly needed for simple input, but can be used for consistency
    # For simple input(), we can't directly style the input() prompt itself with prompt_toolkit's print_formatted_text easily
    # So we print the styled prompt first, then take raw input.
    print_formatted_text(HTML(f"<prompt>{message}</prompt>"), end=" ", style=tui_style)
    return input()


def get_ollama_config() -> Dict:
    config = {}
    host_input = input_styled(
        "Enter Ollama host (leave blank for default, e.g., http://localhost:11434 or OLLAMA_HOST env var): "
    ).strip()
    if host_input:
        config["host"] = host_input
    timeout_input = input_styled(
        "Enter timeout for Ollama requests in seconds (default: 30): ").strip()
    if timeout_input:
        try:
            config["timeout"] = int(timeout_input)
        except ValueError:
            print_html("<error>Invalid timeout value, using default.</error>")
    return config


def get_openai_config() -> Dict:
    config = {}
    print_html(
        "<info>OpenAI API key should be set via the OPENAI_API_KEY environment variable.</info>")
    timeout_input = input_styled(
        "Enter timeout for OpenAI requests in seconds (default: 30.0): ").strip()
    if timeout_input:
        try:
            config["timeout"] = float(timeout_input)
        except ValueError:
            print_html("<error>Invalid timeout value, using default.</error>")
    max_retries_input = input_styled(
        "Enter maximum retries for OpenAI requests (default: 2): ").strip()
    if max_retries_input:
        try:
            config["max_retries"] = int(max_retries_input)
        except ValueError:
            print_html("<error>Invalid retries value, using default.</error>")
    return config


def get_google_config() -> Dict:
    config = {}
    print_html(
        "<info>Google API key should be set via the GOOGLE_API_KEY environment variable.</info>")
    return config


def select_model(llm_instance: LLMChat) -> str | None:
    print_html("\n<status>Fetching available models...</status>")
    try:
        models = llm_instance.get_available_models()
        if not models:
            print_html(
                f"<error>No models found for {llm_instance.__class__.__name__} or an error occurred.</error>")
            if isinstance(llm_instance, OllamaChat):
                print_html(
                    "<info>For Ollama, you might need to pull a model first (e.g., 'ollama pull llama3').</info>")
            manual_model = input_styled(
                "Enter model name manually (or press Enter to cancel): ").strip()
            return manual_model if manual_model else None

        print_html("<menu-prompt>Available models:</menu-prompt>")
        for i, model_name in enumerate(models):
            print_html(
                f"<menu-item-index>{i + 1}.</menu-item-index> <menu-item-text>{model_name}</menu-item-text>")
        print_html("<menu-item-index>0.</menu-item-index> <cancel>Cancel / Enter manually</cancel>")

        while True:
            try:
                choice_input = input_styled(
                    f"Select model (1-{len(models)}, or 0 to cancel/enter manually): "
                ).strip()
                model_choice_idx = int(choice_input)

                if model_choice_idx == 0:
                    manual_model = input_styled(
                        "Enter model name manually (or press Enter to cancel): "
                    ).strip()
                    return manual_model if manual_model else None
                elif 1 <= model_choice_idx <= len(models):
                    return models[model_choice_idx - 1]
                else:
                    print_html(
                        f"<error>Invalid choice. Please enter a number between 0 and {len(models)}.</error>")
            except ValueError:
                print_html("<error>Invalid input. Please enter a number.</error>")
            except (KeyboardInterrupt, EOFError):
                print_html("\n<info>Selection cancelled.</info>")
                return None

    except Exception as e:
        logger.error(
            f"Could not retrieve models for {llm_instance.__class__.__name__}: {e}", exc_info=True
        )
        print_html(f"<error>Error fetching models: {e}</error>")
        manual_model = input_styled(
            "Could not fetch models. Enter model name manually (or press Enter to cancel): "
        ).strip()
        return manual_model if manual_model else None


def run_chat_session(chat_instance: LLMChat):
    ai_name_display = chat_instance.model_name.split('/')[-1]  # Get base name
    print_html(
        f"\n<info>Chatting with</info> <ai-name>{ai_name_display}</ai-name> <info>via {chat_instance.__class__.__name__}.</info>")
    print_html(
        "<info>Type <command>/exit</command> or <command>/quit</command> to end the session.</info>")
    print_html("<info>Type <command>/history</command> to view current session history.</info>")
    print_html("<info>---</info>")

    history_file_name_sanitized = "".join(
        c if c.isalnum() else "_" for c in chat_instance.model_name)
    history_file = (
        f".chat_history_{chat_instance.__class__.__name__}_{history_file_name_sanitized}.txt"
    )

    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        style=tui_style  # Apply style to prompt session (affects input line, completion menu)
    )

    chat_history: List[Dict[str, str]] = []

    while True:
        try:
            # The prompt message itself is styled using HTML
            user_input = session.prompt(HTML("<user-prompt>You: </user-prompt>")).strip()

            if user_input.lower() in ["/exit", "/quit"]:
                print_html("<info>Exiting chat session.</info>")
                break

            if user_input.lower() == "/history":
                if not chat_history:
                    print_html("<info>Chat history is empty.</info>")
                else:
                    print_html("\n<info>--- Chat History ---</info>")
                    for entry in chat_history:
                        role_style = "history-role-user" if entry[
                                                                'role'] == "user" else "history-role-assistant"
                        print_html(
                            f"<{role_style}>{entry['role'].capitalize()}:</{role_style}> <history-content>{entry['content']}</history-content>")
                    print_html("<info>--- End of History ---</info>")
                continue

            if not user_input:
                continue

            current_turn_user_message = {"role": "user", "content": user_input}
            api_history = chat_history.copy()

            print_formatted_text(HTML(f"<ai-name>{ai_name_display}:</ai-name> "), end="",
                                 style=tui_style)

            full_response = ""
            try:
                for chunk in chat_instance.stream_message(user_input, history=api_history):
                    print_formatted_text(HTML(f"<ai-response>{HTML(chunk).value}</ai-response>"),
                                         end="", style=tui_style, flush=True)
                    full_response += chunk
                print()  # Newline after streaming is done
            except Exception as e:
                logger.error(
                    f"Error during streaming with {chat_instance.model_name}: {e}", exc_info=True
                )
                print_html(f"\n<error>Error during streaming: {e}</error>")
                continue

            chat_history.append(current_turn_user_message)
            chat_history.append(
                {"role": "assistant", "content": full_response}
            )

        except KeyboardInterrupt:
            print_html("\n<info>Exiting chat session (KeyboardInterrupt).</info>")
            break
        except EOFError:
            print_html("\n<info>Exiting chat session (EOF).</info>")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in chat session: {e}", exc_info=True)
            print_html(f"\n<error>An unexpected error occurred: {e}</error>")
            break


def main_tui():
    print_html("<welcome>Welcome to the LLM Chat Interface!</welcome>")

    llm_provider_class = select_llm_provider()
    if not llm_provider_class:
        print_html("<info>No provider selected. Exiting.</info>")
        return

    config_params = {}
    temp_instance_for_listing = None
    default_model_for_listing = "dummy-model-for-listing"

    try:
        if llm_provider_class == OllamaChat:
            config_params = get_ollama_config()
            temp_instance_for_listing = llm_provider_class(
                model_name=default_model_for_listing, **config_params
            )
        elif llm_provider_class == OpenAIChat:
            config_params = get_openai_config()
            temp_instance_for_listing = llm_provider_class(
                model_name="gpt-3.5-turbo", **config_params
            )
        elif llm_provider_class == GoogleChat:
            config_params = get_google_config()
            temp_instance_for_listing = llm_provider_class(
                model_name="gemini-1.5-flash-latest", **config_params
            )
        else:
            print_html(
                f"<error>Provider {llm_provider_class.__name__} setup not fully implemented in TUI.</error>")
            return

    except (ValueError, ConnectionError, PermissionError) as e:
        logger.error(
            f"Configuration or Connection Error for {llm_provider_class.__name__}: {e}",
            exc_info=True,
        )
        print_html(f"<error>Error during provider setup: {e}</error>")
        return
    except Exception as e:
        logger.error(
            f"Unexpected error initializing client for model listing ({llm_provider_class.__name__}): {e}",
            exc_info=True)
        print_html(f"<error>An unexpected error occurred during provider setup: {e}</error>")
        return

    selected_model_name = select_model(temp_instance_for_listing)
    if not selected_model_name:
        print_html("<info>No model selected. Exiting.</info>")
        return

    try:
        chat_instance = llm_provider_class(model_name=selected_model_name, **config_params)
        run_chat_session(chat_instance)
    except (ValueError, ConnectionError, PermissionError) as e:
        logger.error(
            f"Error initializing chat with model {selected_model_name}: {e}", exc_info=True
        )
        print_html(f"<error>Error: {e}</error>")
    except Exception as e:
        logger.error(f"Unexpected error initializing chat instance: {e}", exc_info=True)
        print_html(f"<error>An unexpected error occurred: {e}</error>")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )
    main_tui()
