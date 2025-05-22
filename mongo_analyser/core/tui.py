# mongo_analyser/core/tui.py
import logging
import os
import sys
from typing import Dict, Any

from prompt_toolkit import PromptSession, print_formatted_text, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.completion import WordCompleter

# Assuming your core modules are importable like this
from . import extractor
from . import analyser

# from . import shared # If you have shared utilities for core

logger = logging.getLogger(__name__)

# Consistent styling with llm_chat TUI, can be expanded
core_tui_style = Style.from_dict(
    {
        "prompt": "ansigreen bold",
        "menu-item-index": "lightcyan",
        "menu-item-text": "lightblue",
        "info": "ansigray",
        "error": "ansired bold",
        "success": "ansigreen bold",
        "welcome": "ansimagenta bold underline",
        "path": "ansiyellow",
        "parameter": "ansicyan",
    }
)


def print_html_core(text: str):
    """Helper function to print HTML formatted text with the core TUI style."""
    print_formatted_text(HTML(text), style=core_tui_style)


def input_styled_core(message: str, session: PromptSession = None, validator=None, completer=None,
                      default: str = "") -> str:
    """Gets input from the user with a styled prompt, using an existing session if provided."""
    full_prompt = HTML(f"<prompt>{message}</prompt>")
    if session:
        return session.prompt(full_prompt, validator=validator, completer=completer,
                              default=default).strip()
    else:
        # Fallback for simple inputs if no session context is needed (though less ideal for history)
        print_formatted_text(full_prompt, end=" ", style=core_tui_style)
        return input().strip()


class NotEmptyValidator(Validator):
    def validate(self, document):
        text = document.text
        if not text or not text.strip():
            raise ValidationError(message="This input cannot be empty.", cursor_position=len(text))


def get_mongo_connection_details(session: PromptSession) -> Dict[str, str]:
    """Prompts user for MongoDB connection details."""
    print_html_core("\n<info>--- MongoDB Connection Details ---</info>")
    details = {}
    details["uri"] = input_styled_core(
        "Enter MongoDB URI (e.g., mongodb://localhost:27017/): ",
        session,
        validator=NotEmptyValidator(),
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    )
    details["database"] = input_styled_core("Enter Database Name: ", session,
                                            validator=NotEmptyValidator())
    return details


def handle_extraction(session: PromptSession, conn_details: Dict[str, str]):
    print_html_core("\n<info>--- Data Extraction ---</info>")
    collection = input_styled_core("Enter Collection Name to extract from: ", session,
                                   validator=NotEmptyValidator())
    default_output_file = f"{conn_details['database']}_{collection}_data.json.gz"
    output_file = input_styled_core(f"Enter Output File Path (default: {default_output_file}): ",
                                    session, default=default_output_file)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print_html_core(f"<info>Created directory: <path>{output_dir}</path></info>")
        except OSError as e:
            print_html_core(f"<error>Could not create directory {output_dir}: {e}</error>")
            return

    print_html_core(
        f"<info>Starting extraction from <parameter>{conn_details['database']}.{collection}</parameter> to <path>{output_file}</path>...</info>")
    try:
        # Placeholder: Replace with actual call to your extractor
        # Example: extractor.extract_collection_data(uri=conn_details["uri"], db_name=conn_details["database"], coll_name=collection, output_path=output_file)
        print_html_core(f"<info>Simulating extraction for {collection}...</info>")
        # Simulate some work
        if not os.path.exists(
            output_file) and output_file != default_output_file:  # only create if not default and not existing
            with open(output_file, "w") as f:
                f.write('{"message": "Simulated extracted data"}')
        print_html_core(
            f"<success>Extraction successful! Data saved to <path>{output_file}</path></success>")
    except Exception as e:
        logger.error(f"Extraction failed for {collection}: {e}", exc_info=True)
        print_html_core(f"<error>Extraction failed: {e}</error>")


def handle_schema_analysis_live(session: PromptSession, conn_details: Dict[str, str]):
    print_html_core("\n<info>--- Live Schema Analysis ---</info>")
    collection = input_styled_core("Enter Collection Name to analyze: ", session,
                                   validator=NotEmptyValidator())

    print_html_core(
        f"<info>Starting live schema analysis for <parameter>{conn_details['database']}.{collection}</parameter>...</info>")
    try:
        # Placeholder: Replace with actual call to your analyser
        # Example: result = analyser.analyse_live_collection_schema(uri=conn_details["uri"], db_name=conn_details["database"], coll_name=collection)
        result = {"message": f"Simulated schema analysis result for {collection}", "fields": 5,
                  "avg_doc_size": 1024}
        print_html_core("<success>Live schema analysis successful!</success>")
        print_html_core(f"<info>Result: {result}</info>")  # Display actual result appropriately
    except Exception as e:
        logger.error(f"Live schema analysis failed for {collection}: {e}", exc_info=True)
        print_html_core(f"<error>Live schema analysis failed: {e}</error>")


def handle_schema_analysis_file(session: PromptSession):
    print_html_core("\n<info>--- Schema Analysis from File ---</info>")
    file_path_completer = WordCompleter(
        [])  # In a real app, you might populate this or use PathCompleter
    file_path = input_styled_core("Enter path to schema JSON/JSON.GZ file: ", session,
                                  validator=NotEmptyValidator(), completer=file_path_completer)

    if not os.path.exists(file_path):
        print_html_core(f"<error>File not found: <path>{file_path}</path></error>")
        return

    print_html_core(f"<info>Starting schema analysis for file <path>{file_path}</path>...</info>")
    try:
        # Placeholder: Replace with actual call to your analyser
        # Example: result = analyser.analyse_schema_from_file(file_path=file_path)
        result = {
            "message": f"Simulated schema analysis result for file {os.path.basename(file_path)}",
            "types_found": ["string", "int", "object"]}
        print_html_core("<success>Schema analysis from file successful!</success>")
        print_html_core(f"<info>Result: {result}</info>")  # Display actual result appropriately
    except Exception as e:
        logger.error(f"Schema analysis from file {file_path} failed: {e}", exc_info=True)
        print_html_core(f"<error>Schema analysis from file failed: {e}</error>")


def core_main_menu(session: PromptSession):
    """Main menu for the Core Analyser TUI."""
    conn_details = None

    while True:
        print_html_core("\n<welcome>Mongo Analyser - Core Utilities</welcome>")
        if conn_details:
            print_html_core(
                f"<info>Current DB: <parameter>{conn_details['database']}</parameter> on <parameter>{conn_details['uri']}</parameter></info>")
        else:
            print_html_core("<info>Not connected to MongoDB.</info>")

        print_html_core(
            "<menu-item-index>1.</menu-item-index> <menu-item-text>Set/Change MongoDB Connection</menu-item-text>")
        print_html_core(
            "<menu-item-index>2.</menu-item-index> <menu-item-text>Extract Collection Data</menu-item-text>")
        print_html_core(
            "<menu-item-index>3.</menu-item-index> <menu-item-text>Analyze Live Collection Schema</menu-item-text>")
        print_html_core(
            "<menu-item-index>4.</menu-item-index> <menu-item-text>Analyze Schema from File</menu-item-text>")
        print_html_core(
            "<menu-item-index>0.</menu-item-index> <menu-item-text>Exit Core TUI</menu-item-text>")

        choice = input_styled_core("Select an option: ", session).strip()

        if choice == "1":
            conn_details = get_mongo_connection_details(session)
        elif choice == "2":
            if not conn_details:
                print_html_core("<error>Please set MongoDB connection first (Option 1).</error>")
                continue
            handle_extraction(session, conn_details)
        elif choice == "3":
            if not conn_details:
                print_html_core("<error>Please set MongoDB connection first (Option 1).</error>")
                continue
            handle_schema_analysis_live(session, conn_details)
        elif choice == "4":
            handle_schema_analysis_file(session)
        elif choice == "0":
            print_html_core("<info>Exiting Core Analyser TUI.</info>")
            break
        else:
            print_html_core("<error>Invalid option. Please try again.</error>")
        print_html_core("<info>---</info>")  # Separator


def main_core_tui():
    """Entry point for the core functionalities TUI."""
    # Use a persistent session for history across inputs within this TUI
    session = PromptSession(history=FileHistory(".mongo_analyser_core_history.txt"))
    try:
        core_main_menu(session)
    except (KeyboardInterrupt, EOFError):
        print_html_core("\n<info>Exiting Core Analyser TUI.</info>")


if __name__ == "__main__":
    # This allows direct execution for testing:
    # python -m mongo_analyser.core.tui
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main_core_tui()
