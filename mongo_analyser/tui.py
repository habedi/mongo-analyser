import html
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from urllib.parse import urlparse

import pytz
from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import PathCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator
from pymongo.errors import (
    ConnectionFailure as PyMongoConnectionFailure,
)
from pymongo.errors import (
    OperationFailure as PyMongoOperationFailure,
)

# Core and LLM imports
from mongo_analyser.core import analyser as core_analyser
from mongo_analyser.core import db as core_db_manager
from mongo_analyser.core import extractor as core_extractor
from mongo_analyser.core import shared as core_shared
from mongo_analyser.llm_chat.base_chat import LLMChat
from mongo_analyser.llm_chat.google_chat import GoogleChat
from mongo_analyser.llm_chat.ollama_chat import OllamaChat
from mongo_analyser.llm_chat.openai_chat import OpenAIChat

logger = logging.getLogger(__name__)

# --- Unified TUI Styling and Helpers ---

unified_tui_style = Style.from_dict(
    {
        "prompt": "ansigreen bold",
        "user-prompt": "ansiblue bold",  # For LLM chat user input
        "menu-item-index": "lightcyan",
        "menu-item-text": "lightblue",
        "info": "ansigray",
        "error": "ansired bold",
        "success": "ansigreen bold",
        "warning": "ansiyellow bold",
        "welcome": "ansimagenta bold underline",
        "path": "ansiyellow",
        "parameter": "ansicyan",
        "stat-label": "blue bold",
        "stat-value": "white",
        "ai-name": "ansimagenta bold",  # For LLM name
        "ai-response": "ansiyellow",  # For LLM response
        "history-role-user": "ansiblue bold",
        "history-role-assistant": "ansimagenta bold",
        "history-content": "",
        "status": "black bold",  # For status messages like "Fetching..."
        "cancel": "ansiyellow",  # For cancel options
        "command": "cyan bold",  # For TUI commands like /exit
        "mongo-context-header": "magenta bold",
        "mongo-context-body": "magenta",
        "mongo-context-section": "ansimagenta underline",
    }
)


def print_styled_html(text: str):
    try:
        print_formatted_text(HTML(text), style=unified_tui_style)
    except Exception as e:
        logger.error(f"Error rendering HTML string: '{text}'. Error: {e}")
        # Fallback to plain print if styling fails
        import re

        clean_text = re.sub(r"<[^>]+>", "", text)
        print(clean_text)


def input_styled(
    message: str,
    session: PromptSession,  # Made session non-optional for consistency
    validator=None,
    completer=None,
    default: str = "",
    rprompt: str = None,
) -> str:
    full_prompt_html = f"<prompt>{html.escape(message)}</prompt>"
    return session.prompt(
        HTML(full_prompt_html),
        validator=validator,
        completer=completer,
        default=default,
        rprompt=rprompt,
    ).strip()


class NotEmptyValidator(Validator):
    def validate(self, document):
        text = document.text.strip()
        if not text:
            raise ValidationError(
                message="This input cannot be empty.", cursor_position=len(document.text)
            )


class MongoURIValidator(Validator):
    def validate(self, document):
        text = document.text
        if not text.startswith("mongodb://") and not text.startswith("mongodb+srv://"):
            raise ValidationError(
                message="MongoDB URI should start with 'mongodb://' or 'mongodb+srv://'."
            )


def _format_bytes_tui(size_bytes: Optional[Union[int, float]]) -> str:
    if size_bytes is None:
        return "N/A"
    if not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "Invalid"
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


# --- Core TUI Functions (previously in core/tui.py) ---


def _get_mongo_connection_details_tui(
    session: PromptSession, existing_details: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    print_styled_html("\n<info>--- MongoDB Connection Details ---</info>")
    current_details = existing_details.copy() if existing_details else {}

    default_uri = current_details.get("uri", os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    uri_input = input_styled(
        "Enter MongoDB URI: ", session, validator=MongoURIValidator(), default=default_uri
    )
    current_details["uri"] = uri_input if uri_input else default_uri

    default_db_name_from_details = current_details.get("database", "")
    parsed_db_from_uri = ""
    uri_for_parsing = current_details.get("uri", "")
    if not default_db_name_from_details and uri_for_parsing:
        try:
            parsed_uri_obj = urlparse(uri_for_parsing)
            path_db_candidate = parsed_uri_obj.path.lstrip("/")
            if path_db_candidate and "/" not in path_db_candidate:
                parsed_db_from_uri = path_db_candidate.split("?")[0]
        except Exception as e:
            logger.debug(f"Could not parse DB name from URI '{uri_for_parsing}': {e}")

    final_default_db_name = default_db_name_from_details or parsed_db_from_uri
    db_prompt_message = "Enter Database Name"
    db_prompt_message += (
        f" (default: {final_default_db_name})"
        if final_default_db_name
        else " (required if not in URI)"
    )

    db_input = input_styled(
        db_prompt_message + ": ",
        session,
        validator=NotEmptyValidator() if not final_default_db_name else None,
        default=final_default_db_name,
    )
    current_details["database"] = db_input if db_input else final_default_db_name

    uri_to_test = current_details.get("uri")
    db_to_test = current_details.get("database")

    if uri_to_test:
        display_uri = core_shared.redact_uri_password(uri_to_test)
        effective_db_for_test_msg = (
            f" (DB context: {html.escape(db_to_test)})"
            if db_to_test
            else " (DB from URI if specified)"
        )
        print_styled_html(
            f"<info>Attempting to connect to URI: {html.escape(display_uri)}{effective_db_for_test_msg}</info>"
        )
        try:
            if core_db_manager.db_connection_active(uri=uri_to_test, db_name=db_to_test or None):
                actual_db_obj = core_db_manager.get_mongo_db()
                actual_db_name = actual_db_obj.name
                current_details["database"] = actual_db_name
                print_styled_html(
                    f"<success>Connection successful to DB '{html.escape(actual_db_name)}' via URI: {html.escape(display_uri)}</success>"
                )
            else:
                print_styled_html(
                    f"<error>Failed to establish MongoDB connection. URI: {html.escape(display_uri)}. Check logs.</error>"
                )
        except ConnectionError as ce:
            print_styled_html(f"<error>Connection error: {html.escape(str(ce))}</error>")
        except Exception as e:
            print_styled_html(f"<error>Error during connection test: {html.escape(str(e))}</error>")
            logger.error(f"Connection test exception for URI {display_uri}", exc_info=True)
    else:
        print_styled_html("<error>MongoDB URI is empty. Cannot test connection.</error>")
    return current_details


def _handle_list_collections_tui(session: PromptSession, conn_details: Dict[str, str]):
    print_styled_html("\n<info>--- List Collections ---</info>")
    db_name_to_list = conn_details.get("database")
    uri = conn_details.get("uri")
    display_uri = core_shared.redact_uri_password(uri) if uri else "N/A"

    if not uri:
        print_styled_html("<error>MongoDB URI not set. Please set connection first.</error>")
        return

    print_styled_html(
        f"<info>Fetching collections for database '{html.escape(db_name_to_list or 'parsed from URI')}' using URI '{html.escape(display_uri)}'...</info>"
    )
    try:
        collection_names = core_analyser.SchemaAnalyser.list_collection_names(
            uri=uri, db_name=db_name_to_list or ""
        )
        connected_db_name = core_db_manager.get_mongo_db().name
        if collection_names:
            print_styled_html(
                f"<success>Available collections in '{html.escape(connected_db_name)}':</success>"
            )
            for i, name in enumerate(collection_names):
                print_styled_html(
                    f"<menu-item-index>{i + 1}.</menu-item-index> <menu-item-text>{html.escape(name)}</menu-item-text>"
                )
        else:
            print_styled_html(
                f"<warning>No collections found in database '{html.escape(connected_db_name)}'.</warning>"
            )
    except (PyMongoConnectionFailure, PyMongoOperationFailure) as e:
        print_styled_html(f"<error>Could not list collections: {html.escape(str(e))}</error>")
    except Exception as e:
        logger.error(
            f"Failed to list collections for DB related to URI {display_uri}: {e}", exc_info=True
        )
        print_styled_html(f"<error>An unexpected error occurred: {html.escape(str(e))}</error>")


def _handle_extraction_tui(session: PromptSession, conn_details: Dict[str, str]):
    print_styled_html("\n<info>--- Data Extraction ---</info>")
    # ... (Implementation similar to core/tui.py's handle_extraction, using input_styled and print_styled_html)
    # This function would call core_extractor.DataExtractor.extract_data
    uri = conn_details.get("uri")
    db_name = conn_details.get("database")
    display_uri = core_shared.redact_uri_password(uri) if uri else "N/A"

    if not uri or not db_name:
        print_styled_html(
            "<error>MongoDB URI or Database not set. Please set connection first.</error>"
        )
        return

    collection_name = input_styled(
        "Enter Collection Name to extract from: ", session, validator=NotEmptyValidator()
    )
    default_schema_file = f"{db_name}_{collection_name}_schema.json"
    schema_file_path_str = input_styled(
        f"Path to schema JSON file (e.g., {default_schema_file}): ",
        session,
        validator=NotEmptyValidator(),
        completer=PathCompleter(),
        default=default_schema_file,
    )
    schema_file_path = Path(schema_file_path_str)
    if not schema_file_path.is_file():
        print_styled_html(
            f"<error>Schema file not found: <path>{html.escape(str(schema_file_path))}</path></error>"
        )
        return
    try:
        with schema_file_path.open("r", encoding="utf-8") as sf:
            schema_data = json.load(sf)
    except json.JSONDecodeError:
        print_styled_html(
            f"<error>Invalid JSON in schema file: <path>{html.escape(str(schema_file_path))}</path></error>"
        )
        return
    except IOError as e:
        print_styled_html(
            f"<error>Could not read schema file <path>{html.escape(str(schema_file_path))}</path>: {html.escape(str(e))}</error>"
        )
        return

    default_output_file = f"./{db_name}_{collection_name}_data.json.gz"
    output_file_str = input_styled(
        f"Output File Path (default: {default_output_file}): ",
        session,
        default=default_output_file,
        completer=PathCompleter(),
    )
    output_file = Path(output_file_str)
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print_styled_html(
            f"<error>Could not create directory <path>{html.escape(str(output_file.parent))}</path>: {html.escape(str(e))}</error>"
        )
        return

    default_tz = "UTC"
    timezone_str = input_styled(f"Timezone (default: {default_tz}): ", session, default=default_tz)
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        print_styled_html(
            f"<error>Unknown timezone '{html.escape(timezone_str)}'. Using {default_tz}.</error>"
        )
        tz = pytz.timezone(default_tz)

    limit = int(input_styled("Max documents (-1 for all, newest first): ", session, default="-1"))
    batch_size = int(input_styled("Batch size: ", session, default="1000"))
    if batch_size <= 0:
        batch_size = 1000

    print_styled_html(
        f"<info>Starting extraction (newest first): <parameter>{html.escape(db_name)}.{html.escape(collection_name)}</parameter> -> <path>{html.escape(str(output_file))}</path> from URI {html.escape(display_uri)}</info>"
    )
    try:
        core_extractor.DataExtractor.extract_data(
            mongo_uri=uri,
            db_name=db_name,
            collection_name=collection_name,
            schema=schema_data,
            output_file=output_file,
            tz=tz,
            batch_size=batch_size,
            limit=limit,
        )
        print_styled_html(
            f"<success>Extraction successful! Data saved to <path>{html.escape(str(output_file))}</path></success>"
        )
    except (PyMongoConnectionFailure, PyMongoOperationFailure) as e:
        print_styled_html(f"<error>MongoDB error: {html.escape(str(e))}</error>")
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        print_styled_html(f"<error>Extraction failed: {html.escape(str(e))}</error>")


def _handle_schema_analysis_live_tui(session: PromptSession, conn_details: Dict[str, str]):
    print_styled_html("\n<info>--- Live Schema Analysis ---</info>")
    # ... (Implementation similar to core/tui.py's handle_schema_analysis_live)
    # This function would call core_analyser.SchemaAnalyser methods
    uri = conn_details.get("uri")
    db_name = conn_details.get("database")
    display_uri = core_shared.redact_uri_password(uri) if uri else "N/A"

    if not uri or not db_name:
        print_styled_html(
            "<error>MongoDB URI or Database not set. Please set connection first.</error>"
        )
        return

    collection_name = input_styled(
        "Enter Collection Name to analyze: ", session, validator=NotEmptyValidator()
    )
    sample_size_str = input_styled(
        "Sample size for field analysis (-1 for all, default: 1000): ", session, default="1000"
    )
    try:
        sample_size = int(sample_size_str)
    except ValueError:
        print_styled_html("<error>Invalid sample size. Using 1000.</error>")
        sample_size = 1000

    try:
        mongo_collection = core_analyser.SchemaAnalyser.get_collection(
            uri=uri, db_name=db_name, collection_name=collection_name
        )
        actual_db_name_from_conn = core_db_manager.get_mongo_db().name

        print_styled_html(
            f"<info>Fetching general statistics for <parameter>{html.escape(actual_db_name_from_conn)}.{html.escape(collection_name)}</parameter>...</info>"
        )
        general_stats = core_analyser.SchemaAnalyser.get_collection_general_statistics(
            mongo_collection
        )

        print_styled_html("<info>--- General Collection Statistics ---</info>")
        if "error" in general_stats:
            print_styled_html(
                f"<error>Could not fetch general stats: {general_stats['error']}</error>"
            )
        else:
            print_styled_html(
                f"<stat-label>Namespace:</stat-label> <stat-value>{html.escape(str(general_stats.get('ns', 'N/A')))}</stat-value>"
            )
            print_styled_html(
                f"<stat-label>Documents:</stat-label> <stat-value>{general_stats.get('document_count', 'N/A'):,}</stat-value>"
            )
            print_styled_html(
                f"<stat-label>Avg. Object Size:</stat-label> <stat-value>{_format_bytes_tui(general_stats.get('avg_obj_size_bytes'))}</stat-value>"
            )
            # ... more stats
            print_styled_html(
                f"<stat-label>Total Data Size:</stat-label> <stat-value>{_format_bytes_tui(general_stats.get('total_size_bytes'))}</stat-value>"
            )
            print_styled_html(
                f"<stat-label>Total Storage Size:</stat-label> <stat-value>{_format_bytes_tui(general_stats.get('storage_size_bytes'))}</stat-value>"
            )
            print_styled_html(
                f"<stat-label>Indexes:</stat-label> <stat-value>{general_stats.get('nindexes', 'N/A')}</stat-value>"
            )
            print_styled_html(
                f"<stat-label>Total Index Size:</stat-label> <stat-value>{_format_bytes_tui(general_stats.get('total_index_size_bytes'))}</stat-value>"
            )
            print_styled_html(
                f"<stat-label>Capped:</stat-label> <stat-value>{'Yes' if general_stats.get('capped') else 'No'}</stat-value>"
            )

        print_styled_html(
            f"<info>Starting field schema analysis for <parameter>{html.escape(actual_db_name_from_conn)}.{html.escape(collection_name)}</parameter> (sample: {sample_size if sample_size >= 0 else 'all'}) using URI {html.escape(display_uri)}...</info>"
        )
        schema_data, field_stats_data = core_analyser.SchemaAnalyser.infer_schema_and_field_stats(
            mongo_collection, sample_size=sample_size
        )
        print_styled_html(
            "<success>Live schema and field statistics analysis successful!</success>"
        )

        headers = ["Field", "Type", "Cardinality", "Missing (%)"]
        rows = []
        for field, details in schema_data.items():
            field_s_data = field_stats_data.get(field, {})
            cardinality = field_s_data.get("cardinality", "N/A")
            missing_percentage = field_s_data.get("missing_percentage", "N/A")
            rows.append(
                [
                    field,
                    details["type"],
                    cardinality,
                    f"{missing_percentage:.1f}"
                    if isinstance(missing_percentage, (float, int))
                    else "N/A",
                ]
            )

        print_styled_html("<info>--- Field Schema & Statistics ---</info>")
        for row_data in rows:
            print_styled_html(
                f"<parameter>{html.escape(row_data[0])}</parameter>: Type=<info>{html.escape(str(row_data[1]))}</info>, Cardinality=<info>{html.escape(str(row_data[2]))}</info>, Missing%=<info>{html.escape(str(row_data[3]))}</info>"
            )
        print_styled_html("<info>--- End of Field Schema & Statistics ---</info>")

        if input_styled("Save schema to JSON? (y/N): ", session, default="N").lower() == "y":
            sf_def = f"./{html.escape(actual_db_name_from_conn)}_{html.escape(collection_name)}_schema.json"
            sf_path = input_styled(
                f"Schema file path (default: {sf_def}): ",
                session,
                default=sf_def,
                completer=PathCompleter(),
            )
            core_analyser.SchemaAnalyser.save_schema_to_json(schema_data, sf_path)

        if (
            input_styled("Save field statistics to CSV? (y/N): ", session, default="N").lower()
            == "y"
        ):
            statsf_def = f"./{html.escape(actual_db_name_from_conn)}_{html.escape(collection_name)}_field_stats.csv"
            statsf_path = input_styled(
                f"Field Stats CSV path (default: {statsf_def}): ",
                session,
                default=statsf_def,
                completer=PathCompleter(),
            )
            core_analyser.SchemaAnalyser.save_table_to_csv(headers, rows, statsf_path)

    except (PyMongoConnectionFailure, PyMongoOperationFailure) as e:
        print_styled_html(f"<error>MongoDB error: {html.escape(str(e))}</error>")
    except Exception as e:
        logger.error(f"Schema analysis failed: {e}", exc_info=True)
        print_styled_html(f"<error>Schema analysis failed: {html.escape(str(e))}</error>")


def _handle_schema_analysis_file_tui(session: PromptSession):
    print_styled_html("\n<info>--- Display Schema from File ---</info>")
    # ... (Implementation similar to core/tui.py's handle_schema_analysis_file)
    file_path_str = input_styled(
        "Path to schema JSON file: ",
        session,
        validator=NotEmptyValidator(),
        completer=PathCompleter(),
    )
    file_path = Path(file_path_str)
    if not file_path.is_file():
        print_styled_html(
            f"<error>File not found: <path>{html.escape(str(file_path))}</path></error>"
        )
        return
    print_styled_html(
        f"<info>Loading schema from <path>{html.escape(str(file_path))}</path>...</info>"
    )
    try:
        with file_path.open("r", encoding="utf-8") as f:
            schema_content = json.load(f)
        formatted_json = json.dumps(schema_content, indent=2)
        print_styled_html("<info>--- Schema Content ---</info>")
        for line in formatted_json.splitlines():
            print_styled_html(f"<info>{html.escape(line)}</info>")
        print_styled_html("<info>--- End of Schema Content ---</info>")
        print_styled_html("<success>Schema display successful!</success>")
    except json.JSONDecodeError:
        print_styled_html(
            f"<error>Invalid JSON: <path>{html.escape(str(file_path))}</path></error>"
        )
    except IOError as e:
        print_styled_html(
            f"<error>Could not read <path>{html.escape(str(file_path))}</path>: {html.escape(str(e))}</error>"
        )
    except Exception as e:
        logger.error(f"Schema file analysis failed: {e}", exc_info=True)
        print_styled_html(f"<error>Failed: {html.escape(str(e))}</error>")


# --- LLM Chat TUI Functions (previously in llm_chat/tui.py) ---


def _select_llm_provider_tui(session: PromptSession) -> Type[LLMChat] | None:
    print_styled_html("\n<prompt>Select LLM Provider:</prompt>")
    providers = {
        "1": ("Ollama", OllamaChat),
        "2": ("OpenAI", OpenAIChat),
        "3": ("Google", GoogleChat),
    }
    for key, (name, _) in providers.items():
        print_styled_html(
            f"<menu-item-index>{key}.</menu-item-index> <menu-item-text>{name}</menu-item-text>"
        )
    print_styled_html("<menu-item-index>0.</menu-item-index> <cancel>Cancel</cancel>")

    while True:
        choice = input_styled(
            "Enter choice: ",
            session=session,
            completer=WordCompleter(list(providers.keys()) + ["0"]),
        )
        if choice == "0":
            return None
        if choice in providers:
            return providers[choice][1]
        print_styled_html("<error>Invalid choice. Please try again.</error>")


def _get_provider_config_tui(provider_class: Type[LLMChat], session: PromptSession) -> Dict:
    config: Dict[str, Any] = {}
    opts: Dict[str, Any] = {}

    if provider_class == OllamaChat:
        print_styled_html("<info>Configuring Ollama Client:</info>")
        host_input = input_styled(
            "Ollama host (e.g., http://localhost:11434, blank for default/OLLAMA_HOST env): ",
            session=session,
        )
        if host_input:
            config["host"] = host_input
        try:
            config["timeout"] = int(
                input_styled("Timeout (seconds, default 30): ", session=session, default="30")
            )
        except ValueError:
            config["timeout"] = 30
        config["keep_alive"] = input_styled(
            "Keep alive (e.g., 5m, -1 indefinite, 0 session, default 5m): ",
            session=session,
            default="5m",
        )

        temp_str = input_styled("  Temperature (e.g., 0.7, blank for default): ", session=session)
        if temp_str:
            try:
                opts["temperature"] = float(temp_str)
            except ValueError:
                print_styled_html("<warning>Invalid temperature, skipping.</warning>")

        num_ctx_str = input_styled(
            "  Context window size (num_ctx, e.g., 2048, blank for default): ", session=session
        )
        if num_ctx_str:
            try:
                opts["num_ctx"] = int(num_ctx_str)
            except ValueError:
                print_styled_html("<warning>Invalid context window size, skipping.</warning>")

        if opts:
            config["options"] = opts

    elif provider_class == OpenAIChat:
        print_styled_html(
            "<info>OpenAI API key is usually set via OPENAI_API_KEY environment variable.</info>"
        )
        api_key_input = input_styled("OpenAI API Key (blank to rely on env var): ", session=session)
        if api_key_input:
            config["api_key"] = api_key_input
        base_url = input_styled(
            "Azure OpenAI Endpoint / Custom Base URL (blank for default OpenAI): ", session=session
        )
        if base_url:
            config["base_url"] = base_url
        try:
            config["timeout"] = float(
                input_styled("Timeout (seconds, default 30.0): ", session=session, default="30.0")
            )
        except ValueError:
            config["timeout"] = 30.0
        try:
            config["max_retries"] = int(
                input_styled("Max Retries (default 2): ", session=session, default="2")
            )
        except ValueError:
            config["max_retries"] = 2

    elif provider_class == GoogleChat:
        print_styled_html(
            "<info>Google API key is usually set via GOOGLE_API_KEY environment variable.</info>"
        )
        api_key_input = input_styled("Google API Key (blank to rely on env var): ", session=session)
        if api_key_input:
            config["api_key"] = api_key_input

        gen_config: Dict[str, Any] = {}
        if (
            input_styled(
                "Configure generation settings (temperature, etc.)? (y/N): ",
                session=session,
                default="N",
            ).lower()
            == "y"
        ):
            temp_str_google = input_styled(
                "  Temperature (e.g., 0.9, blank for default): ", session=session
            )
            if temp_str_google:
                try:
                    gen_config["temperature"] = float(temp_str_google)
                except ValueError:
                    print_styled_html("<warning>Invalid temperature.</warning>")
            if gen_config:
                config["generation_config"] = gen_config

        if (
            input_styled("Configure safety settings? (y/N): ", session=session, default="N").lower()
            == "y"
        ):
            config["safety_settings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            print_styled_html(
                "<info>Using sample safety settings (BLOCK_ONLY_HIGH for common categories).</info>"
            )
    return config


def _select_model_tui(
    provider_class: Type[LLMChat], client_config: Dict[str, Any], session: PromptSession
) -> str | None:
    print_styled_html("\n<status>Fetching available models...</status>")
    try:
        models = provider_class.list_models(client_config=client_config)
        if not models:
            print_styled_html(
                f"<warning>No models found for {provider_class.__name__}. Check config or pull models.</warning>"
            )
            manual_model = input_styled(
                "Enter model name manually (or press Enter to cancel): ", session=session
            )
            return manual_model if manual_model else None

        print_styled_html("<prompt>Available models:</prompt>")
        for i, model_name in enumerate(models):
            print_styled_html(
                f"<menu-item-index>{i + 1}.</menu-item-index> <menu-item-text>{model_name}</menu-item-text>"
            )
        print_styled_html(
            "<menu-item-index>0.</menu-item-index> <cancel>Cancel / Enter manually</cancel>"
        )

        while True:
            choice_input = input_styled(
                f"Select model (1-{len(models)}, or 0): ",
                session=session,
                completer=WordCompleter([str(i) for i in range(len(models) + 1)]),
            )
            try:
                model_choice_idx = int(choice_input)
                if model_choice_idx == 0:
                    manual_model = input_styled(
                        "Enter model name manually (or press Enter to cancel): ", session=session
                    )
                    return manual_model if manual_model else None
                elif 1 <= model_choice_idx <= len(models):
                    return models[model_choice_idx - 1]
                else:
                    print_styled_html(
                        f"<error>Invalid choice. Enter number between 0 and {len(models)}.</error>"
                    )
            except ValueError:
                print_styled_html("<error>Invalid input. Please enter a number.</error>")
    except Exception as e:
        logger.error(f"Could not retrieve models for {provider_class.__name__}: {e}", exc_info=True)
        print_styled_html(f"<error>Error fetching models: {e}</error>")
        manual_model = input_styled(
            "Enter model name manually (or press Enter to cancel): ", session=session
        )
        return manual_model if manual_model else None


def _handle_mongo_context_tui(
    session: PromptSession,
    chat_history: List[Dict[str, str]],
    current_mongo_conn: Optional[Dict[str, str]] = None,
):
    print_styled_html("\n<mongo-context-header>--- Load MongoDB Context ---</mongo-context-header>")
    default_uri = (
        current_mongo_conn.get("uri")
        if current_mongo_conn
        else os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    )
    default_db = current_mongo_conn.get("database") if current_mongo_conn else ""

    mongo_uri = input_styled(
        "MongoDB URI: ", session=session, validator=MongoURIValidator(), default=default_uri
    )
    db_name = input_styled(
        "Database Name: ", session=session, validator=NotEmptyValidator(), default=default_db
    )
    collection_name = input_styled(
        "Collection Name: ", session=session, validator=NotEmptyValidator()
    )
    sample_size_str = input_styled(
        "Number of newest documents to sample for context: ",
        session=session,
        default="3",
        validator=NotEmptyValidator(),
    )
    try:
        sample_size = int(sample_size_str)
        if sample_size <= 0:
            print_styled_html("<error>Sample size must be positive. Aborting.</error>")
            return
    except ValueError:
        print_styled_html("<error>Invalid sample size. Aborting.</error>")
        return

    fields_str = input_styled(
        "Fields to include from sample documents (comma-separated, blank for all): ",
        session=session,
        validator=None,
    )
    fields_list = [f.strip() for f in fields_str.split(",")] if fields_str else None

    raw_sample_docs_for_schema_str = input_styled(
        "Sample size for schema/stats inference (-1 for all, default 100): ",
        session=session,
        default="100",
        validator=None,
    )
    try:
        sample_size_for_schema = (
            100 if raw_sample_docs_for_schema_str == "" else int(raw_sample_docs_for_schema_str)
        )
    except ValueError:
        print_styled_html("<error>Invalid sample size for schema. Using 100.</error>")
        sample_size_for_schema = 100

    try:
        print_styled_html("<status>Fetching context data from MongoDB...</status>")
        sampled_docs = core_extractor.get_newest_documents(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name,
            sample_size=sample_size,
            fields=fields_list,
        )
        mongo_collection = core_analyser.SchemaAnalyser.get_collection(
            uri=mongo_uri, db_name=db_name, collection_name=collection_name
        )
        schema_data, field_stats_data = core_analyser.SchemaAnalyser.infer_schema_and_field_stats(
            mongo_collection, sample_size=sample_size_for_schema
        )
        hierarchical_schema = core_analyser.SchemaAnalyser.schema_to_hierarchical(schema_data)
        general_coll_stats = core_analyser.SchemaAnalyser.get_collection_general_statistics(
            mongo_collection
        )

        context_parts = [
            f"Context for MongoDB collection: {db_name}.{collection_name}\n",
            "--- General Collection Statistics ---",
        ]
        if "error" in general_coll_stats:
            context_parts.append(f"Could not fetch general stats: {general_coll_stats['error']}")
        else:
            context_parts.extend(
                [
                    f"  Documents: {general_coll_stats.get('document_count', 'N/A'):,}",
                    f"  Avg. Object Size: {_format_bytes_tui(general_coll_stats.get('avg_obj_size_bytes'))}",
                    f"  Total Data Size: {_format_bytes_tui(general_coll_stats.get('total_size_bytes'))}",
                    f"  Total Storage Size: {_format_bytes_tui(general_coll_stats.get('storage_size_bytes'))}",
                    f"  Indexes: {general_coll_stats.get('nindexes', 'N/A')}",
                    f"  Total Index Size: {_format_bytes_tui(general_coll_stats.get('total_index_size_bytes'))}",
                ]
            )
        context_parts.extend(
            [
                "\n",
                "--- Inferred Schema (Hierarchical) ---",
                json.dumps(hierarchical_schema, indent=2),
                "\n",
                "--- Field-level Statistics (Cardinality, Missing %) ---",
            ]
        )
        for field, details in schema_data.items():
            f_stats = field_stats_data.get(field, {})
            card, miss = f_stats.get("cardinality", "N/A"), f_stats.get("missing_percentage", "N/A")
            miss_str = f"{miss:.1f}%" if isinstance(miss, (float, int)) else "N/A"
            context_parts.append(
                f"  {field}: Type={details['type']}, Cardinality={card}, Missing={miss_str}"
            )
        context_parts.extend(
            [
                "\n",
                f"--- {sample_size} Newest Sample Documents (Fields: {fields_str if fields_str else 'all'}) ---",
            ]
        )
        context_parts.append(
            json.dumps(sampled_docs, indent=2)
            if sampled_docs
            else "No sample documents were fetched."
        )

        full_context_str = "\n".join(context_parts)
        max_display_len = 1500
        display_context_snippet = (
            (
                full_context_str[:max_display_len]
                + "...\n\n[Context is very long, only showing snippet]"
            )
            if len(full_context_str) > max_display_len
            else full_context_str
        )

        print_styled_html(
            "<mongo-context-section>--- Combined MongoDB Context Snippet ---</mongo-context-section>"
        )
        for line in html.escape(display_context_snippet).splitlines():
            print_styled_html(f"<mongo-context-body>{line}</mongo-context-body>")

        chat_history.append(
            {
                "role": "user",
                "content": f"Based on the following information about the MongoDB collection '{db_name}.{collection_name}', please answer my questions.\n\n{full_context_str}",
            }
        )
        print_styled_html("<success>Full MongoDB context loaded into chat history.</success>")
    except Exception as e:
        logger.error(f"Failed to load MongoDB context: {e}", exc_info=True)
        print_styled_html(f"<error>Failed to load MongoDB context: {e}</error>")


def _run_chat_session_tui(
    chat_instance: LLMChat,
    session: PromptSession,
    initial_history: List[Dict[str, str]] = None,
    mongo_conn_details: Optional[Dict[str, str]] = None,
):
    ai_name_display = chat_instance.model_name.split("/")[-1]
    print_styled_html(
        f"\n<info>Chatting with</info> <ai-name>{ai_name_display}</ai-name> <info>via {chat_instance.__class__.__name__}.</info>"
    )
    print_styled_html(
        "<info>Type <command>/exit</command> or <command>/quit</command> to end session.</info>"
    )
    print_styled_html("<info>Type <command>/history</command> to view session history.</info>")
    print_styled_html(
        "<info>Type <command>/mongo_context</command> to load data, schema, and stats from MongoDB.</info>"
    )
    print_styled_html("<info>---</info>")

    chat_history: List[Dict[str, str]] = initial_history or []
    while True:
        try:
            user_input = session.prompt(HTML("<user-prompt>You: </user-prompt>")).strip()
            if user_input.lower() in ["/exit", "/quit"]:
                print_styled_html("<info>Exiting chat session.</info>")
                break
            if user_input.lower() == "/history":
                if not chat_history:
                    print_styled_html("<info>Chat history is empty.</info>")
                else:
                    print_styled_html("\n<info>--- Chat History ---</info>")
                    for entry in chat_history:
                        role_style = (
                            "history-role-user"
                            if entry["role"] == "user"
                            else "history-role-assistant"
                        )
                        print_styled_html(
                            f"<{role_style}>{entry['role'].capitalize()}:</{role_style}> <history-content>{html.escape(entry['content'])}</history-content>"
                        )
                    print_styled_html("<info>--- End of History ---</info>")
                continue
            if user_input.lower() == "/mongo_context":
                _handle_mongo_context_tui(session, chat_history, mongo_conn_details)
                continue
            if not user_input:
                continue

            current_turn_user_message = {"role": "user", "content": user_input}
            print_formatted_text(
                HTML(f"<ai-name>{html.escape(ai_name_display)}:</ai-name> "),
                end="",
                style=unified_tui_style,
            )
            full_response = ""
            try:
                for chunk in chat_instance.stream_message(user_input, history=chat_history.copy()):
                    print_formatted_text(
                        HTML(f"<ai-response>{html.escape(chunk)}</ai-response>"),
                        end="",
                        style=unified_tui_style,
                        flush=True,
                    )
                    full_response += chunk
                print()
            except Exception as e:
                logger.error(
                    f"Error during streaming with {chat_instance.model_name}: {e}", exc_info=True
                )
                print_styled_html(f"\n<error>Error during streaming: {e}</error>")
                continue
            chat_history.append(current_turn_user_message)
            chat_history.append({"role": "assistant", "content": full_response})
        except (KeyboardInterrupt, EOFError):
            print_styled_html("\n<cancel>Exiting chat session.</cancel>")
            break
        except Exception as e:
            logger.error(f"Unexpected error in chat session: {e}", exc_info=True)
            print_styled_html(f"\n<error>An unexpected error occurred: {e}</error>")
            break


def _llm_chat_entry_point_tui(
    session: PromptSession, initial_mongo_conn_details: Optional[Dict[str, str]] = None
):
    print_styled_html("<welcome>Welcome to the LLM Chat Interface!</welcome>")
    llm_provider_class = _select_llm_provider_tui(session)
    if not llm_provider_class:
        print_styled_html("<info>No provider selected. Returning to main menu.</info>")
        return

    config_params = _get_provider_config_tui(llm_provider_class, session)
    selected_model_name = _select_model_tui(llm_provider_class, config_params, session)
    if not selected_model_name:
        print_styled_html("<info>No model selected. Returning to main menu.</info>")
        return

    try:
        chat_instance = llm_provider_class(model_name=selected_model_name, **config_params)
        _run_chat_session_tui(chat_instance, session, mongo_conn_details=initial_mongo_conn_details)
    except (ValueError, ConnectionError, PermissionError) as e:
        logger.error(
            f"Error initializing chat with model {selected_model_name}: {e}", exc_info=False
        )
        print_styled_html(f"<error>Error initializing LLM client: {e}</error>")
    except Exception as e:
        logger.error(f"Unexpected error initializing chat: {e}", exc_info=True)
        print_styled_html(f"<error>An unexpected error occurred: {e}</error>")
    finally:
        print_styled_html("<info>LLM Chat session ended. Returning to main menu.</info>")


# --- Main Interactive TUI (Entry Point) ---


def interactive_main_menu(session: PromptSession):
    conn_details: Optional[Dict[str, str]] = None
    while True:
        print_styled_html("\n<welcome>Mongo Analyser - Interactive Mode</welcome>")
        if conn_details and conn_details.get("uri") and conn_details.get("database"):
            db_name_display = html.escape(conn_details.get("database", "N/A"))
            uri_display = html.escape(
                core_shared.redact_uri_password(conn_details.get("uri", "N/A"))
            )
            print_styled_html(
                f"<info>Current Connection: <parameter>{uri_display}</parameter> (DB: <parameter>{db_name_display}</parameter>)</info>"
            )
        else:
            print_styled_html(
                "<warning>Not connected to MongoDB. Set connection first (Option 1).</warning>"
            )

        menu_options = {
            "1": "Set/Change MongoDB Connection",
            "2": "List Collections in Current Database",
            "3": "Extract Collection Data (Newest First)",
            "4": "Analyze Live Collection (Schema, Field Stats, General Stats)",
            "5": "Display Schema from File",
            "6": "LLM Chat Tools (with MongoDB Context)",
            "0": "Exit Interactive Mode",
        }
        for k, v in menu_options.items():
            print_styled_html(
                f"<menu-item-index>{k}.</menu-item-index> <menu-item-text>{html.escape(v)}</menu-item-text>"
            )

        choice = input_styled(
            "Select an option: ", session, completer=WordCompleter(menu_options.keys())
        )
        if choice == "1":
            conn_details = _get_mongo_connection_details_tui(session, conn_details)
        elif choice == "2":
            if not conn_details or not conn_details.get("uri") or not conn_details.get("database"):
                print_styled_html("<error>Please set MongoDB connection first.</error>")
                continue
            _handle_list_collections_tui(session, conn_details)
        elif choice == "3":
            if not conn_details or not conn_details.get("uri") or not conn_details.get("database"):
                print_styled_html("<error>Please set MongoDB connection first.</error>")
                continue
            _handle_extraction_tui(session, conn_details)
        elif choice == "4":
            if not conn_details or not conn_details.get("uri") or not conn_details.get("database"):
                print_styled_html("<error>Please set MongoDB connection first.</error>")
                continue
            _handle_schema_analysis_live_tui(session, conn_details)
        elif choice == "5":
            _handle_schema_analysis_file_tui(session)
        elif choice == "6":
            print_styled_html("<info>Launching LLM Chat Tools...</info>")
            _llm_chat_entry_point_tui(
                session, initial_mongo_conn_details=conn_details
            )  # Pass current mongo conn
            print_styled_html("<info>Returned from LLM Chat Tools.</info>")  # Back to main menu
        elif choice == "0":
            print_styled_html("<info>Disconnecting from MongoDB if connected...</info>")
            core_db_manager.disconnect_mongo(core_db_manager.DEFAULT_ALIAS)
            print_styled_html("<info>Exiting Interactive Mode.</info>")
            break
        else:
            print_styled_html("<error>Invalid option. Please try again.</error>")
        print_styled_html("<info>---</info>")


def main_interactive_tui():
    """Main entry point for the interactive TUI."""
    history_file_path = Path(os.path.expanduser("~")) / ".mongo_analyser_interactive_history.txt"
    try:
        history_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Use the unified style for the main session
        session = PromptSession(
            history=FileHistory(str(history_file_path)),
            auto_suggest=AutoSuggestFromHistory(),
            style=unified_tui_style,
        )
    except OSError as e:
        print(
            f"[Warning] Could not create history file at {history_file_path}: {e}. Using session without history/style."
        )
        session = PromptSession()  # Fallback session

    try:
        interactive_main_menu(session)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting Interactive Mode.")
    except Exception as e:
        logger.critical("Critical error in main_interactive_tui loop.", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}. Exiting.")
    finally:
        core_db_manager.disconnect_mongo(core_db_manager.DEFAULT_ALIAS)
        print("Interactive session ended. MongoDB connection closed if active.")


if __name__ == "__main__":  # For testing this TUI module directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main_interactive_tui()
