import argparse
import io
import json
import sys # Import sys for exiting or handling no command

import pytz # Assuming this is still needed for your core functions
from pymongo.errors import ConnectionFailure

# Core functionalities (used by existing CLI commands)
from .core.analyser import SchemaAnalyser
from .core.extractor import DataExtractor

# Import the main TUI functions
from .llm_chat.tui import main_tui as llm_chat_main_tui
from .core.tui import main_core_tui


def extract_data_cli(args) -> None: # Renamed to avoid clash if used elsewhere
    # Build MongoDB URI
    mongo_uri = DataExtractor.build_mongo_uri(
        args.host, args.port, args.username, args.password
    )

    # Load the schema from the schema file
    with io.open(args.schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Get the timezone from the argument
    tz = pytz.timezone(args.timezone)

    # Try to connect and extract data
    try:
        DataExtractor.extract_data(
            mongo_uri,
            args.database,
            args.collection,
            schema,
            args.output_file,
            tz,
            args.batch_size,
            args.limit,
        )
        print(f"Data successfully exported to {args.output_file}")
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
    except FileNotFoundError:
        print(f"Error: Schema file not found at {args.schema_file}")
    except Exception as e:
        print(f"An unexpected error occurred during data extraction: {e}")


def analyse_schema_cli(args) -> None: # Renamed to avoid clash
    mongo_uri = SchemaAnalyser.build_mongo_uri(args.host, args.port, args.username, args.password)
    try:
        collection = SchemaAnalyser.connect_mongo(mongo_uri, args.database, args.collection)
        schema, stats = SchemaAnalyser.infer_schema_and_stats(collection, sample_size=args.sample_size)

        print(f"Using a sample size of {args.sample_size} documents to infer the schema and statistics")

        headers = ["Field", "Type", "Cardinality", "Missing (%)"]
        rows = []
        for field, details in schema.items():
            field_stats = stats.get(field, {})
            cardinality = field_stats.get("cardinality", "N/A")
            missing_percentage = field_stats.get("missing_percentage", "N/A")
            rows.append([field, details["type"], cardinality, round(missing_percentage, 1) if isinstance(missing_percentage, (int, float)) else "N/A"])

        if args.show_table:
            SchemaAnalyser.draw_unicode_table(headers, rows)
        else:
            print("Schema with Cardinality and Missing Percentage:")
            for i, row in enumerate(rows, start=1):
                print(
                    f"Field[{i}]: {row[0]}, Type: {row[1]}, Cardinality: "
                    f"{row[2]}, Missing (%): {row[3]}"
                )

        if args.schema_file:
            SchemaAnalyser.save_schema_to_json(schema, args.schema_file)
            print(f"Schema saved to {args.schema_file}")


        if args.metadata_file:
            SchemaAnalyser.save_table_to_csv(headers, rows, args.metadata_file)
            print(f"Metadata table saved to {args.metadata_file}")

    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during schema analysis: {e}")


def print_custom_help(parser, subparsers_action):
    # subparsers_action is the _SubParsersAction object
    print(f"{parser.description}\n")
    print("Commands:")
    for cmd, subparser in subparsers_action.choices.items():
        print(f"  {cmd:<18} {subparser.description}")
    print("\nUse <command> --help for more information on a specific command.")


def main():
    analyse_schema_description = "CLI: Analyze and infer MongoDB collection structure."
    extract_data_description = "CLI: Export data from MongoDB collection to a file."
    core_tui_description = "TUI: Interactive MongoDB analysis and extraction."
    chat_tui_description = "TUI: Interactive LLM chat interface."

    parser = argparse.ArgumentParser(
        description="Mongo Analyser: Tools for MongoDB analysis and LLM interaction.",
        prog="mongo-analyser", # Added prog for better help messages
        add_help=False # We'll handle help manually to show subparser descriptions
    )
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')


    # Store the subparsers action for later use in custom help
    subparsers_action = parser.add_subparsers(dest="command", title="Available Commands")

    # Subcommand: analyse_schema (CLI)
    parser_analyse_schema = subparsers_action.add_parser(
        "analyse_schema_cli", # Renamed to distinguish from TUI
        description=analyse_schema_description,
        help=analyse_schema_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_analyse_schema.add_argument(
        "--host", type=str, default="localhost", help="IP or hostname of MongoDB server"
    )
    parser_analyse_schema.add_argument("--port", type=int, default=27017, help="MongoDB port")
    parser_analyse_schema.add_argument("--username", type=str, help="MongoDB username (optional)")
    parser_analyse_schema.add_argument("--password", type=str, help="MongoDB password (optional)")
    parser_analyse_schema.add_argument("--database", type=str, default="admin", help="Database name")
    parser_analyse_schema.add_argument(
        "--collection", type=str, default="system.version", help="Collection name"
    )
    parser_analyse_schema.add_argument(
        "--sample_size", type=int, default=10000, help="Documents to sample for schema inference"
    )
    parser_analyse_schema.add_argument(
        "--show_table", type=SchemaAnalyser.str2bool, nargs="?", const=True, default=True,
        help="Show schema as a table"
    )
    parser_analyse_schema.add_argument(
        "--schema_file", type=str, default="schema.json", help="Path to store schema JSON file"
    )
    parser_analyse_schema.add_argument(
        "--metadata_file", type=str, default="metadata.csv", help="Path to store metadata CSV file"
    )
    parser_analyse_schema.set_defaults(func=analyse_schema_cli)


    # Subcommand: extract_data (CLI)
    parser_extract_data = subparsers_action.add_parser(
        "extract_data_cli", # Renamed to distinguish from TUI
        description=extract_data_description,
        help=extract_data_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_extract_data.add_argument(
        "--host", type=str, default="localhost", help="IP or hostname of MongoDB server"
    )
    parser_extract_data.add_argument("--port", type=int, default=27017, help="MongoDB port")
    parser_extract_data.add_argument("--username", type=str, default=None, help="MongoDB username")
    parser_extract_data.add_argument("--password", type=str, default=None, help="MongoDB password")
    parser_extract_data.add_argument("--database", type=str, default="admin", help="Database name")
    parser_extract_data.add_argument(
        "--collection", type=str, default="system.version", help="Collection name"
    )
    parser_extract_data.add_argument(
        "--schema_file", type=str, default="schema.json", help="Path to schema file"
    )
    parser_extract_data.add_argument(
        "--output_file", type=str, default="output.json.gz", help="Path to output compressed JSON"
    )
    parser_extract_data.add_argument(
        "--timezone", type=str, default="CET", help="Timezone for datetime fields"
    )
    parser_extract_data.add_argument(
        "--batch_size", type=int, default=10000, help="Batch size for reading from MongoDB"
    )
    parser_extract_data.add_argument(
        "--limit", type=int, default=-1, help="Max documents to read (-1 for no limit)"
    )
    parser_extract_data.set_defaults(func=extract_data_cli)

    # Subcommand: core_tui
    parser_core_tui = subparsers_action.add_parser(
        "core_tui",
        description=core_tui_description,
        help=core_tui_description
    )
    parser_core_tui.set_defaults(func=lambda args: main_core_tui()) # No other args needed

    # Subcommand: chat_tui
    parser_chat_tui = subparsers_action.add_parser(
        "chat_tui",
        description=chat_tui_description,
        help=chat_tui_description
    )
    parser_chat_tui.set_defaults(func=lambda args: llm_chat_main_tui()) # No other args needed

    # Parse arguments
    args = parser.parse_args()

    if args.help and args.command is None : # if -h is used without a command
        print_custom_help(parser, subparsers_action)
        sys.exit(0)
    elif hasattr(args, 'func'):
        args.func(args)
    else:
        # No command was provided, or -h was used with a specific command (handled by argparse)
        # If -h was used with a specific command, argparse handles it before this point.
        # So, this branch means no command was given.
        print_custom_help(parser, subparsers_action)
        sys.exit(1)


if __name__ == "__main__":
    main()
