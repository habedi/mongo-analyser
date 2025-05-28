import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Union

import pytz
from pymongo.errors import (
    ConnectionFailure as PyMongoConnectionFailure,
)
from pymongo.errors import (
    OperationFailure as PyMongoOperationFailure,
)

from mongo_analyser.core import db as db_manager
from mongo_analyser.core import shared
from mongo_analyser.core.analyser import SchemaAnalyser
from mongo_analyser.core.extractor import DataExtractor
from mongo_analyser.tui import main_interactive_tui  # Updated import

logger = logging.getLogger(__name__)


def _format_bytes_cli(size_bytes: Union[int, float, None]) -> str:
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


def extract_data_command(args: argparse.Namespace) -> None:
    mongo_uri = shared.build_mongo_uri(
        args.host, args.port, args.username, args.password, args.mongo_params
    )

    schema_file = Path(args.schema_file)
    if not schema_file.is_file():
        print(f"Error: Schema file not found at {schema_file}")
        logger.error(f"Schema file not found: {schema_file}")
        sys.exit(1)

    try:
        with schema_file.open("r", encoding="utf-8") as f:
            schema_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in schema file {schema_file}")
        logger.error(f"Invalid JSON: {schema_file}")
        sys.exit(1)
    except IOError as e:
        print(f"Error: Could not read schema file {schema_file}: {e}")
        logger.error(f"Could not read: {schema_file}: {e}")
        sys.exit(1)

    try:
        tz = pytz.timezone(args.timezone)
    except pytz.UnknownTimeZoneError:
        print(f"Error: Unknown timezone '{args.timezone}'. Using UTC.")
        logger.warning(f"Unknown timezone '{args.timezone}'. Default UTC.")
        tz = pytz.utc

    try:
        print(
            f"Starting data extraction (newest documents first) from {args.database}.{args.collection}..."
        )
        DataExtractor.extract_data(
            mongo_uri,
            args.database,
            args.collection,
            schema_data,
            args.output_file,
            tz,
            args.batch_size,
            args.limit,
        )
        print(f"Data successfully exported to {args.output_file}")
    except (PyMongoConnectionFailure, PyMongoOperationFailure, ConnectionError) as e:
        print(f"MongoDB connection or operation error: {e}")
        logger.critical(f"MongoDB error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data extraction: {e}")
        logger.exception("Extraction CLI error.")
        sys.exit(1)
    finally:
        db_manager.disconnect_mongo(db_manager.DEFAULT_ALIAS)


def analyse_schema_command(args: argparse.Namespace) -> None:
    mongo_uri = shared.build_mongo_uri(
        args.host, args.port, args.username, args.password, args.mongo_params
    )
    try:
        collection_obj = SchemaAnalyser.get_collection(mongo_uri, args.database, args.collection)
        actual_db_name = db_manager.get_mongo_db().name

        print(f"\n--- General Statistics for {actual_db_name}.{args.collection} ---")
        general_stats = SchemaAnalyser.get_collection_general_statistics(collection_obj)
        if "error" in general_stats:
            print(f"Could not fetch general statistics: {general_stats['error']}")
        else:
            print(f"  Namespace: {general_stats.get('ns', 'N/A')}")
            print(f"  Documents: {general_stats.get('document_count', 'N/A'):,}")
            print(
                f"  Avg. Object Size: {_format_bytes_cli(general_stats.get('avg_obj_size_bytes'))}"
            )
            print(f"  Total Data Size: {_format_bytes_cli(general_stats.get('total_size_bytes'))}")
            print(
                f"  Total Storage Size: {_format_bytes_cli(general_stats.get('storage_size_bytes'))}"
            )
            print(f"  Indexes: {general_stats.get('nindexes', 'N/A')}")
            print(
                f"  Total Index Size: {_format_bytes_cli(general_stats.get('total_index_size_bytes'))}"
            )
            print(f"  Capped: {'Yes' if general_stats.get('capped') else 'No'}")

        print(
            f"\n--- Field Schema & Statistics for {actual_db_name}.{args.collection} (Sample: {args.sample_size if args.sample_size >= 0 else 'all'}) ---"
        )
        schema, field_stats_data = SchemaAnalyser.infer_schema_and_field_stats(
            collection_obj, sample_size=args.sample_size
        )

        headers = ["Field", "Type", "Cardinality", "Missing (%)"]
        rows = []
        for field, details in schema.items():
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

        if args.show_table:
            SchemaAnalyser.draw_unicode_table(headers, rows)
        else:
            for i, row_data in enumerate(rows, start=1):
                print(
                    f"  Field[{i}]: {row_data[0]}, Type: {row_data[1]}, Card: {row_data[2]}, Miss%: {row_data[3]}"
                )

        if args.schema_file:
            SchemaAnalyser.save_schema_to_json(schema, args.schema_file)
        if args.metadata_file:
            SchemaAnalyser.save_table_to_csv(headers, rows, args.metadata_file)

    except (PyMongoConnectionFailure, PyMongoOperationFailure, ConnectionError) as e:
        print(f"MongoDB connection or operation error: {e}")
        logger.critical(f"MongoDB error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during schema analysis: {e}")
        logger.exception("Schema analysis CLI error.")
        sys.exit(1)
    finally:
        db_manager.disconnect_mongo(db_manager.DEFAULT_ALIAS)


def print_custom_help(parser: argparse.ArgumentParser, subparsers_action=None):
    print(f"{parser.prog} - {parser.description}\n")
    if subparsers_action:
        print("Commands:")
        for cmd, sub_parser in subparsers_action.choices.items():
            print(f"  {cmd:<22} {sub_parser.description}")
    else:
        parser.print_help()
    print("\nUse <command> --help for more info, or 'interactive' for the TUI.")
    print("Example: mongo-analyser analyse_schema --db myDB --coll myColl --help")


def main():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    parser = argparse.ArgumentParser(
        description="MongoDB Analyser: CLI tools and Interactive TUI.",
        prog="mongo-analyser",
        add_help=False,
    )
    parser.add_argument("--help", action="store_true", help="Show this help message and exit")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.5.0",  # Version bump
    )

    mongo_connection_parser = argparse.ArgumentParser(add_help=False)
    mongo_connection_parser.add_argument(
        "-H", "--host", type=str, default="localhost", help="MongoDB server host"
    )
    mongo_connection_parser.add_argument(
        "-P", "--port", type=int, default=27017, help="MongoDB server port"
    )
    mongo_connection_parser.add_argument(
        "-u", "--username", type=str, help="MongoDB username (optional)"
    )
    mongo_connection_parser.add_argument(
        "-p", "--password", type=str, help="MongoDB password (optional)"
    )
    mongo_connection_parser.add_argument(
        "-d", "--database", type=str, required=True, help="Target database name"
    )
    mongo_connection_parser.add_argument(
        "-c", "--collection", type=str, required=True, help="Target collection name"
    )
    mongo_connection_parser.add_argument(
        "-M",
        "--mongo_params",
        type=str,
        help="Additional MongoDB URI parameters (e.g., replicaSet=rs0&authSource=admin)",
    )

    subparsers_action = parser.add_subparsers(
        dest="command", title="Available Commands", metavar="<command>"
    )

    parser_analyse_schema = subparsers_action.add_parser(
        "analyse_schema",
        parents=[mongo_connection_parser],
        description="CLI: Analyze MongoDB collection structure, field stats, and general stats.",
        help="Analyze MongoDB collection (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser_analyse_schema.add_argument(
        "-s",
        "--sample_size",
        type=int,
        default=1000,
        help="Documents to sample for field analysis (-1 for all)",
    )
    parser_analyse_schema.add_argument(
        "-T",
        "--show_table",
        type=shared.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Show field schema & stats as a table",
    )
    parser_analyse_schema.add_argument(
        "-S", "--schema_file", type=str, help="Path to store field schema JSON (e.g., schema.json)"
    )
    parser_analyse_schema.add_argument(
        "-F",
        "--metadata_file",
        type=str,
        help="Path to store field statistics CSV (e.g., field_stats.csv)",
    )
    parser_analyse_schema.add_argument(
        "-h", "--help", action="store_true", help="Show this command's help message and exit"
    )
    parser_analyse_schema.set_defaults(func=analyse_schema_command)

    parser_extract_data = subparsers_action.add_parser(
        "extract_data",
        parents=[mongo_connection_parser],
        description="CLI: Export data (newest first) from MongoDB collection.",
        help="Export MongoDB collection data (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser_extract_data.add_argument(
        "-s",
        "--schema_file",
        type=str,
        required=True,
        help="Path to hierarchical schema JSON (from analyse_schema)",
    )
    parser_extract_data.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="output.json.gz",
        help="Output compressed JSON file",
    )
    parser_extract_data.add_argument(
        "-Z", "--timezone", type=str, default="UTC", help="Timezone for datetime localization"
    )
    parser_extract_data.add_argument(
        "-b", "--batch_size", type=int, default=1000, help="MongoDB read batch size"
    )
    parser_extract_data.add_argument(
        "-l",
        "--limit",
        type=int,
        default=-1,
        help="Max documents to read (-1 for all, newest first)",
    )
    parser_extract_data.add_argument(
        "-h", "--help", action="store_true", help="Show this command's help message and exit"
    )
    parser_extract_data.set_defaults(func=extract_data_command)

    parser_interactive_tui = subparsers_action.add_parser(
        "interactive",
        description="TUI: Launch interactive TUI for MongoDB and LLM operations.",
        help="Launch interactive TUI",
        add_help=False,
    )
    parser_interactive_tui.add_argument(
        "-h", "--help", action="store_true", help="Show this command's help message and exit"
    )
    parser_interactive_tui.set_defaults(func=lambda args_ns: main_interactive_tui())

    if len(sys.argv) == 1:
        print_custom_help(parser, subparsers_action)
        sys.exit(0)

    if "--help" in sys.argv and len(sys.argv) == 2:
        print_custom_help(parser, subparsers_action)
        sys.exit(0)
    if sys.argv[1] == "--help" and len(sys.argv) == 2:
        print_custom_help(parser, subparsers_action)
        sys.exit(0)

    args = parser.parse_args()

    if args.command and hasattr(args, "help") and args.help:
        if args.command in subparsers_action.choices:
            subparsers_action.choices[args.command].print_help()
        else:
            parser.print_help()
        sys.exit(0)
    elif args.help:
        print_custom_help(parser, subparsers_action)
        sys.exit(0)

    if hasattr(args, "func"):
        is_tui_command = args.command == "interactive"
        try:
            args.func(args)
        except Exception as e:
            logger.critical(f"Unhandled error in command '{args.command}': {e}", exc_info=True)
            print(f"Critical error executing command '{args.command}'. Check logs for details.")
        finally:
            if not is_tui_command:
                db_manager.disconnect_all_mongo()
                logger.debug(
                    f"Disconnected all MongoEngine aliases after CLI command '{args.command}'."
                )
    else:
        print_custom_help(parser, subparsers_action)
        sys.exit(1)


if __name__ == "__main__":
    main()
