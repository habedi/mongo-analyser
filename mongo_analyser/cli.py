import argparse
import io
import json

import pytz
from pymongo.errors import ConnectionFailure

import mongo_analyser.extractor as data_extractor
from mongo_analyser import SchemaAnalyser


def extract_data(args) -> None:
    # Build MongoDB URI
    mongo_uri = data_extractor.DataExtractor.build_mongo_uri(
        args.host, args.port, args.username, args.password
    )

    # Load the schema from the schema file
    with io.open(args.schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Get the timezone from the argument
    tz = pytz.timezone(args.timezone)

    # Try to connect and extract data
    try:
        data_extractor.DataExtractor.extract_data(
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


def analyse_schema(args) -> None:
    mongo_uri = SchemaAnalyser.build_mongo_uri(args.host, args.port, args.username, args.password)
    collection = SchemaAnalyser.connect_mongo(mongo_uri, args.database, args.collection)
    schema, stats = SchemaAnalyser.infer_schema_and_stats(collection, sample_size=args.sample_size)

    print(f"Using a sample size of {args.sample_size} documents to infer the schema and statistics")

    headers = ["Field", "Type", "Cardinality", "Missing (%)"]
    rows = []
    for field, details in schema.items():
        field_stats = stats.get(field, {})
        cardinality = field_stats.get("cardinality", "N/A")
        missing_percentage = field_stats.get("missing_percentage", "N/A")
        rows.append([field, details["type"], cardinality, round(missing_percentage, 1)])

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

    if args.metadata_file:
        SchemaAnalyser.save_table_to_csv(headers, rows, args.metadata_file)


def print_custom_help(parser, subparsers):
    print(parser.description)
    print("\nCommands:")
    for cmd, subparser in subparsers.choices.items():
        print(f"  {cmd:<15} {subparser.description}")


def main():
    analyse_schema_description = "Analyze and infer the structure of a MongoDB collection"
    extract_data_description = "Export data from a MongoDB collection to a file"

    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Usage: mongo_analyser <command> [<args>]", add_help=False
    )

    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: analyse_schema
    parser_analyse_schema = subparsers.add_parser(
        "analyse_schema", description=analyse_schema_description
    )
    parser_analyse_schema.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="IP address or hostname of MongoDB server (default: localhost)",
    )
    parser_analyse_schema.add_argument(
        "--port", type=int, default=27017, help="MongoDB port (default: 27017)"
    )
    parser_analyse_schema.add_argument("--username", type=str, help="MongoDB username (optional)")
    parser_analyse_schema.add_argument("--password", type=str, help="MongoDB password (optional)")
    parser_analyse_schema.add_argument(
        "--database", type=str, default="admin", help="Database name (default: admin)"
    )
    parser_analyse_schema.add_argument(
        "--collection",
        type=str,
        default="system.version",
        help="Collection name  (default: system.version)",
    )
    parser_analyse_schema.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of documents to sample for schema inference (default: 10000)",
    )
    parser_analyse_schema.add_argument(
        "--show_table",
        type=SchemaAnalyser.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Show the schema and metadata as a table in the output (default: True)",
    )
    parser_analyse_schema.add_argument(
        "--schema_file",
        type=str,
        default="schema.json",
        help="Path to store the schema file as a JSON file (default: ./schema.json)",
    )
    parser_analyse_schema.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.csv",
        help="Path to store the metadata file as a CSV file (default: ./metadata.csv)",
    )

    # Subcommand: extract_data
    parser_extract_data = subparsers.add_parser(
        "extract_data", description=extract_data_description
    )
    parser_extract_data.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="IP address or hostname of MongoDB server (default: localhost)",
    )
    parser_extract_data.add_argument(
        "--port", type=int, default=27017, help="MongoDB port (default: 27017)"
    )
    parser_extract_data.add_argument(
        "--username", type=str, default=None, help="MongoDB username (optional)"
    )
    parser_extract_data.add_argument(
        "--password", type=str, default=None, help="MongoDB password (optional)"
    )
    parser_extract_data.add_argument(
        "--database", type=str, default="admin", help="Database name (default: admin)"
    )
    parser_extract_data.add_argument(
        "--collection",
        type=str,
        default="system.version",
        help="Collection name (default: system.version)",
    )
    parser_extract_data.add_argument(
        "--schema_file",
        type=str,
        default="schema.json",
        help="Path to the schema file (default: ./schema.json)",
    )
    parser_extract_data.add_argument(
        "--output_file",
        type=str,
        default="output.json.gz",
        help="Path to the output compressed JSON file (default: ./output.json.gz)",
    )
    parser_extract_data.add_argument(
        "--timezone", type=str, default="CET", help="Timezone for datetime fields (default: CET)"
    )
    parser_extract_data.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for reading data from MongoDB (default: 10000)",
    )
    parser_extract_data.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max number of documents to read (default: -1 for no limit)",
    )

    # Parse arguments and dispatch to the appropriate function
    args = parser.parse_args()

    # If no command is provided, display custom help
    if args.command is None:
        print_custom_help(parser, subparsers)
    elif args.command == "analyse_schema":
        analyse_schema(args)
    elif args.command == "extract_data":
        extract_data(args)


if __name__ == "__main__":
    main()
