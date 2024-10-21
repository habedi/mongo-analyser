#!/bin/bash

# Make sure `pipx install mongo-analyser` or `uv tool install mongo-analyser` is run before executing this script :)
export PATH=$PATH:~/.local/bin

ANALYSE_SCHEMA="mongo_analyser analyse_schema"
EXTRACT_DATA="mongo_analyser extract_data"

print_usage() {
    echo "Usage: $0 [DB_NAME] [COLLECTION_NAME] [SAMPLE_SIZE]"
    echo "Default values:"
    echo "  DB_NAME: admin"
    echo "  COLLECTION_NAME: system.version"
    echo "  SAMPLE_SIZE: 100"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

DB_NAME=${1:-admin}
COLLECTION_NAME=${2:-system.version}
SAMPLE_SIZE=${3:-100}
TIME_ZONE="CET"
DATA_DIR="data"

mkdir -p "$DATA_DIR/$DB_NAME"

PREFIX="$COLLECTION_NAME"
SCHEMA_FILE="$DATA_DIR/$DB_NAME/${PREFIX}_schema.json"
METADATA_FILE="$DATA_DIR/$DB_NAME/${PREFIX}_metadata.csv"
OUTPUT_FILE="$DATA_DIR/$DB_NAME/${PREFIX}_data.json.gz"

echo "Database: $DB_NAME"
echo "Collection: $COLLECTION_NAME"
echo "Sample Size: $SAMPLE_SIZE"
echo "Timezone: $TIME_ZONE"
echo "Schema File: $SCHEMA_FILE"
echo "Metadata File: $METADATA_FILE"
echo "Output File: $OUTPUT_FILE"

extract_schema() {
    echo "Extracting schema..."
    time $ANALYSE_SCHEMA --database "$DB_NAME" --collection "$COLLECTION_NAME" --sample_size "$SAMPLE_SIZE" \
    --schema_file "$SCHEMA_FILE" --metadata_file "$METADATA_FILE"
}

extract_data() {
    echo "Extracting data..."
    time $EXTRACT_DATA --database "$DB_NAME" --collection "$COLLECTION_NAME" --timezone "$TIME_ZONE" \
    --output_file "$OUTPUT_FILE" --schema_file "$SCHEMA_FILE" --limit 100000
}

# Extract schema and data
extract_schema
extract_data
