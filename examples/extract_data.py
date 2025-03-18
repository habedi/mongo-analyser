import io
import json

import pytz

from mongo_analyser import DataExtractor

# Build MongoDB URI
mongo_uri = DataExtractor.build_mongo_uri("localhost", 27017)

# Load the schema from the JSON file
with io.open("schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

# Extract data from the MongoDB collection
DataExtractor.extract_data(
    mongo_uri,
    "admin",
    "system.users",
    schema,
    "output.json.gz",
    pytz.timezone("UTC"),
    batch_size=1000,
    limit=100,
)

# Output: Data should be extracted and saved to the output.json.gz file on success
