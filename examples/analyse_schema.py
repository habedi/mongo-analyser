from mongo_analyser import SchemaAnalyser

# MongoDB connection details
mongo_uri = SchemaAnalyser.build_mongo_uri("localhost", 27017)
collection = SchemaAnalyser.connect_mongo(mongo_uri, "admin", "system.version")

# Infer the schema and statistics
schema, stats = SchemaAnalyser.infer_schema_and_stats(collection, sample_size=1000)

# Print the schema and metadata as JSON
print(f"Schema: {schema}")

# Print the statistics as JSON
print(f"Stats: {stats}")

# Print the schema and metadata as a table
headers = ["Field", "Type", "Cardinality", "Missing (%)"]
rows = []
for field, details in schema.items():
    field_stats = stats.get(field, {})
    cardinality = field_stats.get("cardinality", "N/A")
    missing_percentage = field_stats.get("missing_percentage", "N/A")
    rows.append([field, details["type"], cardinality, round(missing_percentage, 1)])

SchemaAnalyser.draw_unicode_table(headers, rows)

# Save the schema to a JSON file
SchemaAnalyser.save_schema_to_json(schema, "schema.json")
