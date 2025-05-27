import csv
import json
import logging
import uuid
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from bson import Binary, Decimal128, Int64, ObjectId
from pymongo.errors import (
    ConnectionFailure as PyMongoConnectionFailure,
)
from pymongo.errors import (
    OperationFailure as PyMongoOperationFailure,
)
from pymongo.synchronous.collection import Collection as PyMongoCollection

from . import db as db_manager
from . import shared

logger = logging.getLogger(__name__)


class SchemaAnalyser:
    @staticmethod
    def _make_hashable(item: Any) -> Any:
        if isinstance(item, dict):
            return frozenset((k, SchemaAnalyser._make_hashable(v)) for k, v in sorted(item.items()))
        elif isinstance(item, list):
            return tuple(SchemaAnalyser._make_hashable(i) for i in item)
        else:
            return item

    @staticmethod
    def extract_schema_and_stats(
        document: Dict,
        schema: Union[Dict, OrderedDict, None] = None,
        stats: Union[Dict, OrderedDict, None] = None,
        prefix: str = "",
    ) -> Tuple[Union[Dict, OrderedDict], Union[Dict, OrderedDict]]:
        if schema is None:
            schema = OrderedDict()
        if stats is None:
            stats = OrderedDict()

        for key, value in document.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if full_key not in stats:
                stats[full_key] = {
                    "values": set(),
                    "count": 0,
                    "type_counts": Counter(),
                    "numeric_min": float("inf"),
                    "numeric_max": float("-inf"),
                    "date_min": None,
                    "date_max": None,
                    "value_frequencies": Counter(),
                    "array_element_stats": {  # For stats of elements within arrays
                        "type_counts": Counter(),
                        "numeric_min": float("inf"),
                        "numeric_max": float("-inf"),
                        "date_min": None,
                        "date_max": None,
                        "value_frequencies": Counter(),
                    },
                }

            stats[full_key]["count"] += 1

            if isinstance(value, dict):
                SchemaAnalyser.extract_schema_and_stats(value, schema, stats, full_key)
            elif isinstance(value, list):
                SchemaAnalyser.handle_array(value, schema, stats, full_key)
            else:
                SchemaAnalyser.handle_simple_value(
                    value, schema, stats, full_key, is_array_element=False
                )
        return schema, stats

    @staticmethod
    def handle_array(
        value: List,
        schema: Union[Dict, OrderedDict],
        stats: Union[Dict, OrderedDict],
        full_key: str,
    ) -> None:
        current_field_stats = stats[full_key]

        if not value:
            schema[full_key] = {"type": "array<empty>"}
        else:
            element_types_for_schema = Counter()
            # Sample elements for schema type diversity, e.g., first 10 or a random sample
            sample_elements_for_schema = value[:10]

            for elem in sample_elements_for_schema:
                if isinstance(elem, dict):
                    element_types_for_schema.update(["dict"])
                elif isinstance(elem, ObjectId):
                    element_types_for_schema.update(["ObjectId"])
                elif isinstance(elem, uuid.UUID):
                    element_types_for_schema.update(["UUID"])
                elif isinstance(elem, Binary):
                    subtype_str = shared.binary_type_map.get(
                        elem.subtype, f"binary<subtype {elem.subtype}>"
                    )
                    element_types_for_schema.update([subtype_str])
                elif isinstance(elem, bool):
                    element_types_for_schema.update(["bool"])
                elif isinstance(elem, Int64):
                    element_types_for_schema.update(["int64"])
                elif isinstance(elem, int):
                    element_types_for_schema.update(["int32"])
                elif isinstance(elem, float):
                    element_types_for_schema.update(["double"])
                elif isinstance(elem, Decimal128):
                    element_types_for_schema.update(["decimal128"])
                elif isinstance(elem, datetime):
                    element_types_for_schema.update(["datetime"])
                else:
                    element_types_for_schema.update([type(elem).__name__])

            if len(element_types_for_schema) == 1:
                dominant_type = element_types_for_schema.most_common(1)[0][0]
                schema[full_key] = {"type": f"array<{dominant_type}>"}
            elif element_types_for_schema:
                schema[full_key] = {"type": "array<mixed>"}
            else:
                schema[full_key] = {"type": "array<unknown>"}

            # Process all elements for detailed statistics
            for elem in value:
                SchemaAnalyser.handle_simple_value(
                    elem,
                    {},
                    current_field_stats["array_element_stats"],
                    full_key,
                    is_array_element=True,
                )

        try:
            hashable_value = SchemaAnalyser._make_hashable(value)
            current_field_stats["values"].add(hashable_value)
        except TypeError:
            current_field_stats["values"].add(f"unhashable_array_len_{len(value)}")

    @staticmethod
    def handle_simple_value(
        value: Any,
        schema: Union[Dict, OrderedDict],
        stats_dict_to_update: Dict,
        full_key: str,
        is_array_element: bool,
    ) -> None:
        # If it's not an array element, schema is updated for the field itself.
        # If it is an array element, schema is not updated here (array type is determined in handle_array).
        # Stats are always updated in stats_dict_to_update.

        value_type_name = ""
        if isinstance(value, ObjectId):
            value_type_name = "binary<ObjectId>"
        elif isinstance(value, uuid.UUID):
            value_type_name = "binary<UUID>"
        elif isinstance(value, Binary):
            value_type_name = shared.binary_type_map.get(
                value.subtype, f"binary<subtype {value.subtype}>"
            )
        elif isinstance(value, bool):
            value_type_name = "bool"
        elif isinstance(value, Int64):
            value_type_name = "int64"
        elif isinstance(value, int):
            value_type_name = "int32"
        elif isinstance(value, float):
            value_type_name = "double"
        elif isinstance(value, Decimal128):
            value_type_name = "decimal128"
        elif isinstance(value, datetime):
            value_type_name = "datetime"
        else:
            value_type_name = type(value).__name__

        if not is_array_element:
            schema[full_key] = {"type": value_type_name}
            # For non-array elements, 'values' set is for the field itself
            try:
                stats_dict_to_update["values"].add(SchemaAnalyser._make_hashable(value))
            except TypeError:
                stats_dict_to_update["values"].add(f"unhashable_value_type_{type(value).__name__}")

        stats_dict_to_update["type_counts"].update([value_type_name])

        if isinstance(value, (int, float, Int64, Decimal128)):
            num_val = float(value.to_decimal()) if isinstance(value, Decimal128) else float(value)
            stats_dict_to_update["numeric_min"] = min(
                stats_dict_to_update.get("numeric_min", float("inf")), num_val
            )
            stats_dict_to_update["numeric_max"] = max(
                stats_dict_to_update.get("numeric_max", float("-inf")), num_val
            )
        elif isinstance(value, str):
            if len(value) < 256:  # Avoid very long strings in frequency counts
                stats_dict_to_update["value_frequencies"].update([value])
        elif isinstance(value, datetime):
            # Naive comparison, assumes datetimes are comparable or already UTC
            current_min_date = stats_dict_to_update.get("date_min")
            current_max_date = stats_dict_to_update.get("date_max")
            if current_min_date is None or value < current_min_date:
                stats_dict_to_update["date_min"] = value
            if current_max_date is None or value > current_max_date:
                stats_dict_to_update["date_max"] = value

    @staticmethod
    def get_collection(
        uri: str, db_name: str, collection_name: str, server_timeout_ms: int = 5000
    ) -> PyMongoCollection:
        if not db_manager.db_connection_active(
            uri=uri, db_name=db_name, server_timeout_ms=server_timeout_ms
        ):
            raise PyMongoConnectionFailure(
                f"Failed to establish or verify active connection to MongoDB for {db_name}"
            )
        database = db_manager.get_mongo_db()
        return database[collection_name]

    @staticmethod
    def list_collection_names(uri: str, db_name: str, server_timeout_ms: int = 5000) -> List[str]:
        if not db_manager.db_connection_active(
            uri=uri, db_name=db_name, server_timeout_ms=server_timeout_ms
        ):
            raise PyMongoConnectionFailure(
                f"Failed to establish or verify active connection to MongoDB for listing collections in {db_name}"
            )

        database = db_manager.get_mongo_db()
        try:
            names = sorted(database.list_collection_names())
            return names
        except PyMongoOperationFailure as e:
            logger.error(f"MongoDB operation failure listing collections for DB '{db_name}': {e}")
            raise

    @staticmethod
    def infer_schema_and_field_stats(
        collection: PyMongoCollection, sample_size: int, batch_size: int = 1000
    ) -> Tuple[Dict, Dict]:
        schema = OrderedDict()
        stats = OrderedDict()
        total_docs = 0

        try:
            if sample_size < 0:  # All documents
                # For very large collections, consider adding a warning or a practical limit
                # as find() without limit can be resource-intensive.
                # total_docs_in_coll = collection.estimated_document_count() # or count_documents({})
                # if total_docs_in_coll > 1_000_000: # Example threshold
                #    logger.warning(f"Analysing all documents in a large collection ({total_docs_in_coll} docs). This might take a long time.")
                documents = collection.find().batch_size(batch_size)
            else:
                documents = collection.aggregate([{"$sample": {"size": sample_size}}]).batch_size(
                    batch_size
                )

            for doc in documents:
                total_docs += 1
                schema, stats = SchemaAnalyser.extract_schema_and_stats(doc, schema, stats)
                if (
                    sample_size > 0 and total_docs >= sample_size
                ):  # Ensure we don't exceed sample_size if specified
                    break
        except PyMongoOperationFailure as e:
            logger.error(
                f"Error during schema inference on {collection.database.name}.{collection.name}: {e}"
            )
            raise

        final_stats_summary = OrderedDict()
        for key, field_stat_data in stats.items():
            cardinality = len(field_stat_data["values"])  # Cardinality of the field itself
            missing_count = total_docs - field_stat_data["count"]
            missing_percentage = (missing_count / total_docs) * 100 if total_docs > 0 else 0

            processed_stat: Dict[str, Any] = {
                "cardinality": cardinality,
                "missing_percentage": missing_percentage,
                "type_distribution": dict(field_stat_data["type_counts"].most_common(5)),
                # Types of the field itself
            }
            if field_stat_data["numeric_min"] != float("inf"):
                processed_stat["numeric_min"] = field_stat_data["numeric_min"]
                processed_stat["numeric_max"] = field_stat_data["numeric_max"]

            if field_stat_data["date_min"] is not None:
                processed_stat["date_min"] = field_stat_data["date_min"].isoformat()
                processed_stat["date_max"] = field_stat_data["date_max"].isoformat()

            if field_stat_data["value_frequencies"]:
                processed_stat["top_values"] = dict(
                    field_stat_data["value_frequencies"].most_common(5)
                )

            # Add array element stats if present
            array_el_stats = field_stat_data.get("array_element_stats", {})
            if array_el_stats.get("type_counts"):  # Check if any array elements were processed
                processed_stat["array_elements"] = {
                    "type_distribution": dict(array_el_stats["type_counts"].most_common(5))
                }
                if array_el_stats["numeric_min"] != float("inf"):
                    processed_stat["array_elements"]["numeric_min"] = array_el_stats["numeric_min"]
                    processed_stat["array_elements"]["numeric_max"] = array_el_stats["numeric_max"]
                if array_el_stats["date_min"] is not None:
                    processed_stat["array_elements"]["date_min"] = array_el_stats[
                        "date_min"
                    ].isoformat()
                    processed_stat["array_elements"]["date_max"] = array_el_stats[
                        "date_max"
                    ].isoformat()
                if array_el_stats["value_frequencies"]:
                    processed_stat["array_elements"]["top_values"] = dict(
                        array_el_stats["value_frequencies"].most_common(5)
                    )

            final_stats_summary[key] = processed_stat

        sorted_schema = OrderedDict(sorted(schema.items()))
        sorted_stats_summary = OrderedDict(sorted(final_stats_summary.items()))
        return dict(sorted_schema), dict(sorted_stats_summary)

    @staticmethod
    def get_collection_general_statistics(collection: PyMongoCollection) -> Dict:
        try:
            stats_cmd_result = collection.database.command("collStats", collection.name)
            general_stats = {
                "ns": stats_cmd_result.get("ns"),
                "document_count": stats_cmd_result.get("count"),
                "avg_obj_size_bytes": stats_cmd_result.get("avgObjSize"),
                "total_size_bytes": stats_cmd_result.get("size"),
                "storage_size_bytes": stats_cmd_result.get("storageSize"),
                "nindexes": stats_cmd_result.get("nindexes"),
                "total_index_size_bytes": stats_cmd_result.get("totalIndexSize"),
                "capped": stats_cmd_result.get("capped", False),
            }
            return general_stats
        except PyMongoOperationFailure as e:
            logger.error(
                f"Error getting collStats for {collection.database.name}.{collection.name}: {e}"
            )
            return {"error": str(e)}

    @staticmethod
    def draw_unicode_table(headers: List[str], rows: List[List[Any]]) -> None:
        if not rows:
            print("No data to display in table.")
            return

        # Calculate column widths safely, handling potential None or non-string items
        str_rows = [[str(item) for item in r] for r in rows]
        str_headers = [str(h) for h in headers]

        col_widths = [
            max(len(item) for item in col) for col in zip(*([str_headers] + str_rows), strict=False)
        ]

        def draw_separator(sep_type: str) -> None:
            parts = {
                "top": ("┌", "┬", "┐"),
                "mid": ("├", "┼", "┤"),
                "bottom": ("└", "┴", "┘"),
                "line": "─",
            }
            start, sep, end = parts[sep_type]
            separator = start + sep.join([parts["line"] * (w + 2) for w in col_widths]) + end
            print(separator)

        def draw_row(items: List[Any], is_header: bool = False) -> None:
            # Ensure items are strings for formatting
            str_items = [str(item) for item in items]
            row_str = (
                "│ "
                + " │ ".join(f"{item:<{w}}" for item, w in zip(str_items, col_widths, strict=False))
                + " │"
            )
            print(row_str)

        draw_separator("top")
        draw_row(headers, is_header=True)
        draw_separator("mid")
        for r in rows:
            draw_row(r)
        draw_separator("bottom")

    @staticmethod
    def schema_to_hierarchical(schema: Dict) -> Dict:
        hierarchical_schema: Dict = {}
        for field, details in schema.items():
            parts = field.split(".")
            current_level = hierarchical_schema
            for part in parts[:-1]:
                current_level = current_level.setdefault(part, {})
            current_level[parts[-1]] = {"type": details["type"]}
        return hierarchical_schema

    @staticmethod
    def save_schema_to_json(schema: Dict, schema_file: Union[str, Path]) -> None:
        hierarchical_schema = SchemaAnalyser.schema_to_hierarchical(schema)
        output_path = Path(schema_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(hierarchical_schema, f, indent=4)
            logger.info(f"Schema has been saved to {output_path}")
            print(f"Schema has been saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save schema to {output_path}: {e}")
            print(f"Error: Could not save schema to {output_path}.")

    @staticmethod
    def save_table_to_csv(
        headers: List[str], rows: List[List[Any]], csv_file: Union[str, Path]
    ) -> None:
        output_path = Path(csv_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open(mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(rows)
            logger.info(f"Table has been saved to {output_path}")
            print(f"Table has been saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save CSV to {output_path}: {e}")
            print(f"Error: Could not save table to {output_path}.")
