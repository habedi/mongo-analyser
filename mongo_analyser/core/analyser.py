import csv
import json
import logging
import uuid
from collections import OrderedDict
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
            return frozenset((k, SchemaAnalyser._make_hashable(v)) for k, v in item.items())
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
                stats[full_key] = {"values": set(), "count": 0}

            if isinstance(value, dict):
                SchemaAnalyser.extract_schema_and_stats(value, schema, stats, full_key)
            elif isinstance(value, list):
                SchemaAnalyser.handle_array(value, schema, stats, full_key)
            else:
                SchemaAnalyser.handle_simple_value(value, schema, stats, full_key)

            stats[full_key]["count"] += 1
        return schema, stats

    @staticmethod
    def handle_array(
        value: List,
        schema: Union[Dict, OrderedDict],
        stats: Union[Dict, OrderedDict],
        full_key: str,
    ) -> None:
        if len(value) > 0:
            first_elem = value[0]
            if isinstance(first_elem, dict):
                schema[full_key] = {"type": "array<dict>"}
            elif isinstance(first_elem, ObjectId):
                schema[full_key] = {"type": "array<ObjectId>"}
            elif isinstance(first_elem, uuid.UUID):
                schema[full_key] = {"type": "array<UUID>"}
            elif isinstance(first_elem, Binary):
                shared.handle_binary_type_representation(
                    first_elem, schema, full_key, is_array=True
                )
            elif isinstance(first_elem, bool):
                schema[full_key] = {"type": "array<bool>"}
            elif isinstance(first_elem, int):
                schema[full_key] = {
                    "type": "array<int64>" if isinstance(first_elem, Int64) else "array<int32>"
                }
            elif isinstance(first_elem, float):
                schema[full_key] = {"type": "array<double>"}
            else:
                elem_type = type(first_elem).__name__
                schema[full_key] = {"type": f"array<{elem_type}>"}
        else:
            schema[full_key] = {"type": "array<empty>"}

        if full_key not in stats:
            stats[full_key] = {"values": set(), "count": 0}

        hashable_value = SchemaAnalyser._make_hashable(value)
        stats[full_key]["values"].add(hashable_value)

    @staticmethod
    def handle_simple_value(
        value: Any, schema: Union[Dict, OrderedDict], stats: Union[Dict, OrderedDict], full_key: str
    ) -> None:
        if isinstance(value, ObjectId):
            schema[full_key] = {"type": "binary<ObjectId>"}
        elif isinstance(value, uuid.UUID):
            schema[full_key] = {"type": "binary<UUID>"}
        elif isinstance(value, Binary):
            shared.handle_binary_type_representation(value, schema, full_key)
        elif isinstance(value, bool):
            schema[full_key] = {"type": "bool"}
        elif isinstance(value, int):
            schema[full_key] = {"type": "int64" if isinstance(value, Int64) else "int32"}
        elif isinstance(value, float):
            schema[full_key] = {"type": "double"}
        elif isinstance(value, Decimal128):
            schema[full_key] = {"type": "decimal128"}
        else:
            schema[full_key] = {"type": type(value).__name__}

        if full_key not in stats:
            stats[full_key] = {"values": set(), "count": 0}
        stats[full_key]["values"].add(value)

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
            if sample_size < 0:
                documents = collection.find().batch_size(batch_size)
            else:
                documents = collection.aggregate([{"$sample": {"size": sample_size}}]).batch_size(
                    batch_size
                )

            for doc in documents:
                total_docs += 1
                schema, stats = SchemaAnalyser.extract_schema_and_stats(doc, schema, stats)
                if sample_size > 0 and total_docs >= sample_size:
                    break
        except PyMongoOperationFailure as e:
            logger.error(
                f"Error during schema inference on {collection.database.name}.{collection.name}: {e}"
            )
            raise

        final_stats = OrderedDict()
        for key, stat_data in stats.items():
            cardinality = len(stat_data["values"])
            missing_count = total_docs - stat_data["count"]
            missing_percentage = (missing_count / total_docs) * 100 if total_docs > 0 else 0
            final_stats[key] = {
                "cardinality": cardinality,
                "missing_percentage": missing_percentage,
            }

        sorted_schema = OrderedDict(sorted(schema.items()))
        sorted_stats = OrderedDict(sorted(final_stats.items()))
        return dict(sorted_schema), dict(sorted_stats)

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
        col_widths = [
            max(len(str(item)) for item in col) for col in zip(*([headers] + rows), strict=False)
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

        def draw_row(items: List[Any]) -> None:
            row_str = (
                "│ "
                + " │ ".join(f"{item!s:<{w}}" for item, w in zip(items, col_widths, strict=False))
                + " │"
            )
            print(row_str)

        draw_separator("top")
        draw_row(headers)
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
