import logging
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
        schema: Optional[Dict] = None,
        stats: Optional[Dict] = None,
        prefix: str = "",
    ) -> Tuple[Dict, Dict]:
        if schema is None:
            schema = {}
        if stats is None:
            stats = {}

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
                    "array_element_stats": {
                        "values": set(),
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
                    value, schema, stats[full_key], full_key, is_array_element=False
                )

        return schema, stats

    @staticmethod
    def handle_array(
        value: List,
        schema: Dict,
        stats: Dict,
        full_key: str,
    ) -> None:
        current_field_stats = stats[full_key]

        if not value:
            schema[full_key] = {"type": "array<empty>"}
        else:
            element_types_for_schema = Counter()
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
        schema: Dict,
        stats_dict_to_update: Dict,
        full_key: str,
        is_array_element: bool,
    ) -> None:
        stats_dict_to_update.setdefault("values", set())
        stats_dict_to_update.setdefault("type_counts", Counter())
        stats_dict_to_update.setdefault("numeric_min", float("inf"))
        stats_dict_to_update.setdefault("numeric_max", float("-inf"))
        stats_dict_to_update.setdefault("date_min", None)
        stats_dict_to_update.setdefault("date_max", None)
        stats_dict_to_update.setdefault("value_frequencies", Counter())

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
            try:
                stats_dict_to_update["values"].add(SchemaAnalyser._make_hashable(value))
            except:
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
            if len(value) < 256:
                stats_dict_to_update["value_frequencies"].update([value])
        elif isinstance(value, datetime):
            cur_min = stats_dict_to_update.get("date_min")
            cur_max = stats_dict_to_update.get("date_max")
            if cur_min is None or value < cur_min:
                stats_dict_to_update["date_min"] = value
            if cur_max is None or value > cur_max:
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
            return sorted(database.list_collection_names())
        except PyMongoOperationFailure as e:
            logger.error(f"MongoDB operation failure listing collections for DB '{db_name}': {e}")
            raise

    @staticmethod
    def infer_schema_and_field_stats(
        collection: PyMongoCollection, sample_size: int, batch_size: int = 1000
    ) -> Tuple[Dict, Dict]:
        schema: Dict = {}
        stats: Dict = {}
        total_docs = 0

        try:
            docs = (
                collection.find().batch_size(batch_size)
                if sample_size < 0
                else collection.aggregate([{"$sample": {"size": sample_size}}]).batch_size(
                    batch_size
                )
            )
            for doc in docs:
                total_docs += 1
                schema, stats = SchemaAnalyser.extract_schema_and_stats(doc, schema, stats)
                if 0 < sample_size <= total_docs:
                    break
        except PyMongoOperationFailure as e:
            logger.error(
                f"Error during schema inference on {collection.database.name}.{collection.name}: {e}"
            )
            raise

        final_stats_summary: Dict = {}
        for key, field_stat_data in stats.items():
            values_set = field_stat_data.get("values", set())
            count = field_stat_data.get("count", 0)
            type_counts = field_stat_data.get("type_counts", Counter())
            val_freqs = field_stat_data.get("value_frequencies", Counter())
            arr_stats = field_stat_data.get("array_element_stats", {})
            arr_type_counts = arr_stats.get("type_counts", Counter())
            arr_val_freqs = arr_stats.get("value_frequencies", Counter())

            cardinality = len(values_set)
            missing_count = total_docs - count
            missing_pct = (missing_count / total_docs * 100) if total_docs else 0

            processed: Dict[str, Any] = {
                "cardinality": cardinality,
                "missing_percentage": missing_pct,
                "type_distribution": dict(type_counts.most_common(5)),
            }

            num_min = field_stat_data.get("numeric_min", float("inf"))
            if num_min != float("inf"):
                processed["numeric_min"] = num_min
                processed["numeric_max"] = field_stat_data.get("numeric_max", num_min)

            dmin = field_stat_data.get("date_min")
            if dmin is not None:
                processed["date_min"] = dmin.isoformat()
                processed["date_max"] = field_stat_data.get("date_max", dmin).isoformat()

            if val_freqs:
                processed["top_values"] = dict(val_freqs.most_common(5))

            if arr_type_counts:
                arr_processed: Dict[str, Any] = {
                    "type_distribution": dict(arr_type_counts.most_common(5))
                }
                arr_num_min = arr_stats.get("numeric_min", float("inf"))
                if arr_num_min != float("inf"):
                    arr_processed["numeric_min"] = arr_num_min
                    arr_processed["numeric_max"] = arr_stats.get("numeric_max", arr_num_min)
                arr_dmin = arr_stats.get("date_min")
                if arr_dmin is not None:
                    arr_processed["date_min"] = arr_dmin.isoformat()
                    arr_processed["date_max"] = arr_stats.get("date_max", arr_dmin).isoformat()
                if arr_val_freqs:
                    arr_processed["top_values"] = dict(arr_val_freqs.most_common(5))
                processed["array_elements"] = arr_processed

            final_stats_summary[key] = processed

        sorted_schema = dict(sorted(schema.items()))
        sorted_stats = dict(sorted(final_stats_summary.items()))
        return sorted_schema, sorted_stats

    @staticmethod
    def schema_to_hierarchical(schema: Dict) -> Dict:
        hierarchical: Dict = {}
        for field, details in schema.items():
            parts = field.split(".")
            cur = hierarchical
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = {"type": details.get("type")}
        return hierarchical
