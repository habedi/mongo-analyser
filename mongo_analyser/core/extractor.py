import gzip
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytz
from bson import Binary, Decimal128, Int64, ObjectId
from pymongo import DESCENDING
from pymongo.errors import (
    ConnectionFailure as PyMongoConnectionFailure,
)
from pymongo.errors import (
    OperationFailure as PyMongoOperationFailure,
)
from textual.worker import (
    Worker,
    WorkerCancelled,
    get_current_worker,
)

from . import db as db_manager
from . import shared

logger = logging.getLogger(__name__)


class DataExtractor:
    @staticmethod
    def _infer_type_val(value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, Int64):
            return "int64"
        elif isinstance(value, int):
            return "int32"
        elif isinstance(value, float):
            return "double"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, datetime):
            return "datetime"
        elif isinstance(value, ObjectId):
            return "ObjectId"
        elif isinstance(value, uuid.UUID):
            return "UUID"
        elif isinstance(value, Binary):
            return shared.binary_type_map.get(value.subtype, f"binary<subtype {value.subtype}>")
        elif isinstance(value, list):
            if not value:
                return "array<empty>"
            # For conversion, a single type hint is usually sufficient,
            # schema analysis handles detailed mixed type reporting.
            first_elem_type = DataExtractor._infer_type_val(value[0])
            return f"array<{first_elem_type}>"
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, Decimal128):
            return "decimal128"
        elif value is None:
            return "null"
        else:
            return f"unknown<{type(value).__name__}>"

    @staticmethod
    def infer_types_from_document(document: Dict) -> Dict[str, str]:
        field_types: Dict[str, str] = {}
        for key, value in document.items():
            field_types[key] = DataExtractor._infer_type_val(value)
            if isinstance(value, dict):
                for sub_key, sub_type in DataExtractor.infer_types_from_document(value).items():
                    field_types[f"{key}.{sub_key}"] = sub_type
        return field_types

    @staticmethod
    def _convert_single_value(
        val: Any,
        schema_type_str: Optional[str],  # Type hint from schema (e.g. "array<dict>", "datetime")
        tz: Union[pytz.timezone, None],
        items_schema_for_array_elements: Optional[Dict] = None,  # Schema for dicts inside array
    ) -> Any:
        if val is None:
            return None

        # Use schema_type_str if available, otherwise infer.
        # This helps guide conversion, esp. for ambiguous types or when schema is more precise.
        type_to_check = schema_type_str or DataExtractor._infer_type_val(val)

        if isinstance(val, list):  # Array handling
            # If schema says array<dict> and provides items_schema, recurse for elements
            if (
                type_to_check == "array<dict>"  # or schema_type_str == "array<dict>"
                and items_schema_for_array_elements
                and isinstance(items_schema_for_array_elements, dict)
            ):
                return [
                    DataExtractor.convert_to_json_compatible(
                        item, items_schema_for_array_elements, tz
                    )
                    for item in val
                    if isinstance(item, dict)  # Ensure item is a dict before recursing
                ]
            else:  # Generic array conversion
                item_type_str_for_elements = None
                if (
                    type_to_check  # Check if type_to_check (from schema or inferred) is an array type
                    and type_to_check.startswith("array<")
                    and type_to_check.endswith(">")
                ):
                    item_type_str_for_elements = type_to_check[len("array<") : -1]

                return [
                    DataExtractor._convert_single_value(item, item_type_str_for_elements, tz)
                    for item in val
                ]

        # Handle specific types based on schema hint or inference
        if type_to_check == "binary<UUID>" or type_to_check == "UUID" or isinstance(val, uuid.UUID):
            return str(val)
        if (
            type_to_check == "binary<ObjectId>"
            or type_to_check == "ObjectId"
            or isinstance(val, ObjectId)
        ):
            return str(val)
        if type_to_check == "datetime" or isinstance(val, datetime):
            # Ensure val is timezone-aware before astimezone if tz is provided
            if tz:
                if val.tzinfo is None or val.tzinfo.utcoffset(val) is None:  # Naive datetime
                    # logger.debug(f"Localizing naive datetime {val} to tz {tz} before isoformat.")
                    # This assumes naive datetimes are in local system time if tz is provided for conversion.
                    # A more robust solution might require knowing the source timezone of naive datetimes.
                    # For now, if tz is given, we try to make it aware using system's idea or pytz's localize.
                    # However, directly using astimezone on naive datetime with a tz object is problematic.
                    # Let's assume if tz is provided, user wants output in that tz.
                    # If val is naive, it's safer to localize to UTC first, then convert to target tz.
                    # Or, if schema implies it should be a certain tz, that's more complex.
                    # Simplification: if val is naive and tz is given, assume val is UTC for conversion.
                    # val_aware = val.replace(tzinfo=pytz.utc) if val.tzinfo is None else val
                    # return val_aware.astimezone(tz).isoformat()
                    # A simpler approach: if naive, make it tz-aware with the target tz, then format.
                    # This might not be correct if the naive datetime isn't in that tz originally.
                    # Safest: if naive, output as is. If aware, convert.
                    if val.tzinfo is None:
                        return val.isoformat()  # output naive as is
                    return val.astimezone(tz).isoformat()  # convert aware to target tz
            return val.isoformat()  # No tz provided, or val is already aware and tz is None

        if type_to_check == "str":
            return str(val)  # Handles non-string types if schema says str
        if type_to_check in ("int32", "int64") or isinstance(val, (int, Int64)):
            return int(val)
        if type_to_check == "bool" or isinstance(val, bool):
            return bool(val)  # Handles non-bool if schema says bool
        if type_to_check == "double" or isinstance(val, float):
            return float(val)
        if type_to_check == "decimal128" or isinstance(val, Decimal128):
            return str(val.to_decimal())  # Convert Decimal128 to string representation of decimal

        # Handle generic binary data (not ObjectId or UUID)
        if (
            type_to_check
            and "binary<" in type_to_check
            and type_to_check not in ("binary<ObjectId>", "binary<UUID>")  # Already handled
        ) or isinstance(val, Binary):  # If it's a Binary type not caught above
            return val.hex()  # Convert to hex string

        # Fallback for types that are directly JSON serializable or if schema doesn't guide otherwise
        if isinstance(
            val, (str, int, float, bool, dict)
        ):  # dict is handled by recursion in convert_to_json_compatible
            return val

        # If no specific conversion rule matched and not a basic JSON type
        logger.warning(
            f"Value {str(val)[:50]} of type {type(val)} with schema type hint '{schema_type_str}' fell through to string conversion."
        )
        return str(val)  # Fallback to string

    @staticmethod
    def convert_to_json_compatible(
        document: Dict, schema_for_current_level: Dict, tz: Union[pytz.timezone, None]
    ) -> Dict:
        processed_document: Dict = {}
        for key, value in document.items():
            if value is None:
                processed_document[key] = None
                continue

            field_schema_definition = schema_for_current_level.get(key)
            type_str_from_schema: Optional[str] = None
            items_sub_schema: Optional[Dict] = None  # For array<dict>

            if isinstance(field_schema_definition, dict):  # Schema provides info for this key
                type_str_from_schema = field_schema_definition.get("type")

                # If it's an array of dicts, get the schema for the dict items
                if type_str_from_schema == "array<dict>" and isinstance(
                    field_schema_definition.get("items"), dict
                ):
                    items_sub_schema = field_schema_definition.get("items")

                # If the value is a dictionary and the schema defines nested fields for it (not just a type like "dict")
                elif not type_str_from_schema and isinstance(value, dict):  # Nested object schema
                    processed_document[key] = DataExtractor.convert_to_json_compatible(
                        value,
                        field_schema_definition,
                        tz,  # Recurse with sub-schema
                    )
                    continue  # Skip further processing for this key

            # If not a nested object schema handled above, process the value directly
            processed_document[key] = DataExtractor._convert_single_value(
                value, type_str_from_schema, tz, items_sub_schema
            )
        return processed_document

    @staticmethod
    def extract_data(
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        schema: Dict,  # This should be the hierarchical schema
        output_file: Union[str, Path],
        tz: Union[None, pytz.timezone],
        batch_size: int,
        limit: int,
        server_timeout_ms: int = 5000,
    ) -> None:
        output_p = Path(output_file)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        worker_instance: Optional[Worker] = None
        try:
            worker_instance = get_current_worker()
        except Exception:  # get_current_worker can raise if not in a worker context
            pass

        if worker_instance and worker_instance.is_cancelled:
            logger.info("Data extraction cancelled before database connection.")
            raise WorkerCancelled()

        if not db_manager.db_connection_active(
            uri=mongo_uri, db_name=db_name, server_timeout_ms=server_timeout_ms
        ):
            if (
                worker_instance and worker_instance.is_cancelled
            ):  # Check again after potentially long op
                logger.info("Data extraction cancelled during/after database connection attempt.")
                raise WorkerCancelled()
            raise PyMongoConnectionFailure(
                f"MongoDB connection failed for data extraction: {mongo_uri}, DB: {db_name}"
            )

        database = db_manager.get_mongo_db()
        collection = database[collection_name]
        data_cursor = None

        try:
            if worker_instance and worker_instance.is_cancelled:
                logger.info("Data extraction cancelled before finding documents.")
                raise WorkerCancelled()

            data_cursor = (
                collection.find(no_cursor_timeout=True)  # Potentially long running
                .sort("_id", DESCENDING)  # Newest first
                .batch_size(batch_size)
            )
            if limit >= 0:  # Apply limit if specified (0 or positive)
                data_cursor = data_cursor.limit(limit)
                logger.info(
                    f"Reading up to {limit} newest records from {db_name}.{collection_name}..."
                )
            else:  # limit is -1, read all
                logger.info(
                    f"Reading all records (newest first) from {db_name}.{collection_name}..."
                )

            count = 0
            with gzip.open(output_p, "wt", encoding="utf-8") as f:
                f.write("[\n")  # Start of JSON array
                first_doc = True
                for doc in data_cursor:  # Iterate over cursor
                    if worker_instance and worker_instance.is_cancelled:
                        logger.info(f"Data extraction cancelled by worker at document {count + 1}.")
                        if not first_doc:
                            f.write("\n")  # Ensure valid JSON if abruptly ended
                        # No need to write "]" here, file will be incomplete
                        raise WorkerCancelled()

                    count += 1
                    # Pass the root of the hierarchical schema for conversion
                    converted_doc = DataExtractor.convert_to_json_compatible(doc, schema, tz)

                    if not first_doc:
                        f.write(",\n")  # Comma before next document

                    json.dump(converted_doc, f, indent=None)  # Write document
                    first_doc = False

                    if count % batch_size == 0:  # Log progress
                        logger.info(f"Processed {count} documents...")
                        if worker_instance:  # Give worker a chance to update UI if needed
                            worker_instance.update_message(f"Processed {count} documents...")

                if not first_doc:  # If any document was written
                    f.write("\n")
                f.write("]\n")  # End of JSON array
                logger.info(f"Successfully extracted {count} documents to {output_p}")

        except WorkerCancelled:
            logger.warning(
                f"Extraction process cancelled. Output file {output_p} may be incomplete or not a valid JSON array."
            )
            # Do not re-raise if we want the TUI to handle it gracefully without crashing worker
            # The calling TUI part should check worker.is_cancelled
            # However, for CLI, re-raising is fine. For TUI, this might be too abrupt.
            # Let's assume the TUI will check the worker state.
            raise
        except PyMongoOperationFailure as e:
            if worker_instance and worker_instance.is_cancelled:  # If cancelled during op
                logger.warning(f"Extraction cancelled during MongoDB operation: {e}")
                raise WorkerCancelled()
            logger.error(
                f"MongoDB operation failure during data extraction from {db_name}.{collection_name}: {e}"
            )
            raise  # Re-raise for TUI/CLI to handle
        except IOError as e:
            if worker_instance and worker_instance.is_cancelled:
                logger.warning(f"Extraction cancelled during file IO: {e}")
                raise WorkerCancelled()
            logger.error(f"Failed to write to output file {output_p}: {e}")
            raise
        finally:
            if data_cursor is not None:
                data_cursor.close()
                logger.debug("MongoDB cursor closed for data extraction.")


def get_newest_documents(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    sample_size: int,
    fields: Optional[List[str]] = None,
    # Currently not used for projection in this simplified version
    server_timeout_ms: int = 5000,
) -> List[Dict]:
    if not db_manager.db_connection_active(
        uri=mongo_uri, db_name=db_name, server_timeout_ms=server_timeout_ms
    ):
        raise PyMongoConnectionFailure(
            f"MongoDB connection failed for sampling ({db_name}.{collection_name})"
        )

    database = db_manager.get_mongo_db()
    collection = database[collection_name]
    query_cursor = None

    try:
        if sample_size <= 0:
            logger.warning(
                "Sample size must be positive for fetching newest documents. Returning empty list."
            )
            return []

        # Projection logic can be added here if `fields` parameter is to be used
        projection_doc: Optional[Dict[str, int]] = None
        # if fields:
        # projection_doc = {field: 1 for field in fields}
        # if "_id" not in fields: projection_doc["_id"] = 0 # or 1 if always needed

        query_cursor = (
            collection.find(projection=projection_doc)  # Pass projection if implemented
            .sort("_id", DESCENDING)
            .limit(sample_size)
        )

        raw_documents = list(query_cursor)  # Materialize cursor

        # Convert to JSON-compatible types for display in TUI (basic conversion)
        # A more robust way would be to use a simplified version of convert_to_json_compatible
        # or pass a very generic schema. For now, direct conversion of common types.
        processed_docs: List[Dict] = []
        for doc in raw_documents:
            processed_doc = {}
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    processed_doc[key] = str(value)
                elif isinstance(value, datetime):
                    processed_doc[key] = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    processed_doc[key] = str(value)
                elif isinstance(value, Binary):
                    hex_val = value.hex()
                    # Truncate long binary hex strings for display
                    processed_doc[key] = (
                        f"binary_hex:{hex_val[:64]}{'...' if len(hex_val) > 64 else ''}"
                    )
                elif isinstance(value, Decimal128):
                    processed_doc[key] = str(value.to_decimal())
                elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                    # For lists and dicts, this is a shallow conversion.
                    # A recursive converter would be needed for deep conversion for display.
                    # For TUI display, often a string representation is enough for complex nested structures.
                    if (
                        isinstance(value, (list, dict))
                        and len(json.dumps(value, default=str)) > 500
                    ):  # Heuristic for large nested
                        processed_doc[key] = f"{type(value).__name__}(too large to display inline)"
                    else:
                        processed_doc[key] = value
                else:
                    processed_doc[key] = f"unhandled_type:{type(value).__name__}:{str(value)[:50]}"
            processed_docs.append(processed_doc)

        logger.info(
            f"Fetched {len(processed_docs)} newest documents from {db_name}.{collection_name}"
        )
        return processed_docs

    except PyMongoOperationFailure as e:
        logger.error(
            f"MongoDB operation failed during document fetching ({db_name}.{collection_name}): {e}"
        )
        raise
    except Exception as e:  # Catch any other unexpected error
        logger.error(
            f"Unexpected error during document fetching ({db_name}.{collection_name}): {e}",
            exc_info=True,
        )
        raise
    finally:
        if query_cursor is not None:
            query_cursor.close()
            logger.debug("MongoDB cursor closed for get_newest_documents.")
