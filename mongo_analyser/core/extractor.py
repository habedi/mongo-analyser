import gzip
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytz
from bson import Binary, Decimal128, Int64, ObjectId

# Use integer literals for subtypes instead of potentially missing named constants
from bson.binary import UuidRepresentation  # Keep this for UuidRepresentation enum

# from bson.binary import UUID_SUBTYPE, UUID_LEGACY_SUBTYPE # Remove this problematic import
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

# Define constants for subtypes if you prefer not to use magic numbers directly
_BSON_UUID_SUBTYPE_STANDARD = 4
_BSON_UUID_SUBTYPE_LEGACY_PYTHON = 3


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
            if (
                value.subtype == _BSON_UUID_SUBTYPE_LEGACY_PYTHON
            ):  # Use defined constant or literal 3
                return "binary<UUID (legacy)>"
            if value.subtype == _BSON_UUID_SUBTYPE_STANDARD:  # Use defined constant or literal 4
                return "binary<UUID>"
            return shared.binary_type_map.get(value.subtype, f"binary<subtype {value.subtype}>")
        elif isinstance(value, list):
            if not value:
                return "array<empty>"

            element_types = set()
            has_actual_value = False
            for item in value:
                inferred_item_type = DataExtractor._infer_type_val(item)
                element_types.add(inferred_item_type)
                if inferred_item_type != "null":
                    has_actual_value = True

            if not has_actual_value:
                return "array<null>"

            distinct_types_no_null = {t for t in element_types if t != "null"}

            if len(distinct_types_no_null) == 1:
                return f"array<{list(distinct_types_no_null)[0]}>"
            elif not distinct_types_no_null and "null" in element_types:
                return "array<null>"
            else:
                return "array<mixed>"
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, Decimal128):
            return "decimal128"
        elif value is None:
            return "null"
        else:
            return f"unknown<{type(value).__name__}>"

    @staticmethod
    def _convert_single_value(
        val: Any,
        schema_type_str: Optional[str],
        tz: Union[pytz.timezone, None],
        items_schema_for_array_elements: Optional[Dict] = None,
    ) -> Any:
        if val is None:
            return None

        type_to_check = schema_type_str or DataExtractor._infer_type_val(val)

        if isinstance(val, list):
            if (
                type_to_check == "array<dict>"
                and items_schema_for_array_elements
                and isinstance(items_schema_for_array_elements, dict)
            ):
                return [
                    DataExtractor.convert_to_json_compatible(
                        item, items_schema_for_array_elements, tz
                    )
                    if isinstance(item, dict)
                    else DataExtractor._convert_single_value(item, None, tz)
                    for item in val
                ]
            else:
                item_type_str_for_elements = None
                if (
                    type_to_check
                    and type_to_check.startswith("array<")
                    and type_to_check.endswith(">")
                ):
                    item_type_str_for_elements = type_to_check[len("array<") : -1]

                return [
                    DataExtractor._convert_single_value(item, item_type_str_for_elements, tz)
                    for item in val
                ]

        if isinstance(val, uuid.UUID):
            return str(val)

        if isinstance(val, Binary) and val.subtype in (
            _BSON_UUID_SUBTYPE_STANDARD,
            _BSON_UUID_SUBTYPE_LEGACY_PYTHON,
        ):
            try:
                py_uuid = None
                if val.subtype == _BSON_UUID_SUBTYPE_STANDARD:
                    py_uuid = val.as_uuid(UuidRepresentation.STANDARD)
                elif val.subtype == _BSON_UUID_SUBTYPE_LEGACY_PYTHON:
                    try:
                        py_uuid = val.as_uuid(UuidRepresentation.PYTHON_LEGACY)
                    except ValueError:
                        py_uuid = val.as_uuid()

                if py_uuid:
                    return str(py_uuid)
                else:
                    logger.warning(
                        f"Binary UUID subtype {val.subtype} could not be converted to Python UUID, falling to hex."
                    )
                    return val.hex()
            except Exception as e:
                logger.warning(
                    f"Could not convert Binary UUID subtype {val.subtype} (Exception: {e}), falling back to hex."
                )
                return val.hex()

        if type_to_check == "binary<UUID>" or type_to_check == "binary<UUID (legacy)>":
            if isinstance(val, str) and len(val) == 36:
                try:
                    uuid.UUID(val)
                    return val
                except ValueError:
                    pass
            if isinstance(val, Binary):
                return val.hex()
            return str(val)

        if (
            type_to_check == "binary<ObjectId>"
            or type_to_check == "ObjectId"
            or isinstance(val, ObjectId)
        ):
            return str(val)

        if type_to_check == "datetime" or isinstance(val, datetime):
            dt_to_convert = val
            if val.tzinfo is None or val.tzinfo.utcoffset(val) is None:
                dt_to_convert = pytz.utc.localize(val)

            if tz:
                return dt_to_convert.astimezone(tz).isoformat()
            return dt_to_convert.isoformat()

        if type_to_check == "str":
            return str(val)
        if type_to_check in ("int32", "int64") or isinstance(val, (int, Int64)):
            return int(val)
        if type_to_check == "bool" or isinstance(val, bool):
            return bool(val)
        if type_to_check == "double" or isinstance(val, float):
            return float(val)
        if type_to_check == "decimal128" or isinstance(val, Decimal128):
            return str(val.to_decimal())

        if isinstance(val, Binary):
            return val.hex()

        if isinstance(val, (str, int, float, bool, dict)):
            return val

        logger.warning(
            f"Value {str(val)[:50]} of type {type(val)} with schema type hint '{schema_type_str}' fell through to string conversion."
        )
        return str(val)

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
            items_sub_schema: Optional[Dict] = None

            if isinstance(field_schema_definition, dict):
                type_str_from_schema = field_schema_definition.get("type")

                if (
                    type_str_from_schema
                    and type_str_from_schema.startswith("array<")
                    and isinstance(field_schema_definition.get("items"), dict)
                ):
                    if type_str_from_schema == "array<dict>":
                        items_sub_schema = field_schema_definition.get("items")

                elif not type_str_from_schema and isinstance(value, dict):
                    processed_document[key] = DataExtractor.convert_to_json_compatible(
                        value,
                        field_schema_definition,
                        tz,
                    )
                    continue

            processed_document[key] = DataExtractor._convert_single_value(
                value, type_str_from_schema, tz, items_sub_schema
            )
        return processed_document

    @staticmethod
    def extract_data(
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        schema: Dict,
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
        except Exception:
            pass

        if worker_instance and worker_instance.is_cancelled:
            logger.info("Data extraction cancelled before database connection.")
            raise WorkerCancelled()

        if not db_manager.db_connection_active(
            uri=mongo_uri, db_name=db_name, server_timeout_ms=server_timeout_ms
        ):
            if worker_instance and worker_instance.is_cancelled:
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
                collection.find(no_cursor_timeout=True)
                .sort("_id", DESCENDING)
                .batch_size(batch_size)
            )
            if limit >= 0:
                data_cursor = data_cursor.limit(limit)
                logger.info(
                    f"Reading up to {limit} newest records from {db_name}.{collection_name}..."
                )
            else:
                logger.info(
                    f"Reading all records (newest first) from {db_name}.{collection_name}..."
                )

            count = 0
            with gzip.open(output_p, "wt", encoding="utf-8") as f:
                f.write("[\n")
                first_doc = True
                for doc in data_cursor:
                    if worker_instance and worker_instance.is_cancelled:
                        logger.info(f"Data extraction cancelled by worker at document {count + 1}.")
                        if not first_doc:
                            f.write("\n")
                        raise WorkerCancelled()

                    count += 1
                    converted_doc = DataExtractor.convert_to_json_compatible(doc, schema, tz)

                    if not first_doc:
                        f.write(",\n")

                    json.dump(converted_doc, f, indent=None)
                    first_doc = False

                    if count % batch_size == 0:
                        logger.info(f"Processed {count} documents...")
                        if worker_instance:
                            worker_instance.update_message(f"Processed {count} documents...")

                if not first_doc:
                    f.write("\n")
                f.write("]\n")
                logger.info(f"Successfully extracted {count} documents to {output_p}")

        except WorkerCancelled:
            logger.warning(
                f"Extraction process cancelled. Output file {output_p} may be incomplete or not a valid JSON array."
            )
            raise
        except PyMongoOperationFailure as e:
            if worker_instance and worker_instance.is_cancelled:
                logger.warning(f"Extraction cancelled during MongoDB operation: {e}")
                raise WorkerCancelled()
            logger.error(
                f"MongoDB operation failure during data extraction from {db_name}.{collection_name}: {e}"
            )
            raise
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

    @staticmethod
    def get_newest_documents(
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        sample_size: int,
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

            projection_doc: Optional[Dict[str, int]] = None

            query_cursor = (
                collection.find(projection=projection_doc)
                .sort("_id", DESCENDING)
                .limit(sample_size)
            )

            raw_documents = list(query_cursor)
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
                        if value.subtype in (
                            _BSON_UUID_SUBTYPE_STANDARD,
                            _BSON_UUID_SUBTYPE_LEGACY_PYTHON,
                        ):
                            try:
                                representation = (
                                    UuidRepresentation.STANDARD
                                    if value.subtype == _BSON_UUID_SUBTYPE_STANDARD
                                    else UuidRepresentation.PYTHON_LEGACY
                                )
                                processed_doc[key] = str(value.as_uuid(representation))
                            except Exception:
                                processed_doc[key] = value.hex()[:64] + (
                                    "..." if len(value.hex()) > 64 else ""
                                )
                        else:
                            hex_val = value.hex()
                            processed_doc[key] = (
                                f"binary_hex:{hex_val[:64]}{'...' if len(hex_val) > 64 else ''}"
                            )
                    elif isinstance(value, Decimal128):
                        processed_doc[key] = str(value.to_decimal())
                    elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                        try:
                            val_str_for_size_check = (
                                json.dumps(value, default=str)
                                if isinstance(value, (dict, list))
                                else str(value)
                            )
                            if len(val_str_for_size_check) > 500:
                                processed_doc[key] = (
                                    f"{type(value).__name__}(too large to display inline)"
                                )
                            else:
                                processed_doc[key] = value
                        except TypeError:
                            processed_doc[key] = (
                                f"unserializable_type:{type(value).__name__}:{str(value)[:50]}"
                            )

                    else:
                        processed_doc[key] = (
                            f"unhandled_type:{type(value).__name__}:{str(value)[:50]}"
                        )
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
        except Exception as e:
            logger.error(
                f"Unexpected error during document fetching ({db_name}.{collection_name}): {e}",
                exc_info=True,
            )
            raise
        finally:
            if query_cursor is not None:
                query_cursor.close()
                logger.debug("MongoDB cursor closed for get_newest_documents.")
