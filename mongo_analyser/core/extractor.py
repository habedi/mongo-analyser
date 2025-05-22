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
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                field_types[key] = "array<dict>"
        return field_types

    @staticmethod
    def convert_to_json_compatible(
        document: Dict, schema: Dict, tz: Union[pytz.timezone, None]
    ) -> Dict:
        processed_document: Dict = {}

        def _convert_value_recursive(
            key_path: str, val: Any, schema_type_info: Optional[Union[str, Dict]]
        ) -> Any:
            if val is None:
                return None

            current_schema_type_str: Optional[str] = None
            items_schema_info: Optional[Union[str, Dict]] = None

            if isinstance(schema_type_info, str):
                current_schema_type_str = schema_type_info
                if schema_type_info.startswith("array<") and schema_type_info.endswith(">"):
                    items_schema_info = schema_type_info[len("array<") : -1]
            elif isinstance(schema_type_info, dict) and "type" in schema_type_info:
                current_schema_type_str = schema_type_info["type"]
                if current_schema_type_str == "array" and "items" in schema_type_info:
                    items_schema_info = schema_type_info["items"]
                elif isinstance(current_schema_type_str, dict):
                    pass

            try:
                if isinstance(val, list) and items_schema_info:
                    return [
                        _convert_value_recursive(f"{key_path}[{i}]", item, items_schema_info)
                        for i, item in enumerate(val)
                    ]

                if isinstance(val, dict) and isinstance(current_schema_type_str, dict):
                    return DataExtractor.convert_to_json_compatible(
                        val, current_schema_type_str, tz
                    )

                type_to_check = current_schema_type_str or DataExtractor._infer_type_val(val)

                if (
                    type_to_check == "binary<UUID>"
                    or isinstance(val, uuid.UUID)
                    or type_to_check == "UUID"
                    or type_to_check == "binary<ObjectId>"
                    or type_to_check == "ObjectId"
                    or isinstance(val, ObjectId)
                ):
                    return str(val)
                elif type_to_check == "datetime" or isinstance(val, datetime):
                    return val.astimezone(tz).isoformat() if tz else val.isoformat()
                elif type_to_check == "str":
                    return str(val)
                elif type_to_check in ("int32", "int64") or isinstance(val, (int, Int64)):
                    return int(val)
                elif type_to_check == "bool" or isinstance(val, bool):
                    return bool(val)
                elif type_to_check == "double" or isinstance(val, float):
                    return float(val)
                elif type_to_check == "decimal128" or isinstance(val, Decimal128):
                    return str(val.to_decimal())
                elif (
                    type_to_check
                    and "binary<" in type_to_check
                    and type_to_check != "binary<ObjectId>"
                    and type_to_check != "binary<UUID>"
                ) or isinstance(val, Binary):
                    return val.hex()

                if isinstance(val, (str, int, float, bool, list, dict)) and val is not None:
                    return val
                return str(val)
            except Exception as e:
                logger.warning(
                    f"Conversion failed for '{key_path}' (value: {str(val)[:50]}..., type: {type(val)}) to '{current_schema_type_str}': {e}"
                )
                return str(val)

        for top_key, top_value in document.items():
            schema_details_for_key = schema.get(top_key)

            if (
                isinstance(top_value, dict)
                and isinstance(schema_details_for_key, dict)
                and isinstance(schema_details_for_key.get("type"), dict)
            ):
                processed_document[top_key] = DataExtractor.convert_to_json_compatible(
                    top_value, schema_details_for_key["type"], tz
                )
            else:
                processed_document[top_key] = _convert_value_recursive(
                    top_key, top_value, schema_details_for_key
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

        if not db_manager.db_connection_active(
            uri=mongo_uri, db_name=db_name, server_timeout_ms=server_timeout_ms
        ):
            raise PyMongoConnectionFailure(
                f"MongoDB connection failed for data extraction: {mongo_uri}"
            )

        database = db_manager.get_mongo_db()
        collection = database[collection_name]

        try:
            data_cursor = (
                collection.find(no_cursor_timeout=False)
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
                    count += 1
                    converted_doc = DataExtractor.convert_to_json_compatible(doc, schema, tz)
                    if not first_doc:
                        f.write(",\n")
                    json.dump(converted_doc, f, indent=None)
                    first_doc = False
                    if count % batch_size == 0:
                        logger.info(f"Processed {count} documents...")
                f.write("\n]\n")
            logger.info(f"Successfully extracted {count} documents to {output_p}")

        except PyMongoOperationFailure as e:
            logger.error(
                f"MongoDB operation failure during data extraction from {db_name}.{collection_name}: {e}"
            )
            raise
        except IOError as e:
            logger.error(f"Failed to write to output file {output_p}: {e}")
            raise


def get_newest_documents(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    sample_size: int,
    fields: Optional[List[str]] = None,
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

    try:
        if sample_size <= 0:
            logger.warning(
                "Sample size must be positive for fetching newest documents. Returning empty list."
            )
            return []

        projection_doc: Optional[Dict[str, int]] = None
        # Filter out empty strings from fields list if it exists
        valid_fields = [f for f in fields if f] if fields else None

        if valid_fields:
            projection_doc = {field: 1 for field in valid_fields}
            if "_id" not in valid_fields:
                projection_doc["_id"] = 0
        elif (
            fields is not None and not valid_fields
        ):  # User provided fields, but all were empty (e.g. ",,")
            # This case means user intended to select specific fields but provided only empty strings.
            # To avoid fetching all fields by default when projection_doc is None,
            # we can project only _id, or return an empty list, or log a warning.
            # For now, let's project only _id to be safe and minimal.
            logger.warning("Fields list contained only empty strings. Projecting only _id.")
            projection_doc = {"_id": 1}

        query = (
            collection.find(projection=projection_doc).sort("_id", DESCENDING).limit(sample_size)
        )

        raw_documents = list(query)

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
                    processed_doc[key] = (
                        f"binary_hex:{hex_val}"
                        if len(hex_val) < 100
                        else "binary_data(too_long_to_display)"
                    )
                elif isinstance(value, Decimal128):
                    processed_doc[key] = str(value.to_decimal())
                elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
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
    except Exception as e:
        logger.error(
            f"Unexpected error during document fetching ({db_name}.{collection_name}): {e}",
            exc_info=True,
        )
        raise
