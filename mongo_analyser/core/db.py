import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database as PyMongoDatabase
from pymongo.errors import ConnectionFailure, OperationFailure

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db: Optional[PyMongoDatabase] = None
_current_uri: Optional[str] = None
_current_db_name_arg: Optional[str] = None


def db_connection_active(
    uri: str,
    db_name: Optional[str] = None,  # db_name can be part of URI or specified
    server_timeout_ms: int = 5000,
    force_reconnect: bool = False,
    **kwargs,
) -> bool:
    global _client, _db, _current_uri, _current_db_name_arg

    if not force_reconnect and _client is not None and _db is not None and _current_uri == uri:
        target_db_name_for_check = db_name if db_name else _client.get_database().name

        if _db.name == target_db_name_for_check or (db_name and _db.name == db_name):
            try:
                _client.admin.command("ping")
                logger.debug(
                    f"Already connected to MongoDB (URI: {_current_uri}, DB: {_db.name}). Ping successful."
                )
                return True
            except (ConnectionFailure, OperationFailure) as e:
                logger.warning(
                    f"Existing MongoDB connection ping failed: {e}. Attempting to re-establish."
                )
                _client = None
                _db = None
        elif db_name and _db.name != db_name:
            logger.info(
                f"Switching DB context on existing client from '{_db.name}' to '{db_name}'."
            )
            try:
                _db = _client[db_name]
                _current_db_name_arg = db_name
                logger.info(f"Successfully switched DB context to '{_db.name}'.")
                return True
            except Exception as e:
                logger.error(f"Failed to switch DB context to '{db_name}': {e}")
                _client = None
                _db = None

    if _client is not None:
        _client.close()
        _client = None
        _db = None

    try:
        logger.info(f"Attempting to connect to MongoDB: {uri}, target DB specified: {db_name}")
        client_options = {"serverSelectionTimeoutMS": server_timeout_ms, **kwargs}
        temp_client = MongoClient(uri, **client_options)

        db_to_use_name: Optional[str] = None

        uri_parts = uri.split("/")
        path_part = uri_parts[3].split("?")[0] if len(uri_parts) > 3 else None

        if db_name:
            db_to_use_name = db_name
        elif path_part and path_part != "":
            db_to_use_name = path_part
        else:
            logger.error(
                f"No database name specified in URI ('{uri}') or as an argument. Cannot determine database context."
            )
            temp_client.close()
            return False

        if not db_to_use_name:
            logger.error(
                f"Critical: Database name could not be resolved for URI '{uri}' and db_name arg '{db_name}'."
            )
            temp_client.close()
            return False

        temp_client.admin.command("ping")

        _client = temp_client
        _db = _client[db_to_use_name]
        _current_uri = uri
        _current_db_name_arg = db_name

        logger.info(f"Successfully connected to MongoDB. URI: '{uri}', Effective DB: '{_db.name}'.")
        return True
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(f"MongoDB connection failed for URI '{uri}', target DB '{db_name}': {e}")
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        _current_uri = None
        _current_db_name_arg = None
        return False
    except Exception as e:
        logger.error(f"Unexpected error during MongoDB connection: {e}", exc_info=True)
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        _current_uri = None
        _current_db_name_arg = None
        return False


def get_mongo_db() -> PyMongoDatabase:
    global _current_uri, _current_db_name_arg, _client, _db
    if _db is None or _client is None:
        raise ConnectionError("Not connected to MongoDB. Call db_connection_active first.")
    try:
        # Explicitly send the ping command as a dictionary
        _client.admin.command({"ping": 1})
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(f"MongoDB connection lost when trying to get DB: {e}")
        _current_uri = None
        _current_db_name_arg = None
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        raise ConnectionError("MongoDB connection lost. Reconnect needed.") from e
    return _db


def disconnect_mongo() -> None:
    global _client, _db, _current_uri, _current_db_name_arg
    if _client is not None:
        _client.close()
        logger.info("MongoDB client connection closed.")
    _client = None
    _db = None
    _current_uri = None
    _current_db_name_arg = None


def disconnect_all_mongo() -> None:
    disconnect_mongo()
