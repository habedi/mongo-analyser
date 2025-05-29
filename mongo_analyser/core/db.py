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
_current_resolved_db_name: Optional[str] = None


def db_connection_active(
    uri: str,
    db_name: Optional[str] = None,
    server_timeout_ms: int = 5000,
    force_reconnect: bool = False,
    **kwargs,
) -> bool:
    global _client, _db, _current_uri, _current_db_name_arg, _current_resolved_db_name

    if not force_reconnect and _client is not None and _db is not None and _current_uri == uri:
        target_db_name_for_check = (
            db_name if db_name else (_current_resolved_db_name or _client.get_database().name)
        )

        if _db.name == target_db_name_for_check:
            try:
                _client.admin.command("ping")
                logger.debug(
                    f"Already connected to MongoDB (URI: {_current_uri}, DB: {_db.name})."
                    f" Ping was successful."
                )
                _current_resolved_db_name = _db.name
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
                _current_resolved_db_name = _db.name
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

    _current_uri = None
    _current_db_name_arg = None
    _current_resolved_db_name = None

    try:
        logger.info(f"Attempting to connect to MongoDB: {uri}, target DB specified: {db_name}")
        client_options = {"serverSelectionTimeoutMS": server_timeout_ms, **kwargs}

        temp_client = MongoClient(uri, **client_options)

        db_to_use_name: Optional[str] = None
        parsed_uri = temp_client.HOST

        if db_name:
            db_to_use_name = db_name
        elif parsed_uri and isinstance(parsed_uri, tuple) and len(parsed_uri) > 2 and parsed_uri[2]:
            if temp_client.get_database().name != "test" or "test" in uri.lower():
                db_to_use_name = temp_client.get_database().name

            uri_path_part = uri.split("//", 1)[-1].split("/", 1)[-1].split("?", 1)[0]
            if uri_path_part and uri_path_part != temp_client.get_database().name and not db_name:
                if "/" not in uri_path_part:
                    db_to_use_name = uri_path_part

        if not db_to_use_name and not db_name:
            default_db_from_client = temp_client.get_database()
            if default_db_from_client:
                db_to_use_name = default_db_from_client.name
                logger.info(
                    f"No explicit DB in URI path or args, using default from client: {db_to_use_name}"
                )
            else:
                logger.error(
                    f"No database name specified in URI ('{uri}') or as an argument,"
                    f" and no default resolvable. Cannot determine database context."
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
        _current_resolved_db_name = _db.name

        logger.info(
            f"Successfully connected to MongoDB server. URI: '{uri}', Effective DB: '{_current_resolved_db_name}'."
        )
        return True
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(
            f"MongoDB connection/ping failed for URI '{uri}', target DB '{db_name or db_to_use_name}': {e}"
        )
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        _current_uri = None
        _current_db_name_arg = None
        _current_resolved_db_name = None
        return False
    except Exception as e:
        logger.error(f"Unexpected error during MongoDB connection: {e}", exc_info=True)
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        _current_uri = None
        _current_db_name_arg = None
        _current_resolved_db_name = None
        return False


def get_mongo_db() -> PyMongoDatabase:
    global _current_uri, _db, _client
    if _db is None or _client is None:
        raise ConnectionError("Not connected to MongoDB. Call db_connection_active first.")
    try:
        _client.admin.command({"ping": 1})
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(f"MongoDB connection lost when trying to get DB: {e}")
        disconnect_mongo()
        raise ConnectionError("MongoDB connection lost. Reconnect needed.") from e
    return _db


def get_mongo_client() -> Optional[MongoClient]:
    if _client:
        try:
            _client.admin.command("ping")
            return _client
        except (ConnectionFailure, OperationFailure) as e:
            logger.warning(f"Ping failed for existing client, considered disconnected: {e}")
            disconnect_mongo()
            return None
    return None


def get_current_uri() -> Optional[str]:
    return _current_uri


def get_current_resolved_db_name() -> Optional[str]:
    return _current_resolved_db_name


def disconnect_mongo() -> None:
    global _client, _db, _current_uri, _current_db_name_arg, _current_resolved_db_name
    if _client is not None:
        _client.close()
        logger.info("MongoDB client connection closed.")
    _client = None
    _db = None
    _current_uri = None
    _current_db_name_arg = None
    _current_resolved_db_name = None


def disconnect_all_mongo() -> None:
    disconnect_mongo()
