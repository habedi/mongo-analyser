import logging
from typing import Optional, Tuple

from pymongo import MongoClient
from pymongo.database import Database as PyMongoDatabase
from pymongo.errors import ConfigurationError, ConnectionFailure, OperationFailure
from pymongo.server_api import ServerApi

from mongo_analyser.core.shared import redact_uri_password

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db: Optional[PyMongoDatabase] = None
_current_uri: Optional[str] = None
_current_db_name_arg: Optional[str] = None
_current_resolved_db_name: Optional[str] = None
_last_connection_error: Optional[Tuple[str, Optional[int]]] = None


def db_connection_active(
    uri: str,
    db_name: Optional[str] = None,
    server_timeout_ms: int = 5000,
    force_reconnect: bool = False,
    **kwargs,
) -> bool:
    global \
        _client, \
        _db, \
        _current_uri, \
        _current_db_name_arg, \
        _current_resolved_db_name, \
        _last_connection_error
    _last_connection_error = None

    redacted_uri_for_log = redact_uri_password(uri)

    if not force_reconnect and _client is not None and _db is not None and _current_uri == uri:
        target_db_name_for_check = db_name if db_name else _current_resolved_db_name

        if _db.name == target_db_name_for_check:
            try:
                _client.admin.command("ping")
                logger.debug(
                    f"Already connected to MongoDB (URI: {redacted_uri_for_log}, DB: {_db.name}). Ping OK."
                )
                return True
            except (ConnectionFailure, OperationFailure) as e:
                logger.warning(f"Existing MongoDB connection ping failed: {e}. Reconnecting.")
                _last_connection_error = (str(e), getattr(e, "code", None))
                _client, _db = None, None
        elif db_name and _db.name != db_name:
            logger.info(
                f"Switching DB context on existing client from '{_db.name}' to '{db_name}'."
            )
            try:
                _db = _client[db_name]
                _client.admin.command("ping")
                _current_db_name_arg = db_name
                _current_resolved_db_name = _db.name
                logger.info(f"Successfully switched DB context to '{_db.name}'.")
                return True
            except Exception as e:
                logger.error(f"Failed to switch DB context to '{db_name}' or ping failed: {e}")
                _last_connection_error = (str(e), getattr(e, "code", None))
                _client, _db = None, None
        else:
            _client, _db = None, None

    if _client is not None:
        _client.close()
        _client, _db = None, None

    _current_uri = None
    _current_db_name_arg = None
    _current_resolved_db_name = None

    try:
        logger.info(
            f"Attempting to connect to MongoDB: {redacted_uri_for_log}, TUI DB specified: {db_name}"
        )

        client_connect_options = {"serverSelectionTimeoutMS": server_timeout_ms, **kwargs}
        if uri.startswith("mongodb+srv://"):
            client_connect_options["server_api"] = ServerApi("1")
            logger.debug("SRV URI detected, applying ServerApi('1') option.")

        temp_client = MongoClient(uri, **client_connect_options)

        db_to_connect_to: Optional[str] = None

        if db_name:
            db_to_connect_to = db_name
            logger.info(
                f"Using database name explicitly provided via TUI/argument: '{db_to_connect_to}'"
            )
        else:
            try:
                db_from_uri_path = temp_client.get_database().name
                logger.info(f"Database name resolved from URI path: '{db_from_uri_path}'")
                db_to_connect_to = db_from_uri_path

                if db_from_uri_path.lower() in ("admin", "test", "config", "local"):
                    logger.warning(
                        f"URI resolved to default/system database '{db_from_uri_path}'. "
                        "If this is not your target data DB, please specify one in the TUI."
                    )
            except ConfigurationError:
                err_msg = (
                    f"MongoDB URI ('{redacted_uri_for_log}') does not specify a default database path, "
                    "and no database name was provided via the TUI. Cannot determine database context."
                )
                logger.error(err_msg)
                _last_connection_error = (err_msg, None)
                temp_client.close()

                return False

        if not db_to_connect_to:
            err_msg = (
                f"Fatal: Database name could not be determined for URI '{redacted_uri_for_log}'."
            )
            logger.error(err_msg)
            _last_connection_error = (err_msg, None)
            temp_client.close()
            return False

        temp_client.admin.command("ping")
        logger.debug("MongoDB server ping successful.")

        target_db_object = temp_client[db_to_connect_to]

        _client = temp_client
        _db = target_db_object
        _current_uri = uri
        _current_db_name_arg = db_name
        _current_resolved_db_name = _db.name

        logger.info(
            f"Successfully connected to MongoDB server. URI: '{redacted_uri_for_log}', "
            f"Effective DB Context: '{_current_resolved_db_name}'."
        )
        return True

    except ConfigurationError as e:
        logger.error(f"MongoDB client configuration error for URI '{redacted_uri_for_log}': {e}")
        _last_connection_error = (str(e), None)
        if _client:
            _client.close()
        _client, _db, _current_uri, _current_db_name_arg, _current_resolved_db_name = (
            None,
            None,
            None,
            None,
            None,
        )
        return False
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(
            f"MongoDB connection/operation failure for URI '{redacted_uri_for_log}',"
            f" target DB '{db_name or 'from URI'}': {e}"
        )
        _last_connection_error = (str(e), getattr(e, "code", None))
        if _client:
            _client.close()
        _client, _db, _current_uri, _current_db_name_arg, _current_resolved_db_name = (
            None,
            None,
            None,
            None,
            None,
        )
        return False
    except Exception as e:
        logger.error(f"Unexpected error during MongoDB connection: {e}", exc_info=True)
        _last_connection_error = (str(e), None)
        if _client:
            _client.close()
        _client, _db, _current_uri, _current_db_name_arg, _current_resolved_db_name = (
            None,
            None,
            None,
            None,
            None,
        )
        return False


def get_last_connection_error_details() -> Optional[Tuple[str, Optional[int]]]:
    return _last_connection_error


def get_mongo_db() -> PyMongoDatabase:
    global _client, _db, _last_connection_error
    if _db is None or _client is None:
        raise ConnectionError(
            "Not connected to MongoDB or connection lost. Call db_connection_active first or reconnect."
        )
    try:
        _client.admin.command({"ping": 1})
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(f"MongoDB connection lost when trying to get DB: {e}")
        _last_connection_error = (str(e), getattr(e, "code", None))
        disconnect_mongo()
        raise ConnectionError("MongoDB connection lost. Reconnect needed.") from e
    return _db


def get_mongo_client() -> Optional[MongoClient]:
    global _last_connection_error
    if _client:
        try:
            _client.admin.command("ping")
            return _client
        except (ConnectionFailure, OperationFailure) as e:
            logger.warning(f"Ping failed for existing client, considered disconnected: {e}")
            _last_connection_error = (str(e), getattr(e, "code", None))
            disconnect_mongo()
            return None
    return None


def get_current_uri() -> Optional[str]:
    return _current_uri


def get_current_resolved_db_name() -> Optional[str]:
    return _current_resolved_db_name


def disconnect_mongo() -> None:
    global \
        _client, \
        _db, \
        _current_uri, \
        _current_db_name_arg, \
        _current_resolved_db_name, \
        _last_connection_error
    if _client is not None:
        _client.close()
        logger.info("MongoDB client connection closed.")
    _client = None
    _db = None
    _current_uri = None
    _current_db_name_arg = None
    _current_resolved_db_name = None
    _last_connection_error = None


def disconnect_all_mongo() -> None:
    disconnect_mongo()
