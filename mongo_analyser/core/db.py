import logging
from typing import Optional

from mongoengine import connect as me_connect
from mongoengine import disconnect as me_disconnect
from mongoengine import get_db as me_get_db
from mongoengine.connection import ConnectionFailure as MEConnectionLibFailure
from mongoengine.connection import get_connection
from mongoengine.errors import NotRegistered as MENotRegisteredError
from pymongo.database import Database as PyMongoDatabase
from pymongo.errors import (
    ConnectionFailure as PyMongoNetworkConnectionFailure,
)
from pymongo.errors import (
    OperationFailure as PyMongoOperationFailure,
)

logger = logging.getLogger(__name__)

DEFAULT_ALIAS = "mongo_analyser_core_connection"


def db_connection_active(
    uri: str,
    db_name: Optional[str] = None,
    alias: str = DEFAULT_ALIAS,
    server_timeout_ms: int = 5000,
    **kwargs,
) -> bool:
    try:
        get_connection(alias)
        db_to_ping = me_get_db(alias)
        db_to_ping.client.admin.command("ping")
        logger.debug(f"Already connected and live with alias '{alias}'.")
        return True
    except (MEConnectionLibFailure, PyMongoNetworkConnectionFailure, PyMongoOperationFailure) as e:
        logger.info(
            f"No existing live MongoEngine connection for alias '{alias}' (Error: {e}). Attempting new connection."
        )
        try:
            me_disconnect(alias)
        except (MENotRegisteredError, MEConnectionLibFailure):
            pass

    try:
        connect_params = {
            "host": uri,
            "alias": alias,
            "serverSelectionTimeoutMS": server_timeout_ms,
            **kwargs,
        }
        if db_name:
            connect_params["db"] = db_name

        me_connect(**connect_params)

        db_instance_to_ping = me_get_db(alias)
        db_instance_to_ping.client.admin.command("ping")

        logger.info(
            f"MongoEngine connected successfully with alias '{alias}' to DB '{db_instance_to_ping.name}'."
        )
        return True
    except (PyMongoNetworkConnectionFailure, PyMongoOperationFailure) as e:
        logger.error(
            f"MongoEngine connection failed (network/auth) for alias '{alias}'. URI: {uri}. DB: {db_name}. Error: {e}",
            exc_info=False,
        )
        try:
            me_disconnect(alias)
        except (MENotRegisteredError, MEConnectionLibFailure):
            pass
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error during MongoEngine connection for alias '{alias}': {e}",
            exc_info=True,
        )
        try:
            me_disconnect(alias)
        except (MENotRegisteredError, MEConnectionLibFailure):
            pass
        return False


def get_mongo_db(alias: str = DEFAULT_ALIAS) -> PyMongoDatabase:
    try:
        get_connection(alias)
        db = me_get_db(alias)
        return db
    except MEConnectionLibFailure as e:
        logger.error(
            f"Attempted to get database for alias '{alias}' but no active connection registered for it."
        )
        raise ConnectionError(
            f"Not connected via MongoEngine for alias '{alias}'. Call db_connection_active first."
        ) from e


def disconnect_mongo(alias: str = DEFAULT_ALIAS) -> None:
    try:
        me_disconnect(alias)
        logger.info(f"MongoEngine connection with alias '{alias}' disconnected.")
    except MENotRegisteredError:
        logger.debug(
            f"No active MongoEngine connection with alias '{alias}' to disconnect (NotRegistered)."
        )
    except MEConnectionLibFailure as e:
        logger.warning(f"Connection library failure during disconnect for alias '{alias}': {e}")


def disconnect_all_mongo() -> None:
    try:
        from mongoengine import connection as me_connection_module

        aliases_to_disconnect = list(getattr(me_connection_module, "_connections", {}).keys())
        for alias_name in aliases_to_disconnect:
            try:
                me_disconnect(alias_name)
            except MENotRegisteredError:
                logger.debug(f"Alias {alias_name} was already gone during disconnect_all.")
            except MEConnectionLibFailure as e:
                logger.warning(
                    f"Connection library failure during disconnect_all for alias '{alias_name}': {e}"
                )

        if getattr(me_connection_module, "_connection_settings", {}).get(DEFAULT_ALIAS) or getattr(
            me_connection_module, "_connections", {}
        ).get(DEFAULT_ALIAS):
            if DEFAULT_ALIAS not in aliases_to_disconnect:
                try:
                    me_disconnect(DEFAULT_ALIAS)
                except (MENotRegisteredError, MEConnectionLibFailure):
                    pass

        if "default" not in aliases_to_disconnect and (
            getattr(me_connection_module, "_connection_settings", {}).get("default")
            or getattr(me_connection_module, "_connections", {}).get("default")
        ):
            try:
                me_disconnect()
            except (MENotRegisteredError, MEConnectionLibFailure):
                pass

    except Exception as e:
        logger.error(f"Error during disconnect_all_mongo: {e}", exc_info=True)
    logger.info("Attempted to disconnect all known MongoEngine connections.")
