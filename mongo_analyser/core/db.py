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
            f"No existing live MongoEngine connection for alias '{alias}' (Error: {type(e).__name__}). Attempting new connection."
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
    except (MEConnectionLibFailure, MENotRegisteredError) as e:
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
    # Reverted to using internal attributes as get_aliases() might not be available
    try:
        from mongoengine import connection as me_connection_module

        # Accessing internal attributes directly - this can be fragile
        aliases_to_disconnect = list(getattr(me_connection_module, "_connections", {}).keys())
        for alias_name in aliases_to_disconnect:
            try:
                me_disconnect(alias_name)
                logger.info(
                    f"MongoEngine connection with alias '{alias_name}' disconnected via _connections."
                )
            except MENotRegisteredError:
                logger.debug(
                    f"Alias {alias_name} from _connections was already gone during disconnect_all."
                )
            except MEConnectionLibFailure as e:
                logger.warning(
                    f"Connection library failure during disconnect_all for alias '{alias_name}' from _connections: {e}"
                )

        # Also check _connection_settings as a fallback, as an alias might be configured but not actively connected
        settings_aliases = list(getattr(me_connection_module, "_connection_settings", {}).keys())
        for alias_name in settings_aliases:
            if (
                alias_name not in aliases_to_disconnect
            ):  # Avoid trying to disconnect again if already handled
                try:
                    me_disconnect(alias_name)
                    logger.info(
                        f"MongoEngine connection with alias '{alias_name}' disconnected via _connection_settings."
                    )
                except MENotRegisteredError:
                    # This is expected if the connection was never established
                    logger.debug(
                        f"Alias {alias_name} from _connection_settings was not an active connection to disconnect."
                    )
                except MEConnectionLibFailure as e:
                    logger.warning(
                        f"Connection library failure during disconnect_all for alias '{alias_name}' from _connection_settings: {e}"
                    )

        # Explicitly try to disconnect DEFAULT_ALIAS and 'default' if they weren't in the lists,
        # as they are common aliases that might have specific handling or states.
        common_aliases_to_check = [DEFAULT_ALIAS, "default"]
        for ca in common_aliases_to_check:
            if ca not in aliases_to_disconnect and ca not in settings_aliases:
                try:
                    me_disconnect(ca)
                    logger.info(f"Explicit disconnect attempt for common alias '{ca}' successful.")
                except (MENotRegisteredError, MEConnectionLibFailure):
                    logger.debug(
                        f"Explicit disconnect attempt for common alias '{ca}' found no active connection."
                    )
                    pass

    except Exception as e:
        logger.error(f"Error during disconnect_all_mongo: {e}", exc_info=True)
    logger.info("Attempted to disconnect all known MongoEngine connections.")
