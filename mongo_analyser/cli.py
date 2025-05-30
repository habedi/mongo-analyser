import argparse
import getpass
import logging
import os
import sys

from mongo_analyser.core.shared import build_mongo_uri, redact_uri_password
from mongo_analyser.tui import main_interactive_tui

from . import __version__ as mongo_analyser_version

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Mongo Analyser: Analyze and Understand Your Data in MongoDB",
        prog="mongo_analyser",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Mongo Analyser version {mongo_analyser_version}",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"],
        default="OFF",
        help="Set the logging level for the application. Default is OFF (no logs).",
    )

    conn_group = parser.add_argument_group(title="MongoDB Connection Options")
    conn_group.add_argument(
        "--mongo-uri",
        type=str,
        default=os.getenv("MONGO_URI"),
        help="MongoDB connection URI. If provided, it's used directly. "
        "This can be overridden by inputs in the TUI later.",
    )
    conn_group.add_argument(
        "--mongo-host",
        type=str,
        default=os.getenv("MONGO_HOST"),
        help="MongoDB host. Used if --mongo-uri is not provided.",
    )
    conn_group.add_argument(
        "--mongo-port",
        type=int,
        default=os.getenv("MONGO_PORT"),
        help="MongoDB port. Used if --mongo-uri is not provided.",
    )
    conn_group.add_argument(
        "--mongo-username", type=str, default=os.getenv("MONGO_USERNAME"), help="MongoDB username."
    )
    conn_group.add_argument(
        "--mongo-password-env",
        type=str,
        metavar="ENV_VAR_NAME",
        help="Environment variable name to read MongoDB password from. "
        "If --mongo-username is provided and password is not in URI or via this env var, "
        "password will be prompted.",
    )
    conn_group.add_argument(
        "--mongo-database",
        type=str,
        default=os.getenv("MONGO_DATABASE"),
        help="MongoDB database name to pre-fill in the TUI.",
    )

    args = parser.parse_args()

    if args.log_level == "OFF":
        logging.disable(logging.CRITICAL + 1)

    else:
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,
        )

    logger.debug("CLI arguments parsed: %s", args)

    effective_mongo_uri = args.mongo_uri
    initial_target_db_name = args.mongo_database

    if not effective_mongo_uri:
        host_to_use = args.mongo_host or "localhost"
        port_to_use = args.mongo_port or 27017
        username_to_use = args.mongo_username
        password_to_use = None

        if username_to_use:
            if args.mongo_password_env:
                password_to_use = os.getenv(args.mongo_password_env)
                if password_to_use is None:
                    logger.warning(
                        "Environment variable '%s' for MongoDB password not set. Prompting for password.",
                        args.mongo_password_env,
                    )
                    password_to_use = getpass.getpass(
                        f"Enter password for MongoDB user '{username_to_use}' on {host_to_use}:{port_to_use}: "
                    )
            else:
                password_to_use = getpass.getpass(
                    f"Enter password for MongoDB user '{username_to_use}' on {host_to_use}:{port_to_use}: "
                )

        effective_mongo_uri = build_mongo_uri(
            host=host_to_use,
            port=port_to_use,
            username=username_to_use,
            password=password_to_use,
        )
        logger.info(
            "Constructed MongoDB URI from parts: %s", redact_uri_password(effective_mongo_uri)
        )
    else:
        logger.info("Using provided MongoDB URI: %s", redact_uri_password(effective_mongo_uri))

    try:
        main_interactive_tui(
            log_level_override=args.log_level,
            initial_mongo_uri=effective_mongo_uri,
            initial_db_name=initial_target_db_name,
        )
    except Exception as e:
        log_file_path = "mongo_analyser_tui.log"

        app_logger = logging.getLogger("mongo_analyser")
        for handler in app_logger.handlers:
            if isinstance(handler, logging.FileHandler) and hasattr(handler, "baseFilename"):
                log_file_path = handler.baseFilename
                break

        print("\nCRITICAL ERROR: Mongo Analyser TUI unexpectedly quit.", file=sys.stderr)
        print(f"Exception: {type(e).__name__}: {e}", file=sys.stderr)

        if args.log_level != "OFF":
            print(
                f"Please check the log file for detailed traceback: {log_file_path}",
                file=sys.stderr,
            )
        else:
            print(
                "Logging was set to OFF. For a detailed error trace, re-run with a specific --log-level (e.g., --log-level DEBUG)",
                file=sys.stderr,
            )

        if logger.isEnabledFor(logging.CRITICAL):
            logger.critical(
                "Unhandled error launching or during TUI execution: %s", e, exc_info=True
            )
        sys.exit(1)
    finally:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("CLI launcher finished.")


if __name__ == "__main__":
    main()
