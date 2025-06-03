import argparse
import getpass
import os
import sys
from pathlib import Path

from mongo_analyser.core.shared import build_mongo_uri, redact_uri_password
from mongo_analyser.tui import main_interactive_tui
from . import __version__ as mongo_analyser_version


def main():
    parser = argparse.ArgumentParser(
        description="Mongo Analyser: Analyze and Understand Your Data in MongoDB from the command line.",
        prog="mongo_analyser",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Mongo Analyser version {mongo_analyser_version}",
    )

    conn_group = parser.add_argument_group(
        title="MongoDB Connection Pre-fill Options",
        description="Provide connection details to pre-fill the TUI. These can be changed within the application.",
    )
    conn_group.add_argument(
        "--uri",
        dest="mongo_uri",
        type=str,
        default=os.getenv("MONGO_URI"),
        help="MongoDB connection URI (e.g., 'mongodb://user:pass@host:port/db?options'). "
             "If provided, other connection parts (host, port, etc.) are often ignored by the URI parser.",
    )
    conn_group.add_argument(
        "--host",
        dest="mongo_host",
        type=str,
        default=os.getenv("MONGO_HOST", "localhost"),
        help="MongoDB host. Default: localhost (if MONGO_HOST env var not set). Used if --uri is not provided.",
    )
    conn_group.add_argument(
        "--port",
        dest="mongo_port",
        type=int,
        default=int(os.getenv("MONGO_PORT", 27017)),  # Ensure int conversion for getenv
        help="MongoDB port. Default: 27017 (if MONGO_PORT env var not set). Used if --uri is not provided.",
    )
    conn_group.add_argument(
        "--username",
        dest="mongo_username",
        type=str,
        default=os.getenv("MONGO_USERNAME"),
        help="MongoDB username. If provided without a password in the URI or via --password-env, you may be prompted.",
    )
    conn_group.add_argument(
        "--password-env",
        dest="mongo_password_env",
        type=str,
        metavar="ENV_VAR_NAME",
        help="Environment variable name to read MongoDB password from. "
             "Used if --username is provided and password is not in URI.",
    )
    conn_group.add_argument(
        "--db",
        dest="mongo_database",
        type=str,
        default=os.getenv("MONGO_DATABASE"),
        help="MongoDB database name to pre-fill in the TUI. Can also be part of the URI path.",
    )

    args = parser.parse_args()

    effective_mongo_uri = args.mongo_uri
    initial_target_db_name = args.mongo_database

    if not effective_mongo_uri:
        host_to_use = args.mongo_host
        port_to_use = args.mongo_port
        username_to_use = args.mongo_username
        password_to_use = None

        if username_to_use:
            if args.mongo_password_env:
                password_to_use = os.getenv(args.mongo_password_env)
                if password_to_use is None:
                    # Minimal feedback to stderr if password needs to be prompted
                    print(
                        f"Info: Environment variable '{args.mongo_password_env}' for MongoDB password not set.",
                        file=sys.stderr
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
        # Example of a CLI-specific debug message if an env var is set
        if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
            print(
                f"CLI Debug: Constructed MongoDB URI: {redact_uri_password(effective_mongo_uri)}",
                file=sys.stderr
            )
    else:
        if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
            print(
                f"CLI Debug: Using provided MongoDB URI: {redact_uri_password(effective_mongo_uri)}",
                file=sys.stderr
            )

    try:
        main_interactive_tui(
            log_level_override=None,  # Logging level is now handled by the TUI/config
            initial_mongo_uri=effective_mongo_uri,
            initial_db_name=initial_target_db_name,
        )
    except Exception as e:
        print("\nCRITICAL ERROR: Mongo Analyser TUI unexpectedly quit.", file=sys.stderr)
        print(f"Exception: {type(e).__name__}: {e}", file=sys.stderr)

        # Determine the default log file path as defined in tui.py's logging setup
        default_log_dir = Path(
            os.getenv(
                "MONGO_ANALYSER_LOG_DIR",  # Check if the same env var is used for custom log dir
                Path.home() / ".local" / "share" / "mongo_analyser" / "logs",
            )
        )
        default_log_file = default_log_dir / "mongo_analyser_tui.log"

        print(
            f"If application logging was enabled (via its internal configuration), "
            f"check the log file for a detailed traceback, typically located at: {default_log_file}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
