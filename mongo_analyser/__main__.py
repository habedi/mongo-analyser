import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Optional

from mongo_analyser.core.config_manager import DEFAULT_CONFIG_FILE_NAME
from mongo_analyser.core.shared import build_mongo_uri
from mongo_analyser.tui import APP_LOGGER_NAME, main_interactive_tui

from . import __version__ as mongo_analyser_version


def main():
    parser = argparse.ArgumentParser(
        description="Mongo Analyser – dig into your MongoDB data right from the terminal.",
        prog="mongo_analyser",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Mongo Analyser {mongo_analyser_version}",
    )

    parser.add_argument(
        "--app-data-dir",
        dest="app_data_dir_cli",
        type=str,
        default=os.getenv("MONGO_ANALYSER_HOME_DIR"),
        help=(
            "Path to store configs, logs, exports, etc. "
            "Overrides XDG default (e.g., ~/.local/share/mongo_analyser). "
            "You can also set MONGO_ANALYSER_HOME_DIR."
        ),
    )

    conn_group = parser.add_argument_group(
        title="MongoDB Connection (pre-fill)",
        description="Pass connection details to start with these values in the TUI.",
    )
    conn_group.add_argument(
        "--uri",
        dest="mongo_uri",
        type=str,
        default=os.getenv("MONGO_URI"),
        help="Full MongoDB URI (overrides host/port/user).",
    )
    conn_group.add_argument(
        "--host",
        dest="mongo_host",
        type=str,
        default=os.getenv("MONGO_HOST", "localhost"),
        help="MongoDB host (default: localhost).",
    )
    conn_group.add_argument(
        "--port",
        dest="mongo_port",
        type=int,
        default=int(os.getenv("MONGO_PORT", 27017)),
        help="MongoDB port (default: 27017).",
    )
    conn_group.add_argument(
        "--username",
        dest="mongo_username",
        type=str,
        default=os.getenv("MONGO_USERNAME"),
        help="MongoDB username (if auth is required).",
    )
    conn_group.add_argument(
        "--password-env",
        dest="mongo_password_env",
        type=str,
        metavar="ENV_VAR_NAME",
        help="Name of env var holding MongoDB password.",
    )
    conn_group.add_argument(
        "--db",
        dest="mongo_database",
        type=str,
        default=os.getenv("MONGO_DATABASE"),
        help="MongoDB database name to open by default.",
    )

    args = parser.parse_args()

    base_app_data_dir_override: Optional[Path] = None
    final_config_file_path: Optional[Path] = None

    if args.app_data_dir_cli:
        custom_dir = Path(args.app_data_dir_cli).expanduser()
        try:
            custom_dir.mkdir(parents=True, exist_ok=True)
            base_app_data_dir_override = custom_dir.resolve()
            final_config_file_path = base_app_data_dir_override / DEFAULT_CONFIG_FILE_NAME
            if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
                print(f"DEBUG: app data dir set to {base_app_data_dir_override}", file=sys.stderr)
                print(f"DEBUG: config path is {final_config_file_path}", file=sys.stderr)
        except OSError as e:
            print(
                f"ERROR: Can't use '{args.app_data_dir_cli}' for app data: {e}. "
                "Falling back to default location.",
                file=sys.stderr,
            )
    else:
        if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
            print("DEBUG: Using default app data directory logic", file=sys.stderr)

    effective_mongo_uri = args.mongo_uri
    initial_db = args.mongo_database

    if not effective_mongo_uri:
        host = args.mongo_host
        port = args.mongo_port
        user = args.mongo_username
        pwd = None

        if user:
            if args.mongo_password_env:
                pwd = os.getenv(args.mongo_password_env)
                if pwd is None:
                    if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
                        print(
                            f"DEBUG: env var '{args.mongo_password_env}' not set. Asking for password.",
                            file=sys.stderr,
                        )
                    pwd = getpass.getpass(f"Password for '{user}' on {host}:{port}: ")
            else:
                pwd = getpass.getpass(f"Password for '{user}' on {host}:{port}: ")

        effective_mongo_uri = build_mongo_uri(host=host, port=port, username=user, password=pwd)

    try:
        main_interactive_tui(
            initial_mongo_uri=effective_mongo_uri,
            initial_db_name=initial_db,
            base_app_data_dir_override=base_app_data_dir_override,
        )
    except Exception as e:
        print("\nCRITICAL: The TUI exited unexpectedly.", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)

        if base_app_data_dir_override:
            log_base = base_app_data_dir_override
        elif os.getenv("XDG_DATA_HOME"):
            log_base = Path(os.getenv("XDG_DATA_HOME")) / APP_LOGGER_NAME
        else:
            log_base = Path.home() / ".local" / "share" / APP_LOGGER_NAME

        log_file = log_base / "logs" / f"{APP_LOGGER_NAME}_tui.log"
        print(f"Check the log at: {log_file}", file=sys.stderr)

        if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
            import traceback

            traceback.print_exc(file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()
