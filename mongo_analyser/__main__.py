import argparse
import getpass
import os
import sys
from pathlib import Path

from mongo_analyser.core.config_manager import DEFAULT_CONFIG_FILE_NAME
from mongo_analyser.core.shared import build_mongo_uri
from mongo_analyser.tui import APP_LOGGER_NAME, main_interactive_tui

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

    parser.add_argument(
        "--app-data-dir",
        dest="app_data_dir_cli",
        type=str,
        default=os.getenv("MONGO_ANALYSER_HOME_DIR"),
        help=f"Path to the root application data directory for configs, logs, exports, etc. "
        f"Overrides default XDG path (e.g., ~/.local/share/{APP_LOGGER_NAME}). "
        f"Environment variable: MONGO_ANALYSER_HOME_DIR.",
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
        help="MongoDB connection URI...",
    )
    conn_group.add_argument(
        "--host",
        dest="mongo_host",
        type=str,
        default=os.getenv("MONGO_HOST", "localhost"),
        help="MongoDB host...",
    )
    conn_group.add_argument(
        "--port",
        dest="mongo_port",
        type=int,
        default=int(os.getenv("MONGO_PORT", 27017)),
        help="MongoDB port...",
    )
    conn_group.add_argument(
        "--username",
        dest="mongo_username",
        type=str,
        default=os.getenv("MONGO_USERNAME"),
        help="MongoDB username...",
    )
    conn_group.add_argument(
        "--password-env",
        dest="mongo_password_env",
        type=str,
        metavar="ENV_VAR_NAME",
        help="Environment variable name for MongoDB password...",
    )
    conn_group.add_argument(
        "--db",
        dest="mongo_database",
        type=str,
        default=os.getenv("MONGO_DATABASE"),
        help="MongoDB database name...",
    )

    args = parser.parse_args()

    base_app_data_dir_override: Optional[Path] = None
    final_config_file_path: Optional[Path] = None

    custom_app_data_dir_str = args.app_data_dir_cli

    if custom_app_data_dir_str:
        try:
            resolved_custom_dir = Path(custom_app_data_dir_str).expanduser().resolve()

            resolved_custom_dir.mkdir(parents=True, exist_ok=True)
            base_app_data_dir_override = resolved_custom_dir
            final_config_file_path = resolved_custom_dir / DEFAULT_CONFIG_FILE_NAME
            if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
                print(
                    f"CLI Debug: Using custom application data directory: {base_app_data_dir_override}",
                    file=sys.stderr,
                )
                print(
                    f"CLI Debug: Custom configuration file path set to: {final_config_file_path}",
                    file=sys.stderr,
                )
        except OSError as e:
            print(
                f"ERROR: Could not create or access specified application data directory '{custom_app_data_dir_str}': {e}. "
                f"Please check path and permissions. Using default path instead.",
                file=sys.stderr,
            )

            base_app_data_dir_override = None
            final_config_file_path = None
    else:
        if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
            print(
                "CLI Debug: Using default application data directory logic within ConfigManager.",
                file=sys.stderr,
            )

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
                    if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
                        print(
                            f"CLI Debug: Environment variable '{args.mongo_password_env}' for MongoDB password not set. Prompting.",
                            file=sys.stderr,
                        )
                    password_to_use = getpass.getpass(
                        f"Enter password for MongoDB user '{username_to_use}' on {host_to_use}:{port_to_use}: "
                    )
            else:
                password_to_use = getpass.getpass(
                    f"Enter password for MongoDB user '{username_to_use}' on {host_to_use}:{port_to_use}: "
                )
        effective_mongo_uri = build_mongo_uri(
            host=host_to_use, port=port_to_use, username=username_to_use, password=password_to_use
        )

    try:
        main_interactive_tui(
            initial_mongo_uri=effective_mongo_uri,
            initial_db_name=initial_target_db_name,
            base_app_data_dir_override=base_app_data_dir_override,
        )
    except Exception as e:
        print("\nCRITICAL ERROR: Mongo Analyser TUI unexpectedly quit.", file=sys.stderr)
        print(f"Exception: {type(e).__name__}: {e}", file=sys.stderr)

        expected_log_dir_base = Path.home() / ".local" / "share"
        if base_app_data_dir_override:
            expected_log_dir_base = base_app_data_dir_override
        elif os.getenv("XDG_DATA_HOME"):
            expected_log_dir_base = Path(os.getenv("XDG_DATA_HOME")) / APP_LOGGER_NAME
        else:
            expected_log_dir_base = Path.home() / ".local" / "share" / APP_LOGGER_NAME

        final_log_file = expected_log_dir_base / "logs" / f"{APP_LOGGER_NAME}_tui.log"

        print(
            f"If application logging was enabled, check the log file for a detailed traceback, "
            f"expected around: {final_log_file}",
            file=sys.stderr,
        )
        if os.getenv("MONGO_ANALYSER_CLI_DEBUG"):
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
