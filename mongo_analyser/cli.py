import argparse
import logging
import os  # Keep os for getenv
import sys

from mongo_analyser.tui import main_interactive_tui

logger = logging.getLogger(__name__)  # Keep logger if you plan to add CLI logs later


def main():
    # Basic logging setup for the CLI launcher itself, if needed.
    # The TUI sets up its own more detailed logging.
    if not logging.getLogger().hasHandlers():
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=log_level_str,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    parser = argparse.ArgumentParser(
        description="Mongo Analyser TUI Launcher.",
        prog="mongo-analyser",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.5.1",  # Example version bump
    )
    # Add any TUI-specific startup arguments here in the future
    # For example:
    # parser.add_argument(
    #     "--theme",
    #     type=str,
    #     choices=["light", "dark"],
    #     default="dark", # Or get from env
    #     help="Specify the initial theme for the TUI.",
    # )
    # parser.add_argument(
    #     "--log-level",
    #     type=str,
    #     default=os.environ.get("LOG_LEVEL", "INFO").upper(),
    #     help="Set the TUI logging level (e.g., DEBUG, INFO, WARNING)."
    # )

    # Parse arguments, but currently, we don't use them to change TUI behavior.
    # This is a placeholder for future CLI config options for the TUI.
    args = parser.parse_args()

    # if args.log_level:
    #     os.environ["LOG_LEVEL"] = args.log_level.upper() # Allow CLI to override env for TUI

    try:
        main_interactive_tui()
    except Exception as e:
        logger.critical(f"Unhandled error during TUI execution: {e}", exc_info=True)
        print(
            f"Critical error during TUI execution. Check logs for details: {getattr(logger.handlers[1], 'baseFilename', 'mongo_analyser_tui.log') if len(logger.handlers) > 1 and hasattr(logger.handlers[1], 'baseFilename') else 'mongo_analyser_tui.log'}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        # Disconnection is handled by main_interactive_tui's finally block
        # If for some reason TUI doesn't launch or crashes before its own finally:
        # db_manager.disconnect_all_mongo() # This is likely redundant now
        logger.debug("CLI launcher finished.")


if __name__ == "__main__":
    main()
