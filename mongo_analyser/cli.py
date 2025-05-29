import argparse
import logging
import os
import sys

from mongo_analyser.tui import main_interactive_tui

from . import __version__ as mongo_analyser_version

logger = logging.getLogger(__name__)


def main():
    if not logging.getLogger().hasHandlers():
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=log_level_str,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    parser = argparse.ArgumentParser(
        description="Mongo Analyser: Analyze and Understand Your Data in MongoDB",
        prog="mongo-analyser",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{mongo_analyser_version}",
    )

    args = parser.parse_args()

    try:
        main_interactive_tui()
    except Exception as e:
        logger.critical(f"Unhandled error during TUI execution: {e}", exc_info=True)
        print(
            f"Critical error during TUI execution. Check logs for details:"
            f" {getattr(logger.handlers[1], 'baseFilename', 'mongo_analyser_tui.log') if len(logger.handlers) > 1 and hasattr(logger.handlers[1], 'baseFilename') else 'mongo_analyser_tui.log'}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        logger.debug("CLI launcher finished.")


if __name__ == "__main__":
    main()
