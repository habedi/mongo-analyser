from mongo_analyser.core import main_core_tui

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Call the main TUI function to start the application
    main_core_tui()
