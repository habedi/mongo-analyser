from mongo_analyser.llm_chat import main_tui

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Call the main TUI function to start the application
    main_tui()
