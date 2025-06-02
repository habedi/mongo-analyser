import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture(autouse=True)
def mock_llm_list_models(mocker):
    mocker.patch("mongo_analyser.llm_chat.ollama.OllamaChat.list_models",
                 return_value=["ollama_model_test_1", "ollama_model_test_2"], create=True)
    mocker.patch("mongo_analyser.llm_chat.openai.OpenAIChat.list_models",
                 return_value=["openai_model_test_1", "openai_model_test_2"], create=True)
    mocker.patch("mongo_analyser.llm_chat.google.GoogleChat.list_models",
                 return_value=["google_model_test_1", "google_model_test_2"], create=True)


@pytest.fixture(autouse=True)
def mock_app_on_mount_heavy_operations(mocker):
    """
    Mocks potentially slow or side-effect-heavy operations that might run during
    MongoAnalyserApp.on_mount or its children's on_mount methods.
    This helps to speed up tests and avoid WaitForScreenTimeout.
    """
    # Mock the method in ChatView that loads models, as it might use a worker
    # or make network calls (though list_models is already mocked above, this prevents
    # the worker scheduling if _load_models_for_provider is complex)
    mocker.patch("mongo_analyser.views.chat_view.ChatView._load_models_for_provider",
                 new_callable=AsyncMock)

    # If MongoAnalyserApp.on_mount directly calls run_worker for anything not covered
    # by the above, you could mock it here. Example:
    # mocker.patch("mongo_analyser.tui.MongoAnalyserApp.run_worker", new_callable=MagicMock())
    # However, it's often better to mock the specific target function of the worker.

    # Mock the config loading part in app.on_mount to prevent actual file I/O
    # if ConfigManager is initialized there and loads.
    # The ConfigManager tests themselves handle mocking file I/O.
    # This prevents on_mount from failing if a config file is unexpectedly absent/corrupt.
    mocker.patch("mongo_analyser.core.config_manager.ConfigManager.load_config", return_value=None)


@pytest.fixture
async def app_pilot(event_loop):  # Added event_loop for pytest-asyncio
    """Provides a pilot for the main app, ensuring it starts and settles."""
    from mongo_analyser.tui import MongoAnalyserApp  # Local import
    app = MongoAnalyserApp()
    async with app.run_test(headless=True, size=(120, 40)) as pilot:  # headless and size can help
        # Add a small delay to ensure the app has fully mounted and processed initial events
        # This is often necessary when on_mount methods trigger further actions.
        await pilot.pause(0.1)
        yield pilot
