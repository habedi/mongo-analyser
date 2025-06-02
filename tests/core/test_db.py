from unittest.mock import MagicMock, patch, ANY

import pytest
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, ConnectionFailure, OperationFailure
from pymongo.server_api import ServerApi

from mongo_analyser.core import db as db_manager


@pytest.fixture(autouse=True)
def reset_db_module_state():
    db_manager.disconnect_mongo()
    yield
    db_manager.disconnect_mongo()


@pytest.fixture
def mock_mongo_client_class():
    with patch("mongo_analyser.core.db.MongoClient") as MockClient:
        yield MockClient


@pytest.fixture
def mock_mongo_client_instance(mock_mongo_client_class):
    mock_instance = MagicMock(spec=MongoClient)

    mock_admin_db = MagicMock()
    mock_admin_db.command.return_value = {"ok": 1}
    mock_instance.admin = mock_admin_db

    mock_default_db = MagicMock()
    mock_default_db.name = "resolved_db_from_uri"
    mock_instance.get_database.return_value = mock_default_db

    mock_db_via_getitem = MagicMock()
    mock_db_via_getitem.name = "resolved_db_from_uri"
    mock_instance.__getitem__.return_value = mock_db_via_getitem

    mock_mongo_client_class.return_value = mock_instance
    return mock_instance


class TestDBManager:

    def test_db_connection_active_new_connection_success_uri_db(
        self, mock_mongo_client_instance
    ):
        uri = "mongodb://localhost/mydb"

        is_active = db_manager.db_connection_active(uri=uri, db_name=None)
        assert is_active is True

        assert db_manager.get_mongo_client() == mock_mongo_client_instance

        result_db = db_manager.get_mongo_db()
        assert result_db.name == "resolved_db_from_uri"

        assert db_manager.get_current_uri() == uri
        assert db_manager.get_current_resolved_db_name() == "resolved_db_from_uri"

        mock_mongo_client_instance.admin.command.assert_any_call("ping")

    def test_db_connection_active_new_connection_success_explicit_db(
        self, mock_mongo_client_instance
    ):
        uri = "mongodb://localhost/"
        db_name_arg = "explicit_db"

        mock_db_for_explicit = MagicMock()
        mock_db_for_explicit.name = db_name_arg
        mock_mongo_client_instance.__getitem__.return_value = mock_db_for_explicit

        is_active = db_manager.db_connection_active(uri=uri, db_name=db_name_arg)
        assert is_active is True

        assert db_manager.get_mongo_db().name == db_name_arg
        mock_mongo_client_instance.__getitem__.assert_called_with(db_name_arg)

    def test_db_connection_active_srv_uri(
        self, mock_mongo_client_class, mock_mongo_client_instance
    ):
        srv_uri = "mongodb+srv://user@cluster/"

        db_manager.db_connection_active(uri=srv_uri, db_name="mydbfromsrv")

        mock_mongo_client_class.assert_called_once_with(
            srv_uri,
            serverSelectionTimeoutMS=5000,
            server_api=ANY
        )
        assert isinstance(mock_mongo_client_class.call_args[1]["server_api"], ServerApi)

    def test_db_connection_active_connection_failure_on_ping(
        self, mock_mongo_client_instance
    ):
        uri = "mongodb://localhost/mydb"
        mock_mongo_client_instance.admin.command.side_effect = ConnectionFailure("Ping failed")

        is_active = db_manager.db_connection_active(uri=uri, db_name="mydb")
        assert is_active is False
        assert db_manager.get_mongo_client() is None

        last_error = db_manager.get_last_connection_error_details()
        assert last_error is not None
        assert "Ping failed" in last_error[0]

    def test_db_connection_active_configuration_error(
        self, mock_mongo_client_class
    ):
        uri = "mongodb://badconfig"
        mock_mongo_client_class.side_effect = ConfigurationError("Bad URI")

        is_active = db_manager.db_connection_active(uri=uri, db_name="anydb")
        assert is_active is False

        last_error = db_manager.get_last_connection_error_details()
        assert last_error is not None
        assert "Bad URI" in last_error[0]

    def test_db_connection_active_no_db_name_resolved_or_provided(
        self, mock_mongo_client_instance
    ):
        uri = "mongodb://localhost/"
        mock_mongo_client_instance.get_database.side_effect = ConfigurationError("No default DB")

        is_active = db_manager.db_connection_active(uri=uri, db_name=None)
        assert is_active is False

        last_error = db_manager.get_last_connection_error_details()
        assert last_error is not None
        assert "does not specify a default database path" in last_error[0]

    def test_db_connection_active_existing_connection_ping_ok(
        self, mock_mongo_client_instance
    ):
        uri = "mongodb://localhost/mydb"
        mock_mongo_client_instance.get_database.return_value.name = "mydb"

        db_manager.db_connection_active(uri=uri, db_name="mydb")

        mock_mongo_client_instance.admin.command.reset_mock()
        is_active_again = db_manager.db_connection_active(uri=uri, db_name="mydb")

        assert is_active_again is True
        mock_mongo_client_instance.admin.command.assert_called_once_with("ping")

    def test_db_connection_active_existing_connection_switch_db(
        self, mock_mongo_client_instance
    ):
        uri = "mongodb://localhost/"
        initial_db_name = "db1"
        new_db_name = "db2"

        mock_db1 = MagicMock()
        mock_db1.name = initial_db_name
        mock_mongo_client_instance.__getitem__.return_value = mock_db1
        mock_mongo_client_instance.get_database.return_value = mock_db1
        db_manager.db_connection_active(uri=uri, db_name=initial_db_name)
        assert db_manager.get_current_resolved_db_name() == initial_db_name

        mock_db2 = MagicMock()
        mock_db2.name = new_db_name
        mock_mongo_client_instance.__getitem__.return_value = mock_db2
        mock_mongo_client_instance.admin.command.reset_mock()

        is_active_switched = db_manager.db_connection_active(uri=uri, db_name=new_db_name)

        assert is_active_switched is True
        assert db_manager.get_current_resolved_db_name() == new_db_name
        mock_mongo_client_instance.__getitem__.assert_called_with(new_db_name)
        mock_mongo_client_instance.admin.command.assert_called_with("ping")

    def test_db_connection_active_force_reconnect(
        self, mock_mongo_client_class, mock_mongo_client_instance
    ):
        uri = "mongodb://localhost/mydb"
        db_name = "mydb"

        mock_mongo_client_instance.get_database.return_value.name = db_name
        db_manager.db_connection_active(uri=uri, db_name=db_name)
        initial_client = db_manager.get_mongo_client()

        new_instance = MagicMock(spec=MongoClient)
        new_admin = MagicMock()
        new_admin.command.return_value = {"ok": 1}
        new_instance.admin = new_admin
        new_default_db = MagicMock()
        new_default_db.name = db_name
        new_instance.get_database.return_value = new_default_db
        new_instance.__getitem__.return_value = new_default_db
        mock_mongo_client_class.return_value = new_instance

        is_reconnect = db_manager.db_connection_active(uri=uri, db_name=db_name,
                                                       force_reconnect=True)
        assert is_reconnect is True
        assert db_manager.get_mongo_client() == new_instance
        assert db_manager.get_mongo_client() != initial_client
        mock_mongo_client_instance.close.assert_called_once()

    def test_get_mongo_db_not_connected(self):
        with pytest.raises(ConnectionError, match="Not connected to MongoDB"):
            db_manager.get_mongo_db()

    def test_get_mongo_db_connection_lost_on_ping(self, mock_mongo_client_instance):
        uri = "mongodb://localhost/mydb"
        mock_mongo_client_instance.get_database.return_value.name = "mydb"
        db_manager.db_connection_active(uri=uri, db_name="mydb")

        mock_mongo_client_instance.admin.command.side_effect = ConnectionFailure("Lost on ping")

        with pytest.raises(ConnectionError, match="MongoDB connection lost. Reconnect needed."):
            db_manager.get_mongo_db()

        assert db_manager.get_last_connection_error_details() is None

    def test_disconnect_mongo_closes_client(self, mock_mongo_client_instance):
        uri = "mongodb://localhost/mydb"
        mock_mongo_client_instance.get_database.return_value.name = "mydb"
        db_manager.db_connection_active(uri=uri, db_name="mydb")
        assert db_manager.get_mongo_client() is not None

        db_manager.disconnect_mongo()
        assert db_manager.get_mongo_client() is None
        with pytest.raises(ConnectionError):
            _ = db_manager.get_mongo_db()
        assert db_manager.get_current_uri() is None
        assert db_manager.get_current_resolved_db_name() is None
        assert db_manager.get_last_connection_error_details() is None
        mock_mongo_client_instance.close.assert_called_once()

    def test_get_last_connection_error_details_flow(self, mock_mongo_client_instance):
        assert db_manager.get_last_connection_error_details() is None

        uri = "mongodb://localhost/fail_db"
        mock_mongo_client_instance.admin.command.side_effect = OperationFailure("DB connect fail",
                                                                                code=123)
        mock_mongo_client_instance.get_database.return_value.name = "fail_db"

        db_manager.db_connection_active(uri=uri, db_name="fail_db")

        last_error = db_manager.get_last_connection_error_details()
        assert last_error is not None
        assert last_error[0] == "DB connect fail"
        assert last_error[1] == 123

        db_manager.disconnect_mongo()
        assert db_manager.get_last_connection_error_details() is None
