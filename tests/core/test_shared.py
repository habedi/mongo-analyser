from urllib.parse import quote_plus

import pytest

from mongo_analyser.core.shared import build_mongo_uri, redact_uri_password


class TestShared:
    @pytest.mark.parametrize(
        "host, port, username, password, params, expected_uri",
        [
            (
                    "localhost",
                    27017,
                    None,
                    None,
                    None,
                    "mongodb://localhost:27017/",
            ),
            (
                    "myhost.com",
                    "27018",
                    "user1",
                    None,
                    None,
                    f"mongodb://{quote_plus('user1')}@myhost.com:27018/",
            ),
            (
                    "127.0.0.1",
                    27017,
                    "test_user",
                    "test_pass",
                    None,
                    f"mongodb://{quote_plus('test_user')}:{quote_plus('test_pass')}@127.0.0.1:27017/",
            ),
            (
                    "db.example.com",
                    27017,
                    "user@example",
                    "pass/word",
                    "replicaSet=rs0&authSource=admin",
                    f"mongodb://{quote_plus('user@example')}:{quote_plus('pass/word')}@db.example.com:27017/?replicaSet=rs0&authSource=admin",
            ),
            (
                    "localhost",
                    27017,
                    None,
                    None,
                    "readPreference=secondary",
                    "mongodb://localhost:27017/?readPreference=secondary",
            ),
        ],
    )
    def test_build_mongo_uri(
        self, host, port, username, password, params, expected_uri
    ):
        uri = build_mongo_uri(host, port, username, password, params)
        assert uri == expected_uri

    @pytest.mark.parametrize(
        "original_uri, expected_redacted_uri, expect_log_error",
        [
            (
                    "mongodb://user:password@host:port/db?options",
                    "mongodb://user:********@host:port/db?options",
                    False,
            ),
            (
                    "mongodb+srv://user:password@cluster.mongodb.net/db?options",
                    "mongodb+srv://user:********@cluster.mongodb.net/db?options",
                    False,
            ),
            (
                    "mongodb://user@host:port/db",
                    "mongodb://user@host:port/db",
                    False,
            ),
            (
                    "mongodb://host:port/db",
                    "mongodb://host:port/db",
                    False,
            ),
            (

                    "mongodb://user:p@ssw%20rd@host/db",

                    "mongodb://user:********@ssw%20rd@host/db",
                    False,
            ),
            (
                    "mongodb://:password@host/db",
                    "mongodb://:********@host/db",
                    False,
            ),
            (
                    "mongodb://user:@host/db",
                    "mongodb://user:@host/db",
                    False,
            ),
            (

                    "mongodb://你好:世界@host:port/db",
                    "mongodb://你好:********@host:port/db",
                    False,
            ),
            (
                    "invaliduri_that_does_not_break_urlparse",
                    "invaliduri_that_does_not_break_urlparse",
                    False,
            ),
            (
                    "",
                    "",
                    False,
            ),
            (
                    None,
                    None,
                    False,
            ),
        ],
    )
    def test_redact_uri_password(self, original_uri, expected_redacted_uri, expect_log_error,
                                 caplog):
        redacted_uri = redact_uri_password(original_uri)
        assert redacted_uri == expected_redacted_uri

        if expect_log_error:
            assert "Error redacting URI password" in caplog.text
        else:
            if original_uri is not None:
                assert "Error redacting URI password" not in caplog.text
