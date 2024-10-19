import argparse

import pytest

from mongo_analyser import BaseAnalyser


def test_str2bool_true():
    # Arrange
    true_values = ['yes', 'true', 't', 'y', '1']

    # Act & Assert
    for value in true_values:
        assert BaseAnalyser.str2bool(value) is True


def test_str2bool_false():
    # Arrange
    false_values = ['no', 'false', 'f', 'n', '0']

    # Act & Assert
    for value in false_values:
        assert BaseAnalyser.str2bool(value) is False


def test_str2bool_invalid():
    # Act & Assert
    with pytest.raises(argparse.ArgumentTypeError):
        BaseAnalyser.str2bool('invalid')


def test_build_mongo_uri_with_credentials():
    # Arrange
    host = "localhost"
    port = 27017
    username = "user"
    password = "pass"

    # Act
    result = BaseAnalyser.build_mongo_uri(host, port, username, password)

    # Assert
    assert result == "mongodb://user:pass@localhost:27017/"


def test_build_mongo_uri_without_credentials():
    # Arrange
    host = "localhost"
    port = 27017

    # Act
    result = BaseAnalyser.build_mongo_uri(host, port)

    # Assert
    assert result == "mongodb://localhost:27017/"


def test_handle_binary_with_known_subtype(mocker):
    # Arrange
    mock_value = mocker.Mock()
    mock_value.subtype = 0  # Known subtype 'binary<generic>'
    schema = {}
    full_key = "binaryField"

    # Act
    BaseAnalyser.handle_binary(mock_value, schema, full_key)

    # Assert
    assert schema == {"binaryField": {"type": "binary<generic>"}}


def test_handle_binary_with_unknown_subtype(mocker):
    """Test for handling binary data with an unknown subtype."""
    # Arrange
    mock_value = mocker.Mock()
    mock_value.subtype = 99  # Unknown subtype
    schema = {}
    full_key = "binaryField"

    # Act
    BaseAnalyser.handle_binary(mock_value, schema, full_key)

    # Assert
    assert schema == {"binaryField": {"type": "binary<subtype 99>"}}


def test_handle_binary_in_array(mocker):
    """Test for handling binary data in an array."""
    # Arrange
    mock_value = mocker.Mock()
    mock_value.subtype = 4  # Known subtype 'binary<UUID>'
    schema = {}
    full_key = "fieldName"

    # Act
    BaseAnalyser.handle_binary(mock_value, schema, full_key, is_array=True)

    # Assert
    assert schema == {"fieldName": {"type": "array<binary<UUID>>"}}
