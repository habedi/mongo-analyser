import uuid
from datetime import datetime

import pytz
from bson.binary import Binary

from mongo_analyser import DataExtractor


def test_infer_type_with_various_types():
    # Test for different data types
    assert DataExtractor.infer_type(True) == "bool"
    assert DataExtractor.infer_type(10) == "int"
    assert DataExtractor.infer_type(10.5) == "float"
    assert DataExtractor.infer_type("string") == "str"
    assert DataExtractor.infer_type(None) == "null"
    assert DataExtractor.infer_type({"key": "value"}) == "dict"

    # Test for a UUID stored in Binary
    binary_uuid = Binary(uuid.uuid4().bytes, 4)
    assert DataExtractor.infer_type(binary_uuid) == "binary<UUID>"

    # Test for a list with uniform types
    assert DataExtractor.infer_type([1, 2, 3]) == "array<int>"

    # Test for a list with mixed types
    assert DataExtractor.infer_type([1, "string", 3.14]) == "array<mixed>"


def test_infer_types_from_array():
    # Arrange
    array_of_dicts = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": "unknown"}]

    # Act
    field_types = DataExtractor.infer_types_from_array(array_of_dicts)

    # Assert
    assert field_types == {"name": "str", "age": "mixed"}


def test_convert_to_json_compatible():
    # Arrange
    timezone = pytz.timezone("UTC")

    document = {
        "uuid_field": Binary(uuid.uuid4().bytes, 4),
        "int_field": 123,
        "date_field": datetime(2024, 1, 1, tzinfo=timezone),
        "str_field": "test"
    }

    schema = {
        "uuid_field": {"type": "binary<UUID>"},
        "int_field": {"type": "int32"},
        "date_field": {"type": "datetime"},
        "str_field": {"type": "str"}
    }

    # Act
    result = DataExtractor.convert_to_json_compatible(document, schema, timezone)

    # Assert
    assert isinstance(result["uuid_field"], str)
    assert result["int_field"] == 123
    assert result["date_field"] == "2024-01-01T00:00:00+00:00"
    assert result["str_field"] == "test"
