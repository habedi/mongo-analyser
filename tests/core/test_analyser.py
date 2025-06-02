import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from bson import Binary, Decimal128, Int64, ObjectId
from pymongo.errors import ConnectionFailure as PyMongoConnectionFailure
from pymongo.errors import OperationFailure as PyMongoOperationFailure

from mongo_analyser.core.analyser import SchemaAnalyser


# from decimal import Decimal as PyDecimal # Not used directly in this file after corrections


# from mongo_analyser.core import shared # Not used directly in this file after corrections


@pytest.fixture
def mock_collection():
    collection_mock = MagicMock()
    collection_mock.database.name = "test_db"
    collection_mock.name = "test_collection"
    return collection_mock


@pytest.fixture
def mock_db_manager():
    with patch("mongo_analyser.core.analyser.db_manager") as mock_db_man:
        yield mock_db_man


class TestSchemaAnalyser:

    @pytest.mark.parametrize(
        "item, expected_hashable_item",
        [
            (1, 1),
            ("abc", "abc"),
            (True, True),
            (None, None),
            ((1, 2), (1, 2)),
            ([1, 2, 3], (1, 2, 3)),
            ({"a": 1, "b": 2}, frozenset({("a", 1), ("b", 2)})),
            (
                    {"c": [1, {"d": 3}], "a": "val"},
                    frozenset(
                        {("c", (1, frozenset({("d", 3)}))), ("a", "val")}
                    ),
            ),
            ([1, [2, 3], {"x": 4}], (1, (2, 3), frozenset({("x", 4)}))),
        ],
    )
    def test_make_hashable(self, item, expected_hashable_item):
        # Arrange (parameters)
        # Act
        result = SchemaAnalyser._make_hashable(item)
        # Assert
        assert result == expected_hashable_item

    def test_extract_schema_and_stats_simple_doc(self):
        # Arrange
        doc = {
            "name": "test",
            "age": 30,
            "verified": True,
            "rating": 4.5,
            "created_at": datetime(2023, 1, 1, 12, 0, 0),
            "_id": ObjectId("60c72b9f9b1d8b3b8f8b4567"),
            "long_num": Int64(1234567890123),
            "decimal_num": Decimal128("10.50"),
            "binary_data": Binary(b"\x01\x02\x03", 0),
            "uuid_val": uuid.UUID("123e4567-e89b-12d3-a456-426614174000"),
        }
        expected_schema = {
            "name": {"type": "str"},
            "age": {"type": "int32"},
            "verified": {"type": "bool"},
            "rating": {"type": "double"},
            "created_at": {"type": "datetime"},
            "_id": {"type": "binary<ObjectId>"},
            "long_num": {"type": "int64"},
            "decimal_num": {"type": "decimal128"},
            "binary_data": {"type": "binary<generic>"},
            "uuid_val": {"type": "binary<UUID>"},
        }
        # Act
        schema, stats = SchemaAnalyser.extract_schema_and_stats(doc.copy())
        # Assert
        assert schema == expected_schema
        assert stats["name"]["count"] == 1
        assert stats["name"]["type_counts"]["str"] == 1
        assert stats["age"]["numeric_min"] == 30
        assert stats["age"]["numeric_max"] == 30
        assert stats["created_at"]["date_min"] == datetime(2023, 1, 1, 12, 0, 0)
        assert stats["decimal_num"]["numeric_min"] == 10.50
        assert stats["_id"]["type_counts"]["binary<ObjectId>"] == 1
        assert stats["uuid_val"]["type_counts"]["binary<UUID>"] == 1
        assert stats["binary_data"]["type_counts"]["binary<generic>"] == 1

    def test_extract_schema_and_stats_nested_doc(self):
        # Arrange
        doc = {"user": {"name": "tester", "details": {"age": 25}}}
        # Act
        schema, stats = SchemaAnalyser.extract_schema_and_stats(doc)
        # Assert
        assert "user.name" in stats
        assert "user.details.age" in stats
        assert stats["user.name"]["type_counts"]["str"] == 1
        assert stats["user.details.age"]["type_counts"]["int32"] == 1
        assert stats["user.details.age"]["numeric_min"] == 25

    def test_extract_schema_and_stats_array_handling(self):
        # Arrange
        doc_empty_array = {"tags": []}
        doc_simple_array = {"scores": [10, 20, 10]}  # 10 appears twice
        doc_mixed_array = {"mixed": [1, "a", True]}
        doc_dict_array = {"items": [{"id": 1}, {"id": 2}]}

        # Act
        schema_empty, stats_empty = SchemaAnalyser.extract_schema_and_stats(doc_empty_array)
        schema_simple, stats_simple = SchemaAnalyser.extract_schema_and_stats(doc_simple_array)
        schema_mixed, stats_mixed = SchemaAnalyser.extract_schema_and_stats(doc_mixed_array)
        schema_dict, stats_dict = SchemaAnalyser.extract_schema_and_stats(doc_dict_array)

        # Assert
        assert schema_empty["tags"]["type"] == "array<empty>"
        assert stats_empty["tags"]["count"] == 1
        assert not stats_empty["tags"]["array_element_stats"]["type_counts"]

        assert schema_simple["scores"]["type"] == "array<int32>"
        assert stats_simple["scores"]["count"] == 1
        assert stats_simple["scores"]["array_element_stats"]["type_counts"]["int32"] == 3
        assert stats_simple["scores"]["array_element_stats"]["numeric_min"] == 10
        assert stats_simple["scores"]["array_element_stats"]["numeric_max"] == 20
        # Corrected: Current code only collects value_frequencies for strings < 256 chars
        assert stats_simple["scores"]["array_element_stats"]["value_frequencies"].get(10, 0) == 0
        assert stats_simple["scores"]["array_element_stats"]["value_frequencies"].get(20, 0) == 0

        assert schema_mixed["mixed"]["type"] == "array<mixed>"
        assert stats_mixed["mixed"]["array_element_stats"]["type_counts"]["int32"] == 1
        assert stats_mixed["mixed"]["array_element_stats"]["type_counts"]["str"] == 1
        assert stats_mixed["mixed"]["array_element_stats"]["type_counts"]["bool"] == 1

        assert schema_dict["items"]["type"] == "array<dict>"

    def test_get_collection_success(self, mock_db_manager):
        # Arrange
        mock_db_manager.db_connection_active.return_value = True
        mock_db_obj = MagicMock()
        mock_collection_obj = MagicMock()
        mock_db_obj.__getitem__.return_value = mock_collection_obj
        mock_db_manager.get_mongo_db.return_value = mock_db_obj
        uri = "mongodb://localhost"
        db_name = "testdb"
        collection_name = "items"

        # Act
        collection = SchemaAnalyser.get_collection(uri, db_name, collection_name)

        # Assert
        mock_db_manager.db_connection_active.assert_called_once_with(
            uri=uri, db_name=db_name, server_timeout_ms=5000
        )
        mock_db_manager.get_mongo_db.assert_called_once()
        mock_db_obj.__getitem__.assert_called_once_with(collection_name)
        assert collection == mock_collection_obj

    def test_get_collection_failure(self, mock_db_manager):
        # Arrange
        mock_db_manager.db_connection_active.return_value = False
        uri = "mongodb://localhost"
        db_name = "testdb"
        collection_name = "items"

        # Act & Assert
        with pytest.raises(PyMongoConnectionFailure):
            SchemaAnalyser.get_collection(uri, db_name, collection_name)

        mock_db_manager.db_connection_active.assert_called_once_with(
            uri=uri, db_name=db_name, server_timeout_ms=5000
        )

    def test_list_collection_names_success(self, mock_db_manager):
        # Arrange
        mock_db_manager.db_connection_active.return_value = True
        mock_db_obj = MagicMock()
        mock_db_obj.list_collection_names.return_value = ["c", "a", "b"]
        mock_db_manager.get_mongo_db.return_value = mock_db_obj
        uri = "mongodb://localhost"
        db_name = "testdb"

        # Act
        names = SchemaAnalyser.list_collection_names(uri, db_name)

        # Assert
        assert names == ["a", "b", "c"]
        mock_db_manager.db_connection_active.assert_called_once()
        mock_db_manager.get_mongo_db.assert_called_once()
        mock_db_obj.list_collection_names.assert_called_once()

    def test_list_collection_names_op_failure(self, mock_db_manager):
        # Arrange
        mock_db_manager.db_connection_active.return_value = True
        mock_db_obj = MagicMock()
        mock_db_obj.list_collection_names.side_effect = PyMongoOperationFailure("failed")
        mock_db_manager.get_mongo_db.return_value = mock_db_obj
        uri = "mongodb://localhost"
        db_name = "testdb"

        # Act & Assert
        with pytest.raises(PyMongoOperationFailure):
            SchemaAnalyser.list_collection_names(uri, db_name)

    def test_infer_schema_and_field_stats_sample_size(self, mock_collection):
        # Arrange
        docs_to_return = [
            {"_id": ObjectId(), "name": "A", "value": 10, "tags": ["x"]},
            {"_id": ObjectId(), "name": "B", "value": 20, "info": {"valid": True}},
            {"_id": ObjectId(), "name": "A", "value": 15, "tags": ["x", "y"]},
        ]

        # Corrected mock for aggregate().batch_size()
        mock_aggregate_result = MagicMock()
        mock_aggregate_result.batch_size.return_value = iter(docs_to_return)
        mock_collection.aggregate.return_value = mock_aggregate_result
        sample_size = 3

        # Act
        schema, stats = SchemaAnalyser.infer_schema_and_field_stats(
            mock_collection, sample_size, batch_size=1000  # Match default or pass explicitly
        )

        # Assert
        mock_collection.aggregate.assert_called_once_with(
            [{"$sample": {"size": sample_size}}]
        )
        mock_aggregate_result.batch_size.assert_called_once_with(1000)  # Check batch_size call

        assert "name" in schema
        assert schema["name"]["type"] == "str"
        assert "value" in schema
        assert schema["value"]["type"] == "int32"

        assert "name" in stats
        assert stats["name"]["cardinality"] == 2
        assert stats["name"]["missing_percentage"] == 0.0
        assert stats["name"]["type_distribution"] == {"str": 3}
        assert stats["name"]["top_values"] == {"A": 2, "B": 1}

        assert "value" in stats
        assert stats["value"]["numeric_min"] == 10
        stats_value_max = stats["value"].get("numeric_max", float("-inf"))
        assert stats_value_max == 20

        assert "tags" in stats
        assert stats["tags"]["missing_percentage"] == (1 / 3 * 100)
        assert stats["tags"]["array_elements"]["type_distribution"] == {"str": 3}

        assert "info.valid" in stats
        assert stats["info.valid"]["type_distribution"] == {"bool": 1}

    def test_infer_schema_and_field_stats_full_scan(self, mock_collection):
        # Arrange
        docs_to_return = [{"data": "full"}]
        mock_find_result = MagicMock()
        mock_find_result.batch_size.return_value = iter(docs_to_return)
        mock_collection.find.return_value = mock_find_result
        sample_size = -1

        # Act
        schema, stats = SchemaAnalyser.infer_schema_and_field_stats(
            mock_collection, sample_size, batch_size=1000  # Match default or pass explicitly
        )

        # Assert
        mock_collection.find.assert_called_once_with()
        mock_find_result.batch_size.assert_called_with(1000)
        assert "data" in schema

    def test_infer_schema_and_field_stats_op_failure(self, mock_collection):
        # Arrange
        mock_collection.aggregate.side_effect = PyMongoOperationFailure("sampling failed")
        sample_size = 10

        # Act & Assert
        with pytest.raises(PyMongoOperationFailure):
            SchemaAnalyser.infer_schema_and_field_stats(mock_collection, sample_size)

    @pytest.mark.parametrize(
        "flat_schema, hierarchical_schema",
        [
            ({}, {}),
            (
                    {"name": {"type": "str"}, "age": {"type": "int"}},
                    {"name": {"type": "str"}, "age": {"type": "int"}},
            ),
            (
                    {
                        "user.name": {"type": "str"},
                        "user.address.city": {"type": "str"},
                        "user.address.zip": {"type": "int"},
                        "items.0.id": {"type": "ObjectId"},
                    },
                    {
                        "user": {
                            "name": {"type": "str"},
                            "address": {
                                "city": {"type": "str"},
                                "zip": {"type": "int"},
                            },
                        },
                        "items": {
                            "0": {
                                "id": {"type": "ObjectId"}
                            }
                        }
                    },
            ),
        ],
    )
    def test_schema_to_hierarchical(self, flat_schema, hierarchical_schema):
        # Arrange (parameters)
        # Act
        result = SchemaAnalyser.schema_to_hierarchical(flat_schema)
        # Assert
        assert result == hierarchical_schema
