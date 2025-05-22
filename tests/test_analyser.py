import uuid
from collections import OrderedDict

from bson import ObjectId

from mongo_analyser import SchemaAnalyser  # Adjust the import path as necessary


def test_extract_schema_and_stats_simple_document():
    # Arrange
    document = {"name": "Alice", "age": 30}
    schema, stats = OrderedDict(), OrderedDict()

    # Act
    extracted_schema, extracted_stats = SchemaAnalyser.extract_schema_and_stats(document, schema,
                                                                                stats)

    # Assert
    assert extracted_schema == {"name": {"type": "str"}, "age": {"type": "int32"}}
    assert extracted_stats["name"]["count"] == 1
    assert extracted_stats["age"]["count"] == 1


def test_handle_array_with_primitive_values():
    # Arrange
    value = [1, 2, 3]
    schema, stats = OrderedDict(), OrderedDict()
    full_key = "numbers"

    # Act
    SchemaAnalyser.handle_array(value, schema, stats, full_key)

    # Assert
    assert schema == {"numbers": {"type": "array<int32>"}}
    assert len(stats[full_key]["values"]) == 1


def test_handle_array_with_objectid():
    # Arrange
    value = [ObjectId()]
    schema, stats = OrderedDict(), OrderedDict()
    full_key = "ids"

    # Act
    SchemaAnalyser.handle_array(value, schema, stats, full_key)

    # Assert
    assert schema == {"ids": {"type": "array<ObjectId>"}}
    assert len(stats[full_key]["values"]) == 1


def test_handle_simple_value_with_uuid():
    # Arrange
    value = uuid.uuid4()
    schema, stats = OrderedDict(), OrderedDict()
    full_key = "unique_id"

    # Act
    SchemaAnalyser.handle_simple_value(value, schema, stats, full_key)

    # Assert
    assert schema == {"unique_id": {"type": "binary<UUID>"}}
    assert len(stats[full_key]["values"]) == 1


def test_connect_mongo(mocker):
    # Arrange
    mock_mongo_client = mocker.patch("mongo_analyser.core.analyser.MongoClient")
    mock_db = mock_mongo_client.return_value.__getitem__.return_value
    mock_collection = mock_db.__getitem__.return_value
    uri = "mongodb://localhost:27017"
    db_name = "test_db"
    collection_name = "test_collection"

    # Act
    result = SchemaAnalyser.connect_mongo(uri, db_name, collection_name)

    # Assert
    mock_mongo_client.assert_called_once_with(uri)
    assert result == mock_collection


def test_infer_schema_and_stats_with_sample(mocker):
    # Arrange
    mock_collection = mocker.MagicMock()
    mock_documents = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
    mock_collection.aggregate.return_value.batch_size.return_value = iter(mock_documents)

    # Act
    schema, stats = SchemaAnalyser.infer_schema_and_stats(mock_collection, sample_size=2)

    # Assert
    assert schema == {"name": {"type": "str"}, "age": {"type": "int32"}}
    assert stats["name"]["cardinality"] == 2
    assert stats["age"]["missing_percentage"] == 0.0


def test_schema_to_hierarchical():
    # Arrange
    flat_schema = {
        "name.first": {"type": "str"},
        "name.last": {"type": "str"},
        "address.city": {"type": "str"}
    }

    # Act
    hierarchical_schema = SchemaAnalyser.schema_to_hierarchical(flat_schema)

    # Assert
    assert hierarchical_schema == {
        "name": {
            "first": {"type": "str"},
            "last": {"type": "str"}
        },
        "address": {
            "city": {"type": "str"}
        }
    }


def test_save_schema_to_json(mocker):
    # Arrange
    mock_open = mocker.patch("mongo_analyser.core.analyser.io.open", mocker.mock_open())
    mock_json_dump = mocker.patch("mongo_analyser.core.analyser.json.dump")
    schema = {"name": {"type": "str"}}
    schema_file = "schema.json"

    # Act
    SchemaAnalyser.save_schema_to_json(schema, schema_file)

    # Assert
    mock_open.assert_called_once_with(schema_file, 'w')
    mock_json_dump.assert_called_once()


def test_save_table_to_csv(mocker):
    # Arrange
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mock_csv_writer = mocker.patch("csv.writer")
    headers = ["Field", "Type"]
    rows = [["name", "str"]]
    csv_file = "metadata.csv"

    # Act
    SchemaAnalyser.save_table_to_csv(headers, rows, csv_file)

    # Assert
    mock_open.assert_called_once_with(csv_file, mode='w', newline='')
    mock_csv_writer.return_value.writerow.assert_called_once_with(headers)
    mock_csv_writer.return_value.writerows.assert_called_once_with(rows)


def test_draw_unicode_table(mocker):
    # Arrange
    mock_print = mocker.patch("builtins.print")
    headers = ["Field", "Type"]
    rows = [["name", "str"]]

    # Act
    SchemaAnalyser.draw_unicode_table(headers, rows)

    # Assert
    assert mock_print.call_count > 0  # Ensure the print function is called to output the table
