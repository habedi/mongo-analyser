import uuid
from datetime import datetime, timezone
from decimal import Decimal as PyDecimal
from unittest.mock import MagicMock, mock_open, patch

import pytest
import pytz
from bson import Binary, Decimal128, Int64, ObjectId
from textual.worker import WorkerCancelled

from mongo_analyser.core.extractor import DataExtractor, _BSON_UUID_SUBTYPE_STANDARD, \
    _BSON_UUID_SUBTYPE_LEGACY_PYTHON


@pytest.fixture
def mock_db_manager_extractor():
    with patch("mongo_analyser.core.extractor.db_manager") as mock_db_man:
        yield mock_db_man


@pytest.fixture
def mock_worker():
    worker = MagicMock()
    worker.is_cancelled = False
    return worker


class TestDataExtractor:

    @pytest.mark.parametrize(
        "value, expected_type_str",
        [
            (True, "bool"),
            (Int64(123), "int64"),
            (123, "int32"),
            (123.45, "double"),
            ("test", "str"),
            (datetime.now(), "datetime"),
            (ObjectId(), "ObjectId"),
            (uuid.uuid4(), "UUID"),
            (Binary(b"\x01", _BSON_UUID_SUBTYPE_LEGACY_PYTHON), "binary<UUID (legacy)>"),
            (Binary(b"\x01" * 16, _BSON_UUID_SUBTYPE_STANDARD), "binary<UUID>"),
            (Binary(b"\x01", 0), "binary<generic>"),
            ([], "array<empty>"),
            ([1, 2], "array<int32>"),
            ([1, "a"], "array<mixed>"),
            ([None, None], "array<null>"),
            ([1, None], "array<int32>"),
            ({}, "dict"),
            (Decimal128("1.2"), "decimal128"),
            (None, "null"),
            (PyDecimal("1.23"), "unknown<Decimal>"),
        ],
    )
    def test_infer_type_val(self, value, expected_type_str):
        result = DataExtractor._infer_type_val(value)

        assert result == expected_type_str

    @pytest.mark.parametrize(
        "val, schema_type_str, tz_obj, expected_converted_val",
        [
            (None, "any", None, None),
            (uuid.UUID("123e4567-e89b-12d3-a456-426614174000"), "UUID", None,
             "123e4567-e89b-12d3-a456-426614174000"),
            (Binary(uuid.UUID("123e4567-e89b-12d3-a456-426614174000").bytes,
                    _BSON_UUID_SUBTYPE_STANDARD), "binary<UUID>", None,
             "123e4567-e89b-12d3-a456-426614174000"),
            (Binary(b'\x12\x3e\x45\x67\xe8\x9b\x12\xd3\xa4\x56\x42\x66\x14\x17\x40\x00',
                    _BSON_UUID_SUBTYPE_LEGACY_PYTHON), "binary<UUID (legacy)>", None,
             "123e4567-e89b-12d3-a456-426614174000"),
            (ObjectId("60c72b9f9b1d8b3b8f8b4567"), "ObjectId", None, "60c72b9f9b1d8b3b8f8b4567"),
            (datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc), "datetime", None,
             "2023-01-01T12:00:00+00:00"),
            (datetime(2023, 1, 1, 12, 0, 0), "datetime", pytz.timezone("Europe/Oslo"),
             "2023-01-01T13:00:00+01:00"),
            ("test", "str", None, "test"),
            (123, "int32", None, 123),
            (Int64(456), "int64", None, 456),
            (True, "bool", None, True),
            (12.3, "double", None, 12.3),
            (Decimal128("50.25"), "decimal128", None, "50.25"),
            (Binary(b"\xDE\xAD\xBE\xEF", 0), "binary<generic>", None, "deadbeef"),
            ([1, 2], "array<int32>", None, [1, 2]),
            ({"a": 1}, "dict", None, {"a": 1}),

        ],
    )
    def test_convert_single_value(self, val, schema_type_str, tz_obj, expected_converted_val):
        result = DataExtractor._convert_single_value(val, schema_type_str, tz_obj)

        assert result == expected_converted_val

    def test_convert_to_json_compatible_simple(self):
        doc = {"_id": ObjectId("60c72b9f9b1d8b3b8f8b4567"), "name": "Test"}
        schema = {"_id": {"type": "ObjectId"}, "name": {"type": "str"}}

        converted = DataExtractor.convert_to_json_compatible(doc, schema, None)

        assert converted["_id"] == "60c72b9f9b1d8b3b8f8b4567"
        assert converted["name"] == "Test"

    def test_convert_to_json_compatible_nested_and_array(self):
        doc = {
            "event_time": datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            "details": {"values": [Decimal128("10.1"), Decimal128("20.2")]},
            "logs": [{"ts": datetime(2024, 6, 1, 10, 0, 5), "msg": "hello"}],
            "empty_list": []
        }
        schema = {
            "event_time": {"type": "datetime"},
            "details": {
                "values": {"type": "array<decimal128>"}
            },
            "logs": {
                "type": "array<dict>",
                "items": {
                    "ts": {"type": "datetime"},
                    "msg": {"type": "str"}
                }
            },
            "empty_list": {"type": "array<empty>"}
        }
        oslo_tz = pytz.timezone("Europe/Oslo")

        converted = DataExtractor.convert_to_json_compatible(doc, schema, oslo_tz)

        assert converted["event_time"] == "2024-06-01T12:00:00+02:00"
        assert converted["details"]["values"] == ["10.1", "20.2"]
        assert len(converted["logs"]) == 1
        assert converted["logs"][0][
                   "ts"] == "2024-06-01T12:00:05+02:00"
        assert converted["logs"][0]["msg"] == "hello"
        assert converted["empty_list"] == []

    @patch("mongo_analyser.core.extractor.gzip.open", new_callable=mock_open)
    @patch("mongo_analyser.core.extractor.get_current_worker")
    def test_extract_data_success(
        self, mock_get_worker, mock_gzip_open, mock_db_manager_extractor, tmp_path
    ):
        mock_db_manager_extractor.db_connection_active.return_value = True
        mock_db_obj = MagicMock()
        mock_collection_obj = MagicMock()
        mock_cursor = MagicMock()
        docs_to_return = [
            {"_id": ObjectId(), "a": 1},
            {"_id": ObjectId(), "b": "data"},
        ]
        mock_cursor.__iter__.return_value = iter(docs_to_return)
        mock_collection_obj.find.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.batch_size.return_value = mock_cursor

        mock_db_obj.__getitem__.return_value = mock_collection_obj
        mock_db_manager_extractor.get_mongo_db.return_value = mock_db_obj

        mock_worker_instance = MagicMock()
        mock_worker_instance.is_cancelled = False
        mock_get_worker.return_value = mock_worker_instance

        output_file = tmp_path / "out.json.gz"
        schema = {"a": {"type": "int32"}, "b": {"type": "str"}}

        DataExtractor.extract_data(
            "uri", "db", "coll", schema, output_file, None, 100, 10
        )

        mock_db_manager_extractor.db_connection_active.assert_called_once()
        mock_collection_obj.find.assert_called_once()
        mock_gzip_open.assert_called_once_with(output_file, "wt", encoding="utf-8")

        handle = mock_gzip_open()

        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert written_content.startswith("[\n")
        assert written_content.endswith("\n]\n")
        assert '"a": 1' in written_content or '"b": "data"' in written_content
        assert mock_cursor.close.called

    @patch("mongo_analyser.core.extractor.get_current_worker")
    def test_extract_data_worker_cancelled_early(self, mock_get_worker, mock_db_manager_extractor,
                                                 tmp_path):
        mock_worker_instance = MagicMock()
        mock_worker_instance.is_cancelled = True
        mock_get_worker.return_value = mock_worker_instance
        output_file = tmp_path / "out.json.gz"

        with pytest.raises(WorkerCancelled):
            DataExtractor.extract_data("uri", "db", "coll", {}, output_file, None, 100, 10)

        assert not mock_db_manager_extractor.db_connection_active.called

    def test_get_newest_documents_success(self, mock_db_manager_extractor):
        mock_db_manager_extractor.db_connection_active.return_value = True
        mock_db_obj = MagicMock()
        mock_collection_obj = MagicMock()
        mock_cursor = MagicMock()
        object_id = ObjectId()
        dt_now = datetime.now(timezone.utc)
        my_uuid = uuid.uuid4()
        bin_uuid_std = Binary(my_uuid.bytes, _BSON_UUID_SUBTYPE_STANDARD)
        bin_data = Binary(b'\x01\x02\x03', 0)
        dec128 = Decimal128("123.45")

        docs_to_return = [
            {
                "_id": object_id,
                "ts": dt_now,
                "uid": my_uuid,
                "bin_uid": bin_uuid_std,
                "data": bin_data,
                "amount": dec128,
                "name": "TestDoc",
                "large_field": "x" * 600,
                "unhandled": PyDecimal("1.0")
            }
        ]
        mock_cursor.__iter__.return_value = iter(docs_to_return)
        mock_collection_obj.find.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.sort.return_value = mock_cursor
        mock_db_obj.__getitem__.return_value = mock_collection_obj
        mock_db_manager_extractor.get_mongo_db.return_value = mock_db_obj

        result_docs = DataExtractor.get_newest_documents("uri", "db", "coll", 1)

        assert len(result_docs) == 1
        doc = result_docs[0]
        assert doc["_id"] == str(object_id)
        assert doc["ts"] == dt_now.isoformat()
        assert doc["uid"] == str(my_uuid)
        assert doc["bin_uid"] == str(my_uuid)
        assert doc["data"].startswith("binary_hex:010203")
        assert doc["amount"] == "123.45"
        assert doc["name"] == "TestDoc"
        assert doc["large_field"] == "str(too large to display inline)"
        assert doc["unhandled"].startswith("unhandled_type:Decimal:")
        assert mock_cursor.close.called

    def test_get_newest_documents_sample_size_zero(self, mock_db_manager_extractor):
        mock_db_manager_extractor.db_connection_active.return_value = True

        docs = DataExtractor.get_newest_documents("uri", "db", "coll", 0)

        assert docs == []
