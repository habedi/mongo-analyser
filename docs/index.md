# Mongo Analyser Documentation

Mongo Analyser consists of a command-line tool and a Python library that helps you analyse the structure of a MongoDB
collection and extract data from it.

## Command-Line Interface

Mongo Analyser can be used as a command-line tool. The general interface for the command-line tool is:

```bash
Usage: mongo_analyser <command> [<args>]

Commands:
  analyse_schema  Analyse the structure of a MongoDB collection and infer schema and statistics from a sample of documents
  extract_data    Extract data from MongoDB and store it to a compressed JSON file
```

Run the following command to get help on a specific command:

```bash
mongo_analyser <command> --help # or -h
```

## Python Interface

See the [examples](examples/) directory for example code snippets on how to use Mongo Analyser as a Python library.

## Supported Field Types

Mongo Analyser supports the following field types:

| Field Type         | Python Equivalent | MongoDB Equivalent   | Comments                                                                                                                                      |
|--------------------|-------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `int32`            | `int`             | `int32`              |                                                                                                                                               |
| `int64`            | `int`             | `int64`              |                                                                                                                                               |
| `double`           | `float`           | `double`             |                                                                                                                                               |
| `str`              | `str`             | `string`             |                                                                                                                                               |
| `bool`             | `bool`            | `bool`               |                                                                                                                                               |
| `datetime`         | `datetime`        | `date`               |                                                                                                                                               |
| `dict`             | `dict`            | `document`           | Equivalent to a BSON document (which is a MongoDB object or subdocument)                                                                      |
| `empty`            | `None` or `[]`    | `null` or `array`    | The empty type is used when a field has no value (`null`) or is an empty array.                                                               |
| `array<type>`      | `list`            | `array`              | The type of the elements in the array is inferred from the sample of documents and can be any of the supported types except for `array<type>` |
| `binary<UUID>`     | `bytes`           | `binary (subtype 4)` | The UUID is stored as a 16-byte binary value                                                                                                  |
| `binary<MD5>`      | `bytes`           | `binary (subtype 5)` | The MD5 hash is stored as a 16-byte binary value                                                                                              |
| `binary<ObjectId>` | `bytes`           | `objectId`           | The ObjectId is stored as a 12-byte binary value                                                                                              |
                                                                                             
## Notes

- At the moment, Mongo Analyser does not support arrays of objects with different types. Such arrays will be treated as
  arrays of objects with a single type. For example, if an array contains both integers and strings, it will be treated
  as either an array of integers or an array of strings.
