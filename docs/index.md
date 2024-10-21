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

See the [examples](examples/) directory for examples on how to use the Mongo Analyser as a Python library.

## Notes

- At the moment, Mongo Analyser does not support arrays of objects with different types. Such arrays will be treated as
  arrays of objects with a single type. For example, if an array contains both integers and strings, it will be treated
  as either an array of integers or an array of strings, depending on the majority type.
