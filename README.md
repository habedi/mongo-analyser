<div align="center">
  <picture>
    <img alt="Mongo Analyser Logo" src="logo.svg" height="25%" width="25%">
  </picture>
<br>

<h2>Mongo Analyser</h2>

[![Tests](https://img.shields.io/github/actions/workflow/status/habedi/mongo-analyser/tests.yml?label=tests&style=flat&labelColor=555555&logo=github)](https://github.com/habedi/mongo-analyser/actions/workflows/tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/habedi/mongo-analyser?style=flat&labelColor=555555&logo=codecov)](https://codecov.io/gh/habedi/mongo-analyser)
[![Code Quality](https://img.shields.io/codefactor/grade/github/habedi/mongo-analyser?style=flat&labelColor=555555&logo=codefactor)](https://www.codefactor.io/repository/github/habedi/mongo-analyser)
[![PyPI](https://img.shields.io/pypi/v/mongo-analyser.svg?style=flat&labelColor=555555&logo=pypi)](https://pypi.org/project/mongo-analyser)
[![Downloads](https://img.shields.io/pypi/dm/mongo-analyser.svg?style=flat&labelColor=555555&logo=pypi)](https://pypi.org/project/mongo-analyser)
[![Python](https://img.shields.io/badge/python-%3E=3.10-3776ab?style=flat&labelColor=555555&logo=python)](https://github.com/habedi/mongo-analyser)
[![License](https://img.shields.io/badge/license-MIT-007ec6?style=flat&labelColor=555555&logo=open-source-initiative)](https://github.com/habedi/mongo-analyser/blob/main/LICENSE)

Analyze and understand data stored in MongoDB from the command line

</div>

---

MongoDB makes it much easier to store data without a predefined schema.
This flexibility allows developers to quickly experiment with different ideas and adapt the data model as
needed, especially during the early stages of development.
However, this advantage can lead to problems over time, like the data can become unorganized, inconsistent, and
difficult
to manage.
Mongo Analyser is a TUI (text user interface) tool designed to help users with this issue by providing a way for users
to understand the structure of their data stored in MongoDB collections.
It allows users to quickly and easily see the core organization of their data, making it easier to
make informed decisions about improving its structure and consistency.

### Features

- Provides a user-friendly TUI with integrated AI assistance
- Works with MongoDB Atlas as well as self-hosted MongoDB instances
- Works with models from Ollama, OpenAI, and Google
- Automatically infers the schema of MongoDB collections
- Identifies the types of fields in the collection

<div align="center">
  <picture>
    <img alt="Chat View" src="docs/screenshots/chat_view_1.png" height="100%" width="100%">
  </picture>
</div>

---

### Installation

You can install Mongo Analyser using `pipx` or `uv`:

```bash
pipx install mongo-analyser
````

```bash
uv tool install mongo-analyser
```

### Usage

Run `mongo_analyser` in the terminal to launch the TUI.

```bash
mongo_analyser
```

The `mongo_analyser` command supports various options to connect to a MongoDB instance:

```bash
# Connect to a local MongoDB instance with a specific database
mongo_analyser --host localhost --port 27017 --db my_database
```

```bash
# Connect using a full MongoDB URI (with password from environment variable)
export MONGO_PASSWORD="your_secure_password"
mongo_analyser --uri "mongodb://user:${MONGO_PASSWORD}@my_host:27017/my_db"
```

```bash
# Or give connection details directly (password will be prompted)
mongo_analyser --host my_host --username my_user --db my_database
```

Run `mongo_analyser --help` to see all available options and their descriptions.

---

<details>
<summary><strong>Documentation</strong></summary>

### Environment Variables

You can set environment variables to configure Mongo Analyser's connection to MongoDB and LLM providers.

* `MONGO_URI`: MongoDB connection URI (e.g., `mongodb://user:password@host:port/database`)
* `MONGO_HOST`: MongoDB host (default: `localhost`)
* `MONGO_PORT`: MongoDB port (default: `27017`)
* `MONGO_USERNAME`: MongoDB username
* `MONGO_DATABASE`: MongoDB database name
* `MONGO_ANALYSER_HOME_DIR`: Directory to store Mongo Analyser data (default: `~/.local/shared/mongo_analyser`)
* `MONGO_ANALYSER_DEBUG=1` (to enable Textual devtools for debugging)

For LLM providers, you can set API keys:

* `OPENAI_API_KEY`: OpenAI API key for accessing OpenAI models
* `GOOGLE_API_KEY`: Google API key for accessing Google models
* `OLLAMA_HOST`: Ollama host URL (default: `http://localhost:11434`)
* `OLLAMA_CONTEXT_LENGTH`: Context length for Ollama models (default: `2048`)

### Supported Field Types

Mongo Analyser supports the following field types when inferring the schema of a MongoDB collection:

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

</details>

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### Logo

The leaf logo is originally from [SVG Repo](https://www.svgrepo.com/svg/258591/clover-leaf).

### License

Mongo Analyser is licensed under the [MIT License](LICENSE).
