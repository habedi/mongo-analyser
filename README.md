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

Mongo Analyser is a TUI (text user interface) application that helps users get a sense of the structure of their data in
MongoDB.
It allows users to extract the schema, metadata, and sample documents from MongoDB collections, and
chat with an AI assistant to explore and understand their data better.

**Why Mongo Analyser?**

A NoSQL database like MongoDB makes it much easier to store data without a predefined schema.
This flexibility allows developers to quickly experiment with different ideas and adapt the data model as
needed, especially during the early stages of development of a project.
However, this can lead to the data becoming unorganized, inconsistent, and difficult to manage over time.
Mongo Analyser aims to help with this problem by making it easier for users to understand the structure of their MongoDB
collections to prevent data from becoming a *big mess* over time.

### Features

- Provides a user-friendly TUI with integrated AI assistant
- Works with MongoDB Atlas as well as self-hosted MongoDB instances
- Works with a wide range of AI models from Ollama, OpenAI, and Google
- Automatically infers the schema of MongoDB collections
- Extracts metadata and sample documents from collections

> [!Note]
> Mongo Analyser is still in its early stages of development.
> Please report bugs and feature requests on the [GitHub Issues page](https://github.com/habedi/mongo-analyser/issues)
> and [Discussions page](https://github.com/habedi/mongo-analyser/discussions).
> Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md) for more details.

### TUI Screenshots

<div align="center">
  <picture>
    <img alt="Chat View" src="docs/screenshots/chat_view_1.png" height="100%" width="100%">
  </picture>
</div>

<details>
<summary>Show more</summary>

<div align="center">
  <picture>
    <img alt="DB Connect View" src="docs/screenshots/db_connect_view_1.png" height="100%" width="100%">
  </picture>
</div>

<div align="center">
    <picture>
        <img alt="Schema Anlysis View" src="docs/screenshots/schema_analysis_view_1.png" height="100%" width="100%">
    </picture>
</div>

<div align="center">
    <picture>
        <img alt="Data Explorer View" src="docs/screenshots/data_explorer_view_1.png" height="100%" width="100%">
    </picture>
</div>

<div align="center">
    <picture>
        <img alt="Chat View with AI Assistant" src="docs/screenshots/chat_view_2.png" height="100%" width="100%">
    </picture>
</div>

<div align="center">
    <picture>
        <img alt="Config View" src="docs/screenshots/config_view_1.png" height="100%" width="100%">
    </picture>
</div>

</details>

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

<details>
<summary>MongoDB Connection Options</summary>

The `mongo_analyser` command supports various options to connect to a MongoDB instance:

```bash
# Connect to a local MongoDB instance
mongo_analyser --host localhost --port 27017 --db my_database
```

```bash
# Connect using a full MongoDB URI (with password from environment variable)
export MONGO_PASSWORD="your_secure_password"
mongo_analyser --uri "mongodb://user:${MONGO_PASSWORD}@my_host:27017/my_db"
```

```bash
# Connect to a MongoDB Atlas instance with authentication promptted interactively
mongo_analyser --host my_host --username my_user --db my_database
```

Run `mongo_analyser --help` to see all available options and their descriptions.
</details>

---

### Documentation

<details>
<summary>Show</summary>

#### Environment Variables

Mongo Analyser can be configured using environment variables to set up the connection to MongoDB and LLM providers.

##### MongoDB Connection

* `MONGO_URI`: MongoDB connection URI (e.g., `mongodb://user:password@host:port/database`)
* `MONGO_HOST`: MongoDB host (default: `localhost`)
* `MONGO_PORT`: MongoDB port (default: `27017`)
* `MONGO_USERNAME`: MongoDB username
* `MONGO_DATABASE`: MongoDB database name

##### AI Model Providers

* **OpenAI**
    * `OPENAI_API_KEY`: OpenAI API key for accessing OpenAI models
* **Google**
    * `GOOGLE_API_KEY`: Google API key for accessing Google models
* **Ollama**
    * `OLLAMA_HOST`: Ollama host URL (default: `http://localhost:11434`)
    * `OLLAMA_CONTEXT_LENGTH`: Context length for Ollama models (default: `2048`)

##### Misc Options

* `MONGO_ANALYSER_HOME_DIR`: Directory to store Mongo Analyser data (default: `~/.local/shared/mongo_analyser`)

#### Supported Field Types

When inferring the schema of a MongoDB collection, Mongo Analyser supports a variety of field types.
The table below summarizes the supported field types, their Python equivalents, and their MongoDB equivalents.

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
