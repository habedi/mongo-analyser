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
It allows users to extract the schema, metadata, and sample documents from MongoDB collections and
chat with an AI assistant to explore and understand their data better.

**Why Mongo Analyser?**

A NoSQL database like MongoDB makes it much easier to store data without a predefined schema.
This flexibility allows developers to quickly experiment with different ideas and adapt the data model as
needed, especially during the early stages of development of a project.
However, this can lead to the data becoming unorganized, inconsistent, and difficult to manage over time.
Mongo Analyser aims to help with this problem by making it easier for users to understand the structure of their MongoDB
collections to prevent data from becoming a *big mess* over time.

### Features

* User-friendly TUI with built-in AI assistant
* Supports MongoDB Atlas and self-hosted MongoDB instances
* Compatible with AI models from Ollama, OpenAI, and Google
* Automatic schema inference from MongoDB collections
* Data and metadata extraction from collections

> [!Note]
> Mongo Analyser is still in its early stages of development.
> For bugs, feature requests, or discussions,
> please use the [GitHub Issues](https://github.com/habedi/mongo-analyser/issues) and
> [Discussions](https://github.com/habedi/mongo-analyser/discussions) pages.
> Contributions are very welcome!
> See [Contributing Guide](CONTRIBUTING.md) for more details.

### TUI Screenshots

<div align="center">
  <img alt="Chat View" src="docs/screenshots/chat_view_1.png" width="100%">
</div>

<details>
<summary>Show more screenshots</summary>

<div align="center">
  <img alt="DB Connect View" src="docs/screenshots/db_connect_view_1.png" width="100%">
  <img alt="Schema Analysis View" src="docs/screenshots/schema_analysis_view_1.png" width="100%">
  <img alt="Data Explorer View" src="docs/screenshots/data_explorer_view_1.png" width="100%">
  <img alt="Chat View with AI Assistant" src="docs/screenshots/chat_view_2.png" width="100%">
  <img alt="Config View" src="docs/screenshots/config_view_1.png" width="100%">
</div>

</details>

### Installation

Install Mongo Analyser using `pipx` or `uv`:

```bash
pipx install mongo-analyser
```

```bash
uv tool install mongo-analyser
```

### Quick Start

Launch Mongo Analyser by running:

```bash
mongo_analyser
```

<details>
<summary>Show more advanced usages</summary>

You can configure Mongo Analyser with additional options:

**Connect with environment variables**

```bash
export OLLAMA_HOST="http://localhost:11434"
export GOOGLE_API_KEY="your_google_api_key"
export OPENAI_API_KEY="your_openai_api_key"
mongo_analyser --db my_database
```

**Connect via MongoDB URI**

```bash
export MONGO_PASSWORD="your_secure_password"
mongo_analyser --uri "mongodb://user:${MONGO_PASSWORD}@host:27017/db"
```

**Connect by prompting for password**

```bash
mongo_analyser --host my_host --port 27017 --username my_user --db my_database
```

Use `mongo_analyser --help` for a full list of commands.

</details>

---

### Documentation

<details>
<summary>Show</summary>

**MongoDB Connection**

* `MONGO_URI`
* `MONGO_HOST` (default: `localhost`)
* `MONGO_PORT` (default: `27017`)
* `MONGO_USERNAME`
* `MONGO_DATABASE`

**AI Providers**

* `OPENAI_API_KEY`
* `GOOGLE_API_KEY`
* `OLLAMA_HOST` (default: `http://localhost:11434`)
* `OLLAMA_CONTEXT_LENGTH` (default: `2048`)

**Misc**

* `MONGO_ANALYSER_HOME_DIR`: Path to the Mongo Analyser home directory, where configuration and data files are stored.
  This can be set to a custom path if you want to change the default location.
  If not set, it defaults to `~/.local/shared/mongo_analyser`.

**Supported Field Types**

Mongo Analyser can infer the schema of a MongoDB collection and show it as a key-value structure in JSON format.
The keys are the field names, and the values are the types of the fields.
The following table shows the supported field types in Mongo Analyser, their Python equivalents, and their MongoDB
equivalents:

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

Check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

### Logo

Leaf logo courtesy of [SVG Repo](https://www.svgrepo.com/svg/258591/clover-leaf).

### License

Mongo Analyser is available under the [MIT License](LICENSE).
