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

Mongo Analyser is a tool for better understanding the structure of MongoDB collections.
It helps users infer the schema of a collection by analyzing a sample of documents, identifying field types, and
providing insights into the data distribution, etc.
It provides a text user interface (TUI) with built-in AI assistance to help users get a better understanding of their
data.

### Features

- Automatically infers the schema of MongoDB collections
- Identifies the types of fields in the collection
- Provides insights into the distribution of data
- Provides a user-friendly TUI with integrated AI assistance
- Works with MongoDB Atlas and self-hosted MongoDB instances
- Works with models from Ollama, OpenAI, and Google

---

### Installation

```bash
pipx install mongo-analyser
```

Or

```bash
uv tool install mongo-analyser
```

---

<details>
<summary><strong>Documentation</strong></summary>

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
