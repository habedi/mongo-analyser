# Mongo Analyser

<img src="assets/logo-v1.png" align="right" width="25%"/>

[![Tests](https://github.com/habedi/mongo-analyser/actions/workflows/tests.yml/badge.svg)](https://github.com/habedi/mongo-analyser/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/habedi/mongo-analyser/graph/badge.svg?token=HOTAZKP3V7)](https://codecov.io/gh/habedi/mongo-analyser)
[![CodeFactor](https://www.codefactor.io/repository/github/habedi/mongo-analyser/badge)](https://www.codefactor.io/repository/github/habedi/mongo-analyser)
[![PyPI version](https://badge.fury.io/py/mongo-analyser.svg)](https://badge.fury.io/py/mongo-analyser)
[![Pip downloads](https://img.shields.io/pypi/dm/mongo-analyser.svg)](https://pypi.org/project/mongo-analyser)
[![Python version](https://img.shields.io/badge/Python-%3E=3.9-blue)](https://github.com/habedi/mongo-analyser)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://github.com/habedi/mongo-analyser/blob/main/docs/index.md)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/habedi/mongo-analyser/blob/main/LICENSE)

Mongo Analyser is a tool that helps you analyse and infer a MongoDB collection's structure. It can help you extract the
schema of a collection, find the data types of the fields and also export data from the collection based on the
schema it found.

Mongo Analyser can be used as a command-line tool or as a Python library.

## Installation

You can install Mongo Analyser using `pip` (mainly to use it as a library):

```bash
pip install mongo-analyser
```

You can also install it as a standalone executable using `pipx` or `uv`:

```bash
pipx install mongo-analyser
```

```bash
uv tool install mongo-analyser
```

After the installation is complete, you can use the `mongo-analyser` command in your terminal.

See the [documentation](https://github.com/habedi/mongo-analyser/blob/main/docs/index.md) for more information and
examples.

## Demo

[![asciicast](https://asciinema.org/a/682346.svg)](https://asciinema.org/a/682346)
