# Mongo Analyser

<img src="https://github.com/habedi/mongo-analyser/blob/main/assets/logo_v1.png" align="right" width="25%"/>

[![Tests](https://github.com/habedi/mongo-analyser/actions/workflows/tests.yml/badge.svg)](https://github.com/habedi/mongo-analyser/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/habedi/mongo-analyser/graph/badge.svg?token=HOTAZKP3V7)](https://codecov.io/gh/habedi/mongo-analyser)
[![PyPI version](https://badge.fury.io/py/mongo-analyser.svg)](https://badge.fury.io/py/mongo-analyser)
[![License](https://img.shields.io/github/license/habedi/mongo-analyser)](https://github.com/habedi/mongo-analyser/blob/main/LICENSE)
[![Python version](https://img.shields.io/badge/Python-%3E=3.9-blue)](https://github.com/habedi/mongo-analyser)
[![Pip downloads](https://img.shields.io/pypi/dm/mongo-analyser.svg)](https://pypi.org/project/mongo-analyser)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://github.com/habedi/mongo-analyser/blob/main/docs/index.md)
[![CodeFactor](https://www.codefactor.io/repository/github/habedi/mongo-analyser/badge)](https://www.codefactor.io/repository/github/habedi/mongo-analyser)

Mongo Analyser is a tool that helps you to analyse the structure of a MongoDB collection. It can help you extract the
schema of a collection, find the data types of the fields, and also extract sample data from the collection based on the
schema.

Mongo Analyser can be used as a command-line tool or as a Python library.

## Installation

You can install Mongo Analyser using pip (mainly to use it as a library):

```bash
pip install mongo-analyser
```

You can also install it as a standalone executable using pipx:

```bash
pipx install mongo-analyser
```

After installing it using pipx, you can run it from the command line:

```bash
mongo-analyser <command> [<args>]
```

See the [documentation](https://github.com/habedi/mongo-analyser/blob/main/docs/index.md) for more information and
examples.
