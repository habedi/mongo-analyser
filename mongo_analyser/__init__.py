import logging
from importlib.metadata import PackageNotFoundError, version

_logger = logging.getLogger(__name__)

try:
    __version__ = version("mongo_analyser")
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"
    _logger.warning(
        "Could not determine package version using importlib.metadata."
        " Is the library installed correctly?"
    )
