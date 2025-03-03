import argparse
from typing import Union


class BaseAnalyser:
    """Shared utility functions for other classes."""

    # Mapping for BSON binary types
    binary_type_map = {
        0: "binary<generic>",
        1: "binary<function>",
        3: "binary<UUID (legacy)>",
        4: "binary<UUID>",
        5: "binary<MD5>",
    }

    @staticmethod
    def str2bool(v: str) -> bool:
        """Helper function to convert string to boolean value for argparse."""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected (like true/false or 1/0).")

    @staticmethod
    def build_mongo_uri(
        host: str,
        port: Union[str, int],
        username: Union[str, None] = None,
        password: Union[str, None] = None,
    ) -> str:
        """Builds a MongoDB URI from host, port, and optional credentials."""
        if username and password:
            return f"mongodb://{username}:{password}@{host}:{port}/"
        return f"mongodb://{host}:{port}/"

    @staticmethod
    def handle_binary(value: dict, schema: dict, full_key: str, is_array: bool = False) -> None:
        """
        Handles the extraction of binary data from a BSON document.
        Note that this is a procedure and cause side effects on the schema dictionary.
        """
        binary_subtype = value.subtype

        subtype_str = BaseAnalyser.binary_type_map.get(
            binary_subtype, f"binary<subtype {binary_subtype}>"
        )
        schema[full_key] = {"type": f"array<{subtype_str}>" if is_array else subtype_str}
