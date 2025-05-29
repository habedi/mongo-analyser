import logging
from typing import Union
from urllib.parse import quote_plus, urlparse, urlunparse

logger = logging.getLogger(__name__)

binary_type_map = {
    0: "binary<generic>",
    1: "binary<function>",
    3: "binary<UUID (legacy)>",
    4: "binary<UUID>",
    5: "binary<MD5>",
}


def build_mongo_uri(
    host: str,
    port: Union[str, int],
    username: Union[str, None] = None,
    password: Union[str, None] = None,
    params: Union[str, None] = None,
) -> str:
    base_uri = "mongodb://"
    if username and password:
        base_uri += f"{quote_plus(username)}:{quote_plus(password)}@"
    elif username:
        base_uri += f"{quote_plus(username)}@"
    base_uri += f"{host}:{port}/"
    if params:
        base_uri += f"?{params}"
    return base_uri


def redact_uri_password(uri: str) -> str:
    try:
        parsed = urlparse(uri)
        if parsed.password:
            netloc_parts = parsed.netloc.split("@", 1)
            auth_part_present = len(netloc_parts) > 1 and ":" in netloc_parts[0]

            if auth_part_present:
                user_pass_part = netloc_parts[0]
                host_part = netloc_parts[1]
                username = user_pass_part.split(":", 1)[0]
                new_netloc = f"{username}:********@{host_part}"
                parsed = parsed._replace(netloc=new_netloc)
                return urlunparse(parsed)
        return uri
    except Exception as e:
        logger.error(f"Error redacting URI password: {e}")
        return uri
