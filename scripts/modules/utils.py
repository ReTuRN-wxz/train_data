"""
Utility helpers: filename sanitization, schema validation, retry decorator.
"""
import functools
import logging
import random
import time
import requests
import re
import time
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r'\s+')
_TRAILING = re.compile(r'[.\s]+$')


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Return *name* with illegal filesystem characters removed/replaced.

    Rules applied (in order):
    1. Replace illegal chars with underscores.
    2. Collapse consecutive whitespace to a single space.
    3. Strip leading/trailing dots and spaces (Windows dislikes them).
    4. Truncate to *max_length* characters.
    5. Fall back to 'unnamed_paper' when the result would be empty.
    """
    name = _ILLEGAL_CHARS.sub("_", name)
    name = _WHITESPACE.sub(" ", name)
    name = _TRAILING.sub("", name).strip()
    name = name[:max_length]
    return name or "unnamed_paper"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

PARSED_JSON_SCHEMA: Dict[str, type] = {
    "file_name": str,
    "title": str,
    "full_text": str,
}

SUMMARY_JSON_SCHEMA: Dict[str, type] = {
    "concept_layer": str,
    "detail_layer": str,
    "application_layer": str,
}


def validate_schema(data: Dict[str, Any], schema: Dict[str, type]) -> List[str]:
    """Return a list of validation error messages (empty list = valid)."""
    errors: List[str] = []
    for field, expected_type in schema.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
        elif not isinstance(data[field], expected_type):
            errors.append(
                f"Field '{field}' expected {expected_type.__name__}, "
                f"got {type(data[field]).__name__}"
            )
    return errors


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def retry(
    max_attempts: int = 5,
    base_wait: float = 2.0,
    max_wait: float = 60.0,
    retry_on=(requests.exceptions.RequestException,),
):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    if attempt >= max_attempts:
                        raise

                    wait = None
                    status = None
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        status = resp.status_code

                    # 429: 优先使用 Retry-After
                    if status == 429 and resp is not None:
                        ra = resp.headers.get("Retry-After")
                        if ra:
                            try:
                                wait = float(ra)
                            except ValueError:
                                wait = None

                    # fallback: 指数退避 + 抖动
                    if wait is None:
                        wait = min(max_wait, base_wait * (2 ** (attempt - 1)))
                        wait += random.uniform(0, 1.5)

                    logger.warning(
                        "Attempt %s/%s for '%s' failed (%s). Retrying in %.1fs ...",
                        attempt,
                        max_attempts,
                        func.__name__,
                        f"HTTP {status}" if status else repr(e),
                        wait,
                    )
                    time.sleep(wait)

        return wrapper
    return deco