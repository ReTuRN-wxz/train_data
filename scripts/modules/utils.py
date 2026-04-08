"""
Utility helpers: filename sanitization, schema validation, retry decorator.
"""

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
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries *func* up to *max_attempts* times on *exceptions*.

    Args:
        max_attempts: Total number of attempts (including the first call).
        delay: Initial sleep time in seconds between attempts.
        backoff: Multiplier applied to *delay* after each failure.
        exceptions: Tuple of exception types that trigger a retry.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = delay
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.warning(
                            "Attempt %d/%d for '%s' failed: %s. "
                            "Retrying in %.1fs …",
                            attempt,
                            max_attempts,
                            func.__name__,
                            exc,
                            wait,
                        )
                        time.sleep(wait)
                        wait *= backoff
                    else:
                        logger.error(
                            "All %d attempts for '%s' failed: %s",
                            max_attempts,
                            func.__name__,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
