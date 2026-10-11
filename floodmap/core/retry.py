"""Bounded retry with exponential backoff for transient network errors.

Used by data-acquisition code (DEM download now; other downloaders can adopt it).
Only errors that a retry can plausibly fix are retried: dropped/reset connections,
timeouts and HTTP 429/5xx. Everything else fails immediately.
"""
from __future__ import annotations

import time
from typing import Any, Callable

TRANSIENT_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

# Indirection so tests can replace waiting.
sleep: Callable[[float], None] = time.sleep


def is_transient(exc: BaseException) -> bool:
    retry_types: tuple = (ConnectionError, TimeoutError)          # incl. ConnectionResetError (WinError 10054)
    try:
        import requests
        retry_types += (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.Timeout)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(exc, retry_types):
        return True
    code = getattr(exc, "http_status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code in TRANSIENT_HTTP_CODES


def with_retries(fn: Callable[[], Any], what: str, attempts: int = 4, base_delay: float = 10.0,
                 log: Callable[[str], None] = print) -> Any:
    """Call fn(); on a transient error wait base_delay * 2**k and retry; at most `attempts` calls."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if not is_transient(e) or k == attempts - 1:
                raise
            delay = base_delay * (2 ** k)
            log(f"  {what}: {type(e).__name__}: {e} -- retry {k + 1}/{attempts - 1} in {delay:.0f} s")
            sleep(delay)
    raise AssertionError("unreachable")
