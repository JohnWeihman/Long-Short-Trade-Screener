"""The single doorway for all SEC traffic: User-Agent, rate limit, retries."""

from __future__ import annotations

import threading
import time
from typing import Callable

import requests

SEC_MAX_REQUESTS_PER_SECOND = 8.0  # the SEC's limit is 10; stay below it
RETRY_STATUSES = {429, 500, 502, 503, 504}


class SecHttpError(RuntimeError):
    """An SEC request failed for good."""


class NotFoundError(SecHttpError):
    """The SEC has no such resource (HTTP 404)."""


class RateLimiter:
    """Spaces calls at least 1/max_per_second apart, across threads."""

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self.interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next_allowed:
                self._sleep(self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + self.interval


class SecClient:
    """GETs from sec.gov with the required User-Agent, throttled and retried."""

    def __init__(
        self,
        user_agent: str,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise ValueError("SEC requests require a User-Agent")
        self.headers = {
            "User-Agent": user_agent.strip(),
            "Accept-Encoding": "gzip, deflate",
        }
        self._session = session or requests.Session()
        self._limiter = limiter or RateLimiter(SEC_MAX_REQUESTS_PER_SECOND)
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._timeout = timeout
        self._sleep = sleep

    def get(self, url: str) -> bytes:
        """Return the response body, or raise NotFoundError / SecHttpError."""
        last_problem = "no attempt made"
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(self._backoff * 2 ** (attempt - 1))
            self._limiter.wait()
            try:
                response = self._session.get(
                    url, headers=self.headers, timeout=self._timeout
                )
            except requests.RequestException as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
                continue

            status = response.status_code
            if status == 200:
                return response.content
            if status == 404:
                raise NotFoundError(f"404 Not Found: {url}")
            if status == 403:
                raise SecHttpError(
                    f"403 Forbidden: {url}. The SEC rejects requests whose "
                    "User-Agent lacks a name and email, and temporarily blocks "
                    "clients that exceed its rate limit."
                )
            if status in RETRY_STATUSES:
                last_problem = f"HTTP {status}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(min(float(retry_after), 60.0))
                continue
            raise SecHttpError(f"HTTP {status}: {url}")
        raise SecHttpError(
            f"Gave up on {url} after {self._max_retries + 1} attempts ({last_problem})"
        )
