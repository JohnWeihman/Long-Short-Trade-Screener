"""Shared test helpers. Tests never touch the network."""

from __future__ import annotations

import json

import pytest

from screener.data.cache import Cache
from screener.data.http import NotFoundError


class FakeFetcher:
    """Stands in for SecClient: serves canned bodies and records calls."""

    def __init__(self, responses: dict[str, bytes | dict]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self.responses:
            raise NotFoundError(f"404 Not Found: {url}")
        body = self.responses[url]
        return body if isinstance(body, bytes) else json.dumps(body).encode()


@pytest.fixture
def cache(tmp_path) -> Cache:
    return Cache(tmp_path / "data")
