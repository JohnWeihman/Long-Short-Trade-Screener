import pytest
import requests

from screener.data.http import NotFoundError, RateLimiter, SecClient, SecHttpError


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def get(self, url, headers=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes, **kwargs):
    session = FakeSession(outcomes)
    sleeps = []
    client = SecClient(
        "Test User test@invalid.test",
        session=session,
        limiter=RateLimiter(1000),
        sleep=sleeps.append,
        **kwargs,
    )
    return client, session, sleeps


def test_rate_limiter_spaces_calls_under_ten_per_second():
    clock = FakeClock()
    limiter = RateLimiter(8, clock=clock.time, sleep=clock.sleep)
    start = clock.now
    for _ in range(81):
        limiter.wait()
    # 81 back-to-back calls = 80 gaps of 1/8 s = 10 s, i.e. 8 per second.
    assert clock.now - start == pytest.approx(10.0)
    assert all(s == pytest.approx(0.125) for s in clock.sleeps)


def test_rate_limiter_does_not_sleep_when_calls_are_already_slow():
    clock = FakeClock()
    limiter = RateLimiter(8, clock=clock.time, sleep=clock.sleep)
    limiter.wait()
    clock.now += 5
    limiter.wait()
    assert clock.sleeps == []


def test_every_request_carries_the_user_agent():
    client, session, _ = make_client([FakeResponse(200, b"a"), FakeResponse(200, b"b")])
    assert client.get("https://data.sec.gov/x") == b"a"
    assert client.get("https://data.sec.gov/y") == b"b"
    assert [r["headers"]["User-Agent"] for r in session.requests] == ["Test User test@invalid.test"] * 2


def test_client_refuses_an_empty_user_agent():
    with pytest.raises(ValueError):
        SecClient("  ")


def test_retries_on_429_then_succeeds():
    client, session, sleeps = make_client(
        [FakeResponse(429, headers={"Retry-After": "2"}), FakeResponse(503), FakeResponse(200, b"ok")]
    )
    assert client.get("https://data.sec.gov/x") == b"ok"
    assert len(session.requests) == 3
    assert 2.0 in sleeps  # honoured Retry-After


def test_retries_on_connection_error():
    client, session, _ = make_client([requests.ConnectionError("boom"), FakeResponse(200, b"ok")])
    assert client.get("https://data.sec.gov/x") == b"ok"
    assert len(session.requests) == 2


def test_gives_up_after_max_retries():
    client, session, _ = make_client([FakeResponse(503)] * 3, max_retries=2)
    with pytest.raises(SecHttpError, match="Gave up"):
        client.get("https://data.sec.gov/x")
    assert len(session.requests) == 3


def test_404_is_not_retried():
    client, session, _ = make_client([FakeResponse(404)])
    with pytest.raises(NotFoundError):
        client.get("https://data.sec.gov/x")
    assert len(session.requests) == 1


def test_403_is_not_retried_and_explains_itself():
    client, session, _ = make_client([FakeResponse(403)])
    with pytest.raises(SecHttpError, match="User-Agent"):
        client.get("https://data.sec.gov/x")
    assert len(session.requests) == 1
