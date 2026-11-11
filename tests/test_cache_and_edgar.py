import duckdb
import pytest

from screener.data.cache import cik10, fetch_to_cache, read_gz, read_gz_json, write_manifest
from screener.data.edgar import fetch_companyfacts, fetch_submissions

from conftest import FakeFetcher

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
SUBS_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
SUBS_EXTRA_URL = "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"


@pytest.mark.parametrize("given", [320193, "320193", "0000320193", "CIK0000320193"])
def test_cik10_pads_to_ten_digits(given):
    assert cik10(given) == "0000320193"


@pytest.mark.parametrize("bad", ["AAPL", "", "12345678901"])
def test_cik10_rejects_non_ciks(bad):
    with pytest.raises(ValueError):
        cik10(bad)


def test_download_is_stored_byte_for_byte(cache):
    body = b'{"cik": 320193,  "odd spacing": true}'
    fetcher = FakeFetcher({FACTS_URL: body})
    record = fetch_companyfacts(fetcher, cache, 320193)
    assert record.status == "downloaded"
    assert record.key == "0000320193"
    assert cache.companyfacts_path(320193).name == "CIK0000320193.json.gz"
    assert read_gz(cache.companyfacts_path(320193)) == body


def test_cached_file_is_never_refetched(cache):
    fetcher = FakeFetcher({FACTS_URL: {"v": 1}})
    fetch_companyfacts(fetcher, cache, 320193)
    fetcher.responses[FACTS_URL] = {"v": 2}
    record = fetch_companyfacts(fetcher, cache, 320193)
    assert record.status == "cached"
    assert len(fetcher.calls) == 1
    assert read_gz_json(cache.companyfacts_path(320193)) == {"v": 1}


def test_refresh_refetches_and_overwrites(cache):
    fetcher = FakeFetcher({FACTS_URL: {"v": 1}})
    fetch_companyfacts(fetcher, cache, 320193)
    fetcher.responses[FACTS_URL] = {"v": 2}
    record = fetch_companyfacts(fetcher, cache, 320193, refresh=True)
    assert record.status == "downloaded"
    assert len(fetcher.calls) == 2
    assert read_gz_json(cache.companyfacts_path(320193)) == {"v": 2}


def test_missing_resource_is_recorded_not_raised(cache):
    record = fetch_companyfacts(FakeFetcher({}), cache, 320193)
    assert record.status == "not_found"
    assert not cache.companyfacts_path(320193).exists()


def test_failed_download_leaves_no_partial_file(cache):
    class Exploding:
        def get(self, url):
            raise RuntimeError("network down")

    path = cache.companyfacts_path(1)
    with pytest.raises(RuntimeError):
        fetch_to_cache(Exploding(), "u", path, source="s", key="k")
    assert not path.exists()


def test_submissions_fetches_overflow_files(cache):
    main = {"cik": "320193", "filings": {"recent": {}, "files": [{"name": "CIK0000320193-submissions-001.json"}]}}
    fetcher = FakeFetcher({SUBS_URL: main, SUBS_EXTRA_URL: {"accessionNumber": []}})
    records = fetch_submissions(fetcher, cache, 320193)
    assert [r.status for r in records] == ["downloaded", "downloaded"]
    assert fetcher.calls == [SUBS_URL, SUBS_EXTRA_URL]
    assert cache.submissions_path("CIK0000320193-submissions-001.json").exists()

    again = fetch_submissions(fetcher, cache, 320193)
    assert [r.status for r in again] == ["cached", "cached"]
    assert len(fetcher.calls) == 2


def test_manifest_is_queryable_with_duckdb(cache):
    fetcher = FakeFetcher({FACTS_URL: {"v": 1}})
    records = [fetch_companyfacts(fetcher, cache, 320193), fetch_companyfacts(fetcher, cache, 320193)]
    path = write_manifest(cache, records)
    rows = duckdb.sql(
        f"select status, count(*) from read_parquet('{path.as_posix()}') group by 1 order by 1"
    ).fetchall()
    assert rows == [("cached", 1), ("downloaded", 1)]


def test_empty_manifest_writes_nothing(cache):
    assert write_manifest(cache, []) is None
