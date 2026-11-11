"""Download raw companyfacts and submissions JSON from EDGAR into the cache."""

from __future__ import annotations

from screener.data.cache import BytesFetcher, Cache, FetchRecord, cik10, fetch_to_cache, read_gz_json

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{file_name}"


def fetch_companyfacts(
    client: BytesFetcher, cache: Cache, cik: int | str, refresh: bool = False
) -> FetchRecord:
    """All XBRL facts the company has ever reported, in one JSON file."""
    key = cik10(cik)
    return fetch_to_cache(
        client,
        COMPANYFACTS_URL.format(cik10=key),
        cache.companyfacts_path(key),
        source="edgar/companyfacts",
        key=key,
        refresh=refresh,
    )


def fetch_submissions(
    client: BytesFetcher, cache: Cache, cik: int | str, refresh: bool = False
) -> list[FetchRecord]:
    """Company metadata and filing index.

    The main file holds the most recent ~1,000 filings; older ones sit in
    overflow files listed under filings.files, which are fetched too.
    """
    key = cik10(cik)
    main_name = f"CIK{key}.json"
    main = _fetch_submissions_file(client, cache, key, main_name, refresh)
    records = [main]
    if main.status in ("downloaded", "cached"):
        index = read_gz_json(cache.submissions_path(main_name))
        for extra in index.get("filings", {}).get("files", []):
            records.append(_fetch_submissions_file(client, cache, key, extra["name"], refresh))
    return records


def _fetch_submissions_file(
    client: BytesFetcher, cache: Cache, key: str, file_name: str, refresh: bool
) -> FetchRecord:
    return fetch_to_cache(
        client,
        SUBMISSIONS_URL.format(file_name=file_name),
        cache.submissions_path(file_name),
        source="edgar/submissions",
        key=key,
        refresh=refresh,
    )
