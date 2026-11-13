import json

import pandas as pd
import pytest

from screener.cleaning.store import table_path, write_table
from screener.publish import TABLES, PublishError, publish, restore


def fill(cache, with_state=True):
    for area, name in TABLES:
        frame = pd.DataFrame({"x": [1, 2]})
        if name == "pairs":
            frame = pd.DataFrame({"pair_id": ["A/B", "C/D"], "passes": [True, False],
                                  "as_of": [pd.Timestamp("2026-11-13")] * 2})
        write_table(cache, name, frame, area)
    quarterly = pd.DataFrame([
        {"cik10": "1", "metric": m, "fiscal_year": fy, "quarter": 1, "period_end": pd.Timestamp(f"{fy}-03-31"),
         "val": 1.0, "method": "reported", "known_on": pd.Timestamp(f"{fy}-05-01"), "val_ttm": 4.0,
         "val_first": 1.0, "source": "tag", "formula_derived": False}
        for m in ("revenue", "cash", "rd_expense") for fy in range(2015, 2027)
    ])
    write_table(cache, "metrics_quarterly", quarterly)
    if with_state:
        write_table(cache, "model_runs", pd.DataFrame({"api_calls": [3]}), "memos")
        folder = cache.root / "memos" / "model_cache"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "local.json").write_text('{"body": "x"}')


def test_publish_writes_the_app_tables_and_a_manifest(cache, tmp_path):
    fill(cache)
    out = tmp_path / "published"
    manifest = publish(cache, out)
    names = {p.stem for p in out.glob("*.parquet")}
    assert {name for _, name in TABLES} | {"metrics_quarterly", "model_runs"} == names
    assert (manifest["as_of"], manifest["passing_pairs"], manifest["model_cache_entries"]) == ("2026-11-13", 1, 1)
    assert json.loads((out / "manifest.json").read_text())["tables"]["pairs"] == 2
    quarterly = pd.read_parquet(out / "metrics_quarterly.parquet")
    assert set(quarterly["metric"]) == {"revenue"}  # only the charted lines
    assert quarterly["fiscal_year"].min() == 2021  # last six fiscal years
    assert "val_first" not in quarterly.columns
    assert (out / "model_cache" / "local.json").exists()


def test_publish_keeps_earlier_cache_entries_and_replaces_stale_tables(cache, tmp_path):
    fill(cache)
    out = tmp_path / "published"
    publish(cache, out)
    (out / "model_cache" / "from_an_earlier_run.json").write_text("{}")
    (out / "old_table.parquet").write_text("stale")
    publish(cache, out)
    assert {p.name for p in (out / "model_cache").glob("*.json")} == {"local.json", "from_an_earlier_run.json"}
    assert not (out / "old_table.parquet").exists()
    assert not (tmp_path / "published.tmp").exists()


def test_publish_refuses_when_a_table_is_missing(cache, tmp_path):
    fill(cache)
    table_path(cache, "scores", "scores").unlink()
    out = tmp_path / "published"
    with pytest.raises(PublishError, match="scores/scores"):
        publish(cache, out)
    assert not out.exists()


def test_restore_fills_a_fresh_machine_without_overwriting(cache, tmp_path):
    fill(cache)
    out = tmp_path / "published"
    publish(cache, out)

    fresh = type(cache)(tmp_path / "fresh_data")
    assert restore(fresh, out) == ["memos/model_runs", "1 AI memo cache entries"]
    assert (fresh.root / "memos" / "model_cache" / "local.json").exists()
    assert pd.read_parquet(table_path(fresh, "model_runs", "memos"))["api_calls"].tolist() == [3]
    write_table(fresh, "model_runs", pd.DataFrame({"api_calls": [9]}), "memos")
    assert restore(fresh, out) == []  # nothing missing any more; local data wins
    assert pd.read_parquet(table_path(fresh, "model_runs", "memos"))["api_calls"].tolist() == [9]
    assert restore(fresh, tmp_path / "nothing_here") == []
