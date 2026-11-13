"""Copy what the web app needs from data/ (git-ignored) to published/ (committed).

    python -m screener.publish             # write published/ from data/
    python -m screener.publish --restore   # fill data/ from published/ (run state only)

published/ holds small results tables only, never raw downloads: about a
megabyte. It also carries the run state a fresh machine (the scheduled job)
needs to continue where the last run left off: the AI memo cache, so unchanged
pairs are not paid for twice, and the AI and pipeline run logs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from screener import config
from screener.cleaning.store import read_table, table_path
from screener.data.cache import Cache

DEFAULT_PUBLISHED = config.PROJECT_ROOT / "published"

# (area under data/, table name). Published as published/<name>.parquet.
TABLES = [
    ("pairs", "pairs"), ("pairs", "pair_spreads"), ("pairs", "deals"),
    ("scores", "scores"), ("scores", "metrics"), ("scores", "metric_ranks"),
    ("scores", "fundamentals"), ("scores", "peer_groups"),
    ("clean", "companies"), ("clean", "validation"),
    ("memos", "memos"), ("memos", "documents"),
]
QUARTERLY_METRICS = ("revenue", "gross_profit", "operating_income", "net_income",
                     "operating_cash_flow", "capex", "diluted_shares")
QUARTERLY_YEARS = 6
# Run state: copied both ways (published on publish, restored before a run).
STATE_TABLES = [("memos", "model_runs"), ("logs", "pipeline_runs")]
MODEL_CACHE = ("memos", "model_cache")


class PublishError(RuntimeError):
    pass


def _quarterly_for_app(cache: Cache) -> pd.DataFrame:
    """The quarterly table cut down to the lines and years the app charts."""
    quarterly = read_table(cache, "metrics_quarterly")
    keep = quarterly[quarterly["metric"].isin(QUARTERLY_METRICS)]
    latest = keep.groupby("cik10")["fiscal_year"].transform("max")
    keep = keep[keep["fiscal_year"] > latest - QUARTERLY_YEARS]
    columns = ["cik10", "metric", "fiscal_year", "quarter", "period_end", "val", "method", "known_on", "val_ttm"]
    return keep[columns].reset_index(drop=True)


def publish(cache: Cache, out: Path = DEFAULT_PUBLISHED) -> dict:
    """Write published/ and return its manifest. Raises PublishError if a
    table the app needs is missing (nothing is written in that case)."""
    missing = [f"{area}/{name}" for area, name in TABLES if not table_path(cache, name, area).exists()]
    if missing:
        raise PublishError(f"cannot publish, these tables are missing: {', '.join(missing)}")
    frames = {name: read_table(cache, name, area) for area, name in TABLES}
    frames["metrics_quarterly"] = _quarterly_for_app(cache)
    for area, name in STATE_TABLES:
        if table_path(cache, name, area).exists():
            frames[name] = read_table(cache, name, area)

    staging = out.with_name(out.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(staging / f"{name}.parquet", index=False)
    cache_dir = cache.root.joinpath(*MODEL_CACHE)
    published_cache = out / "model_cache"
    (staging / "model_cache").mkdir()
    # Union of what was published before and what this machine has: the
    # scheduled job starts with an empty data/ and must not lose entries.
    for source in (published_cache, cache_dir):
        if source.exists():
            for file in source.glob("*.json"):
                shutil.copy2(file, staging / "model_cache" / file.name)

    pairs = frames["pairs"]
    manifest = {
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "as_of": f"{pd.Timestamp(pairs['as_of'].iloc[0]):%Y-%m-%d}" if len(pairs) else None,
        "passing_pairs": int(pairs["passes"].sum()) if len(pairs) else 0,
        "tables": {name: len(frame) for name, frame in sorted(frames.items())},
        "model_cache_entries": len(list((staging / "model_cache").glob("*.json"))),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    if out.exists():
        shutil.rmtree(out)
    staging.replace(out)
    return manifest


def restore(cache: Cache, source: Path = DEFAULT_PUBLISHED) -> list[str]:
    """Copy the published run state into data/ where data/ does not have it.
    Never overwrites a local file. Returns what was restored."""
    restored = []
    if not source.exists():
        return restored
    for area, name in STATE_TABLES:
        src, dest = source / f"{name}.parquet", table_path(cache, name, area)
        if src.exists() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            restored.append(f"{area}/{name}")
    cache_dir = cache.root.joinpath(*MODEL_CACHE)
    copied = 0
    for file in (source / "model_cache").glob("*.json") if (source / "model_cache").exists() else []:
        target = cache_dir / file.name
        if not target.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            copied += 1
    if copied:
        restored.append(f"{copied} AI memo cache entries")
    return restored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--restore", action="store_true", help="fill data/ with the published run state")
    args = parser.parse_args(argv)
    cache = Cache(config.data_dir())
    if args.restore:
        restored = restore(cache)
        print("Restored: " + (", ".join(restored) if restored else "nothing (data/ already has it, or nothing published)"))
        return 0
    try:
        manifest = publish(cache)
    except PublishError as exc:
        print(exc)
        return 2
    size = sum(p.stat().st_size for p in DEFAULT_PUBLISHED.rglob("*") if p.is_file())
    print(f"Published {len(manifest['tables'])} tables and {manifest['model_cache_entries']} AI memo cache entries "
          f"to {DEFAULT_PUBLISHED} ({size / 1e6:.2f} MB), data as of {manifest['as_of']}, "
          f"{manifest['passing_pairs']} passing pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
