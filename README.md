# Pair-trade screener

A fundamental long/short pair-trade screener: group companies into peer
groups, score them on valuation, quality and direction, and surface pairs
where one company scores well and a close peer scores poorly.

Data is free only: SEC EDGAR for fundamentals and filings, yfinance for prices.

**Status: Stages 1 to 3 (raw data pull; cleaned quarterly and annual
financials; peer groups and scores).** Pairing, memos, the Streamlit app and
the scheduled pipeline come later. See
[CLAUDE.md](CLAUDE.md) for the full design, constraints and known data quirks.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Then edit `.env` and set `SEC_USER_AGENT` to your name and email. The SEC
requires this on every request. `.env` is git-ignored; never commit it.

## Pull data

```
python -m screener.data.pull
```

For every ticker in `config/universe.csv` this downloads, into `data/`:

| What | Where |
|------|-------|
| SEC ticker -> CIK map | `data/raw/edgar/company_tickers.json.gz` |
| XBRL company facts | `data/raw/edgar/companyfacts/CIK##########.json.gz` |
| Filing index and company metadata | `data/raw/edgar/submissions/CIK##########.json.gz` |
| Daily price history | `data/raw/prices/yfinance/<TICKER>.parquet` |
| Log of every fetch attempt | `data/manifest/run_<timestamp>.parquet` |

Anything already cached is skipped. To re-download:

```
python -m screener.data.pull --refresh prices          # or: tickers, edgar, all
python -m screener.data.pull --tickers NVDA AMD        # just these tickers
python -m screener.data.pull --skip-prices
```

SEC requests are throttled to 8 per second (the SEC's limit is 10).

Price columns: `close` is split-adjusted but not dividend-adjusted;
`adj_close` is adjusted for both.

## Check a company by hand

```
python scripts/check_company.py NVDA
```

Prints revenue, net income and share counts for the most recent fiscal year,
each with its XBRL tag, period, accession number and a link to the filing, so
the numbers can be compared against the 10-K.

## Build the cleaned tables

```
python -m screener.data.pull --refresh prices    # split history comes from prices
python -m screener.cleaning.build
```

Rebuilds everything under `data/clean/` from the local cache (no network):

| Table | Contents |
|-------|----------|
| `companies` | one row per company, with `filer_type` and `include` |
| `filings` | every filing: accession number, form, dates |
| `splits` | split history taken from the price files |
| `facts_raw` | every filing's version of every period, for the mapped tags |
| `facts` | one row per period: first-reported and latest values |
| `metric_facts` | the tag map applied: one value per metric per reported period |
| `fiscal_years`, `fiscal_quarters` | each company's own fiscal calendar |
| `metrics_quarterly` | standalone fiscal quarters, with trailing twelve months |
| `metrics_annual` | reported fiscal years and year-end balances |
| `validation` | findings from the consistency checks |

How a number gets there:

1. `config/tag_map.yaml` lists, for each metric, the XBRL tags to try in
   order and formulas to fall back on. Edit it to add a metric or a tag.
2. Share counts and per-share figures are restated to today's share basis.
3. When several filings report the same period, both the first-reported and
   the latest value are kept. Proxy statements and wrong-scale values are
   set aside.
4. A quarter is the reported three-month figure, or else year-to-date minus
   the prior year-to-date (Q4 = full year - nine months), and only when both
   come from the same tag or formula.

Query the tables with DuckDB, for example:

```
python -c "import duckdb; print(duckdb.sql(\"select fiscal_year, quarter, val, method from 'data/clean/metrics_quarterly.parquet' q join 'data/clean/companies.parquet' c using (cik10) where c.ticker='NVDA' and metric='revenue' order by 1,2\").df().tail(8))"
```

## Check the cleaned data

```
python -m screener.cleaning.validate                    # all findings, grouped by check
python -m screener.cleaning.validate --ticker COHR
python -m screener.cleaning.audit                       # tag-map coverage by company and metric
python scripts/trace_metric.py NVDA revenue --fy 2026 --quarter 4
python scripts/trace_metric.py KLAC operating_income --annual
```

`trace_metric.py` prints a metric's recent values and, for one of them, the
reported periods and raw EDGAR facts (tag, filing, accession number) it was
built from. Validation never changes a value; it only reports.

## Score the companies

```
python -m screener.scoring.peers                 # check peer groups against price behaviour
python -m screener.scoring.build                 # as of today
python -m screener.scoring.build --as-of 2026-06-30
```

Writes to `data/scores/`:

| Table | Contents |
|-------|----------|
| `peer_groups` | ticker -> peer group, from `config/peer_groups.csv` |
| `fundamentals` | market cap, net debt, enterprise value, trailing-twelve-month lines, EBITDA, free cash flow, ROIC |
| `metrics` | the ratios: valuation yields, margins, leverage, direction, plus display multiples |
| `metric_ranks` | each metric's value and rank within the peer group (0 worst, 1 best) |
| `scores` | pillar scores (valuation, quality, direction), composite, rank in group |

Everything is computed "as of" a date: only quarters filed by then, and the
last price on or before it. Pillar membership, weights and thresholds are in
`config/scoring.yaml`; the ratio formulas are in `src/screener/scoring/metrics.py`.

```
python scripts/show_scores.py                    # every group, ranked
python scripts/show_scores.py analog_power
python scripts/show_scores.py TXN                # the metrics and ranks behind one score
python scripts/show_fundamentals.py TXN          # market cap, EV, EBITDA, ROIC line by line
```

## Tests

```
pytest
```

Tests use fakes and fixtures only; they never touch the network.

## Layout

```
config/universe.csv     tickers to pull
config/tag_map.yaml     metric -> XBRL tags and fallback formulas
src/screener/data/      fetching and the local cache
src/screener/cleaning/  raw facts -> clean quarterly and annual financials
config/peer_groups.csv  ticker -> peer group
config/scoring.yaml     pillars, weights and thresholds
src/screener/scoring/   peer groups, fundamentals, ratios, ranks and scores
src/screener/pairing/  memos/  app/     later stages
scripts/                command-line helpers
tests/
data/                   local cache (git-ignored)
```
