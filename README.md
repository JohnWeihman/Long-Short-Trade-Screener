# Pair-trade screener

A fundamental long/short pair-trade screener: group companies into peer
groups, score them on valuation, quality and direction, and surface pairs
where one company scores well and a close peer scores poorly.

Data is free only: SEC EDGAR for fundamentals and filings, yfinance for prices.

**Status: Stages 1 to 5 (raw data pull; cleaned quarterly and annual
financials; peer groups and scores; pairs with hedge ratios; template
memos).** The Streamlit app and the scheduled pipeline come later. See
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

## Build the pairs

```
python -m screener.pairing.deals              # flag companies in mergers and acquisitions
python -m screener.pairing.build
```

Uses the scores table's as-of date. Writes to `data/pairs/`:

| Table | Contents |
|-------|----------|
| `deals` | each company's deal status, read from its latest earnings press release, with the sentence that triggered it |
| `pairs` | every candidate pair within a peer group: score gaps, correlation, betas, hedge ratio, spread statistics, whether it passes, and why not |
| `pair_spreads` | daily hedged-spread history for the passing pairs |

A pair is long the higher composite score and short the lower. It passes
when the score gap is large enough, enough pillars favor the long, and the
two stocks' weekly returns are correlated enough. The hedge ratio (dollars
short per dollar long) is the long's beta divided by the short's, measured
against an equal-weighted index of the universe. Thresholds are in
`config/pairing.yaml`.

A company that is being acquired trades on the deal terms, not on its
fundamentals, so pairs with an acquisition target on either side are
excluded (they stay in the table with the reason). Acquirers and recently
completed deals are noted but kept. Detection reads only the latest earnings
release and is keyword-based, so it can miss or misread a deal:
`config/deals.csv` is where you correct it by hand.

```
python scripts/show_pairs.py                  # ranked pairs that pass
python scripts/show_pairs.py --all            # every candidate, with the reason it failed
python scripts/show_pairs.py TER AMAT         # one pair side by side, with the hedge arithmetic
```

## Write the memos

```
python -m screener.memos.build
```

For the companies in the top-ranked pairs this downloads (once, then cached)
the latest 10-K and the last four earnings press releases, extracts the
sections a memo needs, and writes one memo per pair. Writes to `data/memos/`:

| Table | Contents |
|-------|----------|
| `documents` | each filing document used, with its fetch status and link |
| `sections` | extracted text: 10-K business, risk factors, MD&A and risk headings; press-release headline, summary and outlook; each with `ok` / `problem` |
| `memos` | one row per passing pair: the template memo, the AI memo if there is one, its status and model, its fact sheet and sources as JSON |
| `model_runs` | one row per run: API calls, cache hits, regenerations, fallbacks, tokens, estimated cost |

The memos are also saved as Markdown files in `data/memos/md/`.

Each memo is built from a fact sheet (scores, ratios, hedge ratio, spread,
caveats), which holds every number the memo may use, plus quoted excerpts
from the filings. Section extraction is heuristic and reports what it could
not find; `config/memos.yaml` sets how many pairs get a memo.

There are two kinds of memo:

- **Template memo** (the default, free): written for every passing pair on
  every run, so its numbers are always current. Sections: why this pair
  (pillar scores and every metric against the peer-group median), recent
  quarters (revenue growth against the year-ago quarter, gross and
  operating margin, trailing free cash flow margin), what each company said,
  risks and caveats, trade construction, sources.
- **AI memo** (optional, paid): when `ANTHROPIC_API_KEY` is set in `.env`,
  Claude Haiku 5.5 writes an analyst view (thesis, what management said,
  what could go wrong) for the top 10 pairs. It sees only facts that change
  when a company files, never prices or valuation multiples, and is
  regenerated only when a pair is new, either company files a 10-K, 10-Q or
  earnings 8-K, or a deal status changes; otherwise the cached view is
  reused at no cost. The AI memo is that dated view plus the template's
  current figures. Every figure in the view is checked against the material
  it was given; a view that fails the check, or any API failure, leaves the
  pair with its template memo, and `ai_note` says why. `--template-only`
  skips the API for a run.

Every run appends a row to `data/memos/model_runs.parquet`: API calls,
cached and regenerated views, fallbacks, tokens and estimated cost.

```
python scripts/show_memo.py                   # list the memos
python scripts/show_memo.py TER AMAT          # one template memo
python scripts/show_memo.py 1                 # by rank
python scripts/show_memo.py TER AMAT --ai     # the AI memo, if there is one
python scripts/show_memo.py --log             # API calls per run
```

## Update everything

```
python -m screener.pipeline                   # refresh EDGAR and prices, rebuild every table and memo
python -m screener.pipeline --no-refresh      # rebuild from the local cache only
```

Runs the data pull, cleaning, scoring, deal check, pairs and memos in order,
and prints how long each took. This is the command a scheduled job runs.

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
config/pairing.yaml     pair thresholds and windows
src/screener/pairing/   candidate pairs, correlation filter, hedge ratios, spreads
config/deals.csv        manual merger and acquisition flags
config/memos.yaml       how many pairs get a memo
src/screener/memos/     filing text extraction, fact sheets and memos
src/screener/app/       later stage
scripts/                command-line helpers
tests/
data/                   local cache (git-ignored)
```
