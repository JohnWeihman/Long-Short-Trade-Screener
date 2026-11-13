# Pair-trade screener

Fundamental long/short pair-trade screener that surfaces ideas like those a
discretionary L/S equity hedge fund would run. Built in stages; **only build
the stage the user asks for.**

## End goal (design for it from day one)

- Cover every operating company that files with the SEC. We start with a small
  universe; scaling up must be an expansion, not a rewrite.
- A pipeline on a schedule (GitHub Actions) that refreshes data, recomputes
  scores and pairs, and saves results.
- A private Streamlit site (Streamlit Community Cloud) showing ranked pairs,
  the metrics behind them, spread charts, and short long/short memos.

## How the screener works

1. Group companies into true peer groups.
2. Within each group, rank on valuation (EV/EBITDA, P/E, FCF yield, EV/sales),
   quality (margins, ROIC, net debt/EBITDA), and direction (margin trends,
   revenue growth acceleration, FCF trend).
3. Flag pairs where one company scores well and a close peer scores poorly.
4. Keep only pairs with reasonable historical price correlation; compute
   beta-adjusted hedge ratios.
5. Generate short memos for top pairs from 10-K sections and 8-K earnings
   press releases.

## Stages

| Stage | Scope | Status |
|-------|-------|--------|
| 1 | Project setup; raw data pull (ticker->CIK, companyfacts, submissions, prices) into local cache; hand-check script | built and committed (on `main`) |
| 2 | Cleaning: tag fallback mapping, Q4 derivation, restatement dedup, filer-type handling | built and committed (on `main`). Known limits are listed under "Known problems" below and in the `validation` table |
| 3 | Peer groups, derived fundamentals, metrics, within-group scores | built and committed (on `main`) |
| 4 | Candidate pairs, correlation filter, beta-adjusted hedge ratios, spread series | built and committed (on `main`) |
| 5 | Memos from 10-K sections and 8-K earnings press releases | built on branch `stage-5-memos`; template layer committed (dd74eff); peer medians, quarterly trends, Haiku AI layer, run log and `screener.pipeline` not yet committed. AI layer not yet run against the API (no ANTHROPIC_API_KEY on this machine as of 2026-11-13) |
| 6 | Streamlit app, publish step, GitHub Actions schedule, hosting on Streamlit Community Cloud | built on branch `stage-6-app`, not committed; app checked locally; workflow not yet run (no GitHub repository yet) |

## Hard constraints

- **Free data only.** SEC EDGAR for fundamentals and filings; yfinance for
  prices. Only `src/screener/data/prices.py` may import yfinance, so the price
  source can be swapped.
- **SEC rules.** Every SEC request sends a `User-Agent` with the user's name
  and email, and stays under 10 requests/second. All SEC traffic goes through
  `SecClient` in `src/screener/data/http.py` (throttled to 8/s). Never call
  `requests` against sec.gov anywhere else.
- **Cache every raw download.** Never re-fetch something already saved unless
  the user explicitly refreshes (`--refresh`).
- **No secrets or the user's email in committed code.** They live in `.env`
  (git-ignored), read via `src/screener/config.py`. `.env.example` holds
  placeholders only.

## Tech stack

Python, pandas, Parquet files queried with DuckDB, pytest, Streamlit,
GitHub Actions. Dependencies in `requirements.txt`; the package is installed
editable from `pyproject.toml`.

## Layout

```
config/universe.csv        starting universe (one ticker per row)
src/screener/config.py     .env loading, paths, SEC_USER_AGENT validation
src/screener/data/         FETCHING ONLY - no cleaning here
  http.py                  SecClient: User-Agent, rate limit, retries
  cache.py                 cache paths, gzip read/write, fetch-if-missing, manifest
  tickers.py               ticker -> CIK from SEC company_tickers.json
  edgar.py                 companyfacts + submissions downloads
  prices.py                PriceSource interface + yfinance implementation
  pull.py                  CLI: python -m screener.data.pull
  rawfacts.py              read-only lookups on raw companyfacts (hand checks)
config/tag_map.yaml        metric -> XBRL tags (priority order) + fallback formulas
src/screener/cleaning/     Stage 2
  tagmap.py                loads and validates the tag map
  audit.py                 coverage audit: python -m screener.cleaning.audit
  store.py                 the only place cleaned tables are read/written
  companies.py             company table, filing index, filer_type classification
  splits.py                restate share and per-share facts to today's share basis
  facts.py                 flatten mapped tags; de-dup to first-reported + latest
  metrics.py               tag map applied to `facts`: one value per metric per reported period
  periods.py               fiscal calendar per company; classify periods as Q / YTD / FY / instant
  quarters.py              standalone quarters (reported, derived, ratio), annual table, TTM
  validate.py              consistency checks -> `validation` table (reports, never corrects)
  trace.py                 the reported periods and raw facts behind one cleaned value
  build.py                 rebuild data/clean/*: python -m screener.cleaning.build
config/peer_groups.csv     curated ticker -> peer_group (blank = unplaced, not ranked)
config/scoring.yaml        tax rate and staleness limits for Stage 3
src/screener/scoring/      Stage 3
  peers.py                 peer file loader + residual-return correlation check
  fundamentals.py          as-of fundamentals: market cap, net debt, EV, TTM lines, EBITDA, FCF, ROIC
  metrics.py               ratios from fundamentals (valuation as yields; display multiples)
  scores.py                rank within peer group -> pillar scores -> composite
  report.py                plain-text group table and per-company detail
  build.py                 writes data/scores/*: python -m screener.scoring.build [--as-of DATE]
scripts/show_fundamentals.py   one company's fundamentals line by line (hand check)
scripts/show_scores.py         ranked group tables, or the ranks behind one company's score
config/pairing.yaml        Stage 4 thresholds and windows
src/screener/pairing/      Stage 4
  config.py                loads config/pairing.yaml
  candidates.py            every within-group pair, long the higher composite; records gap/pillar rules
  stats.py                 correlation, beta, hedge ratio, share ratio
  spread.py                daily hedged spread, z-score and recent moves
  report.py                plain-text ranked table and pair detail
  deals.py                 M&A flags from each company's latest earnings release: python -m screener.pairing.deals
  build.py                 writes data/pairs/*: python -m screener.pairing.build
config/deals.csv           manual deal flags; override the detection
scripts/show_pairs.py          ranked pairs, all candidates with reasons, or one pair in detail
config/memos.yaml          Stage 5: how many pairs, press-release quarters, risk headings
src/screener/data/filings.py   fetch filing documents (10-K main document, 8-K complete submission); pick which
src/screener/memos/        Stage 5
  extract.py               filing HTML -> paragraphs; 10-K Items 1 / 1A / 7; press-release headline, summary, outlook
  sources.py               select, fetch and extract the documents for the top pairs' companies
  facts.py                 fact sheet per pair: every number a memo may use (plain JSON-able data)
  write.py                 template memo (Markdown) from fact sheet + filing excerpts; inputs hash
  model_writer.py          optional AI analyst view (Haiku): filing-based cache key, number check, run cost
src/screener/pipeline.py   one command for everything: python -m screener.pipeline [--no-refresh]
  build.py                 writes data/memos/* and md files: python -m screener.memos.build [--refresh]
scripts/show_memo.py           list memos, or print one by tickers or rank
src/screener/app/          Stage 6 web app (reads published/ only)
  data.py                  load published tables; shape them for pages (pure functions, tested)
  charts.py                Altair charts: spread, quarterly trends (dataviz reference palette)
  main.py                  pages: Pairs, Pair detail, Peer groups, Data status
streamlit_app.py           Streamlit entry point (adds src/ to sys.path)
src/screener/publish.py    data/ -> published/ (+ --restore of run state); python -m screener.publish
published/                 committed app data (~0.7 MB) + AI memo cache + run logs
.github/workflows/pipeline.yml   weekdays 23:00 UTC: tests, pipeline --publish, commit published/
scripts/check_company.py   prints revenue / net income / shares for latest FY (raw data)
scripts/trace_metric.py    a cleaned metric next to the raw facts behind it
tests/                     pytest, no network access
data/                      git-ignored cache
```

Cache layout (keyed by 10-digit zero-padded CIK, never by ticker, because
tickers change and get reused):

```
data/raw/edgar/company_tickers.json.gz
data/raw/edgar/companyfacts/CIK##########.json.gz
data/raw/edgar/submissions/CIK##########.json.gz   (+ -submissions-00N overflow files)
data/raw/prices/yfinance/<TICKER>.parquet
data/manifest/run_<UTC timestamp>.parquet          one row per fetch attempt
```

Cleaned tables (rebuilt from the raw cache, never edited by hand):

```
data/clean/companies.parquet   one row per company; filer_type, include
data/clean/filings.parquet     every filing (accession, form, dates)
data/clean/facts_raw.parquet   every filing's version of every period, mapped tags only
data/clean/facts.parquet       one row per (company, tag, unit, period):
                               val_first / val_latest + the filing behind each, restated flag
data/clean/metric_facts.parquet       tag map applied: (company, metric, start, end) -> val, val_first, source
data/clean/fiscal_years.parquet       per company: fiscal_year, fy_start, fy_end, complete
data/clean/fiscal_quarters.parquet    per company: fiscal_year, quarter, period_end
data/clean/metrics_quarterly.parquet  standalone quarters: val, val_first, method, known_on, val_ttm
data/clean/metrics_annual.parquet     reported full years / year-end balances
data/clean/splits.parquet             split history from the price cache
data/clean/validation.parquet         findings: check, severity, ticker, metric, period, value, expected
data/clean/audit/              tag coverage audit output
```

Raw EDGAR responses are stored byte-for-byte (gzipped), with the same file
names the SEC uses in its nightly bulk `companyfacts.zip` / `submissions.zip`,
so moving to bulk downloads only changes the fetcher.

## EDGAR data quirks (Stage 2 must handle these)

- The same metric uses different XBRL tags across companies and years
  (`Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax` vs
  `SalesRevenueNet`): needs a tag fallback mapping.
- Q4 is not reported separately: derive it as annual minus nine-month figures
  for income statement items.
- The same period appears in multiple filings, including restatements. Select
  by period `start`/`end`, not by the `fy`/`fp` fields, which describe the
  filing the fact appeared in, not the period of the fact.
- Income statement items are durations; balance sheet items are point-in-time.
- Fiscal years do not follow the calendar (NVDA ends late January, MU late
  August, AVGO around the start of November) and may be 52/53 weeks.
- Banks, insurers, REITs, BDCs, foreign IFRS filers (20-F/40-F), SPACs, and
  shell companies need separate handling or exclusion.
- At full scale use the SEC's nightly bulk companyfacts download instead of
  per-company API calls.
- Found in the Stage 2 audit of the semis universe:
  - A tag that means something different is worse than a gap. Do not add
    "close enough" fallbacks: `CommonStockSharesIssued` includes treasury
    shares (TXN 1,741M vs 908M outstanding); `Depreciation` alone misses
    billions of intangibles amortization (AVGO).
  - Some metrics are components, not alternatives (short-term debt = current
    part of long-term debt + commercial paper + ...): sum them via a formula.
  - Some companies report no operating-income line (KLAC, COHR), no gross
    profit (QCOM, KLAC, COHR) or no total liabilities (AMD, INTC, ADI, ...);
    these are derived by formula and flagged `derived`.
  - Even a standard D&A tag can exclude acquired-intangibles amortization
    (SYNA, MRVL pre-FY2024). Company-specific extension tags are not in
    companyfacts at all. Treat D&A / EBITDA as lower confidence.
  - Raw EDGAR values can simply be wrong: COHR FY2023
    `DepreciationAndAmortization` = 681,687M (a scale error in the filing).
    Validation must catch this; never patch a single value by hand.
  - Proxy statements (DEF 14A) reuse us-gaap tags in pay-versus-performance
    tables with rounded or differently defined values, and are filed after
    the 10-K. Only forms in `facts.STATEMENT_FORMS` count as fact sources.
  - "Latest filed" is not always the correct value. Some filings report a
    number in the wrong scale (SYNA's FY2024 10-K: 39.7 diluted shares
    instead of 39.7M; COHR's FY2025 10-K: prior-year D&A of 681,687M).
    `facts.flag_scale_errors` sets these aside when another filing of the
    same period differs by 1,000x or 1,000,000x, keeping the value nearer
    the tag's usual size for that company (85 rows in 80 periods). It only
    acts with that corroboration; a lone outlier is validation's job.
  - `facts.large_disagreement` marks periods where filings differ by 10x+.
    Most are 10-for-1 splits restating share counts and EPS (NVDA, AVGO,
    LRCX, KLAC): legitimate, to be explained by the split step. The rest
    are real conflicts, unresolved: MCHP FY2015-17 `StockholdersEquity`
    (one 10-K tagged an unrelated number), INTC 2021 securities
    reclassification, SLAB 2020 tax after a discontinued-ops recast.
  - Fiscal year label = year of (fy_end - 15 days): a year ending 3 Jan 2026
    is fiscal 2025 (LSCC, SLAB), one ending 25 Jan 2026 is fiscal 2026 (NVDA).
  - Quarter rules: a reported three-month value beats a derived one; derived
    = year-to-date minus prior year-to-date (Q4 = FY - 9 months). Cash flow
    items are year-to-date only, so Q2-Q4 are always derived. Weighted
    average shares: Q4 = 4 x FY - 3 x nine-month average (approximation).
    EPS Q4 = derived net income / derived shares, never a subtraction.
  - A quarter is derived only when the two figures being subtracted have
    the same `source` (same tag or same formula). The tag map falls back
    per period, so without this rule AMD/MRVL/ENTG D&A quarters came out
    negative. Cost: 258 derived values (2.7%) became gaps, most visibly
    pretax income and D&A quarters for some companies.
  - Cover-page share counts are assigned to the latest fiscal quarter ending
    within 150 days before the cover date.
  - Scoring will need one share count per company: prefer the cover-page
    count, fall back to the balance-sheet count (TXN, NXPI, COHR have only
    the former; ADI only the latter).
  - Known problems in the quarterly table, reported by validation
    (159-190 findings over each company's last five fiscal years):
    (a) Q4 EPS by ratio uses total net income, so it is overstated for
        companies with preferred dividends (AVGO FY2022: 0.87 vs 0.78 by
        annual-minus-quarters; COHR FY2022-25), and can be far off in loss
        quarters (SYNA FY2023). NVDA FY2009 Q4 derived shares are negative.
    (b) Mixed vintages after a recast: `val` uses each period's latest
        value, but a later 10-K can recast the full year (discontinued
        operations) while the nine-month figure never gets recast. SLAB
        FY2019 Q4 revenue comes out -144M. `val_first` is consistent.
    (c) About 2% of company-metric-years have four quarters that do not sum
        to the reported year (reported Q1-Q3 disagree with the nine-month
        figure, e.g. stock comp, or recasts).
  - `cleaning/audit.py` reads raw companyfacts with its own "latest filed"
    rule and does not yet benefit from the proxy or scale fixes; the period
    logic must read the `facts` table instead.
  - Share counts are as-reported and not split-adjusted (KLAC 10-for-1 on
    2026-06-12: 130.6M -> 1,306.5M); prices are. `cleaning/splits.py`
    restates every 'shares' and 'USD/shares' fact to today's basis using the
    split history in the price cache, at the `facts_raw` level, before
    de-dup. Rule: apply splits dated after the FILING date for us-gaap
    facts (statements issued after a split restate all periods), and after
    the fact's own date for dei cover-page counts. Verified on the 8 splits
    since 2009 in this universe: no share or per-share period ending 2019+
    is left restated by more than 3%. `facts_raw.val_reported` keeps the
    filed value. Consequence: clean share counts change whenever a new
    split appears in the price data, so prices must be refreshed before a
    rebuild.
  - Companies reorganized under a new CIK have short histories under the
    current one (AVGO from FY2016, MRVL from FY2020).
  - Net income uses `NetIncomeLoss` (attributable to the parent), not
    `ProfitLoss` (includes minority interests).
- yfinance `close` is split-adjusted but not dividend-adjusted; `adj_close`
  is adjusted for both. Neither is the price actually printed on the day.

## Stage 3 decisions and findings

- User decisions (2026-11): curated peer groups, equal pillar weights, flat
  21% tax rate for ROIC, latest restated values for current scores
  (first-reported reserved for backtests), unplaced companies left unranked.
- Peer groups: ai_datacenter (NVDA, AVGO, MRVL, COHR), processors (AMD,
  INTC, QCOM - thin), analog_power, mixed_signal_connectivity,
  equipment_materials (incl. AMKR); MU unplaced. The check removes the
  sector's weekly move and correlates the residuals: 25 of 26 placed
  companies are most correlated with their own group (MPWR is not).
- Everything in scoring is "as of" a date: only quarters with known_on <=
  as_of, and the last price <= as_of. TTM is recomputed from quarters, not
  read from `val_ttm`.
- Net debt = short + long-term debt - cash - short-term investments. Items a
  company does not report are taken as 0 and noted. Leases, preferred stock
  and minority interests are excluded (understates EV for COHR, INTC).
- EBITDA is blank when D&A is missing (COHR, TXN, LSCC, SLAB, SYNA as of
  2026-11): rank those on EV/EBIT.
- Open: KLAC (3,176M) and TER (96M) hold securities under
  `AvailableForSaleSecuritiesDebtSecurities`, which is not limited to
  current holdings and so is not in the tag map; their net debt is
  overstated by that much. NVDA's 42.8B of equity securities
  (`EquitySecuritiesFvNi`) is also not counted as cash-like.
- ROIC is distorted when invested capital moves sharply within a year
  (NVDA: 52B -> 206B, ROIC 121%).
- Scoring: each metric is a percentile rank within the peer group (0 worst,
  1 best, ties averaged); pillar = mean of its ranks (needs 2); composite =
  weighted mean of pillars (needs 2). Ranks are robust to outliers, so no
  clipping is applied. Valuation is ranked on yields (EBITDA/EV, earnings /
  market cap, FCF / market cap, revenue/EV) so losses rank last.
- Fallback slots: a company lacking EBITDA is ranked on EBIT yield and net
  debt/EBIT in those slots, among all group members. That mixes two scales
  within one slot; `metric_ranks.used_metric` records which was used.
- Known weaknesses of the composite: equal weights let strong direction
  offset extreme valuation (LSCC ranks 2nd of 5 at 225x EV/EBIT); groups of
  3-5 give coarse ranks; `processors` is flagged thin.
- Data tables for Stage 4: `data/scores/scores.parquet` (composite, pillars,
  rank_in_group per ticker) and `metric_ranks.parquet`.

## Stage 4 decisions and findings

- User decisions (2026-11): beta benchmark is the equal-weighted universe
  index (sector-neutral, no extra data); minimum composite gap 0.25;
  minimum weekly correlation 0.50 over 104 weeks; at least 2 of 3 pillars
  must favor the long; thin groups included but flagged.
- Hedge ratio = beta_long / beta_short = dollars short per $1 long. Share
  ratio = hedge ratio x price_long / price_short. `pair_beta` (long
  regressed on short) is stored for comparison only.
- The spread series applies TODAY's hedge ratio to the whole history, reset
  daily. It shows how the pair as it would be put on now has behaved; it is
  not a backtest. Positive spread move / z-score = long already outperformed.
- Pairs use the as-of date stored in the scores table; there is no separate
  --as-of on the pairing build.
- `pairs.parquet` keeps every candidate with `passes` and `reasons`; nothing
  is dropped. As of 2026-11-11: 62 candidates, 22 clear gap + pillars, 15
  also clear correlation. Seven fail on correlation, several narrowly
  (0.47-0.50), including every ai_datacenter pair.
- Raw weekly correlations in this universe over two years are only
  0.38-0.68, lower than the "all semis move together" intuition.
- Not built: cointegration, liquidity/borrow screens, earnings dates,
  backtesting. The last weekly bar can be a partial week.

## Stage 5 decisions and findings

- User decisions (2026-11): template memo first, then an optional
  model-written layer (off unless an API key is in `.env`; numbers only from
  the fact sheet; cached; template is the fallback); top 10 passing pairs;
  `beautifulsoup4` (+ `lxml`) for HTML.
- Sources: the latest 10-K's main document, and for the last four earnings
  8-Ks (item code 2.02 in the filing index's new `items` column) the
  complete submission `<accession>.txt`, from which exhibit EX-99.1 is
  taken. Cached under `data/raw/edgar/filings/CIK##########/<accession>/`.
- Tables: `data/memos/documents.parquet` (one row per document, with fetch
  status) and `data/memos/sections.parquet` (section text, `ok`, `problem`).
- Extraction is heuristic. 10-K items are found by line-start "Item N"
  headings, taking the occurrence followed by the most text (skips the table
  of contents). Risk headings are bold passages inside Item 1A (none found
  for SLAB and POWI, which do not bold them). Press-release outlook falls
  back to expectation sentences when there is no "Outlook"/"Guidance"
  heading, so it can pick up a quote instead of the guidance table (TER,
  NXPI); MPWR's exhibit is an earnings commentary, not a press release.
- IMPORTANT, found in the press releases as of 2026-11: SLAB has a pending
  acquisition by Texas Instruments and SYNA by onsemi (both are short legs
  of top-ranked pairs: CRUS/SYNA, CRUS/SLAB, LSCC/SYNA). SWKS's latest
  release confirms the Qorvo combination closed, so its financials are no
  longer comparable with prior periods.
- Deal handling (user decision 2026-11-13: flag and exclude targets).
  `pairing/deals.py` reads the latest earnings release of every scored
  company and sets a status: `pending_target` (pairs excluded when
  `exclude_pending_targets` is true), `pending_other` (acquirer or other
  party; note only), `closed` (recent completed deal; note only), `none`.
  A target needs both pending-deal language in a sentence about M&A and a
  sign it is the one being bought (guidance or call suspended, "to be
  acquired by", or the deal phrase naming the company's first name word).
  Legal boilerplate paragraphs are ignored. `config/deals.csv` overrides.
  As of 2026-11-11: targets SLAB, SYNA; acquirer MRVL (Lumentum), plus TXN
  and ON by manual override (their own releases did not trigger); closed
  SWKS (Qorvo), LSCC (AMI), MCHP (Hailo, Vorago), DIOD (ElevATE). Passing
  pairs went from 15 to 12.
- Limits of deal handling: only the latest earnings release is read, so a
  deal announced since then is missed until the next release or a manual
  row, and a flag can disappear when a company files a second release
  (DIOD's 2026-11-12 8-K dropped its "closed" flag for ElevATE); scoring still ranks targets within their peer group, which shifts
  other companies' percentile ranks; `pairs.deals_checked` is False and the
  build warns if the deals table has not been built.
- User decisions (2026-11-13, second round): memos update automatically via
  `screener.pipeline`; template memos are the default, for EVERY passing
  pair, rebuilt each run; AI memos optional, top 10 only, Claude Haiku 5.5,
  regenerated only when a pair is new or a company has a new filing; a new
  filing = 10-K, 10-Q or earnings 8-K (item 2.02), amendments included;
  deal-status changes also regenerate; scheduling (GitHub Actions) is Stage 6.
- `memos.parquet`: one row per passing pair. `memo` is always the template
  memo. `memo_ai` is the AI memo or empty; `ai_status` is generated /
  cached / fallback / not top / off, with the reason in `ai_note`;
  `ai_model`, `ai_written`, `filings_through`, `ai_key`. The fact sheet is
  the only source of numbers.
- Template memo sections: why this pair (pillar table; every metric for
  long, short and peer median with who is stronger; biggest edges), recent
  quarters (last four known quarters per company: revenue growth vs the
  year-ago quarter, gross and operating margin, TTM FCF margin; derived
  quarters marked *, recent deals flagged), what each company said, risks
  and caveats, trade construction, sources.
- Peer median = median across all scored companies in the group, both legs
  included, of the metric the LONG was ranked on (so a fallback slot uses the
  fallback metric); infinite leverage counts as highest. Needs the scores
  area's `metrics` table.
- Template weaknesses: outlook excerpts are sometimes a CEO quote or a
  lead-in to a guidance table because tables are not prose; memos are
  600-900 words with the new tables.
- AI layer (`memos/model_writer.py`, Anthropic Python SDK >= 2.8, plain
  `client.messages.create`):
  - Gate: `model_writer.enabled` AND `ANTHROPIC_API_KEY` set
    (`config.anthropic_api_key_present`; only the SDK reads the key).
    `--template-only` skips it. Only ranks 1..`ai_pairs` (10) are sent.
  - Request: `claude-haiku-5-5`, `output_config.effort` medium (also
    Haiku 5.5's default), no `thinking` param (adaptive by default), no
    temperature. Haiku has no server-side refusal fallback: a refusal falls
    back to the template memo.
  - Input is `facts.filing_facts(fact_sheet)`: quality and direction metrics
    (with who is stronger), quarterly trends, deal status/evidence, data
    notes, plus tagged filing excerpts. Valuation, prices, scores, ranks,
    peer medians and trade statistics are deliberately excluded because they
    move daily while the cached view does not.
  - Output: three `###` sections (Thesis / What management said / What could
    go wrong). `assemble` builds the AI memo: template header, "## Analyst
    view" with a dated note, the narrative, then the template's CURRENT
    sections (why this pair, recent quarters, risks, trade construction,
    sources). The template's "What each company said" is dropped there.
  - Check: every number in the narrative must match a number in the request
    (also x100 or /100, rounded to the precision written; integers 0-12
    always allowed). Any miss -> fallback, reason in `ai_note`. A valuation
    multiple in the narrative fails the check because it is not in the input.
  - Cache: `data/memos/model_cache/<key>.json` holds the narrative only.
    key = pair_id (long/short order matters) + each leg's latest relevant
    accession (`data.filings.latest_relevant_filing`) + each leg's deal
    status + model + effort + max_words + PROMPT_VERSION. Bump
    PROMPT_VERSION when the prompt changes. Failures are not cached.
    Auth/permission errors stop further calls for the run.
  - Run log: `data/memos/model_runs.parquet`, one row per memo build (even
    with AI off): pairs, ai_pairs, cached, generated, fallbacks and reasons,
    api_calls, tokens, estimated cost (prices in config/memos.yaml).
    `scripts/show_memo.py --log`.
- Pipeline (`screener.pipeline`): pull with `--refresh all` (ticker map,
  companyfacts, submissions, prices: ~3 SEC requests per company), then
  cleaning, scoring (as of today), deal check, pairs, memos. Filing
  documents are fetched only when new (keyed by accession). Exit code 1 from
  a step (finished with problems) does not stop the run; >= 2 or an
  exception stops it. `--no-refresh` skips the pull. Cleaning and scoring
  rebuild fully every run (about a minute each on 27 companies).
- Pipeline order is now: data.pull -> cleaning.build -> scoring.build ->
  pairing.deals -> pairing.build -> memos.build.

## Stage 6 decisions and findings

- User decisions (2026-11-13): published data committed to `main` under
  `published/`; schedule weekdays 23:00 UTC; access via Streamlit's viewer
  list only (private repo -> private app; one private app per free account,
  confirmed in Streamlit docs 2026-11-13); tests run before every publish.
- `published/` holds only small results tables (metrics_quarterly cut to 7
  lines and 6 fiscal years) plus run state: the AI memo cache
  (`model_cache/*.json`, union of published and local so entries are never
  lost), `model_runs`, `pipeline_runs`, and `manifest.json` (as_of,
  published_at, row counts). `publish.restore` fills data/ from published/
  before every pipeline run, never overwriting local files, so a fresh
  runner does not pay again for AI memos.
- Pipeline additions: "price check" step after the pull fails the run (exit
  2) when more than 10% of the universe has no price within 5 days, so a
  Yahoo block on cloud runners never publishes stale results; `--publish`
  adds a final publish step, then republishes once more so published/
  includes this run's log line. Exit 1 ("problems") still publishes; the
  workflow only fails on >= 2.
- App: reads `published/` (or SCREENER_PUBLISHED_DIR). Pages via a sidebar
  radio (key "page") rather than st.navigation, so AppTest can drive them.
  Data cached with st.cache_data keyed on manifest.published_at. Charts use
  `theme=None` and their own palette (slot 1 blue = long, slot 2 orange =
  short; light/dark steps chosen from `st.context.theme.type`). The dataviz
  validator needs Node, which is not installed here; the two slots are the
  reference palette's documented-validated slots, used unchanged.
- Streamlit 1.67: `use_container_width` is deprecated; use width="stretch".
- Known display quirk: a peer median of a valuation multiple is the inverse
  of the median yield, so with loss-makers in the group it can be extreme
  (CRUS/SWKS group median P/E 354x = LSCC's).
- The workflow pins actions/checkout@v4, setup-python@v5, cache@v4 and
  Python 3.14 to match local; `data/raw` is carried in actions/cache keyed
  per run with a prefix restore key.

## Working rules

- Propose a plan first and wait for the user's approval before writing code.
- Keep modules small and separated: data fetching, cleaning, scoring, pairing,
  memos, and the web app each live in their own module.
- Write tests for anything that transforms data. Tests must not hit the
  network.
- When a number looks wrong, show the user the raw EDGAR value behind it
  (tag, period, accession number) before changing any code. Fix root causes,
  not symptoms.
- Tell the user when a stage is complete and working so they can commit it.
  The user makes the commits.

## Commands

```
.venv\Scripts\activate
pip install -r requirements.txt
pytest
python -m screener.data.pull                     # pull the universe (skips cached files)
python -m screener.data.pull --refresh prices    # re-download prices only
python scripts/check_company.py NVDA
python -m screener.cleaning.build                # rebuild data/clean (refresh prices first: splits)
python -m screener.cleaning.validate             # findings; --ticker X, --check NAME, --limit N
python -m screener.cleaning.audit                # tag-map coverage matrix
python scripts/trace_metric.py NVDA revenue --fy 2026 --quarter 4
python -m screener.scoring.peers                 # peer groups vs residual return correlation
python -m screener.scoring.build                 # fundamentals, metrics, ranks, scores (--as-of DATE)
python scripts/show_scores.py [GROUP|TICKER]
python scripts/show_fundamentals.py NVDA
python -m screener.pairing.deals                 # M&A flags from latest earnings releases (--refresh)
python -m screener.pairing.build                 # pairs for the scores' as-of date
python scripts/show_pairs.py [--all | LONG SHORT]
python -m screener.memos.build                   # fetch filings for top pairs, extract sections, write memos (--refresh, --template-only)
python scripts/show_memo.py [LONG SHORT | RANK] [--ai] | --log
python -m screener.pipeline                      # everything, in order (--no-refresh to skip downloads, --publish for the app)
python -m screener.publish                       # data/ -> published/ (--restore: published/ -> data/)
streamlit run streamlit_app.py                   # the app, http://localhost:8501
```
