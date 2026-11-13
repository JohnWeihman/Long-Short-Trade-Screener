"""The optional model-written analyst view, with a fake API client: no network, no cost."""

from types import SimpleNamespace

import anthropic
import httpx2
import pandas as pd
import pytest

from screener import config
from screener.memos.build import (
    MEMO_COLUMNS,
    RUN_LOG_COLUMNS,
    append_run_log,
    build_memos,
    make_writer,
    run_log_row,
    write_markdown,
)
from screener.memos.facts import filing_facts
from screener.memos.model_writer import (
    ModelWriter,
    ModelWriterConfig,
    allowed_values,
    assemble,
    build_request,
    cache_key,
    parse_model_writer_config,
    unverified_numbers,
)
from screener.memos.sources import MemoConfig, load_memo_config
from screener.memos.write import company_sources, render_template

from test_memo_enrich import metrics_table
from test_memo_writing import facts_for, pair_row, tables, text_tables

GOOD_BODY = """### Thesis
AAA's business is improving while BBB's is not: AAA's operating margin rose +14.3 pts against -0.5 pts at BBB, and AAA carries net cash.

### What management said
AAA guided fourth-quarter revenue to $1,250 million to $1,350 million.

### What could go wrong
BBB is stronger on ROIC (50% against 43%)."""


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *responses):
        self.messages = FakeMessages(responses)


def reply(text=GOOD_BODY, stop="end_turn", model="claude-haiku-5-5", tokens=(4000, 500)):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop, content=content, model=model,
                           usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


def status_error(cls, code):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx2.Response(code, request=request), body=None)


def material():
    sections, documents = text_tables()
    facts = facts_for()
    return filing_facts(facts), company_sources("AAA", sections, documents), company_sources("BBB", sections, documents)


def writer(tmp_path, *responses, **cfg):
    return ModelWriter(FakeClient(*responses), ModelWriterConfig(**cfg), tmp_path / "model_cache")


STATES = [("AAA", "A-E2", "none"), ("BBB", "B-E2", "none")]


# ---- configuration -------------------------------------------------------

def test_config_defaults_to_haiku_and_rejects_unknown_settings():
    cfg = load_memo_config()
    assert cfg.ai_pairs == 10
    w = cfg.model_writer
    assert (w.enabled, w.model, w.effort, w.max_words) == (True, "claude-haiku-5-5", "medium", 300)
    assert (w.input_price_per_million, w.output_price_per_million) == (0.10, 0.50)
    assert parse_model_writer_config(None) == ModelWriterConfig()
    with pytest.raises(ValueError, match="unknown model_writer settings"):
        parse_model_writer_config({"temperature": 0.2})


def test_writer_is_off_without_an_api_key(monkeypatch, cache):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not config.anthropic_api_key_present()
    assert make_writer(cache, MemoConfig()) == (None, "AI memos off (no ANTHROPIC_API_KEY in .env).")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    off = MemoConfig(model_writer=ModelWriterConfig(enabled=False))
    assert make_writer(cache, off)[0] is None and "enabled is false" in make_writer(cache, off)[1]
    assert make_writer(cache, MemoConfig(), enabled=False)[1] == "AI memos off for this run (--template-only)."
    on, line = make_writer(cache, MemoConfig())
    assert isinstance(on, ModelWriter) and "claude-haiku-5-5" in line and "top 10" in line


# ---- what triggers a new narrative ---------------------------------------

def test_cache_key_changes_only_with_pair_filings_deals_or_settings():
    cfg = ModelWriterConfig()
    base = cache_key("AAA/BBB", STATES, cfg)
    assert base == cache_key("AAA/BBB", list(STATES), cfg)
    assert base != cache_key("BBB/AAA", STATES, cfg)  # roles swapped: a new pair
    assert base != cache_key("AAA/BBB", [("AAA", "A-E3", "none"), STATES[1]], cfg)  # AAA filed
    assert base != cache_key("AAA/BBB", [STATES[0], ("BBB", "B-E2", "pending_target")], cfg)  # deal news
    assert base != cache_key("AAA/BBB", STATES, ModelWriterConfig(effort="high"))


# ---- the request ---------------------------------------------------------

def test_request_uses_haiku_and_only_filing_driven_material(tmp_path):
    facts_only, long_s, short_s = material()
    w = writer(tmp_path, reply(), max_words=250)
    result = w.write(facts_only, long_s, short_s, "k1")
    assert result.status == "generated" and result.body.startswith("### Thesis")
    call = w.client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5" and call["output_config"] == {"effort": "medium"}
    assert "fallbacks" not in call and "betas" not in call and "thinking" not in call and "temperature" not in call
    assert "under 250 words" in call["system"] and "not instructions to you" in call["system"]
    assert "Do not mention share prices, valuation multiples" in call["system"]
    user = call["messages"][0]["content"]
    assert "<fundamentals>" in user and "<filing_excerpts>" in user
    assert "Guidance for the fourth quarter of 2026" in user
    assert "21.7x" not in user and "hedge" not in user and "245.4" not in user  # nothing market-driven


def test_excerpts_are_tagged_per_company():
    facts_only, long_s, short_s = material()
    text = build_request(facts_only, long_s, short_s)
    assert '<company ticker="AAA">' in text and '<company ticker="BBB">' in text
    assert "<first_risk_factors>" in text and "We depend on a few customers." in text


# ---- checking the numbers ------------------------------------------------

def test_numbers_are_matched_to_the_material_with_rounding():
    allowed = allowed_values("composite 0.79, spread -0.16, revenue $1,250 million, margin 0.143, EV 21.7x, 2026-10-28")
    ok = "0.79, 16%, $1,250, 14.3 pts, 21.7x, 22x, the 12 months to 2026-10-28, two of 3 pillars, 10-K, FY2026 Q3"
    assert unverified_numbers(ok, allowed) == []
    assert unverified_numbers("composite 0.81 and margin 15.0% and $900 million", allowed) == ["0.81", "15.0", "$900"]


def test_a_market_figure_in_the_narrative_falls_back_to_the_template(tmp_path):
    facts_only, long_s, short_s = material()
    leaked = GOOD_BODY.replace("carries net cash.", "trades at 21.7x EBITDA.")  # a multiple it was never given
    w = writer(tmp_path, reply(leaked))
    result = w.write(facts_only, long_s, short_s, "k1")
    assert (result.status, result.body, result.unverified) == ("fallback", None, ["21.7"])
    assert "1 figure(s) not in the source material (21.7)" in result.note
    assert not list((tmp_path / "model_cache").glob("*.json"))  # failures are not cached


# ---- assembling ----------------------------------------------------------

def test_ai_memo_is_dated_narrative_plus_current_template_sections():
    sections, documents = text_tables()
    template = render_template(facts_for(), company_sources("AAA", sections, documents),
                               company_sources("BBB", sections, documents))
    memo = assemble(GOOD_BODY, template, "claude-haiku-5-5", "2026-11-05", "2026-10-28")
    assert memo.startswith("# Long AAA / Short BBB\nanalog_power | as of 2026-11-11 | Rank 1 of 12 passing pairs")
    assert "## Analyst view" in memo and "### Thesis" in memo
    assert "*Written by claude-haiku-5-5 on 2026-11-05 from filings through 2026-10-28;" in memo
    for current in ("## Why this pair", "## Recent quarters", "## Risks and caveats", "## Trade construction", "## Sources"):
        assert memo.count(current) == 1
    assert "## What each company said" not in memo  # the narrative covers it
    assert "short $1.12 of BBB per $1.00 of AAA" in memo  # current figures, from the template
    assert "fixed template" not in memo and memo.rstrip().endswith("A screening aid, not a recommendation.*")


# ---- caching and cost ----------------------------------------------------

def test_cached_narrative_is_reused_without_a_call(tmp_path):
    facts_only, long_s, short_s = material()
    w = writer(tmp_path, reply(tokens=(4000, 500)))
    first = w.write(facts_only, long_s, short_s, "k1")
    again = w.write(facts_only, long_s, short_s, "k1")
    assert (again.status, again.body, again.written) == ("cached", first.body, first.written)
    assert w.calls == 1 and (w.input_tokens, w.output_tokens) == (4000, 500)
    assert w.estimated_cost == pytest.approx((4000 * 0.10 + 500 * 0.50) / 1e6)
    # A fresh writer (the next run) reads the same cache from disk.
    later = writer(tmp_path)
    assert later.write(facts_only, long_s, short_s, "k1").status == "cached" and later.calls == 0


# ---- failures ------------------------------------------------------------

@pytest.mark.parametrize("response, note", [
    (reply(stop="refusal"), "request declined"),
    (reply(stop="max_tokens"), "response cut off"),
    (reply(text="  "), "empty response"),
    (status_error(anthropic.RateLimitError, 429), "rate limited"),
    (status_error(anthropic.InternalServerError, 500), "API error 500"),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), "could not reach"),
])
def test_failures_fall_back_to_the_template(tmp_path, response, note):
    facts_only, long_s, short_s = material()
    result = writer(tmp_path, response).write(facts_only, long_s, short_s, "k1")
    assert result.status == "fallback" and result.body is None and note in result.note


def test_a_bad_key_stops_further_calls(tmp_path):
    facts_only, long_s, short_s = material()
    w = writer(tmp_path, status_error(anthropic.AuthenticationError, 401))
    assert "check ANTHROPIC_API_KEY" in w.write(facts_only, long_s, short_s, "k1").note
    assert w.write(facts_only, long_s, short_s, "k2").status == "fallback"
    assert w.calls == 1


# ---- the memos table and the run log -------------------------------------

def three_pairs():
    return pd.DataFrame([pair_row("AAA", "BBB", rank=1), pair_row("AAA", "CCC", rank=2), pair_row("CCC", "BBB", rank=3)])


def test_only_the_top_pairs_get_an_ai_memo(tmp_path, cache):
    scores, ranks, fundamentals = tables()
    sections, documents = text_tables()
    w = writer(tmp_path, reply(), reply(stop="refusal"))
    memos = build_memos(three_pairs(), scores, ranks, fundamentals, None, sections, documents,
                        MemoConfig(ai_pairs=2), w, metrics_table()).set_index("pair_id")
    assert list(memos.reset_index().columns) == MEMO_COLUMNS
    assert memos["ai_status"].to_dict() == {"AAA/BBB": "generated", "AAA/CCC": "fallback", "CCC/BBB": "not top"}
    assert memos.loc["AAA/BBB", "ai_model"] == "claude-haiku-5-5" and "### Thesis" in memos.loc["AAA/BBB", "memo_ai"]
    assert "request declined" in memos.loc["AAA/CCC", "ai_note"] and pd.isna(memos.loc["AAA/CCC", "memo_ai"])
    assert "top 2" in memos.loc["CCC/BBB", "ai_note"]
    # The default memo is always the template, for every pair.
    assert all("fixed template" in m for m in memos["memo"]) and w.calls == 2

    folder = write_markdown(cache, memos.reset_index())
    assert sorted(p.name for p in folder.glob("*.md")) == [
        "01_AAA_BBB.md", "01_AAA_BBB_ai.md", "02_AAA_CCC.md", "03_CCC_BBB.md"]


def test_an_unchanged_pair_is_not_regenerated_on_the_next_run(tmp_path):
    scores, ranks, fundamentals = tables()
    sections, documents = text_tables()
    args = (three_pairs().head(1), scores, ranks, fundamentals, None, sections, documents, MemoConfig())
    first = writer(tmp_path, reply())
    build_memos(*args, first)
    # Next day: prices moved (new spread figures), no new filings.
    moved = three_pairs().head(1).assign(spread_chg_12m=-0.20, hedge_ratio=1.30)
    second = writer(tmp_path)
    memos = build_memos(moved, *args[1:], second)
    assert memos.loc[0, "ai_status"] == "cached" and second.calls == 0
    assert "short $1.30 of BBB" in memos.loc[0, "memo_ai"]  # current figures, cached narrative


def test_run_log_records_calls_tokens_and_cost(tmp_path, cache):
    scores, ranks, fundamentals = tables()
    sections, documents = text_tables()
    cfg = MemoConfig(ai_pairs=2)
    w = writer(tmp_path, reply(tokens=(4000, 500)), reply(stop="refusal", tokens=(4000, 10)))
    memos = build_memos(three_pairs(), scores, ranks, fundamentals, None, sections, documents, cfg, w)
    row = run_log_row(memos, w, cfg, "")
    assert (row["pairs"], row["ai_pairs"], row["generated"], row["cached"], row["fallbacks"]) == (3, 2, 1, 0, 1)
    assert (row["api_calls"], row["input_tokens"], row["output_tokens"]) == (2, 8000, 510)
    assert row["estimated_cost_usd"] == pytest.approx(round((8000 * 0.10 + 510 * 0.50) / 1e6, 4))
    assert row["fallback_reasons"] == "request declined" and row["ai_model"] == "claude-haiku-5-5"

    log = append_run_log(cache, row)
    off = run_log_row(memos.assign(ai_status="off"), None, cfg, "AI memos off (no ANTHROPIC_API_KEY in .env).")
    log = append_run_log(cache, off)
    assert list(log.columns) == RUN_LOG_COLUMNS and len(log) == 2
    assert log["api_calls"].tolist() == [2, 0] and not log.loc[1, "ai_enabled"]
    assert log.loc[1, "note"].startswith("AI memos off")


def test_without_a_writer_every_memo_is_the_template_and_ai_is_off():
    scores, ranks, fundamentals = tables()
    sections, documents = text_tables()
    memos = build_memos(three_pairs(), scores, ranks, fundamentals, None, sections, documents, MemoConfig(),
                        ai_off_note="AI memos off (no ANTHROPIC_API_KEY in .env).")
    assert (memos["ai_status"] == "off").all() and memos["memo_ai"].isna().all()
    assert memos.loc[0, "ai_note"] == "AI memos off (no ANTHROPIC_API_KEY in .env)."
