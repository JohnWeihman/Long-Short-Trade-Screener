import json

import pandas as pd
import pytest

from screener.data.cache import write_gz
from screener.data.filings import document_url
from screener.pairing.candidates import candidate_pairs
from screener.pairing.config import PairingConfig
from screener.pairing.deals import detect_all, detect_deal, load_overrides

from conftest import FakeFetcher
from test_pairing import CFG, score_row, weekly

FILLER = "Revenue for the quarter was $200 million, up 5 percent from the same quarter a year ago."


# ---- reading one press release -----------------------------------------

@pytest.mark.parametrize("line, name", [
    ("Due to the announced pending acquisition of Silicon Labs by Texas Instruments, Silicon Labs has suspended providing forward-looking guidance.", "SILICON LABORATORIES INC."),
    ("As previously announced, onsemi and Synaptics entered into a definitive agreement under which onsemi agreed to acquire Synaptics in an all-stock transaction.", "SYNAPTICS Inc"),
    ("In light of the proposed merger, the company will not host a conference call to discuss results.", "SOMECO INC"),
    ("The company entered into a merger agreement and expects to be acquired by BigCo in 2027.", None),
])
def test_acquisition_target_is_recognized(line, name):
    status, evidence = detect_deal([FILLER, line], name)
    assert status == "pending_target" and evidence == line


def test_acquirer_is_flagged_but_not_as_a_target():
    line = "Marvell and Lumentum today announced a definitive agreement for Marvell to acquire Lumentum in an all-stock transaction."
    assert detect_deal([FILLER, line], "Marvell Technology, Inc.") == ("pending_other", line)
    # The same sentence in the target's own release names the target.
    assert detect_deal([line], "LUMENTUM HOLDINGS INC")[0] == "pending_target"


@pytest.mark.parametrize("line", [
    "Successfully completed its acquisition of ElevATE Semiconductor, a provider of test circuits.",
    "Lattice completed the acquisition of AMI on July 27, 2026.",
    "Skyworks Begins New Chapter Following Qorvo Combination; Reports Fourth Quarter Results",
    "For the December quarter, our first as a combined company, we anticipate revenue of $1.8 billion.",
])
def test_completed_deal(line):
    assert detect_deal([FILLER, line], "ANYCO INC") == ("closed", line)


@pytest.mark.parametrize("line", [
    # Non-GAAP boilerplate, not a deal:
    "This outlook excludes known charges related to completed acquisitions of $0.01 per share.",
    "The employee payments were established in connection with completed acquisitions, are subject to conditions.",
    # A partnership, not M&A:
    "Announced strategic partnerships to mobilize capital for AI infrastructure, subject to definitive agreements.",
    # Legal boilerplate lists deals that do not exist:
    "This release contains forward-looking statements, including risks related to any pending acquisition or merger agreement.",
    # Guidance suspended for another reason, with no deal anywhere:
    "Given macroeconomic uncertainty, the company has suspended providing quarterly guidance.",
    FILLER,
])
def test_routine_language_is_not_a_deal(line):
    assert detect_deal([line], "ANYCO INC") == ("none", "")


def test_evidence_is_trimmed_and_short_names_are_not_used_for_matching():
    long_line = "The company agreed to acquire TargetCo. " + "x" * 600
    assert len(detect_deal([long_line], None)[1]) == 320
    # 'ON' is too short to anchor a match: "acquisition of On..." must not make ON a target.
    line = "onsemi announced the pending acquisition of Online Devices Corp for $100 million."
    assert detect_deal([line], "ON SEMICONDUCTOR CORP")[0] == "pending_other"


# ---- overrides ---------------------------------------------------------

def test_real_overrides_file_loads():
    frame = load_overrides()
    assert list(frame.columns) == ["ticker", "status", "note"]


def test_overrides_are_validated(tmp_path):
    path = tmp_path / "deals.csv"
    path.write_text("# c\nticker,status,note\nswks,closed,Qorvo\nSWKS,none,cleared\n")
    assert load_overrides(path).to_dict("records") == [{"ticker": "SWKS", "status": "none", "note": "cleared"}]
    path.write_text("ticker,status,note\nSWKS,maybe,\n")
    with pytest.raises(ValueError, match="unknown status"):
        load_overrides(path)
    path.write_text("ticker,state\nSWKS,none\n")
    with pytest.raises(ValueError, match="columns"):
        load_overrides(path)


# ---- all companies -----------------------------------------------------

def submission(paragraphs):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (f"<DOCUMENT>\n<TYPE>8-K\n<FILENAME>f.htm\n<TEXT>\n<p>Item 2.02</p>\n</TEXT>\n</DOCUMENT>\n"
            f"<DOCUMENT>\n<TYPE>EX-99.1\n<FILENAME>r.htm\n<TEXT>\n<html><body>{body}</body></html>\n</TEXT>\n</DOCUMENT>\n")


def setup(cache):
    companies = pd.DataFrame([
        {"cik10": "0000000001", "ticker": "TGT", "name": "Targetco Inc"},
        {"cik10": "0000000002", "ticker": "FINE", "name": "Fineco Inc"},
        {"cik10": "0000000003", "ticker": "NOPR", "name": "Nopress Inc"},
        {"cik10": "0000000004", "ticker": "NONE", "name": "Nofilings Inc"},
    ])
    def filing(cik, accn, filed, items="2.02,9.01"):
        return {"cik10": cik, "accn": accn, "form": "8-K", "filed": pd.Timestamp(filed), "report_date": pd.NaT,
                "primary_document": "f.htm", "file_number": None, "items": items}
    filings = pd.DataFrame([
        filing("0000000001", "T-OLD", "2025-08-01"), filing("0000000001", "T-NEW", "2025-11-01"),
        filing("0000000002", "F-NEW", "2025-11-02"),
        filing("0000000003", "N-NEW", "2025-11-03"),
    ])
    responses = {
        document_url("0000000001", "T-NEW", "T-NEW.txt"): submission(
            [FILLER, "Due to the pending acquisition of Targetco by BigCo, Targetco has suspended providing guidance."]).encode(),
        document_url("0000000001", "T-OLD", "T-OLD.txt"): submission([FILLER]).encode(),
        document_url("0000000002", "F-NEW", "F-NEW.txt"): submission([FILLER]).encode(),
        document_url("0000000003", "N-NEW", "N-NEW.txt"): b"<DOCUMENT>\n<TYPE>8-K\n<TEXT>\nx\n</TEXT>\n</DOCUMENT>\n",
    }
    return companies, filings, FakeFetcher(responses)


NO_OVERRIDES = pd.DataFrame(columns=["ticker", "status", "note"])
AS_OF = pd.Timestamp("2025-12-01")


def test_detect_all_reads_only_the_latest_release(cache):
    companies, filings, fetcher = setup(cache)
    deals, records = detect_all(fetcher, cache, filings, companies, AS_OF, NO_OVERRIDES)
    by = deals.set_index("ticker")
    assert by["status"].to_dict() == {"TGT": "pending_target", "FINE": "none", "NOPR": "none", "NONE": "none"}
    assert by.loc["TGT", "accn"] == "T-NEW" and "pending acquisition of Targetco" in by.loc["TGT", "evidence"]
    assert by.loc["FINE", "evidence"] == ""
    assert by.loc["NOPR", "evidence"] == "the 8-K has no press release exhibit to check"
    assert by.loc["NONE", "evidence"] == "no earnings release found to check"
    assert len(records) == 3 and not any("T-OLD" in call for call in fetcher.calls)
    # A second run downloads nothing.
    detect_all(fetcher, cache, filings, companies, AS_OF, NO_OVERRIDES)
    assert len(fetcher.calls) == 3


def test_before_the_deal_was_announced_the_company_is_clear(cache):
    companies, filings, fetcher = setup(cache)
    deals, _ = detect_all(fetcher, cache, filings, companies, pd.Timestamp("2025-09-01"), NO_OVERRIDES)
    assert deals.set_index("ticker").loc["TGT", ["status", "accn"]].tolist() == ["none", "T-OLD"]


def test_manual_override_wins_and_offline_mode_reports_what_is_missing(cache):
    companies, filings, fetcher = setup(cache)
    overrides = pd.DataFrame([{"ticker": "TGT", "status": "none", "note": "deal terminated"},
                              {"ticker": "FINE", "status": "pending_target", "note": ""}])
    deals, _ = detect_all(fetcher, cache, filings, companies, AS_OF, overrides)
    by = deals.set_index("ticker")
    assert by.loc["TGT", ["status", "source", "evidence"]].tolist() == ["none", "manual", "deal terminated"]
    assert by.loc["FINE", ["status", "source"]].tolist() == ["pending_target", "manual"]

    offline, records = detect_all(None, cache, filings, companies[companies["ticker"] == "FINE"],
                                  pd.Timestamp("2025-12-01"), NO_OVERRIDES)
    assert records == [] and offline.loc[0, "status"] == "none"  # cached from the run above, so it was read


def test_offline_with_nothing_cached_says_so(cache):
    companies, filings, _ = setup(cache)
    deals, _ = detect_all(None, cache, filings, companies, AS_OF, NO_OVERRIDES)
    assert deals.set_index("ticker").loc["TGT", "evidence"] == "earnings release not downloaded yet"


# ---- effect on pairs ---------------------------------------------------

def evaluated(deals, cfg=CFG):
    from screener.pairing.build import evaluate_pairs
    frame, _ = weekly()
    table = pd.DataFrame([
        score_row("AAA", 0.80, 0.9, 0.8, 0.7), score_row("BBB", 0.50, 0.2, 0.9, 0.4), score_row("CCC", 0.20, 0.1, 0.2, 0.3),
    ])
    dates = pd.bdate_range("2025-06-02", periods=120)
    daily = {t: pd.Series(0.001, index=dates) for t in ("AAA", "BBB", "CCC")}
    pairs, spreads = evaluate_pairs(candidate_pairs(table, cfg), frame, daily,
                                    {"AAA": 200.0, "BBB": 50.0, "CCC": 25.0}, {}, cfg, deals)
    return pairs.set_index("pair_id"), spreads


def test_pair_with_an_acquisition_target_is_excluded_with_the_reason():
    pairs, spreads = evaluated({"CCC": "pending_target", "BBB": "closed", "AAA": "pending_other"})
    assert not pairs.loc["AAA/CCC", "passes"] and not pairs.loc["AAA/CCC", "passes_deal"]
    assert pairs.loc["AAA/CCC", "reasons"] == "CCC is being acquired (pending deal)"
    assert pd.isna(pairs.loc["AAA/CCC", "rank"]) and "AAA/CCC" not in set(spreads["pair_id"])
    assert pairs.loc["AAA/CCC", "deal_short"] == "pending_target"
    # Acquirers and completed deals stay in, with a note; the ranking closes up.
    assert pairs.loc["AAA/BBB", "passes"] and pairs.loc["AAA/BBB", "rank"] == 1
    assert pairs.loc["AAA/BBB", "notes"] == (
        "AAA: party to a pending deal | BBB: recently completed a deal; results may not be comparable")
    assert len(pairs) == 3  # still listed


def test_both_legs_targets_and_the_switch_to_keep_them():
    pairs, _ = evaluated({"AAA": "pending_target", "CCC": "pending_target"})
    assert pairs.loc["AAA/CCC", "reasons"] == "AAA and CCC are being acquired (pending deal)"
    keep = PairingConfig(min_overlap_weeks=20, correlation_window_weeks=60, spread_history_days=100,
                         spread_zscore_window_days=40, exclude_pending_targets=False)
    pairs, _ = evaluated({"CCC": "pending_target"}, keep)
    assert pairs.loc["AAA/CCC", "passes"] and pairs.loc["AAA/CCC", "passes_deal"]
    assert "BEING ACQUIRED" in pairs.loc["AAA/CCC", "notes"]


def test_no_deal_check_means_everything_counts_as_clear():
    pairs, _ = evaluated(None)
    assert pairs["passes"].all() and (pairs["deal_long"] == "none").all() and pairs["passes_deal"].all()
