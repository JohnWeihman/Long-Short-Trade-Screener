import pandas as pd
import pytest

from screener.cleaning.companies import filing_index
from screener.data.cache import write_gz
from screener.data.filings import (
    document_url,
    earnings_releases,
    fetch_document,
    latest_annual_report,
    parse_submission,
    press_release,
    read_document,
    submission_file_name,
)
from screener.memos.extract import (
    emphasized_lines,
    html_to_lines,
    risk_headings,
    split_10k,
    split_press_release,
)
from screener.memos.sources import (
    MemoConfig,
    extract_sections,
    fetch_documents,
    load_memo_config,
    select_documents,
    top_pair_tickers,
)

from conftest import FakeFetcher

CIK = "0000001234"
AS_OF = pd.Timestamp("2026-03-01")


def filings():
    rows = [
        ("K-25", "10-K", "2025-02-20", "main25.htm", ""),
        ("K-26", "10-K", "2026-02-20", "main26.htm", ""),
        ("KA-26", "10-K/A", "2026-02-25", "amend.htm", ""),
        ("K-27", "10-K", "2027-02-20", "future.htm", ""),               # after as_of
        ("E-1", "8-K", "2025-04-25", "e1.htm", "2.02,9.01"),
        ("E-2", "8-K", "2025-07-25", "e2.htm", "2.02,9.01"),
        ("E-3", "8-K", "2025-10-24", "e3.htm", "9.01, 2.02"),
        ("E-4", "8-K", "2026-01-30", "e4.htm", "2.02"),
        ("X-1", "8-K", "2026-02-10", "x1.htm", "5.02,9.01"),             # an officer change, not earnings
        ("X-2", "8-K", "2026-02-12", "x2.htm", "12.02"),                 # must not match "2.02" as a substring
        ("X-3", "8-K", "2026-02-14", "x3.htm", None),
    ]
    return pd.DataFrame([{
        "cik10": CIK, "accn": a, "form": f, "filed": pd.Timestamp(d), "report_date": pd.NaT,
        "primary_document": p, "file_number": None, "items": i,
    } for a, f, d, p, i in rows])


# ---- choosing and fetching filings -------------------------------------

def test_filing_index_carries_8k_item_codes():
    main = {"filings": {"recent": {
        "accessionNumber": ["A-1", "A-2"], "form": ["8-K", "10-Q"], "filingDate": ["2026-01-30", "2026-02-10"],
        "reportDate": ["", ""], "primaryDocument": ["a.htm", "b.htm"], "fileNumber": ["1", "1"],
        "items": ["2.02,9.01", ""],
    }}}
    index = filing_index(1234, main)
    assert index.set_index("accn")["items"].to_dict() == {"A-1": "2.02,9.01", "A-2": ""}


def test_document_url_strips_dashes_and_leading_zeros():
    assert document_url(CIK, "0001234567-26-000012", "main.htm") == (
        "https://www.sec.gov/Archives/edgar/data/1234/000123456726000012/main.htm")
    assert submission_file_name("0001234567-26-000012") == "0001234567-26-000012.txt"


def test_latest_annual_report_ignores_amendments_and_the_future():
    assert latest_annual_report(filings(), CIK, AS_OF)["accn"] == "K-26"
    assert latest_annual_report(filings(), CIK, pd.Timestamp("2026-01-01"))["accn"] == "K-25"
    assert latest_annual_report(filings(), "999", AS_OF) is None


def test_earnings_releases_are_item_2_02_only_newest_first():
    got = earnings_releases(filings(), CIK, AS_OF, 3)
    assert got["accn"].tolist() == ["E-4", "E-3", "E-2"]
    assert earnings_releases(filings(), CIK, pd.Timestamp("2025-08-01"), 4)["accn"].tolist() == ["E-2", "E-1"]


def test_fetch_document_caches(cache):
    url = document_url(CIK, "K-26", "main26.htm")
    fetcher = FakeFetcher({url: b"<html>hello</html>"})
    assert fetch_document(fetcher, cache, CIK, "K-26", "main26.htm").status == "downloaded"
    assert fetch_document(fetcher, cache, CIK, "K-26", "main26.htm").status == "cached"
    assert len(fetcher.calls) == 1
    assert read_document(cache, CIK, "K-26", "main26.htm") == "<html>hello</html>"
    assert cache.filing_path(CIK, "K-26", "main26.htm").parent.name == "K-26"


SUBMISSION = """<SEC-DOCUMENT>0001-26-1.txt
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>form8k.htm
<DESCRIPTION>8-K
<TEXT>
<html><body><p>Item 2.02 Results of Operations</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.2
<FILENAME>slides.htm
<TEXT>
<html><body><p>Investor slides</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>release.htm
<DESCRIPTION>PRESS RELEASE
<TEXT>
<html><body><p>Test Semi Reports Record Fourth Quarter Revenue</p></body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def test_parse_submission_and_pick_the_press_release():
    documents = parse_submission(SUBMISSION)
    assert [(d["type"], d["file_name"]) for d in documents] == [
        ("8-K", "form8k.htm"), ("EX-99.2", "slides.htm"), ("EX-99.1", "release.htm")]
    assert documents[2]["description"] == "PRESS RELEASE"
    chosen = press_release(documents)
    assert chosen["file_name"] == "release.htm" and "Record Fourth Quarter" in chosen["text"]
    assert press_release(documents[:2])["file_name"] == "slides.htm"  # no 99.1: first EX-99
    assert press_release(documents[:1]) is None


# ---- HTML to text ------------------------------------------------------

def test_html_to_lines_keeps_sentences_whole_and_drops_noise():
    html = """<html><head><title>x</title><style>p{}</style></head><body>
      <ix:header><ix:hidden>0001234 hidden fact</ix:hidden></ix:header>
      <div style="display:none">invisible</div>
      <p>We design <b>analog</b> chips&nbsp;for <span>cars</span>.</p>
      <div>Second<br>block</div>
      <table><tr><td>Revenue</td><td>100</td></tr></table>
      <p>12</p><p>Table of Contents</p>
      <script>alert(1)</script></body></html>"""
    assert html_to_lines(html) == ["We design analog chips for cars.", "Second", "block", "Revenue 100"]


def test_emphasized_lines_finds_bold_sentences_in_order():
    html = """<body>
      <p><b>Short bold</b></p>
      <p><span style="font-weight:700">Our revenue depends on a small number of customers and could decline.</span></p>
      <p>Plain paragraph that is long enough to count but is not bold at all, so it is ignored here.</p>
      <p><strong>We face intense competition in every market we serve and may lose share.</strong></p>
      <p><span style="font-weight: bold">We face intense competition in every market we serve and may lose share.</span></p>
    </body>"""
    assert emphasized_lines(html) == [
        "Our revenue depends on a small number of customers and could decline.",
        "We face intense competition in every market we serve and may lose share.",
    ]


def ten_k_lines():
    para = "Sentence about the business that goes on for a while to make the paragraph long. " * 10
    return (
        ["FORM 10-K", "Item 1. Business 3", "Item 1A. Risk Factors 9", "Item 7. Management's Discussion 30", "PART I"]
        + ["Item 1. Business"] + [para] * 4
        + ["Item 1A. Risk Factors"] + ["We depend on a few customers.", para] * 6
        + ["Item 1B. Unresolved Staff Comments", "None."]
        + ["Item 2. Properties", "We lease offices."]
        + ["ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"] + [para] * 8
        + ["Item 7A. Quantitative and Qualitative Disclosures About Market Risk", "Rates."]
        + ["Item 8. Financial Statements", para]
    )


def test_split_10k_skips_the_table_of_contents():
    sections = split_10k(ten_k_lines())
    assert all(s.ok for s in sections.values())
    assert sections["business"].text.count("Sentence about the business") == 40
    assert sections["risk_factors"].text.startswith("We depend on a few customers.")
    assert "Unresolved Staff Comments" not in sections["risk_factors"].text
    assert sections["mdna"].chars > 5000 and "Rates." not in sections["mdna"].text


def test_split_10k_reports_missing_and_stub_sections():
    lines = ["Item 1. Business", "See our annual report to shareholders.", "Item 1A. Risk Factors", "None."]
    sections = split_10k(lines)
    assert not sections["business"].ok and "incorporated by reference" in sections["business"].problem
    assert not sections["mdna"].ok and sections["mdna"].problem == "no 'Item 7' heading found"


def test_item_heading_must_start_the_line():
    lines = ten_k_lines()
    lines.insert(8, "As discussed in Item 7 below, results vary.")  # prose inside Item 1
    assert split_10k(lines)["business"].text.count("Sentence about the business") == 40


def test_risk_headings_are_bold_lines_inside_the_risk_section():
    emphasized = ["A bold line from the business section.", "We depend on a few customers.", "Not a sentence"]
    risk_text = "We depend on a few customers.\nDetail...\nNot a sentence"
    assert risk_headings(risk_text, emphasized) == ["We depend on a few customers."]
    many = [f"Risk number {i} could hurt us." for i in range(12)]
    assert len(risk_headings("\n".join(many), many, limit=5)) == 5


PRESS_RELEASE = [
    "Exhibit 99.1",
    "Test Semi Reports Record Fourth Quarter and Fiscal 2026 Revenue",
    "SAN JOSE, Calif., Jan. 30, 2026 - Test Semi Inc. today reported revenue of $1.2 billion for the fourth quarter, up 30% from a year ago.",
    "Revenue 1,200",
    "\"Demand for our power products was exceptional,\" said the chief executive officer of Test Semi in a statement.",
    "Business Outlook",
    "For the first quarter of fiscal 2027, the company expects revenue of $1.25 billion, plus or minus $25 million.",
    "Gross margin is expected to be approximately 62 percent on a GAAP basis for the same period.",
    "CONFERENCE CALL",
    "The company will host a call at 2 p.m. Pacific time today to discuss these results with investors.",
    "This release contains forward-looking statements within the meaning of the Private Securities Litigation Reform Act.",
]


def test_press_release_headline_summary_and_outlook():
    sections = split_press_release(PRESS_RELEASE)
    assert sections["headline"].text == "Test Semi Reports Record Fourth Quarter and Fiscal 2026 Revenue"
    assert sections["summary"].text.startswith("SAN JOSE, Calif.")
    assert "Revenue 1,200" not in sections["summary"].text  # table cell, not prose
    assert "forward-looking statements" not in sections["summary"].text
    assert sections["outlook"].text.splitlines() == PRESS_RELEASE[6:8]
    assert all(s.ok for s in sections.values())


def test_headline_skips_contact_lines_at_the_top():
    lines = [
        "Mike Sullivan (408) 986-7977",
        "Vice President, Investor Relations",
        "investor_relations@testsemi.com",
        "Test Semi Corporation",
        "Test Semi Announces Third Quarter 2026 Results",
    ] + PRESS_RELEASE[2:]
    assert split_press_release(lines)["headline"].text == "Test Semi Announces Third Quarter 2026 Results"
    # With no results-style line, the first plausible line is used.
    assert split_press_release(["(408) 986-7977", "A Statement From Test Semi Corporation Today"])["headline"].text == (
        "A Statement From Test Semi Corporation Today")


def test_outlook_leaves_out_legal_boilerplate():
    lines = PRESS_RELEASE[:6] + [
        "The following statements are based on current expectations. These statements are forward-looking, and actual results may differ materially.",
    ] + PRESS_RELEASE[6:]
    outlook = split_press_release(lines)["outlook"].text
    assert "actual results may differ" not in outlook and "expects revenue of $1.25 billion" in outlook


def test_press_release_without_an_outlook_heading_falls_back_to_expectation_sentences():
    lines = [l for l in PRESS_RELEASE if l != "Business Outlook"]
    outlook = split_press_release(lines)["outlook"]
    assert outlook.ok and "expects revenue of $1.25 billion" in outlook.text
    none = split_press_release(PRESS_RELEASE[:3])["outlook"]
    assert not none.ok and none.problem == "no outlook or guidance language found"
    assert not split_press_release([])["headline"].ok


# ---- orchestration -----------------------------------------------------

def test_memo_config_loads_and_top_pair_tickers():
    cfg = load_memo_config()
    assert (cfg.ai_pairs, cfg.press_release_quarters, cfg.risk_headings) == (10, 4, 3)
    pairs = pd.DataFrame({
        "long": ["A", "A", "C", "E"], "short": ["B", "C", "D", "F"],
        "passes": [True, True, False, True], "rank": pd.array([2, 1, None, 3], dtype="Int64"),
    })
    assert top_pair_tickers(pairs, 2) == ["A", "C", "B"]
    assert top_pair_tickers(pairs, 10) == ["A", "C", "B", "E", "F"]
    assert top_pair_tickers(pairs) == ["A", "C", "B", "E", "F"]  # every passing pair


def test_select_documents_picks_main_10k_document_and_8k_submissions():
    docs = select_documents(filings(), {"TEST": CIK}, AS_OF, 2)
    assert docs[["kind", "accn", "file_name"]].values.tolist() == [
        ["annual_report", "K-26", "main26.htm"],
        ["earnings_release", "E-4", "E-4.txt"],
        ["earnings_release", "E-3", "E-3.txt"],
    ]
    assert docs["url"].iloc[0].endswith("/K-26-index.htm")


def test_fetch_then_extract_end_to_end(cache):
    docs = select_documents(filings(), {"TEST": CIK}, AS_OF, 2)
    para = "<p>" + "Sentence about the business that goes on for a while to make the paragraph long. " * 10 + "</p>"
    risk = '<p><b>We depend on a small number of customers for most of our revenue.</b></p>' + para
    ten_k = ("<html><body><p>Item 1. Business</p>" + para * 4 + "<p>Item 1A. Risk Factors</p>" + risk * 6
             + "<p>Item 2. Properties</p><p>Offices.</p><p>Item 7. Management's Discussion</p>" + para * 8
             + "<p>Item 8. Financial Statements</p></body></html>")
    release = "<html><body>" + "".join(f"<p>{line}</p>" for line in PRESS_RELEASE) + "</body></html>"
    submission = SUBMISSION.replace(
        "<html><body><p>Test Semi Reports Record Fourth Quarter Revenue</p></body></html>", release)
    fetcher = FakeFetcher({
        document_url(CIK, "K-26", "main26.htm"): ten_k.encode(),
        document_url(CIK, "E-4", "E-4.txt"): submission.encode(),
        # E-3 is missing on the server.
    })
    fetched, records = fetch_documents(fetcher, cache, docs)
    assert fetched["status"].tolist() == ["downloaded", "downloaded", "not_found"] and len(records) == 3

    sections = extract_sections(cache, fetched, MemoConfig())
    annual = sections[sections["kind"] == "annual_report"].set_index("section")
    assert annual["ok"].to_dict() == {"business": True, "risk_factors": True, "mdna": True, "risk_headings": True}
    assert annual.loc["risk_headings", "text"] == "We depend on a small number of customers for most of our revenue."
    good = sections[(sections["accn"] == "E-4")].set_index("section")
    assert good.loc["headline", "text"].startswith("Test Semi Reports Record")
    assert "expects revenue of $1.25 billion" in good.loc["outlook", "text"]
    missing = sections[sections["accn"] == "E-3"].iloc[0]
    assert (missing["section"], missing["ok"]) == ("document", False) and "not_found" in missing["problem"]


def test_8k_without_a_press_release_is_reported(cache):
    docs = select_documents(filings(), {"TEST": CIK}, AS_OF, 1)
    docs = docs[docs["kind"] == "earnings_release"]
    write_gz(cache.filing_path(CIK, "E-4", "E-4.txt"), SUBMISSION.split("<DOCUMENT>\n<TYPE>EX-99.2")[0].encode())
    row = extract_sections(cache, docs).iloc[0]
    assert not row["ok"] and row["problem"] == "no press release exhibit (EX-99) in the 8-K"
