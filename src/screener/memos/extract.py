"""Turn filing HTML into text and pull out the parts a memo needs.

10-K: Item 1 (Business), Item 1A (Risk Factors), Item 7 (Management's
Discussion and Analysis), plus the bold risk-factor headings.
Earnings press release: headline, opening summary, outlook paragraphs.

All of this is heuristic: filings have no fixed layout. Every extraction
reports whether it looks right (`ok`, `problem`) instead of failing quietly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline-XBRL filings declare themselves as XML but are HTML documents.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:  # lxml is much faster on multi-megabyte 10-Ks; the built-in parser also works
    import lxml  # noqa: F401
    PARSER = "lxml"
except ImportError:  # pragma: no cover
    PARSER = "html.parser"

BLOCK_TAGS = ["p", "div", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6", "title", "hr"]
CELL_TAGS = ["td", "th"]
DROP_TAGS = ["script", "style", "head", "ix:header", "ix:hidden"]

SECTION_ITEMS = {"business": "1", "risk_factors": "1A", "mdna": "7"}
MIN_SECTION_CHARS = {"business": 2_000, "risk_factors": 5_000, "mdna": 5_000}

_ITEM_HEADING = re.compile(r"^item\s+(\d{1,2}[a-c]?)\b", re.I)
_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(bold|[6-9]00)", re.I)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
_PAGE_FURNITURE = re.compile(r"^(\d{1,3}|table of contents|index to financial statements)$", re.I)
_OUTLOOK_HEADING = re.compile(r"\b(outlook|guidance|forward[- ]looking guidance)\b", re.I)
_OUTLOOK_SENTENCE = re.compile(
    r"\b(expects?|expected|anticipates?|outlook|guidance|forecast|we (?:now )?(?:see|project))\b", re.I)
_SAFE_HARBOR = re.compile(
    r"forward-looking statements|private securities litigation reform act|statements are forward-looking"
    r"|actual results (?:may|could) differ|difficult to estimate|necessary to reconcile", re.I)
_HEADLINE_WORDS = re.compile(r"\b(reports?|announces?|results|quarter|revenue|earnings)\b", re.I)
_CONTACT = re.compile(r"@|\(\d{3}\)\s*\d{3}|\d{3}[-.]\d{3}[-.]\d{4}|investor relations|^contacts?\b|vice president", re.I)


@dataclass(frozen=True)
class Section:
    name: str
    text: str
    ok: bool
    problem: str = ""

    @property
    def chars(self) -> int:
        return len(self.text)


def _clean(text: str) -> str:
    return re.sub(r"[ \t\r\f\v ​]+", " ", text).strip()


def _soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, PARSER)
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE):
        tag.decompose()
    return soup


def html_to_lines(html: str) -> list[str]:
    """The document as a list of paragraphs: one entry per block of text,
    with inline formatting flattened and page furniture removed."""
    soup = _soup(html)
    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    for tag in soup.find_all(CELL_TAGS):
        tag.insert_after(" ")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    lines = (_clean(line) for line in soup.get_text().split("\n"))
    return [line for line in lines if line and not _PAGE_FURNITURE.match(line)]


def emphasized_lines(html: str, min_chars: int = 40, max_chars: int = 400) -> list[str]:
    """Bold passages of sentence length, in document order: in a 10-K's risk
    section these are the risk-factor headings."""
    soup = _soup(html)
    found: dict[str, None] = {}
    is_bold = lambda tag: tag.name in ("b", "strong") or bool(_BOLD_STYLE.search(tag.get("style") or ""))  # noqa: E731
    for tag in soup.find_all(is_bold):  # one pass, so document order is kept
        text = _clean(tag.get_text(" "))
        if min_chars <= len(text) <= max_chars:
            found[text] = None
    return list(found)


def split_10k(lines: list[str]) -> dict[str, Section]:
    """Business, risk factors and MD&A from a 10-K's paragraphs.

    Item headings appear at least twice (table of contents, then the real
    section). For each wanted item, every heading occurrence is tried and
    the one followed by the most text before the next item heading wins.
    """
    headings = [
        (i, m.group(1).upper()) for i, line in enumerate(lines)
        if len(line) < 200 and (m := _ITEM_HEADING.match(line))
    ]
    sections = {}
    for name, item in SECTION_ITEMS.items():
        best: list[str] = []
        for n, (start, found) in enumerate(headings):
            if found != item:
                continue
            end = next((i for i, other in headings[n + 1:] if other != item), len(lines))
            body = lines[start + 1:end]
            if sum(map(len, body)) > sum(map(len, best)):
                best = body
        text = "\n".join(best)
        if not best:
            sections[name] = Section(name, "", False, f"no 'Item {item}' heading found")
        elif len(text) < MIN_SECTION_CHARS[name]:
            sections[name] = Section(name, text, False,
                                     f"only {len(text):,} characters after 'Item {item}' (incorporated by reference?)")
        else:
            sections[name] = Section(name, text, True)
    return sections


def risk_headings(risk_text: str, emphasized: list[str], limit: int = 8) -> list[str]:
    """The first `limit` bold headings that appear inside the risk section."""
    return [line for line in emphasized if line in risk_text and line.rstrip().endswith(".")][:limit]


def split_press_release(lines: list[str], summary_chars: int = 1_500, outlook_chars: int = 1_500) -> dict[str, Section]:
    """Headline, opening summary and outlook from an earnings press release."""
    body = [line for line in lines if not re.match(r"^(exhibit|ex-)\s*99", line, re.I)]
    if not body:
        empty = Section("headline", "", False, "empty document")
        return {"headline": empty, "summary": Section("summary", "", False, "empty document"),
                "outlook": Section("outlook", "", False, "empty document")}

    # Headline: the first line that reads like a results headline; press
    # releases often open with contact names, phone numbers and e-mail addresses.
    def plausible(line: str) -> bool:
        return 25 <= len(line) <= 250 and not _CONTACT.search(line)

    early = body[:25]
    start = next((i for i, line in enumerate(early) if plausible(line) and _HEADLINE_WORDS.search(line)), None)
    if start is None:
        start = next((i for i, line in enumerate(early) if plausible(line)), 0)
    headline = body[start]

    summary: list[str] = []
    for line in body[start + 1:]:
        if _SAFE_HARBOR.search(line) or sum(map(len, summary)) >= summary_chars:
            break
        if len(line) >= 60:  # prose and highlight bullets, not table cells
            summary.append(line)

    outlook: list[str] = []
    for i, line in enumerate(body):
        if len(line) < 80 and _OUTLOOK_HEADING.search(line):  # a heading such as "Business Outlook"
            for following in body[i + 1:i + 12]:
                if len(following) < 80 and following.isupper():
                    break
                if len(following) >= 60 and not _SAFE_HARBOR.search(following):
                    outlook.append(following)
            if outlook:
                break
    if not outlook:  # no heading: fall back to sentences that talk about expectations
        outlook = [line for line in body if len(line) >= 60 and _OUTLOOK_SENTENCE.search(line)
                   and not _SAFE_HARBOR.search(line)][:4]
    outlook_text = "\n".join(outlook)[:outlook_chars]

    return {
        "headline": Section("headline", headline, True),
        "summary": Section("summary", "\n".join(summary)[:summary_chars], bool(summary),
                           "" if summary else "no prose found after the headline"),
        "outlook": Section("outlook", outlook_text, bool(outlook),
                           "" if outlook else "no outlook or guidance language found"),
    }
