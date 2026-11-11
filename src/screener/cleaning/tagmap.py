"""Load and validate config/tag_map.yaml: standardized metric -> XBRL tags,
with optional formulas for when no single tag carries the metric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from screener import config

DEFAULT_TAG_MAP = config.PROJECT_ROOT / "config" / "tag_map.yaml"
KINDS = ("duration", "instant", "cover")
DEFAULT_TAXONOMY = "us-gaap"
COMBINE = ("sum", "average", "ratio")
FORMULA_KEYS = ("add", "subtract", "add_if_present", "subtract_if_present")


class TagMapError(ValueError):
    pass


@dataclass(frozen=True)
class Term:
    """One operand of a formula: another metric, or a raw XBRL tag."""

    is_metric: bool
    name: str  # metric name, or tag name
    taxonomy: str = ""  # for tags only

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class Formula:
    """value = sum(add) - sum(subtract), plus/minus the *_if_present terms
    that have data. Every add/subtract term must have data, and at least one
    term overall must have data."""

    add: tuple[Term, ...] = ()
    subtract: tuple[Term, ...] = ()
    add_if_present: tuple[Term, ...] = ()
    subtract_if_present: tuple[Term, ...] = ()

    def terms(self) -> tuple[Term, ...]:
        return self.add + self.subtract + self.add_if_present + self.subtract_if_present


@dataclass(frozen=True)
class Metric:
    name: str
    kind: str  # 'duration' | 'instant' | 'cover'
    unit: str
    tags: tuple[tuple[str, str], ...]  # (taxonomy, tag), in priority order
    optional: bool = False
    formulas: tuple[Formula, ...] = ()  # tried in order, only if no tag has data
    combine: str = "sum"  # how quarters combine into longer periods: sum | average | ratio
    ratio_of: tuple[str, str] | None = None  # (numerator metric, denominator metric) for 'ratio'


def _split_tag(raw: str) -> tuple[str, str]:
    taxonomy, _, tag = raw.strip().rpartition(":")
    return (taxonomy or DEFAULT_TAXONOMY, tag)


def _parse_term(raw: str, metric_names: set[str], where: str) -> Term:
    text = str(raw).strip()
    if text in metric_names:
        return Term(True, text)
    taxonomy, tag = _split_tag(text)
    if not tag or tag[0].islower():
        # Tags are CamelCase; a lowercase term is a misspelled metric name.
        raise TagMapError(f"{where}: '{text}' is not a metric in the tag map")
    return Term(False, tag, taxonomy)


def _parse_formula(raw: object, metric_names: set[str], where: str) -> Formula:
    if not isinstance(raw, dict) or not raw:
        raise TagMapError(f"{where}: a formula must be a mapping")
    unknown = set(raw) - set(FORMULA_KEYS)
    if unknown:
        raise TagMapError(f"{where}: unknown formula keys {sorted(unknown)}")
    parts = {
        key: tuple(_parse_term(t, metric_names, where) for t in raw.get(key) or [])
        for key in FORMULA_KEYS
    }
    formula = Formula(**parts)
    if not formula.terms():
        raise TagMapError(f"{where}: formula has no terms")
    return formula


def _check_no_cycles(metrics: dict[str, Metric]) -> None:
    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in path:
            raise TagMapError("formulas refer to each other in a loop: " + " -> ".join(path + (name,)))
        for formula in metrics[name].formulas:
            for term in formula.terms():
                if term.is_metric:
                    visit(term.name, path + (name,))

    for name in metrics:
        visit(name, ())


def parse_tag_map(raw: dict) -> dict[str, Metric]:
    """Validated metrics, in file order. Raises TagMapError on any problem."""
    if not isinstance(raw, dict) or not raw:
        raise TagMapError("tag map is empty")
    names = {str(n) for n in raw}
    metrics: dict[str, Metric] = {}
    owner: dict[tuple[str, str], str] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise TagMapError(f"{name}: expected a mapping")
        unknown = set(spec) - {"kind", "unit", "tags", "optional", "formulas", "combine", "ratio_of"}
        if unknown:
            raise TagMapError(f"{name}: unknown keys {sorted(unknown)}")
        if spec.get("kind") not in KINDS:
            raise TagMapError(f"{name}: kind must be one of {KINDS}")
        if not spec.get("unit"):
            raise TagMapError(f"{name}: unit is required")
        tags = tuple(_split_tag(str(t)) for t in spec.get("tags") or [])
        formulas = tuple(_parse_formula(f, names, str(name)) for f in spec.get("formulas") or [])
        if any(not tag for _, tag in tags) or not (tags or formulas):
            raise TagMapError(f"{name}: at least one tag or formula is required")
        for key in tags:
            if key in owner:
                # One tag feeding two metrics directly would double count it.
                raise TagMapError(f"{':'.join(key)} is listed under both {owner[key]} and {name}")
            owner[key] = name
        combine = spec.get("combine", "sum")
        if combine not in COMBINE:
            raise TagMapError(f"{name}: combine must be one of {COMBINE}")
        ratio_of = spec.get("ratio_of")
        if (combine == "ratio") != (ratio_of is not None):
            raise TagMapError(f"{name}: 'combine: ratio' and 'ratio_of' go together")
        if ratio_of is not None:
            if not isinstance(ratio_of, list) or len(ratio_of) != 2 or not set(map(str, ratio_of)) <= names - {str(name)}:
                raise TagMapError(f"{name}: ratio_of must name two other metrics")
            ratio_of = (str(ratio_of[0]), str(ratio_of[1]))
        metrics[str(name)] = Metric(
            str(name), spec["kind"], str(spec["unit"]), tags, bool(spec.get("optional", False)),
            formulas, combine, ratio_of,
        )
    _check_no_cycles(metrics)
    return metrics


def load_tag_map(path: Path = DEFAULT_TAG_MAP) -> dict[str, Metric]:
    with open(path, encoding="utf-8") as handle:
        return parse_tag_map(yaml.safe_load(handle))
