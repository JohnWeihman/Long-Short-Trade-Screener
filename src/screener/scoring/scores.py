"""Rank each metric within the peer group and combine the ranks into pillar
scores (valuation, quality, direction) and a composite.

A rank runs from 0 (worst in the group) to 1 (best); ties share the
average. Ranks, not raw values, are combined, so one extreme number cannot
dominate a score. Companies with no peer group are not ranked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from screener import config
from screener.scoring.metrics import RANKED

DEFAULT_SCORING = config.PROJECT_ROOT / "config" / "scoring.yaml"
RANK_COLUMNS = ["cik10", "ticker", "peer_group", "pillar", "metric", "used_metric", "value", "rank"]


class ScoringConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    lower_is_better: bool = False
    fallback: str | None = None


@dataclass(frozen=True)
class ScoringConfig:
    pillars: dict[str, tuple[MetricSpec, ...]]
    pillar_weights: dict[str, float] = field(default_factory=dict)
    min_group_size: int = 4
    min_metrics_per_pillar: int = 2
    min_pillars: int = 2


def parse_scoring_config(raw: dict) -> ScoringConfig:
    pillars_raw = raw.get("pillars")
    if not isinstance(pillars_raw, dict) or not pillars_raw:
        raise ScoringConfigError("scoring config needs a 'pillars' section")
    pillars: dict[str, tuple[MetricSpec, ...]] = {}
    seen: set[str] = set()
    for pillar, entries in pillars_raw.items():
        specs = []
        for entry in entries or []:
            unknown = set(entry) - {"metric", "lower_is_better", "fallback"}
            if unknown:
                raise ScoringConfigError(f"{pillar}: unknown keys {sorted(unknown)}")
            spec = MetricSpec(str(entry.get("metric")), bool(entry.get("lower_is_better", False)), entry.get("fallback"))
            for name in filter(None, (spec.metric, spec.fallback)):
                if name not in RANKED:
                    raise ScoringConfigError(f"{pillar}: '{name}' is not a metric (see scoring/metrics.py)")
            if spec.metric in seen:
                raise ScoringConfigError(f"'{spec.metric}' is listed twice")
            seen.add(spec.metric)
            specs.append(spec)
        if not specs:
            raise ScoringConfigError(f"pillar '{pillar}' has no metrics")
        pillars[str(pillar)] = tuple(specs)

    weights = {str(k): float(v) for k, v in (raw.get("pillar_weights") or {}).items()}
    weights = {pillar: weights.get(pillar, 1.0) for pillar in pillars}
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ScoringConfigError("pillar_weights must be non-negative and not all zero")
    return ScoringConfig(
        pillars, weights,
        int(raw.get("min_group_size", 4)), int(raw.get("min_metrics_per_pillar", 2)), int(raw.get("min_pillars", 2)),
    )


def load_scoring_config(path: Path = DEFAULT_SCORING) -> ScoringConfig:
    with open(path, encoding="utf-8") as handle:
        return parse_scoring_config(yaml.safe_load(handle) or {})


def percentile_rank(values: pd.Series, lower_is_better: bool = False) -> pd.Series:
    """0 = worst, 1 = best, ties share the average; blank stays blank.

    With fewer than two values there is nothing to rank against: all blank.
    """
    present = values.notna()
    n = int(present.sum())
    if n < 2:
        return pd.Series(np.nan, index=values.index)
    order = values.rank(method="average", ascending=True)  # 1 = smallest
    position = (order - 1) / (n - 1)
    return (1 - position if lower_is_better else position).where(present)


def rank_metrics(metrics: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """One row per (company, metric slot): the value, its rank within the
    peer group, and which metric was actually used (the fallback when the
    main one is missing for that company)."""
    placed = metrics[metrics["peer_group"].notna()]
    out = []
    for group, members in placed.groupby("peer_group", sort=True):
        for pillar, specs in cfg.pillars.items():
            for spec in specs:
                main = members[spec.metric]
                rank = percentile_rank(main, spec.lower_is_better)
                value, used = main.copy(), pd.Series(spec.metric, index=members.index).where(main.notna(), None)
                if spec.fallback:
                    backup = members[spec.fallback]
                    backup_rank = percentile_rank(backup, spec.lower_is_better)
                    use_backup = main.isna() & backup.notna()
                    rank = rank.where(~use_backup, backup_rank)
                    value = value.where(~use_backup, backup)
                    used = used.where(~use_backup, spec.fallback)
                out.append(pd.DataFrame({
                    "cik10": members["cik10"], "ticker": members["ticker"], "peer_group": group,
                    "pillar": pillar, "metric": spec.metric, "used_metric": used, "value": value, "rank": rank,
                }))
    if not out:
        return pd.DataFrame(columns=RANK_COLUMNS)
    return pd.concat(out, ignore_index=True)[RANK_COLUMNS]


def score(metric_ranks: pd.DataFrame, metrics: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """One row per company: pillar scores, composite, rank within the group."""
    base = metrics[["cik10", "ticker", "peer_group", "as_of"]].copy()
    sizes = base.groupby("peer_group")["ticker"].transform("size")
    base["group_size"] = sizes.where(base["peer_group"].notna()).astype("Int64")
    base["thin_group"] = (base["group_size"] < cfg.min_group_size).fillna(False).astype(bool)

    ranked = metric_ranks.dropna(subset=["rank"])
    for pillar in cfg.pillars:
        rows = ranked[ranked["pillar"] == pillar]
        stats = rows.groupby("ticker")["rank"].agg(["mean", "size"])
        enough = stats[stats["size"] >= cfg.min_metrics_per_pillar]
        base[pillar] = base["ticker"].map(enough["mean"])
        base[f"{pillar}_metrics"] = base["ticker"].map(stats["size"]).fillna(0).astype(int)

    pillars = list(cfg.pillars)
    weights = pd.Series(cfg.pillar_weights)[pillars]
    values = base[pillars]
    available = values.notna()
    weighted = values.fillna(0).mul(weights, axis=1).sum(axis=1)
    total_weight = available.mul(weights, axis=1).sum(axis=1)
    base["pillars_scored"] = available.sum(axis=1)
    usable = (base["pillars_scored"] >= cfg.min_pillars) & (total_weight > 0)
    base["composite"] = (weighted / total_weight.where(total_weight > 0)).where(usable)
    base["rank_in_group"] = (
        base.groupby("peer_group")["composite"].rank(method="min", ascending=False).astype("Int64")
    )
    return base
