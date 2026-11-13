"""Load config/pairing.yaml."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from screener import config

DEFAULT_PAIRING = config.PROJECT_ROOT / "config" / "pairing.yaml"
BENCHMARKS = ("universe_equal_weight",)


class PairingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PairingConfig:
    min_composite_gap: float = 0.25
    min_pillars_favoring_long: int = 2
    correlation_window_weeks: int = 104
    min_overlap_weeks: int = 52
    min_correlation: float = 0.50
    beta_benchmark: str = "universe_equal_weight"
    spread_history_days: int = 756
    spread_zscore_window_days: int = 252
    exclude_pending_targets: bool = True


def parse_pairing_config(raw: dict) -> PairingConfig:
    known = {f.name for f in fields(PairingConfig)}
    unknown = set(raw) - known
    if unknown:
        raise PairingConfigError(f"unknown pairing settings: {sorted(unknown)}")
    cfg = PairingConfig(**raw)
    if cfg.beta_benchmark not in BENCHMARKS:
        raise PairingConfigError(f"beta_benchmark must be one of {BENCHMARKS}")
    if not 0 <= cfg.min_composite_gap <= 1 or not -1 <= cfg.min_correlation <= 1:
        raise PairingConfigError("min_composite_gap must be in [0, 1] and min_correlation in [-1, 1]")
    if cfg.min_overlap_weeks > cfg.correlation_window_weeks:
        raise PairingConfigError("min_overlap_weeks cannot exceed correlation_window_weeks")
    return cfg


def load_pairing_config(path: Path = DEFAULT_PAIRING) -> PairingConfig:
    with open(path, encoding="utf-8") as handle:
        return parse_pairing_config(yaml.safe_load(handle) or {})
