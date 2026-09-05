"""Hermes Trading Bot - Strategy.

Represents and validates the trading strategy configuration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class Strategy:
    """Trading strategy with deterministic rules."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default strategy: RSI mean-reversion
        self.version = "0001"
        self.indicators: Dict[str, Dict] = {}
        self.entry: Dict[str, Any] = {}
        self.exit: Dict[str, Any] = {}
        self.position: Dict[str, Any] = {}
        self.filters: Dict[str, Any] = {}

        if config:
            self.from_dict(config)

    def from_dict(self, config: Dict[str, Any]) -> None:
        """Load strategy from dictionary."""
        self.version = config.get("version", "0001")
        self.indicators = config.get("indicators", {})
        self.entry = config.get("entry", {})
        self.exit = config.get("exit", {})
        self.position = config.get("position", {})
        self.filters = config.get("filters", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary."""
        return {
            "version": self.version,
            "indicators": self.indicators,
            "entry": self.entry,
            "exit": self.exit,
            "position": self.position,
            "filters": self.filters,
        }

    def to_yaml(self) -> str:
        """Convert strategy to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @staticmethod
    def load_from_file(path: Path) -> "Strategy":
        """Load strategy from YAML file, creating default if missing."""
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            default_strat = {
                "version": "01",
                "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
                "exit": {"take_profit_pct": 4.0, "stop_loss_pct": 2.0},
                "position": {"size_pct": 5.0},
                "stop_loss_pct": 2.0,
                "position_size_r": 0.5
            }
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(default_strat, f, default_flow_style=False, sort_keys=False)
            s = Strategy()
            s.from_dict(default_strat)
            return s

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        s = Strategy()
        s.from_dict(config)
        return s

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate strategy has all required components.

        Returns:
            (is_valid, errors) tuple
        """
        errors: List[str] = []

        # Check entry conditions
        if not self.entry:
            errors.append("Entry conditions not defined")

        # Check exit conditions
        if not self.exit:
            errors.append("Exit conditions not defined")

        # Check position sizing
        if not self.position:
            errors.append("Position sizing not defined")
        else:
            size_pct = self.position.get("size_pct", None)
            if size_pct is not None and (size_pct <= 0 or size_pct > 100):
                errors.append(f"Invalid position size pct: {size_pct}")

        # Check indicators
        if not self.indicators:
            errors.append("No indicators defined")

        is_valid = len(errors) == 0
        return is_valid, errors