"""Config loading.

Every threshold in this project lives in ``configs/*.yaml``. Loading goes
through here so that a missing key fails loudly at startup rather than
silently falling back to a default buried in a script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(KeyError):
    """A config key is missing or malformed."""


class Config:
    """Read-only nested view over a parsed YAML document.

    Lookups use slash-delimited paths and raise on anything missing. There is
    deliberately no ``.get(key, default)``: a default that lives in code rather
    than in the config file is exactly what this project is trying to avoid.
    """

    def __init__(self, data: dict[str, Any], source: str = "<dict>") -> None:
        self._data = data
        self.source = source

    @classmethod
    def load(cls, name: str, config_dir: Path | None = None) -> Config:
        """Load ``configs/<name>.yaml`` (``name`` may include the extension)."""
        directory = config_dir or CONFIG_DIR
        path = directory / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        with path.open() as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ConfigError(f"{path} did not parse to a mapping")
        return cls(data, source=str(path))

    def __getitem__(self, path: str) -> Any:
        node: Any = self._data
        for part in path.split("/"):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"{self.source}: missing key {path!r}")
            node = node[part]
        return node

    def section(self, path: str) -> Config:
        node = self[path]
        if not isinstance(node, dict):
            raise ConfigError(f"{self.source}: {path!r} is not a mapping")
        return Config(node, source=f"{self.source}:{path}")

    def __contains__(self, path: str) -> bool:
        try:
            self[path]
        except ConfigError:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        """A deep copy, for logging the exact config alongside a run."""
        import copy

        return copy.deepcopy(self._data)

    def resolve_path(self, path: str) -> Path:
        """Interpret a config value as a path relative to the repository root."""
        value = self[path]
        candidate = Path(value)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate


@dataclass(frozen=True)
class GrowthTemperatureBin:
    """One growth-temperature class, i.e. one control token.

    This is the OGT label of the source organism. It is not a decoding
    setting; see the terminology table in README.md.
    """

    name: str
    min_c: float
    max_c: float

    def contains(self, ogt_c: float) -> bool:
        """Half-open on the upper edge, so the bins tile the real line."""
        return self.min_c <= ogt_c < self.max_c


class GrowthTemperatureBinner:
    """Maps an OGT in Celsius to a bin name.

    Validates at construction that the bins are sorted, non-overlapping and
    gapless. A silent gap would drop sequences from the training set without
    anyone noticing, which is the class of bug this project is built to avoid.
    """

    def __init__(self, bins: list[GrowthTemperatureBin]) -> None:
        if not bins:
            raise ConfigError("no growth-temperature bins configured")
        ordered = sorted(bins, key=lambda b: b.min_c)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if earlier.max_c != later.min_c:
                raise ConfigError(
                    f"growth-temperature bins must tile without gaps or overlap: "
                    f"{earlier.name} ends at {earlier.max_c} but "
                    f"{later.name} starts at {later.min_c}"
                )
        self.bins = ordered

    @classmethod
    def from_config(cls, cfg: Config, path: str = "growth_temperature/bins") -> GrowthTemperatureBinner:
        raw = cfg[path]
        bins = []
        for entry in raw:
            try:
                bins.append(
                    GrowthTemperatureBin(
                        name=entry["name"],
                        min_c=float(entry["min"]),
                        max_c=float(entry["max"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                # A quoted infinity ("Infinity" instead of .inf) lands here
                # rather than becoming a string that compares as unequal.
                raise ConfigError(f"malformed growth-temperature bin {entry!r}: {exc}") from exc
        return cls(bins)

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.bins]

    def bin_for(self, ogt_c: float) -> str:
        value = float(ogt_c)
        if math.isnan(value):
            raise ValueError("OGT is NaN; label it or drop the row upstream")
        for bucket in self.bins:
            if bucket.contains(value):
                return bucket.name
        # Only reachable at +inf when the top bin is closed.
        raise ValueError(f"OGT {value} falls outside every configured bin")
