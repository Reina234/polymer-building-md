from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GromacsFiles:
    itp: Path
    gro: Path
    top: Path | None = None

    def __post_init__(self) -> None:
        self._assert_exists(self.itp)
        self._assert_exists(self.gro)
        if self.top is not None:
            self._assert_exists(self.top)

    @staticmethod
    def _assert_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Expected GROMACS output not found: {path}")
