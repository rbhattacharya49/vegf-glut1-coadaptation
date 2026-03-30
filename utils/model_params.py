"""Shared parameter dataclasses and small helpers for the VEGF-GLUT1 model.

This module centralizes default parameter values used across the project.
Keeping these dataclasses in one place makes the notebooks easier to read,
reduces duplication, and ensures that parameter changes propagate cleanly
across fixed-neighborhood, evolving-neighborhood, and therapy simulations.

Main classes
------------
BaseParams
    Shared biological and numerical parameters.
FixedNParams
    Parameter set for the fixed-neighborhood model.
EvolvingParams
    Parameter set for models where neighborhood size N evolves in time.

Main helpers
------------
fixedN_defaults(), evolving_defaults()
    Convenience constructors with optional keyword overrides.
apply_overrides()
    Create a modified copy of a parameter dataclass.
to_dict()
    Convert a parameter dataclass to a plain dictionary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BaseParams:
    """Parameters shared across all model variants.

    Notes
    -----
    The defaults here are project-wide defaults. Individual notebooks can
    override any of these values when defining a simulation scenario.
    """

    # Biological / fitness parameters
    b: float = 1.0
    R: float = 100.0
    a: float = 0.05
    k: float = 0.05
    c: float = 0.01
    h: float = 0.01
    f: float = 0.005

    # Trait evolvabilities
    sigma_u: float = 0.005
    sigma_y: float = 0.005

    # Solver controls
    t_max: float = 30000.0
    atol: float = 1e-9
    rtol: float = 1e-7
    max_step: float = 50.0

    # Therapy parameters (defaults correspond to therapy OFF)
    e: float = 1.0
    d: float = 1.0
    m_G: float = 0.0
    m_A: float = 0.0
    t_drug: float = float("inf")


@dataclass(frozen=True)
class FixedNParams(BaseParams):
    """Parameter set for the fixed-neighborhood model."""


@dataclass(frozen=True)
class EvolvingParams(BaseParams):
    """Parameter set for models with evolving neighborhood size N(t)."""

    N_floor: float = 1.0
    u_eps: float = 1e-8


def fixedN_defaults(**overrides: Any) -> FixedNParams:
    """Return a FixedNParams instance with optional overrides."""
    return replace(FixedNParams(), **overrides) if overrides else FixedNParams()


def evolving_defaults(**overrides: Any) -> EvolvingParams:
    """Return an EvolvingParams instance with optional overrides."""
    return replace(EvolvingParams(), **overrides) if overrides else EvolvingParams()


def apply_overrides(p: T, **overrides: Any) -> T:
    """Return a modified copy of parameter object ``p``."""
    return replace(p, **overrides)


def to_dict(p: Any) -> Dict[str, Any]:
    """Convert a parameter dataclass to a plain dictionary."""
    return asdict(p)


try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def load_yaml_overrides(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load optional parameter overrides from a YAML file.

    The file is expected to have sections such as ``base``, ``fixedN``, and
    ``evolving``. Missing files or a missing PyYAML installation are handled
    gracefully by returning an empty dictionary.
    """
    if yaml is None:
        return {}

    path = path or Path("configs/params.yaml")
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return {k: (v or {}) for k, v in data.items() if isinstance(v, dict)}


def fixedN_defaults_from_file(path: Optional[Path] = None, **extra: Any) -> FixedNParams:
    """Return FixedNParams using optional YAML overrides plus keyword overrides."""
    overrides = load_yaml_overrides(path).get("base", {})
    overrides.update(load_yaml_overrides(path).get("fixedN", {}))
    overrides.update(extra)
    return replace(FixedNParams(), **overrides) if overrides else FixedNParams()


def evolving_defaults_from_file(path: Optional[Path] = None, **extra: Any) -> EvolvingParams:
    """Return EvolvingParams using optional YAML overrides plus keyword overrides."""
    overrides = load_yaml_overrides(path).get("base", {})
    overrides.update(load_yaml_overrides(path).get("evolving", {}))
    overrides.update(extra)
    return replace(EvolvingParams(), **overrides) if overrides else EvolvingParams()
