"""Shared data structures for the Neo4j XAI pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShapTokenFactor:
    """One token-level SHAP contribution."""

    token: str
    value: float
    rank: int
    direction: str


@dataclass(frozen=True)
class MappedConcept:
    """A SHAP token mapped to a clinical graph concept."""

    name: str
    label: str
    matched_alias: str
    shap_value: float


@dataclass(frozen=True)
class Recommendation:
    """Evidence-backed graph recommendation candidate."""

    name: str
    score: float
    concepts: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    resources: list[dict] = field(default_factory=list)
