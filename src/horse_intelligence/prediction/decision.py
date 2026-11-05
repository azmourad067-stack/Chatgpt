
"""Décision SIMPLE_GAGNANT avec abstention conservatrice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Decision = Literal["BET", "NO_BET", "INSUFFICIENT_DATA"]


@dataclass(frozen=True)
class DecisionConfig:
    min_edge: float = 0.02
    min_expected_value: float = 0.03
    probability_sum_tolerance: float = 0.02


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    num_pmu: int | None
    reason: str
    p_win: float | None = None
    p_low: float | None = None
    market_prob: float | None = None
    odds: float | None = None
    conservative_edge: float | None = None
    conservative_ev: float | None = None


def decide_simple_gagnant(
    *,
    num_pmu: list[int],
    p_win: list[float],
    p_low: list[float],
    market_prob: list[float],
    odds: list[float],
    data_status: str,
    model_status: str,
    in_sample: bool,
    uncertainty: str,
    config: DecisionConfig | None = None,
) -> DecisionResult:

    cfg = config or DecisionConfig()

    if data_status != "verified":
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Les données de la course ne sont pas entièrement vérifiées.",
        )

    if model_status != "active":
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Le modèle n'est pas activé et validé.",
        )

    if in_sample:
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "La course appartient à la période d'apprentissage.",
        )

    if uncertainty != "exploitable":
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "L'incertitude du modèle ne permet pas une décision fiable.",
        )

    arrays = [
        np.asarray(num_pmu),
        np.asarray(p_win, dtype=float),
        np.asarray(p_low, dtype=float),
        np.asarray(market_prob, dtype=float),
        np.asarray(odds, dtype=float),
    ]

    if not arrays[0].size or any(a.ndim != 1 for a in arrays):
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Les données des partants sont absentes ou invalides.",
        )

    if len({a.size for a in arrays}) != 1:
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Les tableaux des partants ont des dimensions différentes.",
        )

    numbers, p, low, market, prices = arrays

    if not all(np.all(np.isfinite(a)) for a in arrays):
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Une ou plusieurs valeurs sont manquantes ou non finies.",
        )

    if len(np.unique(numbers)) != len(numbers):
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Les numéros PMU ne sont pas uniques.",
        )

    if (
        np.any(numbers <= 0)
        or np.any(p < 0)
        or np.any(p > 1)
        or np.any(low < 0)
        or np.any(low > p)
        or np.any(market <= 0)
        or np.any(market > 1)
        or np.any(prices <= 1)
    ):
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Les probabilités, bornes ou cotes sont incohérentes.",
        )

    if abs(float(p.sum()) - 1.0) > cfg.probability_sum_tolerance:
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Les probabilités du modèle ne sont pas normalisées.",
        )

    if abs(float(market.sum()) - 1.0) > cfg.probability_sum_tolerance:
        return DecisionResult(
            "INSUFFICIENT_DATA", None,
            "Les probabilités du marché ne sont pas normalisées.",
        )

    edge = low - market
    ev = low * prices - 1.0

    eligible = (
        (edge >= cfg.min_edge)
        & (ev >= cfg.min_expected_value)
    )

    if not np.any(eligible):
        return DecisionResult(
            "NO_BET", None,
            "Aucun partant ne présente un avantage conservateur suffisant.",
        )

    candidates = np.where(eligible)[0]
    best = int(candidates[np.argmax(ev[candidates])])

    return DecisionResult(
        decision="BET",
        num_pmu=int(numbers[best]),
        reason="Avantage conservateur détecté : les deux seuils sont atteints.",
        p_win=float(p[best]),
        p_low=float(low[best]),
        market_prob=float(market[best]),
        odds=float(prices[best]),
        conservative_edge=float(edge[best]),
        conservative_ev=float(ev[best]),
    )
