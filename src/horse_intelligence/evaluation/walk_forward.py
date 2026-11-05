
"""Validation chronologique glissante (walk-forward).

Chaque pli utilise exclusivement les courses antérieures à sa période de test.
Les métadonnées permettent de vérifier la provenance temporelle des prédictions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from horse_intelligence.models.ranking import FloatArray

ModelRunner = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], FloatArray]


@dataclass(frozen=True)
class Fold:
    train_dates: list[date]
    calib_dates: list[date]
    test_dates: list[date]

    @property
    def train_end(self) -> date:
        return self.calib_dates[-1]

    @property
    def base_end(self) -> date:
        return self.train_dates[-1]


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    predictions: pd.DataFrame
    errors: dict[str, list[str]] = field(default_factory=dict)


def make_folds(
    dates: list[date],
    min_train_days: int,
    test_days: int,
    calib_days: int,
) -> list[Fold]:
    dates = sorted(set(dates))

    if min_train_days <= calib_days:
        raise ValueError("min_train_days doit être supérieur à calib_days")
    if test_days < 1 or calib_days < 1:
        raise ValueError("test_days et calib_days doivent être positifs")

    folds = []
    start = min_train_days

    while start < len(dates):
        train = dates[:start]
        test = dates[start:start + test_days]

        folds.append(
            Fold(
                train_dates=train[:-calib_days],
                calib_dates=train[-calib_days:],
                test_dates=test,
            )
        )
        start += test_days

    return folds


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"race_id", "race_date", "num_pmu", "won"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Colonnes obligatoires absentes du dataset : {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Dataset vide")

    if frame.duplicated(["race_id", "num_pmu"]).any():
        raise ValueError("Doublons détectés sur race_id et num_pmu")

    if frame["race_date"].isna().any():
        raise ValueError("Dates de course manquantes")

    if not pd.api.types.is_object_dtype(frame["race_date"]):
        pass


def run_walk_forward(
    frame: pd.DataFrame,
    models: dict[str, ModelRunner],
    min_train_days: int = 21,
    test_days: int = 7,
    calib_days: int = 7,
) -> WalkForwardResult:

    _validate_frame(frame)

    labelled = frame[frame["won"].notna()].copy()

    folds = make_folds(
        list(labelled["race_date"].unique()),
        min_train_days,
        test_days,
        calib_days,
    )

    parts = []
    errors: dict[str, list[str]] = {}

    for i, fold in enumerate(folds):
        base = labelled[labelled["race_date"].isin(fold.train_dates)]
        calib = labelled[labelled["race_date"].isin(fold.calib_dates)]
        test = labelled[labelled["race_date"].isin(fold.test_dates)]

        if base.empty or calib.empty or test.empty:
            continue

        if not (
            base["race_date"].max() < calib["race_date"].min()
            and calib["race_date"].max() < test["race_date"].min()
        ):
            raise RuntimeError(
                "Chevauchement temporel interdit entre les jeux du pli"
            )

        out = test[
            ["race_id", "race_date", "num_pmu", "won"]
        ].copy()

        out["fold"] = i
        out["base_end"] = fold.base_end
        out["calib_end"] = fold.train_end
        out["test_start"] = fold.test_dates[0]
        out["test_end"] = fold.test_dates[-1]

        for name, runner in models.items():
            try:
                predictions = np.asarray(
                    runner(base, calib, test),
                    dtype=np.float64,
                )

                if predictions.ndim != 1 or len(predictions) != len(test):
                    raise ValueError(
                        "Le modèle a retourné un nombre de prédictions incorrect"
                    )

                if not np.isfinite(predictions).all():
                    raise ValueError(
                        "Le modèle a retourné des probabilités non finies"
                    )

                if np.any((predictions < 0) | (predictions > 1)):
                    raise ValueError(
                        "Probabilités hors de l'intervalle [0, 1]"
                    )

                out[name] = predictions

            except Exception as exc:
                errors.setdefault(name, []).append(
                    f"pli {i}: {type(exc).__name__}: {exc}"
                )
                out[name] = np.nan

        parts.append(out)

    predictions = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame()
    )

    return WalkForwardResult(
        folds=folds,
        predictions=predictions,
        errors=errors,
    )
