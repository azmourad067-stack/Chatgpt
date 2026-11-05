
import pytest

from horse_intelligence.prediction.decision import (
    DecisionConfig,
    decide_simple_gagnant,
)


def make_decision(**overrides):
    data = {
        "num_pmu": [1, 2, 3],
        "p_win": [0.60, 0.25, 0.15],
        "p_low": [0.50, 0.20, 0.10],
        "market_prob": [0.30, 0.40, 0.30],
        "odds": [2.50, 3.00, 4.00],
        "data_status": "verified",
        "model_status": "active",
        "in_sample": False,
        "uncertainty": "exploitable",
    }
    data.update(overrides)
    return decide_simple_gagnant(**data)


def test_detects_conservative_betting_opportunity():
    result = make_decision()

    assert result.decision == "BET"
    assert result.num_pmu == 1
    assert result.p_low == pytest.approx(0.50)
    assert result.conservative_edge == pytest.approx(0.20)
    assert result.conservative_ev == pytest.approx(0.25)


def test_refuses_when_no_runner_has_sufficient_edge():
    result = make_decision(
        p_low=[0.25, 0.20, 0.10],
    )

    assert result.decision == "NO_BET"
    assert result.num_pmu is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"data_status": "unverified"},
        {"model_status": "inactive"},
        {"in_sample": True},
        {"uncertainty": "high"},
    ],
)
def test_returns_insufficient_data_when_conditions_are_not_met(overrides):
    result = make_decision(**overrides)

    assert result.decision == "INSUFFICIENT_DATA"
    assert result.num_pmu is None


def test_rejects_non_normalized_model_probabilities():
    result = make_decision(
        p_win=[0.60, 0.30, 0.15],
    )

    assert result.decision == "INSUFFICIENT_DATA"


def test_rejects_non_normalized_market_probabilities():
    result = make_decision(
        market_prob=[0.50, 0.40, 0.30],
    )

    assert result.decision == "INSUFFICIENT_DATA"


def test_rejects_duplicate_runner_numbers():
    result = make_decision(
        num_pmu=[1, 1, 3],
    )

    assert result.decision == "INSUFFICIENT_DATA"


def test_rejects_invalid_odds():
    result = make_decision(
        odds=[2.50, 1.00, 4.00],
    )

    assert result.decision == "INSUFFICIENT_DATA"


def test_rejects_inconsistent_probability_bounds():
    result = make_decision(
        p_low=[0.70, 0.20, 0.10],
    )

    assert result.decision == "INSUFFICIENT_DATA"


def test_rejects_arrays_with_different_lengths():
    result = make_decision(
        odds=[2.50, 3.00],
    )

    assert result.decision == "INSUFFICIENT_DATA"


def test_rejects_missing_values():
    result = make_decision(
        p_low=[0.50, float("nan"), 0.10],
    )

    assert result.decision == "INSUFFICIENT_DATA"


def test_custom_thresholds_can_prevent_a_bet():
    config = DecisionConfig(
        min_edge=0.30,
        min_expected_value=0.20,
    )

    result = make_decision(config=config)

    assert result.decision == "NO_BET"
