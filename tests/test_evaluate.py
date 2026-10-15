"""Tests for the evaluation module, using hand-computed fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from vsm.evaluate import (
    NONE,
    bootstrap_macro_f1_ci,
    confusion_matrix,
    count_matches,
    macro_f1_at,
    macro_micro,
    nominal_metrics,
    per_class_metrics,
    predicted_classes,
    threshold_sweep,
    top_confusions,
)
from vsm.taxonomy import HAZARD_CLASSES


def rec(frame_id: str, *labels: tuple[str, float]) -> dict:
    """Build a prediction record: ``rec("f1", ("road_change", 0.9))``."""
    return {
        "frame_id": frame_id,
        "labels": [
            {"class_name": name, "confidence": conf, "evidence": "test", "bbox_hint": None}
            for name, conf in labels
        ],
    }


# --------------------------------------------------------------------------------------
# TP / FP / FN hand-computed cases
# --------------------------------------------------------------------------------------


def test_known_tp_fp_fn_counts() -> None:
    predictions = [
        rec("f1", ("road_change", 0.9)),  # TP road_change
        rec("f2", ("construction_zone", 0.8)),  # FP construction_zone, FN road_change
        rec("f3"),  # FN construction_zone
        rec("f4", ("construction_zone", 0.9)),  # TP construction_zone
    ]
    gt = {
        "f1": ["road_change"],
        "f2": ["road_change"],
        "f3": ["construction_zone"],
        "f4": ["construction_zone"],
    }
    counts = count_matches(predictions, gt, threshold=0.5)
    assert counts.tp["road_change"] == 1
    assert counts.fp["road_change"] == 0
    assert counts.fn["road_change"] == 1
    assert counts.tp["construction_zone"] == 1
    assert counts.fp["construction_zone"] == 1
    assert counts.fn["construction_zone"] == 1
    assert counts.support("road_change") == 2
    assert counts.support("construction_zone") == 2

    per_class = per_class_metrics(counts)
    # road_change: P=1.0 R=0.5 -> F1 = 2*1*0.5/1.5 = 0.6667
    assert per_class["road_change"]["precision"] == pytest.approx(1.0)
    assert per_class["road_change"]["recall"] == pytest.approx(0.5)
    assert per_class["road_change"]["f1"] == pytest.approx(2 / 3)
    # construction_zone: P=0.5 R=0.5 -> F1 = 0.5
    assert per_class["construction_zone"]["f1"] == pytest.approx(0.5)


def test_perfect_predictions_give_one() -> None:
    gt = {f"f{i}": [name] for i, name in enumerate(HAZARD_CLASSES)}
    predictions = [rec(fid, (classes[0], 0.95)) for fid, classes in gt.items()]
    counts = count_matches(predictions, gt, threshold=0.5)
    per_class = per_class_metrics(counts)
    for name in HAZARD_CLASSES:
        assert per_class[name]["precision"] == pytest.approx(1.0)
        assert per_class[name]["recall"] == pytest.approx(1.0)
        assert per_class[name]["f1"] == pytest.approx(1.0)
    agg = macro_micro(counts)
    assert agg["macro_f1"] == pytest.approx(1.0)
    assert agg["micro_f1"] == pytest.approx(1.0)


def test_empty_predictions_give_zero_recall() -> None:
    gt = {"f1": ["road_change"], "f2": ["emergency_vehicle"]}
    predictions = [rec("f1"), rec("f2")]
    counts = count_matches(predictions, gt, threshold=0.5)
    per_class = per_class_metrics(counts)
    assert per_class["road_change"]["recall"] == 0.0
    assert per_class["road_change"]["f1"] == 0.0
    assert per_class["road_change"]["precision"] == 0.0  # 0/0 defined as 0
    agg = macro_micro(counts)
    assert agg["macro_f1"] == 0.0
    assert agg["tp"] == 0
    assert agg["fn"] == 2


def test_all_false_positives_penalise_precision_only() -> None:
    gt = {"f1": [], "f2": []}
    predictions = [rec("f1", ("road_change", 0.9)), rec("f2", ("construction_zone", 0.9))]
    counts = count_matches(predictions, gt, threshold=0.5)
    agg = macro_micro(counts)
    assert agg["tp"] == 0
    assert agg["fp"] == 2
    assert agg["fn"] == 0
    assert agg["macro_precision"] == 0.0
    assert agg["macro_recall"] == 0.0


def test_below_threshold_predictions_are_ignored() -> None:
    gt = {"f1": ["road_change"]}
    predictions = [rec("f1", ("road_change", 0.49))]
    counts = count_matches(predictions, gt, threshold=0.5)
    assert counts.tp["road_change"] == 0
    assert counts.fn["road_change"] == 1
    assert predicted_classes(predictions[0], 0.49) == ["road_change"]


def test_multi_label_frame_counts_both_classes() -> None:
    gt = {"f1": ["road_change", "construction_zone"]}
    predictions = [rec("f1", ("road_change", 0.9), ("construction_zone", 0.8))]
    counts = count_matches(predictions, gt, threshold=0.5)
    assert counts.tp["road_change"] == 1
    assert counts.tp["construction_zone"] == 1
    assert counts.total_support() == 2


def test_frames_without_predictions_are_excluded() -> None:
    gt = {"f1": ["road_change"], "f2": ["road_change"], "f3": []}
    predictions = [rec("f1", ("road_change", 0.9))]
    counts = count_matches(predictions, gt, threshold=0.5)
    assert counts.n_frames == 1
    assert counts.total_support() == 1


# --------------------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------------------


def test_confusion_row_sums_equal_support() -> None:
    gt = {
        "f1": ["road_change"],
        "f2": ["construction_zone"],
        "f3": [],
        "f4": ["emergency_vehicle"],
        "f5": ["occluded_pedestrian"],
    }
    predictions = [
        rec("f1", ("road_change", 0.9)),
        rec("f2", ("road_change", 0.7)),  # confused with road_change
        rec("f3", ("construction_zone", 0.8)),  # false fire on nominal
        rec("f4"),  # missed
        rec("f5", ("occluded_pedestrian", 0.6)),
    ]
    matrix, labels = confusion_matrix(predictions, gt, threshold=0.5)
    assert labels == list(HAZARD_CLASSES) + [NONE]
    assert matrix.shape == (5, 5)

    # Row sums must equal the number of GT frames whose primary label is that row.
    gt_primary = {
        "road_change": 1,
        "construction_zone": 1,
        NONE: 1,
        "emergency_vehicle": 1,
        "occluded_pedestrian": 1,
    }
    for i, label in enumerate(labels):
        assert matrix[i].sum() == gt_primary[label], label
    assert matrix.sum() == len(predictions)

    # Spot-check cells.
    assert matrix[labels.index("road_change"), labels.index("road_change")] == 1
    assert matrix[labels.index("construction_zone"), labels.index("road_change")] == 1
    assert matrix[labels.index(NONE), labels.index("construction_zone")] == 1
    assert matrix[labels.index("emergency_vehicle"), labels.index(NONE)] == 1

    confusions = top_confusions(matrix, labels, k=3)
    pairs = {(c["gt"], c["predicted"]) for c in confusions}
    assert ("construction_zone", "road_change") in pairs
    assert (NONE, "construction_zone") in pairs
    assert ("emergency_vehicle", NONE) in pairs


def test_confusion_matrix_uses_highest_confidence_label() -> None:
    gt = {"f1": ["road_change"]}
    predictions = [rec("f1", ("road_change", 0.6), ("construction_zone", 0.95))]
    matrix, labels = confusion_matrix(predictions, gt, threshold=0.5)
    assert matrix[labels.index("road_change"), labels.index("construction_zone")] == 1


def test_confusion_matrix_nominal_diagonal() -> None:
    gt = {"f1": [], "f2": []}
    predictions = [rec("f1"), rec("f2")]
    matrix, labels = confusion_matrix(predictions, gt, threshold=0.5)
    assert matrix[labels.index(NONE), labels.index(NONE)] == 2


# --------------------------------------------------------------------------------------
# Threshold sweep / nominal metrics
# --------------------------------------------------------------------------------------


def test_threshold_recall_is_monotone_non_increasing() -> None:
    gt = {f"f{i}": ["road_change"] for i in range(10)}
    predictions = [rec(f"f{i}", ("road_change", 0.1 + 0.09 * i)) for i in range(10)]
    sweep = threshold_sweep(predictions, gt, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
    recalls = [p["recall"] for p in sweep["per_class"]["road_change"]]
    assert recalls == sorted(recalls, reverse=True)
    # Raising the threshold can only remove predictions, so recall cannot increase.
    for earlier, later in zip(recalls, recalls[1:], strict=False):
        assert later <= earlier + 1e-12
    macro = [p["macro_f1"] for p in sweep["macro"]]
    assert len(macro) == 5
    assert sweep["thresholds"] == [0.1, 0.3, 0.5, 0.7, 0.9]


def test_threshold_sweep_is_flat_for_well_separated_confidences() -> None:
    gt = {"f1": ["road_change"], "f2": ["road_change"]}
    predictions = [rec("f1", ("road_change", 0.9)), rec("f2", ("road_change", 0.8))]
    sweep = threshold_sweep(predictions, gt, thresholds=[0.1, 0.5, 0.7])
    f1s = [p["f1"] for p in sweep["per_class"]["road_change"]]
    assert f1s == [1.0, 1.0, 1.0]


def test_nominal_metrics_detect_false_fires() -> None:
    gt = {"f1": [], "f2": [], "f3": ["road_change"]}
    predictions = [rec("f1"), rec("f2", ("road_change", 0.8)), rec("f3", ("road_change", 0.9))]
    metrics = nominal_metrics(predictions, gt, threshold=0.5)
    assert metrics["tp"] == 1  # f1 correctly predicted nominal
    assert metrics["fp"] == 1  # f2 false fired
    assert metrics["fn"] == 0
    assert metrics["tn"] == 1  # f3 correctly predicted as hazard
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["support"] == 1


# --------------------------------------------------------------------------------------
# Bootstrap CI
# --------------------------------------------------------------------------------------


def test_bootstrap_identical_predictions_zero_delta() -> None:
    gt = {f"f{i}": [HAZARD_CLASSES[i % 4]] for i in range(12)}
    predictions = [rec(f"f{i}", (HAZARD_CLASSES[i % 4], 0.9)) for i in range(12)]
    result = bootstrap_macro_f1_ci(predictions, predictions, gt, n_samples=200, seed=1)
    assert result["delta_macro_f1"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)
    assert result["significant"] is False
    assert result["n_frames"] == 12
    assert result["n_samples"] == 200


def test_bootstrap_improvement_is_significant() -> None:
    gt = {f"f{i}": ["road_change"] for i in range(20)}
    good = [rec(f"f{i}", ("road_change", 0.9)) for i in range(20)]
    bad = [rec(f"f{i}") for i in range(20)]  # misses everything
    result = bootstrap_macro_f1_ci(bad, good, gt, n_samples=300, seed=7)
    # macro-F1 averages over all four hazard classes, so a perfect single-class
    # predictor scores 1/4.
    assert result["delta_macro_f1"] == pytest.approx(0.25)
    assert result["ci_lower"] > 0
    assert result["significant"] is True
    assert result["prob_delta_le_zero"] == 0.0


def test_bootstrap_no_shared_frames_reports_error() -> None:
    result = bootstrap_macro_f1_ci(
        [rec("a", ("road_change", 0.9))], [rec("b", ("road_change", 0.9))], {"a": ["road_change"]}
    )
    assert "error" in result
    assert result["n_frames"] == 0


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    gt = {f"f{i}": ["road_change"] for i in range(10)}
    a = [rec(f"f{i}", ("road_change", 0.9)) if i % 3 else rec(f"f{i}") for i in range(10)]
    b = [rec(f"f{i}", ("road_change", 0.55 + 0.01 * i)) for i in range(10)]
    first = bootstrap_macro_f1_ci(a, b, gt, n_samples=100, seed=3)
    second = bootstrap_macro_f1_ci(a, b, gt, n_samples=100, seed=3)
    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]


def test_macro_f1_at_helper_matches_manual() -> None:
    gt = {"f1": ["road_change"], "f2": ["road_change"]}
    predictions = [rec("f1", ("road_change", 0.9)), rec("f2")]
    value = macro_f1_at(predictions, gt, threshold=0.5)
    # Only road_change is defined: P=1.0, R=0.5 -> F1 = 0.6667; other classes 0.
    assert value == pytest.approx((2 / 3) / 4, rel=1e-5)


def test_counts_totals_are_consistent() -> None:
    gt = {f"f{i}": [HAZARD_CLASSES[i % 4]] for i in range(8)}
    predictions = [rec(f"f{i}", (HAZARD_CLASSES[(i + 1) % 4], 0.9)) for i in range(8)]
    counts = count_matches(predictions, gt, threshold=0.5)
    # Every prediction is a FP for one class and a FN for another.
    assert sum(counts.tp.values()) == 0
    assert sum(counts.fp.values()) == 8
    assert sum(counts.fn.values()) == 8
    assert np.isclose(macro_micro(counts)["macro_f1"], 0.0)
