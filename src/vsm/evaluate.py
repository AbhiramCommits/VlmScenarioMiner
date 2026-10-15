"""Multi-label scenario evaluation.

Everything here works on the flat prediction records written by
``vsm.pipeline.run_batch`` plus a ground-truth mapping ``{frame_id: [classes]}``.

Conventions
-----------
* ``nominal`` is *not* a label. A frame with no labels is a *nominal prediction* and
  a ground-truth frame with an empty class list is a *nominal frame*.
* Precision/recall/F1 are therefore computed over the four hazard classes, and the
  nominal decision is reported separately (plus inside the confusion matrix).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vsm.taxonomy import HAZARD_CLASSES

NONE = "none"


@dataclass(slots=True)
class ClassCounts:
    """TP/FP/FN tallies for a multi-label problem."""

    classes: tuple[str, ...] = HAZARD_CLASSES
    tp: dict[str, int] = field(default_factory=dict)
    fp: dict[str, int] = field(default_factory=dict)
    fn: dict[str, int] = field(default_factory=dict)
    n_frames: int = 0
    n_predicted_empty: int = 0
    n_gt_nominal: int = 0

    def __post_init__(self) -> None:
        for name in self.classes:
            self.tp.setdefault(name, 0)
            self.fp.setdefault(name, 0)
            self.fn.setdefault(name, 0)

    def support(self, name: str) -> int:
        return self.tp[name] + self.fn[name]

    def total_support(self) -> int:
        return sum(self.support(name) for name in self.classes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "tp": dict(self.tp),
            "fp": dict(self.fp),
            "fn": dict(self.fn),
            "support": {name: self.support(name) for name in self.classes},
            "total_support": self.total_support(),
            "n_frames": self.n_frames,
            "n_predicted_empty": self.n_predicted_empty,
            "n_gt_nominal": self.n_gt_nominal,
        }


def predicted_classes(record: dict[str, Any], threshold: float) -> list[str]:
    """Classes in a prediction record at or above ``threshold``."""
    return [
        str(label["class_name"])
        for label in record.get("labels", [])  # type: ignore[union-attr]
        if float(label["confidence"]) >= threshold  # type: ignore[index]
    ]


def align_frames(
    predictions: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Restrict both sides to the frames present in *both*.

    Frames skipped by dedup have no prediction, so they are excluded from scoring
    (and the excluded count is reported by the caller).
    """
    kept = [record for record in predictions if str(record["frame_id"]) in ground_truth]
    frame_ids = {str(record["frame_id"]) for record in kept}
    return kept, {fid: classes for fid, classes in ground_truth.items() if fid in frame_ids}


def count_matches(
    predictions: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    threshold: float = 0.5,
    classes: Iterable[str] = HAZARD_CLASSES,
) -> ClassCounts:
    """Per-class TP/FP/FN at a single confidence threshold."""
    class_list = tuple(classes)
    counts = ClassCounts(classes=class_list)
    for record in predictions:
        frame_id = str(record["frame_id"])
        if frame_id not in ground_truth:
            continue
        gt = set(ground_truth[frame_id])
        pred = set(predicted_classes(record, threshold)) & set(class_list)
        counts.n_frames += 1
        if not pred:
            counts.n_predicted_empty += 1
        if not gt:
            counts.n_gt_nominal += 1
        for name in class_list:
            in_gt, in_pred = name in gt, name in pred
            if in_gt and in_pred:
                counts.tp[name] += 1
            elif in_pred:
                counts.fp[name] += 1
            elif in_gt:
                counts.fn[name] += 1
    return counts


def per_class_metrics(counts: ClassCounts) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support per hazard class."""
    metrics: dict[str, dict[str, float]] = {}
    for name in counts.classes:
        tp, fp, fn = counts.tp[name], counts.fp[name], counts.fn[name]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": counts.support(name),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return metrics


def macro_micro(counts: ClassCounts) -> dict[str, float]:
    """Macro (unweighted mean over hazard classes) and micro (pooled) aggregates."""
    per_class = per_class_metrics(counts)
    macro_p = (
        float(np.mean([per_class[name]["precision"] for name in counts.classes]))
        if counts.classes
        else 0.0
    )
    macro_r = (
        float(np.mean([per_class[name]["recall"] for name in counts.classes]))
        if counts.classes
        else 0.0
    )
    macro_f1 = (
        float(np.mean([per_class[name]["f1"] for name in counts.classes]))
        if counts.classes
        else 0.0
    )

    tp = sum(counts.tp.values())
    fp = sum(counts.fp.values())
    fn = sum(counts.fn.values())
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    return {
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "micro_precision": round(micro_p, 6),
        "micro_recall": round(micro_r, 6),
        "micro_f1": round(micro_f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def nominal_metrics(
    predictions: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    threshold: float = 0.5,
) -> dict[str, float]:
    """How well the model separates "nothing happening" from "something happening"."""
    tp = fp = fn = tn = 0
    for record in predictions:
        frame_id = str(record["frame_id"])
        if frame_id not in ground_truth:
            continue
        gt_nominal = not ground_truth[frame_id]
        pred_nominal = not predicted_classes(record, threshold)
        if gt_nominal and pred_nominal:
            tp += 1
        elif gt_nominal and not pred_nominal:
            fp += 1
        elif not gt_nominal and pred_nominal:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6)
        if precision + recall
        else 0.0,
        "support": tp + fn,
        "accuracy": round((tp + tn) / (tp + tn + fp + fn), 6) if tp + tn + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def confusion_matrix(
    predictions: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    threshold: float = 0.5,
    classes: Sequence[str] = HAZARD_CLASSES,
) -> tuple[np.ndarray, list[str]]:
    """Multi-label confusion matrix using each frame's primary label.

    Rows are ground truth, columns are predictions. A frame whose ground truth is
    nominal, and a frame the model answered with no labels, both map to the shared
    ``none`` row/column, so the diagonal ``none -> none`` cell counts correct nominal
    frames while ``hazard -> none`` counts missed hazards and ``none -> hazard``
    counts false fires.

    Returns ``(matrix, labels)`` where ``labels`` names rows and columns.
    """
    labels = list(classes) + [NONE]
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for record in predictions:
        frame_id = str(record["frame_id"])
        if frame_id not in ground_truth:
            continue
        gt_classes = [c for c in ground_truth[frame_id] if c in index]
        gt_row = gt_classes[0] if gt_classes else NONE
        pred_classes = [c for c in predicted_classes(record, threshold) if c in index]
        if pred_classes:
            confidences = {
                str(label["class_name"]): float(label["confidence"])  # type: ignore[index]
                for label in record.get("labels", [])  # type: ignore[union-attr]
            }
            pred_col = max(pred_classes, key=lambda c: confidences.get(c, 0.0))
        else:
            pred_col = NONE
        matrix[index[gt_row], index[pred_col]] += 1
    return matrix, labels


def top_confusions(matrix: np.ndarray, labels: Sequence[str], k: int = 5) -> list[dict[str, Any]]:
    """Off-diagonal cells with the largest counts."""
    entries: list[dict[str, Any]] = []
    for i, gt_name in enumerate(labels):
        for j, pred_name in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                entries.append({"gt": gt_name, "predicted": pred_name, "count": int(matrix[i, j])})
    entries.sort(key=lambda e: (-e["count"], str(e["gt"]), str(e["predicted"])))
    return entries[:k]


def threshold_sweep(
    predictions: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    thresholds: Sequence[float] = tuple(round(0.1 * i, 1) for i in range(1, 10)),
    classes: Sequence[str] = HAZARD_CLASSES,
) -> dict[str, Any]:
    """Precision/recall per class (and macro-F1) across a confidence sweep."""
    sweeps: dict[str, list[dict[str, float]]] = {name: [] for name in classes}
    macro: list[dict[str, float]] = []
    for threshold in thresholds:
        counts = count_matches(predictions, ground_truth, threshold=threshold, classes=classes)
        per_class = per_class_metrics(counts)
        for name in classes:
            sweeps[name].append(
                {
                    "threshold": float(threshold),
                    "precision": per_class[name]["precision"],
                    "recall": per_class[name]["recall"],
                    "f1": per_class[name]["f1"],
                }
            )
        agg = macro_micro(counts)
        macro.append(
            {
                "threshold": float(threshold),
                "macro_precision": agg["macro_precision"],
                "macro_recall": agg["macro_recall"],
                "macro_f1": agg["macro_f1"],
            }
        )
    return {"per_class": sweeps, "macro": macro, "thresholds": [float(t) for t in thresholds]}


def macro_f1_at(
    predictions: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    threshold: float = 0.5,
) -> float:
    counts = count_matches(predictions, ground_truth, threshold=threshold)
    return macro_micro(counts)["macro_f1"]


def bootstrap_macro_f1_ci(
    predictions_a: Sequence[dict[str, Any]],
    predictions_b: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    threshold: float = 0.5,
    n_samples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap CI on the macro-F1 difference between two prompt versions.

    Frames are resampled with replacement; both versions are re-scored on the *same*
    resample, so the interval reflects the paired comparison rather than two
    independent estimates.

    Bootstrap resamples contain repeated frames. To keep every drawn sample counted
    (rather than collapsed by frame id), each drawn position is relabelled with its
    position index, which the scoring functions treat as a fresh frame id.
    """
    a_by_id = {str(record["frame_id"]): record for record in predictions_a}
    b_by_id = {str(record["frame_id"]): record for record in predictions_b}
    shared = sorted(fid for fid in a_by_id if fid in b_by_id and fid in ground_truth)
    if not shared:
        return {
            "error": "no shared frames between the two prediction sets",
            "n_frames": 0,
            "n_samples": 0,
        }

    gt = {fid: ground_truth[fid] for fid in shared}
    point_a = macro_f1_at([a_by_id[fid] for fid in shared], gt, threshold)
    point_b = macro_f1_at([b_by_id[fid] for fid in shared], gt, threshold)

    rng = np.random.default_rng(seed)
    n = len(shared)
    deltas = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        positions = rng.integers(0, n, size=n)
        resample_gt = {str(p): gt[shared[p]] for p in positions}
        resample_a = [dict(a_by_id[shared[p]], frame_id=str(p)) for p in positions]
        resample_b = [dict(b_by_id[shared[p]], frame_id=str(p)) for p in positions]
        deltas[i] = macro_f1_at(resample_b, resample_gt, threshold) - macro_f1_at(
            resample_a, resample_gt, threshold
        )

    lower = float(np.percentile(deltas, 100 * alpha / 2))
    upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {
        "version_a_macro_f1": round(point_a, 6),
        "version_b_macro_f1": round(point_b, 6),
        "delta_macro_f1": round(point_b - point_a, 6),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "ci_level": 1 - alpha,
        "n_samples": n_samples,
        "n_frames": n,
        "seed": seed,
        "prob_delta_le_zero": round(float((deltas <= 0).mean()), 6),
        "significant": bool(lower > 0 or upper < 0),
    }


def evaluate_run(
    predictions: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[str]],
    threshold: float = 0.5,
    sweep_thresholds: Sequence[float] = tuple(round(0.1 * i, 1) for i in range(1, 10)),
) -> dict[str, Any]:
    """Full metric bundle for one prompt version."""
    aligned_preds, aligned_gt = align_frames(predictions, ground_truth)
    counts = count_matches(aligned_preds, aligned_gt, threshold=threshold)
    matrix, matrix_labels = confusion_matrix(aligned_preds, aligned_gt, threshold=threshold)
    return {
        "threshold": threshold,
        "n_frames_scored": counts.n_frames,
        "n_predictions": len(predictions),
        "n_frames_excluded_no_prediction": max(0, len(ground_truth) - counts.n_frames),
        "counts": counts.to_dict(),
        "per_class": per_class_metrics(counts),
        "macro_micro": macro_micro(counts),
        "nominal": nominal_metrics(aligned_preds, aligned_gt, threshold=threshold),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": matrix_labels,
        "top_confusions": top_confusions(matrix, matrix_labels, k=6),
        "threshold_sweep": threshold_sweep(aligned_preds, aligned_gt, thresholds=sweep_thresholds),
    }
