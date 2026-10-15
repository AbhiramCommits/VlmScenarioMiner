#!/usr/bin/env python
"""Evaluate prompt versions against the ground-truth set.

Writes:
    out/eval/metrics.json          machine-readable metrics for every version
    out/eval/metrics.md            markdown tables
    out/eval/confusion_<v>.csv     confusion matrix per version
    out/eval/pr_curves.png         precision/recall vs confidence per class
    out/eval/experiment_log.md     prompt-iteration log with measured metric movement

Example:
    python scripts/evaluate.py --runs v1,v2,v3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsm.config import load_settings  # noqa: E402
from vsm.evaluate import bootstrap_macro_f1_ci, evaluate_run  # noqa: E402
from vsm.pipeline import load_predictions  # noqa: E402
from vsm.prompts import load_prompt  # noqa: E402
from vsm.taxonomy import HAZARD_CLASSES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="v1,v2,v3", help="comma-separated prompt versions")
    parser.add_argument("--gt", type=Path, default=Path("data/ground_truth.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def write_markdown(metrics: dict, out_path: Path, gt_meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Scenario mining metrics\n")
    lines.append(f"Confidence threshold: **{metrics['threshold']}**  ")
    lines.append(f"Ground truth: `{metrics['gt_path']}` ({metrics['n_gt_frames']} frames, "
                 f"produced by `{gt_meta.get('mode', 'unknown')}` with "
                 f"{gt_meta.get('adjudication_noise_rate', 0.0):.0%} adjudication noise)\n")

    lines.append("## Per-class metrics (all versions)\n")
    header = "| class | " + " | ".join(f"{v} P / R / F1" for v in metrics["versions"]) + " | support |"
    lines.append(header)
    lines.append("|" + "---|" * (len(metrics["versions"]) + 2))
    for name in HAZARD_CLASSES:
        cells = []
        for version in metrics["versions"]:
            per_class = metrics["runs"][version]["per_class"][name]
            cells.append(f"{_fmt(per_class['precision'])} / {_fmt(per_class['recall'])} / {_fmt(per_class['f1'])}")
        support = metrics["runs"][metrics["versions"][0]]["per_class"][name]["support"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {support} |")
    lines.append("")

    lines.append("## Macro / micro aggregates\n")
    lines.append("| version | macro P | macro R | macro F1 | micro P | micro R | micro F1 | nominal acc | frames |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for version in metrics["versions"]:
        run = metrics["runs"][version]
        mm = run["macro_micro"]
        lines.append(
            f"| {version} | {_fmt(mm['macro_precision'])} | {_fmt(mm['macro_recall'])} | {_fmt(mm['macro_f1'])} "
            f"| {_fmt(mm['micro_precision'])} | {_fmt(mm['micro_recall'])} | {_fmt(mm['micro_f1'])} "
            f"| {_fmt(run['nominal']['accuracy'])} | {run['n_frames_scored']} |"
        )
    lines.append("")

    lines.append("## Run cost / reliability\n")
    lines.append("| version | schema-failure rate | retries recovered | tokens | est. cost (USD) | p50 latency (ms) | p95 latency (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for version in metrics["versions"]:
        meta = metrics["run_meta"].get(version, {})
        lines.append(
            f"| {version} | {_fmt(meta.get('schema_failure_rate', 0.0), 4)} | {meta.get('retries_recovered', 0)} "
            f"| {meta.get('total_tokens', 0)} | {meta.get('est_cost_usd', 0.0):.6f} "
            f"| {_fmt(meta.get('p50_latency_ms', 0.0), 1)} | {_fmt(meta.get('p95_latency_ms', 0.0), 1)} |"
        )
    lines.append("")

    lines.append("## Confusion matrix (best version: "
                 f"{metrics['best_version']})\n")
    best = metrics["runs"][metrics["best_version"]]
    labels = best["confusion_labels"]
    lines.append("Rows = ground truth, columns = prediction (`none` = no labels emitted).\n")
    lines.append("| gt \\ pred | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels) + 1))
    for i, label in enumerate(labels):
        row = best["confusion_matrix"][i]
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    lines.append("### Top confusions (best version)\n")
    lines.append("| ground truth | predicted | count |")
    lines.append("|---|---|---|")
    for entry in best["top_confusions"]:
        lines.append(f"| {entry['gt']} | {entry['predicted']} | {entry['count']} |")
    lines.append("")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def write_confusion_csv(matrix: list[list[int]], labels: list[str], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt\\pred", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def write_pr_curves(metrics: dict, out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib always present in this repo
        return False

    versions = metrics["versions"]
    ncols = min(3, len(versions)) or 1
    nrows = (len(versions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    colors = ["#c0392b", "#e67e22", "#2980b9", "#27ae60", "#8e44ad"]
    for ax, version in zip(axes.ravel(), versions, strict=False):
        sweep = metrics["runs"][version]["threshold_sweep"]
        for color, name in zip(colors, HAZARD_CLASSES, strict=False):
            points = sweep["per_class"][name]
            ax.plot(
                [p["recall"] for p in points],
                [p["precision"] for p in points],
                marker="o",
                ms=3,
                color=color,
                label=name,
            )
        mm = metrics["runs"][version]["macro_micro"]
        ax.set_title(f"{version}: PR vs confidence threshold (macro-F1 {mm['macro_f1']:.3f})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    for ax in axes.ravel()[len(versions):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def write_experiment_log(metrics: dict, out_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Prompt-iteration experiment log\n")
    lines.append(
        "Each row is one prompt version evaluated on the same frame set, the same "
        "ground truth and the same confidence threshold. The last column is a paired "
        f"bootstrap CI ({metrics['bootstrap_samples']} resamples, seed {metrics['seed']}) "
        "on the macro-F1 difference versus the previous version.\n"
    )
    lines.append("| version | change (changelog summary) | macro P | macro R | macro F1 | schema-failure rate | est. cost (USD) | delta macro-F1 vs previous (95% CI) | delta macro-F1 vs v1 (95% CI) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for version in metrics["versions"]:
        run = metrics["runs"][version]
        meta = metrics["run_meta"].get(version, {})
        mm = run["macro_micro"]
        summary = metrics["changelogs"].get(version, "")
        ci = metrics["bootstrap"].get(version)
        if ci is None:
            ci_text = "- (baseline)"
        elif "error" in ci:
            ci_text = ci["error"]
        else:
            ci_text = (
                f"{ci['delta_macro_f1']:+.3f} [{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}]"
                f" {'significant' if ci['significant'] else 'not significant'}"
            )
        base_ci = metrics.get("bootstrap_vs_first", {}).get(version)
        if version == metrics.get("baseline_version"):
            base_text = "- (baseline)"
        elif base_ci and "error" not in base_ci:
            base_text = (
                f"{base_ci['delta_macro_f1']:+.3f} [{base_ci['ci_lower']:+.3f}, {base_ci['ci_upper']:+.3f}]"
                f" {'significant' if base_ci['significant'] else 'not significant'}"
            )
        else:
            base_text = "-"
        lines.append(
            f"| {version} | {summary} | {_fmt(mm['macro_precision'])} | {_fmt(mm['macro_recall'])} "
            f"| {_fmt(mm['macro_f1'])} | {_fmt(meta.get('schema_failure_rate', 0.0), 4)} "
            f"| {meta.get('est_cost_usd', 0.0):.6f} | {ci_text} | {base_text} |"
        )
    lines.append("")
    baseline = metrics.get("baseline_version", metrics["versions"][0])
    if baseline in metrics["runs"] and metrics["best_version"] in metrics["runs"]:
        delta = (
            metrics["runs"][metrics["best_version"]]["macro_micro"]["macro_f1"]
            - metrics["runs"][baseline]["macro_micro"]["macro_f1"]
        )
        headline = (
            f"**Headline:** macro-F1 moved from {_fmt(metrics['runs'][baseline]['macro_micro']['macro_f1'])} "
            f"({baseline}) to {_fmt(metrics['runs'][metrics['best_version']]['macro_micro']['macro_f1'])} "
            f"({metrics['best_version']}), a delta of {delta:+.3f}"
        )
        ci = metrics.get("bootstrap_vs_first", {}).get(metrics["best_version"])
        if ci and "error" not in ci:
            headline += (
                f" with a paired bootstrap 95% CI of [{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}] "
                f"over {ci['n_samples']} resamples of {ci['n_frames']} frames "
                f"(P(delta<=0)={ci['prob_delta_le_zero']:.3f})."
            )
        else:
            headline += "."
        lines.append(headline + "\n")
    lines.append("## Threshold sweep (macro-F1 by confidence threshold)\n")
    header = "| version | " + " | ".join(f"{t:.1f}" for t in metrics["runs"][metrics["versions"][0]]["threshold_sweep"]["thresholds"]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(metrics["runs"][metrics["versions"][0]]["threshold_sweep"]["thresholds"]) + 1))
    for version in metrics["versions"]:
        sweep = metrics["runs"][version]["threshold_sweep"]["macro"]
        lines.append(f"| {version} | " + " | ".join(_fmt(p["macro_f1"]) for p in sweep) + " |")
    lines.append("")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    settings.eval_dir.mkdir(parents=True, exist_ok=True)

    from vsm.gt import load_ground_truth


    ground_truth, gt_meta = load_ground_truth(args.gt)
    if not ground_truth:
        print(f"ERROR: no ground truth at {args.gt}; run scripts/label_tool.py first", file=sys.stderr)
        return 1

    versions = [v.strip() for v in args.runs.split(",") if v.strip()]
    runs: dict[str, dict] = {}
    run_meta: dict[str, dict] = {}
    predictions_by_version: dict[str, list] = {}
    changelogs: dict[str, str] = {}

    for version in versions:
        run_dir = settings.run_dir(version)
        pred_path = run_dir / "predictions.jsonl"
        if not pred_path.exists():
            print(f"ERROR: missing {pred_path}; run scripts/run_inference.py --prompt-version {version}", file=sys.stderr)
            return 1
        predictions = load_predictions(pred_path)
        predictions_by_version[version] = predictions
        runs[version] = evaluate_run(predictions, ground_truth, threshold=args.threshold)
        meta_path = run_dir / "run_meta.json"
        run_meta[version] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        changelogs[version] = load_prompt(version).changelog_summary()

    bootstrap: dict[str, dict] = {}
    for previous, current in zip(versions, versions[1:], strict=False):
        bootstrap[current] = bootstrap_macro_f1_ci(
            predictions_by_version[previous],
            predictions_by_version[current],
            ground_truth,
            threshold=args.threshold,
            n_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    baseline = versions[0]
    bootstrap_vs_first: dict[str, dict] = {}
    for current in versions[1:]:
        bootstrap_vs_first[current] = bootstrap_macro_f1_ci(
            predictions_by_version[baseline],
            predictions_by_version[current],
            ground_truth,
            threshold=args.threshold,
            n_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    best_version = max(versions, key=lambda v: runs[v]["macro_micro"]["macro_f1"])
    metrics = {
        "threshold": args.threshold,
        "gt_path": str(args.gt),
        "gt_meta": gt_meta,
        "n_gt_frames": len(ground_truth),
        "versions": versions,
        "best_version": best_version,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "runs": runs,
        "run_meta": run_meta,
        "bootstrap": bootstrap,
        "bootstrap_vs_first": bootstrap_vs_first,
        "baseline_version": baseline,
        "changelogs": changelogs,
        "macro_over": list(HAZARD_CLASSES),
    }
    (settings.eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    md = write_markdown(metrics, settings.eval_dir / "metrics.md", gt_meta)
    del md
    for version in versions:
        write_confusion_csv(
            runs[version]["confusion_matrix"],
            runs[version]["confusion_labels"],
            settings.eval_dir / f"confusion_{version}.csv",
        )
    plotted = write_pr_curves(metrics, settings.eval_dir / "pr_curves.png")
    write_experiment_log(metrics, settings.eval_dir / "experiment_log.md")

    print(f"evaluated {len(versions)} prompt versions on {len(ground_truth)} GT frames "
          f"(scored {runs[best_version]['n_frames_scored']} frames with predictions)")
    print(f"{'version':8s} {'macro-P':>8s} {'macro-R':>8s} {'macro-F1':>9s} {'schema-fail':>12s} {'cost USD':>10s}")
    for version in versions:
        mm = runs[version]["macro_micro"]
        meta = run_meta.get(version, {})
        print(f"{version:8s} {mm['macro_precision']:8.3f} {mm['macro_recall']:8.3f} {mm['macro_f1']:9.3f} "
              f"{meta.get('schema_failure_rate', 0.0):12.4f} {meta.get('est_cost_usd', 0.0):10.6f}")
    for version, ci in bootstrap.items():
        if "error" not in ci:
            print(f"  {version} vs previous: delta macro-F1 {ci['delta_macro_f1']:+.3f} "
                  f"[{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}] "
                  f"P(delta<=0)={ci['prob_delta_le_zero']:.3f} significant={ci['significant']}")
    for version, ci in bootstrap_vs_first.items():
        if "error" not in ci:
            print(f"  {version} vs {baseline}: delta macro-F1 {ci['delta_macro_f1']:+.3f} "
                  f"[{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}] "
                  f"P(delta<=0)={ci['prob_delta_le_zero']:.3f} significant={ci['significant']}")
    print(f"best version: {best_version}")
    print(f"artifacts   : {settings.eval_dir} (metrics.json, metrics.md, experiment_log.md, "
          f"confusion_*.csv, pr_curves.png={plotted})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
