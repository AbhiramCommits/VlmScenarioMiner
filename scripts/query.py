#!/usr/bin/env python
"""Query the scenario manifest with DuckDB.

    python scripts/query.py --counts
    python scripts/query.py --tag construction_zone --min-conf 0.6
    python scripts/query.py --sql "SELECT location, COUNT(*) FROM scenarios GROUP BY 1"
    python scripts/query.py --agreement
    python scripts/query.py --frames

The `scenarios` view is the partitioned Parquet manifest; the `gt` view is the
ground-truth label file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vsm.config import load_settings  # noqa: E402
from vsm.store import (  # noqa: E402
    agreement_report,
    frame_summary,
    open_db,
    query_by_tag,
    run_sql,
    scenario_counts,
)
from vsm.taxonomy import HAZARD_CLASSES  # noqa: E402

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tag", choices=list(HAZARD_CLASSES), help="scenario tag to filter on")
    group.add_argument("--counts", action="store_true", help="per-class manifest counts")
    group.add_argument("--agreement", action="store_true", help="manifest vs ground-truth agreement")
    group.add_argument("--frames", action="store_true", help="frames with the most tags")
    group.add_argument("--sql", help="raw SQL against the scenarios / gt views")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--store", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    connection = open_db(store_dir=args.store, settings=settings)

    if args.counts:
        print("scenario manifest counts (DuckDB over partitioned Parquet)\n")
        print(scenario_counts(connection).to_string(index=False))
        total = connection.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]
        print(f"\ntotal detections: {total}")
    elif args.agreement:
        print("manifest vs ground truth (all GT frames)\n")
        print(agreement_report(connection).to_string(index=False))
    elif args.frames:
        print(f"frames with the most scenario tags (top {args.limit})\n")
        print(frame_summary(connection, limit=args.limit).to_string(index=False))
    elif args.sql:
        print(args.sql)
        print()
        print(run_sql(connection, args.sql).to_string(index=False))
    elif args.tag:
        print(f"tag={args.tag} min_conf={args.min_conf} limit={args.limit}\n")
        frame = query_by_tag(connection, args.tag, min_conf=args.min_conf, limit=args.limit)
        display = frame[
            ["frame_id", "scene_id", "class_name", "confidence", "ego_speed_mps", "location", "evidence"]
        ].copy()
        display["evidence"] = display["evidence"].str.slice(0, 44)
        print(display.to_string(index=False))
        print(f"\nrows returned: {len(frame)}")
    else:
        print("nothing to do; pass --counts, --tag, --agreement, --frames or --sql", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
