"""Evaluate approved external data; refuse development-pattern overlap."""
import argparse
import json
from pathlib import Path

from healthdl.data import ROOT
from healthdl.evaluation import evaluate_external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--group-column")
    args = parser.parse_args()
    try:
        report = evaluate_external(args.data, args.artifacts, args.group_column)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Status: {report['status']}; rows: {report['rows']}; overlapping rows: {report['overlapping_rows']}")
    if report["status"] != "evaluated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
