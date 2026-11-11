"""Compare a candidate binary symptom table without guessing labels or features."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from healthdl.data import file_hash, load_catalog, load_dataset, normalize_disease, normalize_symptom, pattern_id


def audit(path):
    original = load_dataset()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2:
        raise ValueError("Candidate has no data rows")
    header, rows = rows[0], rows[1:]
    if header[-1] != "prognosis" or any(len(row) != len(header) for row in rows):
        raise ValueError("Expected complete rows with a final prognosis column")
    groups = {}
    for index, name in enumerate(header[:-1]):
        groups.setdefault(normalize_symptom(name), []).append(index)
    if set(groups) != set(original.features):
        raise ValueError("Candidate feature schema is incompatible; no approximate mapping performed")
    raw = np.array([row[:-1] for row in rows], dtype=np.float32)
    if not np.isin(raw, [0, 1]).all():
        raise ValueError("Candidate features must be binary and finite")
    features = np.stack([raw[:, groups[name]].max(axis=1) for name in original.features], axis=1)
    keys = {pattern_id(row) for row in features}
    known = {pattern_id(row) for row in original.x}
    catalog = load_catalog()
    aliases = {normalize_disease(name) for item in catalog.values() for name in [item["label"], *item["aliases"]]}
    return {"sha256": file_hash(path), "raw_rows": len(rows), "unique_binary_patterns": len(keys),
            "shared_with_original": len(keys & known), "new_binary_patterns": len(keys - known),
            "unknown_labels": sorted({row[-1] for row in rows if normalize_disease(row[-1]) not in aliases}),
            "note": "Counts compare symptom vectors only. They do not validate labels, license, clinical provenance, or independence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    print(json.dumps(audit(parser.parse_args().data), indent=2))
