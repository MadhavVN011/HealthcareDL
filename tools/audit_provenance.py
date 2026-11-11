"""Compare the supplied data against public publisher downloads; no credentials needed."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "HealthcareDL provenance audit/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read(10_000_001)
    if len(content) > 10_000_000:
        raise ValueError("Public archive exceeds the audit size limit")
    return content


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    findings = []
    sources = []
    for owner, slug in [("kaushil268", "disease-prediction-using-machine-learning"),
                         ("itachi9604", "disease-symptom-description-dataset")]:
        ref = f"{owner}/{slug}"
        base = "https://www.kaggle.com/api/v1/datasets/"
        metadata = next(item for item in json.loads(fetch(base + "list?search=" + slug)) if item.get("ref") == ref)
        archive_bytes = fetch(base + "download/" + ref)
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        source = {"ref": ref, "publisher_name": metadata["creatorName"],
                  "title": metadata["title"].strip(), "version": metadata["currentVersionNumber"],
                  "url": metadata["url"], "declared_license": metadata["licenseName"],
                  "metadata_api": base + "list?search=" + slug,
                  "archive_sha256": sha(archive_bytes)}
        sources.append(source)
        if owner == "kaushil268":
            local = list(csv.reader((ROOT / "Training.csv").open()))
            remote_bytes = archive.read("Training.csv")
            remote = list(csv.reader(io.StringIO(remote_bytes.decode("utf-8-sig"))))
            # Publisher adds a final empty column; no other transformation is allowed.
            trailing_empty = all(len(row) == len(local[0]) + 1 and row[-1] == "" for row in remote)
            matches = trailing_empty and [row[:-1] for row in remote] == local
            if not matches:
                raise ValueError("Training.csv no longer matches the publisher table")
            known = {tuple(row) for row in local[1:]}
            testing = list(csv.reader(io.StringIO(archive.read("Testing.csv").decode("utf-8-sig"))))
            overlap = sum(tuple(row) in known for row in testing[1:])
            findings.append({"material": "Training.csv", "source_label": "Kaggle · KAUSHIL268", "source_url": source["url"],
                             "summary": "All 4,920 rows and headers match after removing the publisher's final empty column.",
                             "comparison": "table_exact_after_empty_column_removal", "rows": 4920,
                             "local_sha256": sha((ROOT / "Training.csv").read_bytes()), "remote_sha256": sha(remote_bytes)})
            findings.append({"material": "Publisher's Testing.csv", "source_label": "Same publisher", "source_url": source["url"],
                             "summary": f"{overlap} of {len(testing)-1} rows already appear in this project's original training table. Not an independent patient benchmark.",
                             "comparison": "overlap_audit", "rows": len(testing)-1, "overlap_rows": overlap,
                             "remote_sha256": sha(archive.read("Testing.csv"))})
        else:
            remote = archive.read("Symptom-severity.csv")
            local = (ROOT / "data/legacy/Symptom-severity.csv").read_bytes()
            if remote != local:
                raise ValueError("Archived severity table does not match publisher")
            findings.append({"material": "Archived severity table", "source_label": "Kaggle · Pranay Patil", "source_url": source["url"],
                             "summary": "Exact byte-for-byte match with the publisher's Symptom-severity.csv.",
                             "comparison": "byte_exact", "sha256": sha(local)})
    report = {"checked_on": datetime.now(timezone.utc).date().isoformat(), "method": "Public publisher API metadata and complete archive comparison",
              "findings": findings, "sources": sources,
              "limits": ["File identity does not establish original clinical collection, consent, or ownership.",
                         "Publisher-declared licenses are recorded as evidence, not legal certification.",
                         "The associated Testing.csv is not independent clinical validation."]}
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/provenance.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(findings)} comparisons; wrote {target}")


if __name__ == "__main__":
    main()
