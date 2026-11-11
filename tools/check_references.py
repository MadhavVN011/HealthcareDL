"""Check public reference availability. This is not a clinical content review."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def check(item):
    request = urllib.request.Request(item["source"], headers={"User-Agent": "HealthcareDL reference availability check/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return {"condition_id": item["id"], "url": item["source"], "status": response.status, "final_url": response.url}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"condition_id": item["id"], "url": item["source"], "status": "unavailable", "error_type": type(error).__name__}


def main():
    catalog = json.loads((ROOT / "data/conditions.json").read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, catalog))
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Reference URL availability only; not factual or clinical approval.", "results": results}
    (ROOT / "data/reference_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = [row["condition_id"] for row in results if row["status"] != 200]
    print(f"References reachable: {len(results)-len(failures)}/{len(results)}")
    if failures:
        print("Unavailable:", ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
