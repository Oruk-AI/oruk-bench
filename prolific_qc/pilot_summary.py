"""Separate summary of the 2026-07-03 pilot export (pre-cleanup).

The pilot batch predates the studio-metacorpus pool and the 3.6.0+ QC
mechanics; per instructions it is analyzed separately and NEVER merged into
the main aggregation. This script only produces descriptive stats.

Output (PRIVATE): private_bench/pilot_summary.json

Usage:  python pilot_summary.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

from qc_common import download_inputs, utf8_stdout, write_json

PRIVATE_DIR = os.environ.get("ORUK_PRIVATE_DIR", "private_bench")


def main() -> None:
    utf8_stdout()
    paths = download_inputs()
    n_pkg = 0
    participants = set()
    pool_versions = Counter()
    schema_versions = Counter()
    roles = Counter()
    tiers = Counter()
    n_rows = 0
    with open(paths["pilot_submissions"], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pkg = json.loads(line)
            md = pkg.get("metadata") or {}
            n_pkg += 1
            participants.add(md.get("participant_id"))
            pool_versions[(md.get("trial_pool") or {}).get("poolVersion") or "<none>"] += 1
            schema_versions[md.get("schema_version")] += 1
            tiers[(md.get("server_quality") or {}).get("bonus_tier")] += 1
            for r in pkg.get("rows") or []:
                n_rows += 1
                roles[r.get("trial_role") or "<missing>"] += 1
    out = {
        "export": "2026-07-03T19-49-01-submissions.ndjson (pilot, pre-cleanup)",
        "note": "analyzed separately; never merged into the main v1 labels",
        "n_packages": n_pkg,
        "n_rows": n_rows,
        "n_participants": len(participants),
        "pool_versions": dict(pool_versions),
        "schema_versions": dict(schema_versions),
        "bonus_tiers": {str(k): v for k, v in tiers.items()},
        "trial_roles": dict(roles),
    }
    write_json(os.path.join(PRIVATE_DIR, "pilot_summary.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
