"""
Concatenate the per-job Rosetta Flex ddG CSVs (from the SLURM array) into one
shared-schema dataset, de-duplicating and reporting coverage.

Each array task writes ``rosetta_flex/results/parts/job_<i>.csv`` (one ddG row).
This merges them into a single ``Synthetic_FlexddG_TCR_pMHC.csv`` with the same
columns the MadraX generator emits, so the two tiers concatenate cleanly:

    PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid, ddG, source

Coverage report: because a run is sharded across nodes and resumable, some jobs.csv
rows may not have landed yet (still running, or failed). We cross-reference the part
files against ``jobs.csv`` and report exactly which job indices are missing, so gaps
are visible before the data is used for training (re-run the array to fill them).
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("merge_results")

FIELDS = ["PDB_ID", "Chain", "Residue_Position", "WT_Amino_Acid", "Mutant_Amino_Acid", "ddG", "source"]


def report_coverage(parts_dir: Path, jobs_csv: Path, n_merged: int, n_structures: int) -> None:
    """Log rows merged, structures covered, and which jobs.csv indices are still missing."""
    LOGGER.info("Coverage: %d unique ddG rows across %d structures.", n_merged, n_structures)
    if not jobs_csv.exists():
        LOGGER.info("(%s not found; skipping missing-job check.)", jobs_csv)
        return
    with open(jobs_csv, newline="") as fh:
        n_jobs = sum(1 for _ in csv.DictReader(fh))
    missing = [i for i in range(n_jobs)
               if not (parts_dir / f"job_{i}.csv").exists() or (parts_dir / f"job_{i}.csv").stat().st_size == 0]
    done = n_jobs - len(missing)
    LOGGER.info("Jobs complete: %d/%d (%.1f%%).", done, n_jobs, 100.0 * done / max(1, n_jobs))
    if missing:
        preview = ", ".join(str(i) for i in missing[:20])
        LOGGER.warning("%d job(s) missing/empty (e.g. indices: %s%s). Re-submit the array (resumable) to fill.",
                       len(missing), preview, " ..." if len(missing) > 20 else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-job Flex ddG CSVs into one dataset.")
    ap.add_argument("--parts_dir", type=str, default="rosetta_flex/results/parts")
    ap.add_argument("--output", type=str, default="rosetta_flex/results/Synthetic_FlexddG_TCR_pMHC.csv")
    ap.add_argument("--jobs_csv", type=str, default="rosetta_flex/jobs/jobs.csv",
                    help="jobs.csv to cross-reference for the missing-job coverage check.")
    args = ap.parse_args()

    parts_dir = Path(args.parts_dir)
    parts = sorted(glob.glob(str(parts_dir / "*.csv")))
    if not parts:
        LOGGER.error("No part files found in %s.", args.parts_dir)
        return

    seen = set()
    structures = set()
    n_rows = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for part in parts:
            with open(part, newline="") as fh:
                for row in csv.DictReader(fh):
                    key = (row["PDB_ID"], row["Chain"], row["Residue_Position"], row["WT_Amino_Acid"], row["Mutant_Amino_Acid"])
                    if key in seen:
                        continue
                    seen.add(key)
                    writer.writerow({k: row[k] for k in FIELDS})
                    structures.add(row["PDB_ID"])
                    n_rows += 1

    LOGGER.info("Merged %d part files into %d unique ddG rows -> %s", len(parts), n_rows, out_path)
    report_coverage(parts_dir, Path(args.jobs_csv), n_rows, len(structures))


if __name__ == "__main__":
    main()
