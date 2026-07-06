"""
Concatenate per-shard FoldX (or Rosetta) ddG CSVs into one deduplicated dataset.

Each array task in submit_foldx.sbatch writes <stem>.shard<N>.csv for its slice of PDB
files. Once all tasks complete, run this script to merge them:

    python merge_shards.py \
        --output runs/foldx/tcr_pmhc_foldx.csv \
        --pattern "runs/foldx/tcr_pmhc_ddg.shard*.csv"

Deduplication key: (PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid).
The full nine-column schema is preserved; missing columns in older shards get empty strings.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
from pathlib import Path

from generate_tcr_pmhc_dataset import CSV_FIELDNAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("merge_shards")

_KEY = ("PDB_ID", "Chain", "Residue_Position", "WT_Amino_Acid", "Mutant_Amino_Acid")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-shard ddG CSVs into one dataset.")
    ap.add_argument("--output", default="tcr_pmhc_merged.csv",
                    help="Destination CSV. Shards default to <stem>.shard*<suffix>.")
    ap.add_argument("--pattern", default=None,
                    help="Glob for the per-shard CSVs (overrides the auto-derived pattern).")
    ap.add_argument("--target_samples", type=int, default=None,
                    help="Stop after writing this many rows (cap the merged dataset).")
    args = ap.parse_args()

    out_path = Path(args.output)
    pattern = args.pattern or str(
        out_path.with_name(f"{out_path.stem}.shard*{out_path.suffix}")
    )
    parts = sorted(glob.glob(pattern))
    if not parts:
        LOGGER.error("No shard files matched: %s", pattern)
        return

    LOGGER.info("Merging %d shard files → %s", len(parts), out_path)
    seen: set = set()
    n_rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for part in parts:
            if args.target_samples is not None and n_rows >= args.target_samples:
                break
            with open(part, newline="") as fh:
                for row in csv.DictReader(fh):
                    key = tuple(row.get(k, "") for k in _KEY)
                    if key in seen:
                        continue
                    seen.add(key)
                    writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})
                    n_rows += 1
                    if args.target_samples is not None and n_rows >= args.target_samples:
                        break

    LOGGER.info("Wrote %d unique rows to %s.", n_rows, out_path)


if __name__ == "__main__":
    main()
