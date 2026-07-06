"""
Run (or parse) a single Rosetta Flex ddG job and emit a shared-schema CSV row.

One invocation == one line of ``jobs.csv`` (selected by ``--job-index``),
matching the SLURM array model in submit_array.sbatch:

    1. Build the rosetta_scripts command for ddG_backrub.xml with the job's
       per-mutation resfile and chains-to-move, run it in a per-job work dir.
    2. Parse the resulting ddG.db3 SQLite ensemble into a single interface ddG.
    3. Append one row to the output CSV in the shared schema:
       PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid,
       ddG, source(="rosetta_flex").

This script is designed to run on HPC; locally it is exercised with:
    --parse-only <ddG.db3>   parse an existing db3 without invoking Rosetta
    --self-test              build a synthetic db3 and verify the parser

ddG sign convention matches the project: mutant - wild type.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("run_flex_ddg")

THREE_TO_ONE_REV = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

OUTPUT_FIELDS = [
    "PDB_ID", "Chain", "Residue_Position", "WT_Amino_Acid",
    "Mutant_Amino_Acid", "ddG", "source",
]
SOURCE_TAG = "rosetta_flex"

# Default location for the official Kortemme flex_ddG GAM/reweighting coefficients.
# fetch_gam_coeffs.py writes here; if the file exists it is used automatically (no
# fabricated numbers -- absent file simply falls back to raw total_score, "nogam").
DEFAULT_GAM_COEFFS = Path(__file__).resolve().parent / "gam_coeffs" / "flex_ddg_gam.json"


# --------------------------------------------------------------------------- #
# Rosetta command construction                                                #
# --------------------------------------------------------------------------- #
def build_command(job: Dict[str, str], args) -> List[str]:
    """rosetta_scripts command line reproducing the flex_ddG protocol."""
    return [
        args.rosetta_bin,
        "-s", job["pdb_path"],
        "-parser:protocol", args.protocol_xml,
        "-parser:script_vars",
        f"chainstomove={job['chains_to_move']}",
        f"pathtoresfile={job['resfile_path']}",
        f"backrubntrials={args.backrub_trials}",
        "-nstruct", str(args.nstruct),
        "-ignore_unrecognized_res",
        "-ignore_zero_occupancy", "false",
        "-ex1", "-ex2",
        "-extrachi_cutoff", "0",
        "-restore_talaris_behavior",
        "-out:path:all", job["_workdir"],
        "-out:prefix", f"{job['job_id']}_",
    ]


def run_rosetta(job: Dict[str, str], args) -> Path:
    workdir = Path(job["_workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(job, args)
    LOGGER.info("Running Rosetta: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=workdir)
    db3 = workdir / "ddG.db3"
    if not db3.exists():
        raise FileNotFoundError(f"Expected {db3} after Rosetta run; not found.")
    return db3


# --------------------------------------------------------------------------- #
# ddG.db3 parsing                                                             #
# --------------------------------------------------------------------------- #
def _total_scores_by_struct(conn: sqlite3.Connection, gam_coeffs: Optional[Dict[str, float]]) -> Dict[int, float]:
    """Per-structure score: either the stored total_score, or a reweighted sum.

    Without GAM coefficients we use Rosetta's own `total_score`. With a
    coefficients dict {score_type_name: weight} we recompute a reweighted total
    (the published Flex ddG GAM reweighting; supply the official coefficients
    from the flex_ddG_tutorial repo). This keeps the GAM hook explicit instead
    of fabricating biophysics constants.
    """
    cur = conn.cursor()
    if gam_coeffs:
        cur.execute(
            """
            SELECT ss.struct_id, st.score_type_name, ss.score_value
            FROM structure_scores ss
            JOIN score_types st ON st.score_type_id = ss.score_type_id
            """
        )
        totals: Dict[int, float] = {}
        for struct_id, name, value in cur.fetchall():
            w = gam_coeffs.get(name)
            if w is not None:
                totals[struct_id] = totals.get(struct_id, 0.0) + w * float(value)
        return totals

    cur.execute(
        """
        SELECT ss.struct_id, ss.score_value
        FROM structure_scores ss
        JOIN score_types st ON st.score_type_id = ss.score_type_id
        WHERE st.score_type_name = 'total_score'
        """
    )
    return {sid: float(v) for sid, v in cur.fetchall()}


def parse_ddg_db3(db3_path: str, gam_coeffs: Optional[Dict[str, float]] = None) -> float:
    """Average the backrub ensemble into one interface ddG (mutant - WT).

    Structure tags written by the protocol encode the pose identity. We classify
    each struct by its tag containing 'wt'/'mut' and 'bound'/'unbound', then:

        ddG = (mean mut_bound - mean mut_unbound)
            - (mean wt_bound  - mean wt_unbound)

    If bound/unbound are not distinguishable in the tags we fall back to the
    simpler total-score difference (mean mut - mean wt), which still respects the
    mutant-minus-WT sign convention.
    """
    conn = sqlite3.connect(db3_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT struct_id, tag FROM structures")
        tags = {sid: (tag or "").lower() for sid, tag in cur.fetchall()}
        totals = _total_scores_by_struct(conn, gam_coeffs)
    finally:
        conn.close()

    buckets: Dict[str, List[float]] = {"wt_bound": [], "wt_unbound": [], "mut_bound": [], "mut_unbound": []}
    simple: Dict[str, List[float]] = {"wt": [], "mut": []}
    for sid, score in totals.items():
        tag = tags.get(sid, "")
        side = "mut" if "mut" in tag else ("wt" if "wt" in tag else None)
        if side is None:
            continue
        simple[side].append(score)
        if "unbound" in tag:
            buckets[f"{side}_unbound"].append(score)
        elif "bound" in tag:
            buckets[f"{side}_bound"].append(score)

    def mean(xs: List[float]) -> float:
        if not xs:
            raise ValueError("empty score bucket")
        return sum(xs) / len(xs)

    if all(buckets[k] for k in buckets):
        ddg = (mean(buckets["mut_bound"]) - mean(buckets["mut_unbound"])) - (
            mean(buckets["wt_bound"]) - mean(buckets["wt_unbound"])
        )
        return ddg

    if simple["wt"] and simple["mut"]:
        LOGGER.warning("bound/unbound tags not found in %s; using total-score difference fallback.", db3_path)
        return mean(simple["mut"]) - mean(simple["wt"])

    raise ValueError(f"Could not classify any WT/mutant structures in {db3_path}.")


def load_gam_coeffs(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    import json
    with open(path) as fh:
        coeffs = json.load(fh)
    LOGGER.info("Loaded %d GAM reweighting coefficients from %s", len(coeffs), path)
    return {str(k): float(v) for k, v in coeffs.items()}


# --------------------------------------------------------------------------- #
# Job dispatch + output                                                        #
# --------------------------------------------------------------------------- #
def read_job(jobs_csv: str, job_index: int) -> Dict[str, str]:
    with open(jobs_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not 0 <= job_index < len(rows):
        raise IndexError(f"job-index {job_index} out of range (0..{len(rows) - 1}).")
    return rows[job_index]


def append_row(out_csv: Path, job: Dict[str, str], ddg: float) -> None:
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "PDB_ID": job["pdb_id"],
                "Chain": job["chain"],
                "Residue_Position": job["resnum"],
                "WT_Amino_Acid": THREE_TO_ONE_REV.get(job["wt_aa"], job["wt_aa"]),
                "Mutant_Amino_Acid": THREE_TO_ONE_REV.get(job["mut_aa"], job["mut_aa"]),
                "ddG": round(ddg, 4),
                "source": SOURCE_TAG,
            }
        )


def build_synthetic_db3(path: str, wt_total: float = -500.0, mut_total: float = -495.0) -> None:
    """Create a minimal ddG.db3 matching the queried schema, for offline tests."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE structures (struct_id INTEGER PRIMARY KEY, batch_id INTEGER, tag TEXT);
        CREATE TABLE score_types (score_type_id INTEGER PRIMARY KEY, batch_id INTEGER, score_type_name TEXT);
        CREATE TABLE structure_scores (struct_id INTEGER, score_type_id INTEGER, score_value REAL);
        """
    )
    cur.execute("INSERT INTO score_types VALUES (1, 1, 'total_score')")
    # 3 backrub rounds x {wt,mut} x {bound,unbound}. The mutation perturbs only
    # the *bound* state here (wt_bound=-500, mut_bound=-495), while both unbound
    # states share the same energy, so the interface ddG isolates to mut-wt = +5.
    unbound_total = -300.0
    sid = 1
    for rnd in range(3):
        for side, bound_base in (("wt", wt_total), ("mut", mut_total)):
            for state, value in (("bound", bound_base), ("unbound", unbound_total)):
                cur.execute("INSERT INTO structures VALUES (?, 1, ?)", (sid, f"{side}_{state}_round{rnd}"))
                cur.execute("INSERT INTO structure_scores VALUES (?, 1, ?)", (sid, value + 0.1 * rnd))
                sid += 1
    conn.commit()
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run/parse a single Rosetta Flex ddG job into a shared-schema CSV row.")
    ap.add_argument("--jobs_csv", type=str, default="rosetta_flex/jobs/jobs.csv")
    ap.add_argument("--job-index", type=int, default=None, help="0-based row in jobs.csv (e.g. $SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--output", type=str, default="rosetta_flex/results/flex_ddg.csv")
    ap.add_argument("--rosetta_bin", type=str, default=os.environ.get("ROSETTA_SCRIPTS_BIN", "rosetta_scripts.default.linuxgccrelease"))
    ap.add_argument("--protocol_xml", type=str, default="rosetta_flex/ddG_backrub.xml")
    ap.add_argument("--backrub_trials", type=int, default=1500,
                    help="Backrub MC steps. Default 1500 = throughput-tuned for a <4h / 20k run (see "
                         "estimate_runtime.py). Graphinity/Hummer 2025 validated 3500 (pass that for the "
                         "reference-accuracy set); Barlow 2018 used 35000. Below ~1000 risks under-relaxing "
                         "large mutations, so 1500 is a modest, not aggressive, reduction.")
    ap.add_argument("--nstruct", type=int, default=1,
                    help="Ensemble size. Default 1 (Hummer et al./Graphinity 2025) — vs Barlow 2018's 35. "
                         "The mutant-minus-WT difference cancels most single-model noise; ~35x cheaper. Pass 35 for the full protocol.")
    ap.add_argument("--workdir", type=str, default=None, help="Per-job scratch dir. Defaults to a temp dir.")
    ap.add_argument("--gam-coeffs", type=str, default=None,
                    help=f"JSON of {{score_type: weight}} for GAM reweighting. If omitted, "
                         f"{DEFAULT_GAM_COEFFS} is used when present, else raw total_score (nogam).")
    ap.add_argument("--parse-only", type=str, default=None, help="Parse an existing ddG.db3 and print the ddG, no Rosetta run.")
    ap.add_argument("--self-test", action="store_true", help="Build a synthetic db3 and verify the parser end-to-end.")
    args = ap.parse_args()

    # Resolve GAM coefficients: explicit flag wins, else auto-use the shipped default if
    # fetch_gam_coeffs.py has populated it, else None (raw total_score).
    gam_path = args.gam_coeffs
    if gam_path is None and DEFAULT_GAM_COEFFS.exists():
        gam_path = str(DEFAULT_GAM_COEFFS)
        LOGGER.info("Using default GAM coefficients at %s.", gam_path)
    gam = load_gam_coeffs(gam_path)

    if args.self_test:
        with tempfile.TemporaryDirectory() as td:
            db3 = str(Path(td) / "ddG.db3")
            build_synthetic_db3(db3)
            ddg = parse_ddg_db3(db3, gam)
            # Synthetic: bound mut-wt = +5, unbound identical -> ddG ~ +5.
            LOGGER.info("self-test parsed ddG = %.4f (expected ~5.0)", ddg)
            assert abs(ddg - 5.0) < 1e-6, f"self-test failed: {ddg}"
            LOGGER.info("self-test PASSED.")
        return

    if args.parse_only:
        ddg = parse_ddg_db3(args.parse_only, gam)
        LOGGER.info("Parsed ddG = %.4f from %s", ddg, args.parse_only)
        print(ddg)
        return

    if args.job_index is None:
        LOGGER.error("--job-index is required for a real run (or use --parse-only / --self-test).")
        sys.exit(2)

    # Idempotent resume: if this job's per-task CSV already has a row, skip the (expensive)
    # Rosetta run. Belt-and-suspenders with the sbatch driver's own skip filter, so a requeued
    # array task never recomputes finished mutations.
    out_path = Path(args.output)
    if out_path.exists() and out_path.stat().st_size > 0:
        LOGGER.info("job-index %d: output %s already present; skipping.", args.job_index, out_path)
        return

    job = read_job(args.jobs_csv, args.job_index)
    job["_workdir"] = args.workdir or tempfile.mkdtemp(prefix=f"flexddg_{job['job_id']}_")
    db3 = run_rosetta(job, args)
    ddg = parse_ddg_db3(str(db3), gam)
    append_row(Path(args.output), job, ddg)
    LOGGER.info("job %s (%s %s%s%s->%s) -> ddG=%.4f", job["job_id"], job["pdb_id"], job["chain"], job["resnum"], job["wt_aa"], job["mut_aa"], ddg)


if __name__ == "__main__":
    main()
