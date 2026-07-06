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
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        "-in:file:fullatom",
        "-ignore_unrecognized_res",
        "-ignore_zero_occupancy", "false",
        "-ex1", "-ex2",
        "-restore_talaris_behavior",
        "-out:path:all", job["_workdir"],
        "-out:prefix", f"{job['job_id']}_",
    ]


def run_rosetta(job: Dict[str, str], args) -> Path:
    workdir = Path(job["_workdir"])
    # A killed/preempted attempt at this same job-index can leave a half-written ddG.db3 or
    # other Rosetta output behind (resume only checks the output CSV, not this dir -- see
    # main()). Rosetta's ReportToDB opens the db3 with normal sqlite semantics, so a stale
    # file here would have new rows appended into old (possibly partial) batches rather than
    # a clean run, corrupting parse_ddg_db3's per-batch means. Always start from an empty dir.
    if workdir.exists():
        shutil.rmtree(workdir)
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
# InterfaceDdGMover (with db_reporter="dbreport" set in ddG_backrub.xml) clones the
# `dbreport` ReportToDB mover 4x internally and applies each clone to one of the 4
# ddG-relevant poses, renaming each clone's batch to "<STATE>_<batch_description>"
# (batch_description="interface_ddG" here). So the state lives in `batches.name`
# (joined to structure_scores via batch_id) -- NOT in `structures.tag`, which this
# protocol does not populate meaningfully. Verified against the actual Rosetta
# source (protocols/features/InterfaceDdGMover.cc) and the Kortemme-Lab reference
# analyze_flex_ddG.py, which parses ddG.db3 the same way.
BATCH_STATE_PREFIXES = {
    "bound_wt_": "wt_bound",
    "unbound_wt_": "wt_unbound",
    "bound_mut_": "mut_bound",
    "unbound_mut_": "mut_unbound",
}


def _classify_batch_name(name: str) -> Optional[str]:
    for prefix, state in BATCH_STATE_PREFIXES.items():
        if name.startswith(prefix):
            return state
    return None


def _total_scores_by_struct(
    conn: sqlite3.Connection, gam_coeffs: Optional[Dict[str, float]]
) -> Dict[int, Tuple[str, float]]:
    """Per-structure (state, score): either Rosetta's total_score, or a GAM-reweighted sum.

    Without GAM coefficients we use Rosetta's own `total_score`. With a coefficients
    dict {score_type_name: weight} we recompute a reweighted total (the published
    Flex ddG GAM reweighting; supply the official coefficients from the
    flex_ddG_tutorial repo). This keeps the GAM hook explicit instead of fabricating
    biophysics constants. State is `batches.name` (see BATCH_STATE_PREFIXES) joined
    in via batch_id -- struct_id alone is not state-identifying.
    """
    cur = conn.cursor()
    cur.execute("SELECT batch_id, name FROM batches")
    batch_names = {bid: name for bid, name in cur.fetchall()}

    if gam_coeffs:
        cur.execute(
            """
            SELECT ss.struct_id, ss.batch_id, st.score_type_name, ss.score_value
            FROM structure_scores ss
            JOIN score_types st ON st.score_type_id = ss.score_type_id AND st.batch_id = ss.batch_id
            """
        )
        totals: Dict[int, float] = {}
        struct_batch: Dict[int, int] = {}
        for struct_id, batch_id, name, value in cur.fetchall():
            struct_batch[struct_id] = batch_id
            w = gam_coeffs.get(name)
            if w is not None:
                totals[struct_id] = totals.get(struct_id, 0.0) + w * float(value)
        return {
            sid: (batch_names.get(struct_batch[sid], ""), total)
            for sid, total in totals.items()
        }

    cur.execute(
        """
        SELECT ss.struct_id, ss.batch_id, ss.score_value
        FROM structure_scores ss
        JOIN score_types st ON st.score_type_id = ss.score_type_id AND st.batch_id = ss.batch_id
        WHERE st.score_type_name = 'total_score'
        """
    )
    return {sid: (batch_names.get(bid, ""), float(v)) for sid, bid, v in cur.fetchall()}


def parse_ddg_db3(db3_path: str, gam_coeffs: Optional[Dict[str, float]] = None) -> float:
    """Average the backrub ensemble into one interface ddG (mutant - WT).

    Each scored structure's state (wt_bound / wt_unbound / mut_bound / mut_unbound) is
    given by its `batches.name` prefix (see BATCH_STATE_PREFIXES) -- InterfaceDdGMover
    writes one such batch per state, per backrub trajectory checkpoint. ddG follows the
    protocol's own formula (InterfaceDdGMover.cc / analyze_flex_ddG.py's calc_ddg()):

        ddG = (mean mut_bound - mean mut_unbound) - (mean wt_bound - mean wt_unbound)

    Raises ValueError with a diagnostic dump of what batches WERE found if any of the
    4 required states is missing -- this should never silently produce a wrong number.
    """
    conn = sqlite3.connect(db3_path)
    try:
        totals = _total_scores_by_struct(conn, gam_coeffs)
    finally:
        conn.close()

    buckets: Dict[str, List[float]] = {"wt_bound": [], "wt_unbound": [], "mut_bound": [], "mut_unbound": []}
    unclassified_batches: set = set()
    for _sid, (batch_name, score) in totals.items():
        state = _classify_batch_name(batch_name)
        if state is None:
            unclassified_batches.add(batch_name)
            continue
        buckets[state].append(score)

    missing = [k for k, v in buckets.items() if not v]
    if missing:
        seen_batches = sorted({bn for bn, _ in totals.values()})
        raise ValueError(
            f"{db3_path}: missing required state(s) {missing} in structure_scores/batches. "
            f"Batches actually present: {seen_batches or '<none>'}. "
            f"Unclassified batch names: {sorted(unclassified_batches) or '<none>'}. "
            "This means db_reporter=\"dbreport\" and/or trajectory_apply_mover wiring in "
            "ddG_backrub.xml did not execute as expected -- check the Rosetta stdout/stderr "
            "for this job, not just this parse error."
        )

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs)

    return (mean(buckets["mut_bound"]) - mean(buckets["mut_unbound"])) - (
        mean(buckets["wt_bound"]) - mean(buckets["wt_unbound"])
    )


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
    """Create a minimal ddG.db3 matching the REAL schema InterfaceDdGMover writes, for offline
    tests: one `batches` row per state (bound_wt_/unbound_wt_/bound_mut_/unbound_mut_ + the
    XML's batch_description), each with its own batch-scoped score_types + structure_scores
    rows -- exactly how 4 internally-cloned ReportToDB movers would each populate their own
    batch (see BATCH_STATE_PREFIXES / parse_ddg_db3's docstring for the verified reference).
    """
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE batches (batch_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE structures (struct_id INTEGER PRIMARY KEY, batch_id INTEGER, tag TEXT);
        CREATE TABLE score_types (score_type_id INTEGER, batch_id INTEGER, score_type_name TEXT);
        CREATE TABLE structure_scores (struct_id INTEGER, batch_id INTEGER, score_type_id INTEGER, score_value REAL);
        """
    )
    # 3 backrub-trajectory checkpoints x {wt,mut} x {bound,unbound} = 4 batches, 3 structs each.
    # The mutation perturbs only the *bound* state here (wt_bound=-500, mut_bound=-495), while
    # both unbound states share the same energy, so the interface ddG isolates to mut-wt = +5.
    unbound_total = -300.0
    states = [
        ("bound_wt_interface_ddG", wt_total),
        ("unbound_wt_interface_ddG", unbound_total),
        ("bound_mut_interface_ddG", mut_total),
        ("unbound_mut_interface_ddG", unbound_total),
    ]
    sid = 1
    for batch_id, (batch_name, base_value) in enumerate(states, start=1):
        cur.execute("INSERT INTO batches VALUES (?, ?)", (batch_id, batch_name))
        cur.execute("INSERT INTO score_types VALUES (1, ?, 'total_score')", (batch_id,))
        for rnd in range(3):
            cur.execute("INSERT INTO structures VALUES (?, ?, ?)", (sid, batch_id, f"{batch_name}_round{rnd}"))
            cur.execute(
                "INSERT INTO structure_scores VALUES (?, ?, 1, ?)",
                (sid, batch_id, base_value + 0.1 * rnd),
            )
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
    ap.add_argument("--backrub_trials", type=int, default=3500,
                    help="Backrub MC steps. Default 3500 = Graphinity/Hummer et al. 2025's validated "
                         "reference-accuracy setting. Barlow 2018 used 35000. Pass 1500 for a throughput-tuned "
                         "reduction if node/time budget is constrained (see estimate_runtime.py); below ~1000 "
                         "risks under-relaxing large mutations.")
    ap.add_argument("--nstruct", type=int, default=1,
                    help="Ensemble size. Default 1 (Hummer et al./Graphinity 2025) — vs Barlow 2018's 35. "
                         "The mutant-minus-WT difference cancels most single-model noise; ~35x cheaper. Pass 35 for the full protocol.")
    ap.add_argument("--workdir", type=str, default=None, help="Per-job scratch dir. Defaults to a temp dir.")
    ap.add_argument("--keep-workdir", action="store_true",
                    help="Keep the Rosetta scratch dir (structures, ddG.db3, struct.db3) after a "
                         "successful parse. Default is to delete it once its ddG row is safely in "
                         "--output, since ~20k jobs' worth of Rosetta output would otherwise fill "
                         "the node's local disk. Pass this to inspect a specific job's raw output.")
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

    # The ddG row is safely on disk in --output now, so the (large: structures + backrub
    # ensemble + db3s) Rosetta scratch dir is no longer needed. At ~20k jobs this is the
    # difference between a bounded and an unbounded disk footprint on the node.
    if not args.keep_workdir:
        shutil.rmtree(job["_workdir"], ignore_errors=True)


if __name__ == "__main__":
    main()
