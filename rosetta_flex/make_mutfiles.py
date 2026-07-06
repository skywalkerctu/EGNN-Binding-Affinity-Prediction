"""
Enumerate Rosetta Flex ddG jobs for the Tier-2 dataset (budgeted + contact-ranked).

For every structure in ``--pdb_dir`` we reuse the *exact same* interface detection
as the MadraX generator (``interface_utils.find_interface_targets``) so Tier 1 and
Tier 2 mutate comparable positions, then write one Rosetta resfile per selected
(residue, mutant amino acid) and a single ``jobs.csv`` describing every job.

Sampling (why this isn't full saturation)
------------------------------------------
Full saturation (19 mutants x every interface residue x every structure) is ~1M
jobs -- far more than a backbone-flexible Flex ddG run can afford. Tier 2 instead
targets a fixed budget (``--target_samples``, default 20000) of the *highest-value*
mutations, chosen to maximise training signal per Rosetta CPU-hour:

1. Keep only **genuine inter-chain contacts**: residues whose minimum atom-atom
   distance to the opposite binding partner (``MutationTarget.min_interchain_dist``,
   already computed by the shared interface detector) is within
   ``--max_interchain_dist`` A. Second-shell residues pulled in by the broad CDR
   radius are dropped -- their binding ddG is mostly noise.
2. **Rank within a structure by contact tightness** (smallest interchain distance
   first) and cap per structure (``--per_structure_cap``) so one large interface
   cannot dominate the budget.
3. **Round-robin across structures** to fill the budget, so the set spans as many
   distinct complexes as possible. When a structure's quota is small, mutants are
   interleaved by slot (Ala first, then a seeded order) so the budget spreads over
   many *positions* (an alanine scan of the tightest contacts) before saturating any
   single residue -- diversity beats per-label precision for an EGNN training set.

Everything is deterministic under ``--seed``. Pass ``--target_samples 0`` (or a
negative value) to recover the old full-saturation enumeration.

A SLURM array (see submit_array.sbatch) then runs ``jobs.csv`` rows through
``run_flex_ddg.py``. Nothing here invokes Rosetta -- it only prepares inputs, so it
is safe to run anywhere (including to dry-run the pipeline).

Resfile format (Rosetta): default all positions to repack-as-native (NATAA), then
force the single target position to the mutant identity with PIKAA. The PDB
insertion code (if any) is appended directly to the residue number, exactly as
Rosetta expects, so CDR3 insertion-coded positions are targeted correctly.

    NATAA
    start
    <resnum><icode> <chain> PIKAA <mut_one_letter>
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

# Allow `python rosetta_flex/make_mutfiles.py` from the repo root to import the
# shared helper that lives one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Bio.PDB import PDBParser

from interface_utils import (
    DEFAULT_COMPLEX_CHAINS,
    DEFAULT_TCR_CHAINS,
    THREE_TO_ONE,
    MutationTarget,
    find_interface_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("make_mutfiles")

ONE_LETTER = [THREE_TO_ONE[a] for a in THREE_TO_ONE]

JOBS_FIELDS = [
    "job_id",
    "pdb_id",
    "pdb_path",
    "chain",
    "resnum",
    "wt_aa",      # 1-letter
    "mut_aa",     # 1-letter
    "resfile_path",
    "chains_to_move",
    "min_interchain_dist",  # provenance: how tight the contact this sample came from is
]

# A selected sample: which structure, which residue, and the mutant identity (1-letter).
Sample = Tuple[str, str, MutationTarget, str]  # (pdb_id, pdb_path, target, mut_one)


def write_resfile(path: Path, chain: str, resnum: int, icode: str, mut_one: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rosetta resfile: insertion code is appended to the number with no space
    # (e.g. "111A B PIKAA L"); icode is "" for ordinary residues.
    path.write_text(f"NATAA\nstart\n{resnum}{icode} {chain} PIKAA {mut_one}\n")


def _mutant_order(target: MutationTarget, rng: random.Random) -> List[str]:
    """Order the 19 non-WT mutants for one residue: Ala first, then a seeded shuffle.

    Putting alanine first means a residue that only receives a *single* slot from the
    budget contributes the canonical hotspot probe (X->A); the seeded shuffle of the
    rest keeps mutation-type coverage unbiased as the per-residue quota grows.
    """
    wt_one = target.wt_aa_one
    muts = [m for m in ONE_LETTER if m != wt_one]
    rng.shuffle(muts)
    if "A" in muts and wt_one != "A":
        muts.remove("A")
        muts.insert(0, "A")
    return muts


def _structure_queue(targets: Sequence[MutationTarget], per_structure_cap: int, rng: random.Random) -> List[Tuple[MutationTarget, str]]:
    """Ordered (target, mut_one) candidates for one structure, capped and slot-interleaved.

    ``targets`` must already be sorted by contact tightness (smallest interchain distance
    first). We interleave by mutant *slot* across residues so that a small draw spans many
    positions (tightest contacts first) rather than saturating a single residue.
    """
    mut_orders = [(t, _mutant_order(t, rng)) for t in targets]
    queue: List[Tuple[MutationTarget, str]] = []
    max_slots = max((len(m) for _, m in mut_orders), default=0)
    for slot in range(max_slots):
        for t, muts in mut_orders:
            if slot < len(muts):
                queue.append((t, muts[slot]))
    if per_structure_cap and per_structure_cap > 0:
        queue = queue[:per_structure_cap]
    return queue


def select_samples(
    struct_targets: List[Tuple[str, str, List[MutationTarget]]],
    target_samples: int,
    per_structure_cap: int,
    seed: int,
) -> List[Sample]:
    """Choose up to ``target_samples`` (structure, residue, mutant) samples.

    ``struct_targets`` is a list of ``(pdb_id, pdb_path, targets)`` where ``targets`` is
    already contact-filtered and sorted by tightness. Round-robins across structures so the
    budget spreads over as many complexes as possible; deterministic under ``seed``.
    """
    rng = random.Random(seed)
    queues: List[Tuple[str, str, List[Tuple[MutationTarget, str]]]] = []
    for pdb_id, pdb_path, targets in struct_targets:
        q = _structure_queue(targets, per_structure_cap, rng)
        if q:
            queues.append((pdb_id, pdb_path, q))

    selected: List[Sample] = []
    cursors = [0] * len(queues)
    made_progress = True
    while made_progress and (target_samples <= 0 or len(selected) < target_samples):
        made_progress = False
        for si, (pdb_id, pdb_path, q) in enumerate(queues):
            if cursors[si] < len(q):
                t, mut = q[cursors[si]]
                selected.append((pdb_id, pdb_path, t, mut))
                cursors[si] += 1
                made_progress = True
                if target_samples > 0 and len(selected) >= target_samples:
                    break
    return selected


def collect_struct_targets(
    pdb_files: Sequence[str],
    tcr_chains: Sequence[str],
    complex_chains: Sequence[str],
    interface_radius: float,
    max_interchain_dist: float,
) -> Tuple[List[Tuple[str, str, List[MutationTarget]]], int]:
    """Parse every structure once and return contact-filtered, tightness-sorted targets.

    Returns ``(struct_targets, n_dropped_residues)``. A residue is kept only if it is a
    *genuine* inter-chain contact: ``0 <= min_interchain_dist <= max_interchain_dist``
    (``-1`` means the distance could not be computed, so it is dropped).
    """
    parser = PDBParser(QUIET=True)
    struct_targets: List[Tuple[str, str, List[MutationTarget]]] = []
    n_dropped = 0
    for pdb_path in pdb_files:
        pdb_id = Path(pdb_path).stem.lower()
        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            targets = find_interface_targets(structure, tcr_chains, complex_chains, interface_radius)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s: interface detection failed (%s); skipping.", pdb_id, exc)
            continue
        if not targets:
            LOGGER.warning("%s: no interface targets found; skipping.", pdb_id)
            continue
        if max_interchain_dist and max_interchain_dist > 0:
            kept = [t for t in targets if 0.0 <= t.min_interchain_dist <= max_interchain_dist]
            n_dropped += len(targets) - len(kept)
        else:
            kept = list(targets)
        if not kept:
            continue
        # Tightest genuine contacts first; ties broken by proximity to the CDR centre.
        kept.sort(key=lambda t: (t.min_interchain_dist, t.distance_to_cdr))
        struct_targets.append((pdb_id, pdb_path, kept))
    return struct_targets, n_dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Rosetta Flex ddG resfiles + jobs.csv for TCR-pMHC interfaces.")
    ap.add_argument("--pdb_dir", type=str, default="stcrdab_structures/", help="Directory of standardized TCR-pMHC PDBs (default: the same Tier-1 STCRDab set).")
    ap.add_argument("--out_dir", type=str, default="rosetta_flex/jobs", help="Where resfiles + jobs.csv are written.")
    ap.add_argument("--interface_radius", type=float, default=10.0, help="Interface radius (must match the MadraX run for comparable positions).")
    ap.add_argument("--target_samples", type=int, default=20000, help="Total number of mutations to enumerate (budget). Pass 0 (or negative) for the old full-saturation enumeration.")
    ap.add_argument("--per_structure_cap", type=int, default=100, help="Max mutations drawn from any single structure (diversity guard). 0 = no cap.")
    ap.add_argument("--max_interchain_dist", type=float, default=8.0, help="Keep only residues within this many Angstrom of the opposite binding partner (genuine contacts). 0 = keep all interface residues.")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic seed for mutant ordering / tie-breaking.")
    ap.add_argument("--tcr_chains", type=str, default=",".join(DEFAULT_TCR_CHAINS))
    ap.add_argument("--complex_chains", type=str, default=",".join(DEFAULT_COMPLEX_CHAINS))
    ap.add_argument("--chains_to_move", type=str, default="".join(DEFAULT_TCR_CHAINS),
                    help="Chains separated to compute binding ddG (default 'DE' = the TCR moves off the pMHC).")
    ap.add_argument("--limit", type=int, default=None, help="Only parse the first N PDBs (dry runs).")
    args = ap.parse_args()

    tcr_chains = tuple(c.strip() for c in args.tcr_chains.split(",") if c.strip())
    complex_chains = tuple(c.strip() for c in args.complex_chains.split(",") if c.strip())
    out_dir = Path(args.out_dir)
    resfile_dir = out_dir / "resfiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(glob.glob(str(Path(args.pdb_dir) / "*.pdb")))
    if args.limit:
        pdb_files = pdb_files[: args.limit]
    if not pdb_files:
        LOGGER.error("No PDB files found in %s.", args.pdb_dir)
        return

    struct_targets, n_dropped = collect_struct_targets(
        pdb_files, tcr_chains, complex_chains, args.interface_radius, args.max_interchain_dist,
    )
    if not struct_targets:
        LOGGER.error("No structures with genuine interface contacts found (check chains / radius / --max_interchain_dist).")
        return

    n_contacts = sum(len(t) for _, _, t in struct_targets)
    LOGGER.info(
        "Parsed %d PDBs -> %d structures with contacts, %d contact residues kept "
        "(dropped %d second-shell residues beyond %.1f A).",
        len(pdb_files), len(struct_targets), n_contacts, n_dropped, args.max_interchain_dist,
    )

    selected = select_samples(struct_targets, args.target_samples, args.per_structure_cap, args.seed)
    if not selected:
        LOGGER.error("Sampling produced no jobs.")
        return

    # Write resfiles + jobs.csv in the selected order.
    jobs_path = out_dir / "jobs.csv"
    per_struct = Counter()
    dist_hist = Counter()
    with open(jobs_path, "w", newline="") as jf:
        writer = csv.DictWriter(jf, fieldnames=JOBS_FIELDS)
        writer.writeheader()
        for job_id, (pdb_id, pdb_path, t, mut_one) in enumerate(selected):
            wt_one = t.wt_aa_one
            resfile = resfile_dir / pdb_id / f"{t.chain}{t.resnum}{t.icode}{wt_one}{mut_one}.resfile"
            write_resfile(resfile, t.chain, t.resnum, t.icode, mut_one)
            writer.writerow(
                {
                    "job_id": job_id,
                    "pdb_id": pdb_id,
                    "pdb_path": pdb_path,
                    "chain": t.chain,
                    "resnum": t.resnum,
                    "wt_aa": wt_one,
                    "mut_aa": mut_one,
                    "resfile_path": str(resfile),
                    "chains_to_move": args.chains_to_move,
                    "min_interchain_dist": round(t.min_interchain_dist, 3),
                }
            )
            per_struct[pdb_id] += 1
            dist_hist[min(int(t.min_interchain_dist), 9)] += 1

    n_jobs = len(selected)
    n_struct = len(per_struct)
    LOGGER.info("Wrote %d jobs across %d structures to %s", n_jobs, n_struct, jobs_path)
    if args.target_samples > 0 and n_jobs < args.target_samples:
        LOGGER.warning(
            "Only %d/%d requested samples were available. Raise --per_structure_cap (now %d), "
            "--max_interchain_dist (now %.1f A), or add more structures to reach the target.",
            n_jobs, args.target_samples, args.per_structure_cap, args.max_interchain_dist,
        )
    LOGGER.info("Per-structure draw: min/median/max = %d / %d / %d",
                min(per_struct.values()), sorted(per_struct.values())[len(per_struct) // 2], max(per_struct.values()))
    LOGGER.info("Contact-distance histogram (floor A -> count): %s",
                ", ".join(f"{k}:{dist_hist[k]}" for k in sorted(dist_hist)))
    LOGGER.info("Submit with the node-packed launcher: see rosetta_flex/plan_run.py to size NODES/--time,")
    LOGGER.info("then `NJOBS=%d NODES=<n> sbatch --array=0-<NODES-1> rosetta_flex/submit_array.sbatch`.", n_jobs)


if __name__ == "__main__":
    main()
