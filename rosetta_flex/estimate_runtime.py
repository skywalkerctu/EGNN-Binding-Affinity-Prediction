#!/usr/bin/env python
"""
Estimate the Flex ddG run's CPU cost and wall time WITHOUT running Rosetta.

Why this can be done a priori
-----------------------------
Per-mutation cost decomposes into (see ddG_backrub.xml):
  * backrub  -- ~fixed for a given ``ntrials`` (size-independent to first order),
  * repack   -- scales with the 8 A mutation neighbourhood,
  * minimize -- scales with the whole-pose size.
Both size-dependent terms are computable straight from the PDBs with BioPython (residue
counts + an 8 A neighbour search) -- no force field, no Rosetta. The only quantity pure
geometry cannot give is the absolute per-trial constant on this CPU, so we anchor it to a
literature central value (``--anchor_min_per_mut``) and let you rescale exactly with one real
timing (``--benchmark_min_per_mut`` from ``plan_run.py --benchmark``).

Crucially, TCR-pMHC complexes are size-homogeneous (~500-600 residues each), so the
size-dependent spread is small and this estimate generalises across the whole 20k set from
just a handful of parsed structures.

Cost model (transparent, two-component)
---------------------------------------
    t_mut(job) = T_ref * [ alpha * (ntrials / anchor_ntrials)
                         + (1 - alpha) * (n_pose(job) / anchor_pose_res) ]

  alpha         = size-independent (backrub) fraction of a reference mutation (~0.6).
  T_ref         = per-mutation minutes at the reference pose size & anchor_ntrials
                  (literature anchor, or rescaled to a real benchmark mean).
  n_pose(job)   = standard-residue count of that job's complex.

Neighbourhood size is reported as a diagnostic (it drives repack/backrub locality) but pose
size is used as the single size proxy to avoid over-parameterising an a-priori model.

Usage
-----
    # from a real jobs.csv (exact), optimising for a 10h wall clock:
    python rosetta_flex/estimate_runtime.py --jobs_csv rosetta_flex/jobs/jobs.csv --target_hours 10
    # project a small parsed set (e.g. 2 structures) to the full 20k run:
    python rosetta_flex/estimate_runtime.py --jobs_csv rosetta_flex/jobs/jobs.csv \
        --project_samples 20000 --target_hours 10
    # rescale to a measured per-mutation time (removes the hardware unknown):
    python rosetta_flex/estimate_runtime.py --benchmark_min_per_mut 9.4 --target_hours 10
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Bio.PDB import NeighborSearch, PDBParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("estimate_runtime")


def parse_structure_sizes(pdb_path: str, neighborhood_radius: float):
    """Return (n_pose_residues, all_atoms NeighborSearch, residue_index) for one complex.

    residue_index maps (chain_id, resnum) -> Residue so job rows can be located to count the
    8 A mutation neighbourhood. Only standard residues (blank hetflag) are counted, matching
    what the pipeline mutates.
    """
    structure = PDBParser(QUIET=True).get_structure(Path(pdb_path).stem, pdb_path)
    model = next(structure.get_models())
    atoms = []
    res_index: Dict[Tuple[str, int], object] = {}
    n_pose = 0
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            n_pose += 1
            res_index.setdefault((chain.id, residue.id[1]), residue)
            atoms.extend(residue.get_atoms())
    ns = NeighborSearch(atoms) if atoms else None
    return n_pose, ns, res_index


def neighborhood_size(ns, residue, radius: float) -> int:
    """# standard residues within ``radius`` A of any atom of ``residue`` (incl. itself)."""
    if ns is None or residue is None:
        return 0
    nbr = set()
    for atom in residue.get_atoms():
        nbr.update(ns.search(atom.coord, radius, level="R"))
    return len([r for r in nbr if r.id[0] == " "])


def load_jobs(jobs_csv: Path) -> List[dict]:
    with open(jobs_csv, newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_pdb_path(row: dict, pdb_dir: Optional[Path]) -> Optional[str]:
    """jobs.csv stores an absolute-ish pdb_path; fall back to <pdb_dir>/<pdb_id>.pdb."""
    p = row.get("pdb_path", "")
    if p and Path(p).exists():
        return p
    if pdb_dir is not None:
        cand = pdb_dir / f"{row['pdb_id']}.pdb"
        if cand.exists():
            return str(cand)
    return None


def measure_jobs(jobs: List[dict], pdb_dir: Optional[Path], radius: float):
    """Per-job (n_pose, n_nbr), parsing each unique PDB once. Skips rows whose PDB is missing."""
    per_pdb_cache: Dict[str, Tuple[int, object, dict]] = {}
    pose_sizes: List[int] = []
    nbr_sizes: List[int] = []
    n_missing = 0
    for row in jobs:
        path = resolve_pdb_path(row, pdb_dir)
        if path is None:
            n_missing += 1
            continue
        if path not in per_pdb_cache:
            per_pdb_cache[path] = parse_structure_sizes(path, radius)
        n_pose, ns, res_index = per_pdb_cache[path]
        try:
            resnum = int(row["resnum"])
        except (KeyError, ValueError):
            resnum = None
        residue = res_index.get((row.get("chain", ""), resnum)) if resnum is not None else None
        pose_sizes.append(n_pose)
        nbr_sizes.append(neighborhood_size(ns, residue, radius))
    return pose_sizes, nbr_sizes, n_missing, len(per_pdb_cache)


def per_mut_minutes(n_pose: int, ntrials: int, T_ref: float, alpha: float,
                    anchor_pose_res: float, anchor_ntrials: int) -> float:
    return T_ref * (alpha * (ntrials / anchor_ntrials)
                    + (1.0 - alpha) * (n_pose / anchor_pose_res))


def plan_nodes(n_jobs: int, mean_min: float, p95_min: float, cores: int,
               target_hours: float, max_nodes: int, margin: float) -> Optional[Tuple[int, float]]:
    """Minimum node count whose per-core serial slice fits target_hours (using p95 for safety)."""
    ceiling = target_hours * 60.0
    for nodes in range(1, max_nodes + 1):
        serial = math.ceil(n_jobs / (nodes * cores))
        wall = serial * p95_min * margin
        if wall <= ceiling:
            return nodes, wall
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs_csv", type=Path, default=Path("rosetta_flex/jobs/jobs.csv"))
    ap.add_argument("--pdb_dir", type=Path, default=Path("stcrdab_structures"),
                    help="Fallback dir to locate <pdb_id>.pdb when jobs.csv paths are absent.")
    ap.add_argument("--project_samples", type=int, default=None,
                    help="Report totals for this many jobs (default: actual jobs.csv rows). Use to "
                         "project a few parsed structures to the full run.")
    ap.add_argument("--ntrials", type=int, default=3500, help="backrub trials for THIS run (default 3500 = Graphinity full accuracy).")
    ap.add_argument("--neighborhood_radius", type=float, default=8.0, help="8 A matches the protocol.")
    # Literature anchor (central per-mutation minutes at the reference size/ntrials).
    ap.add_argument("--anchor_min_per_mut", type=float, default=12.0)
    ap.add_argument("--anchor_pose_res", type=float, default=560.0)
    ap.add_argument("--anchor_ntrials", type=int, default=3500)
    ap.add_argument("--alpha", type=float, default=0.6, help="Size-independent (backrub) fraction ~0.5-0.7.")
    ap.add_argument("--benchmark_min_per_mut", type=float, default=None,
                    help="If given, rescale so the predicted MEAN equals this measured value (exact anchor).")
    # Cluster / target.
    ap.add_argument("--cores_per_node", type=int, default=64)
    ap.add_argument("--target_hours", type=float, default=8.0, help="Wall-clock ceiling to optimise for.")
    ap.add_argument("--max_nodes", type=int, default=64,
                    help="Node ceiling for this ESTIMATE's tradeoff table (submit_all_nodes.sh sizes "
                         "the real launch off whatever's actually idle in your partition, not this).")
    ap.add_argument("--margin", type=float, default=1.2, help="Safety multiplier on estimated wall time.")
    ap.add_argument("--ntrials_grid", type=str, default="3500,2500,2000,1500,1000,750",
                    help="ntrials values to show in the <target_hours> tradeoff table.")
    args = ap.parse_args()

    if not args.jobs_csv.exists():
        LOGGER.error("jobs.csv not found at %s -- run make_mutfiles.py first.", args.jobs_csv)
        sys.exit(1)
    jobs = load_jobs(args.jobs_csv)
    if not jobs:
        LOGGER.error("jobs.csv is empty.")
        sys.exit(1)

    pose_sizes, nbr_sizes, n_missing, n_pdbs = measure_jobs(jobs, args.pdb_dir, args.neighborhood_radius)
    if not pose_sizes:
        LOGGER.error("Could not locate any referenced PDBs (looked in jobs.csv paths and --pdb_dir %s).", args.pdb_dir)
        sys.exit(1)
    if n_missing:
        LOGGER.warning("%d/%d job rows referenced a PDB that could not be found; estimating from the %d measured.",
                       n_missing, len(jobs), len(pose_sizes))

    mean_pose = statistics.mean(pose_sizes)
    LOGGER.info("Parsed %d unique complexes | pose residues min/mean/max = %d / %.0f / %d | "
                "8A neighbourhood residues min/mean/max = %d / %.1f / %d",
                n_pdbs, min(pose_sizes), mean_pose, max(pose_sizes),
                min(nbr_sizes), statistics.mean(nbr_sizes), max(nbr_sizes))

    # Per-job predicted minutes at the requested ntrials.
    T_ref = args.anchor_min_per_mut
    def times_for(ntrials, tref):
        return [per_mut_minutes(p, ntrials, tref, args.alpha, args.anchor_pose_res, args.anchor_ntrials)
                for p in pose_sizes]

    times = times_for(args.ntrials, T_ref)
    if args.benchmark_min_per_mut is not None:
        # Rescale T_ref so the predicted mean matches the measured benchmark (keeps size spread shape).
        cur_mean = statistics.mean(times)
        T_ref *= args.benchmark_min_per_mut / cur_mean
        times = times_for(args.ntrials, T_ref)
        LOGGER.info("Rescaled to benchmark %.2f min/mut (T_ref -> %.2f).", args.benchmark_min_per_mut, T_ref)

    mean_min = statistics.mean(times)
    p95_min = sorted(times)[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)]
    n_jobs = args.project_samples or len(jobs)
    total_core_hours = n_jobs * mean_min / 60.0

    src = "measured jobs.csv" if args.project_samples is None else f"projected to {n_jobs} samples"
    LOGGER.info("Per-mutation estimate (%s): mean=%.2f min, p95=%.2f min%s.",
                src, mean_min, p95_min,
                "" if args.benchmark_min_per_mut is not None else " (literature-anchored; +-~2x until benchmarked)")
    LOGGER.info("Total work for %d jobs: %.0f CPU-hours (%.0f CPU-min).",
                n_jobs, total_core_hours, n_jobs * mean_min)

    # ---- <target_hours> optimisation: min nodes per ntrials choice --------------------------
    # Objective: keep the MOST backbone sampling (highest ntrials) that still fits target_hours
    # within the --max_nodes budget. Lowering ntrials past the repack/minimize floor barely saves
    # nodes, so this picks the accuracy-preserving knee automatically for a given node budget.
    grid = [int(x) for x in args.ntrials_grid.split(",") if x.strip()]
    print(f"\n========== fitting under {args.target_hours:.1f} h "
          f"(<= {args.max_nodes} nodes, cores/node={args.cores_per_node}, margin={args.margin}, {n_jobs} jobs) ==========")
    print("  ntrials | mean min/mut | min nodes | cores | est wall (h) | within budget")
    print("  --------+--------------+-----------+-------+--------------+--------------")
    rows = []
    for nt in sorted(grid, reverse=True):
        t = times_for(nt, T_ref)
        m = statistics.mean(t)
        p95 = sorted(t)[min(len(t) - 1, math.ceil(0.95 * len(t)) - 1)]
        # min nodes to fit the TIME (ignore budget here so we can show the true requirement)
        res = plan_nodes(n_jobs, m, p95, args.cores_per_node, args.target_hours, 999, args.margin)
        if res is None:
            print(f"  {nt:7d} | {m:12.2f} | {'>999':>9} | {'-':>5} | {'-':>12} | no")
            continue
        nodes, wall = res
        within = nodes <= args.max_nodes
        print(f"  {nt:7d} | {m:12.2f} | {nodes:9d} | {nodes*args.cores_per_node:5d} | {wall/60:12.1f} | "
              f"{'yes' if within else 'NO (>'+str(args.max_nodes)+')'}")
        rows.append((nt, nodes, wall, within))

    fitting = [r for r in rows if r[3]]
    if not fitting:
        cheapest = min(rows, key=lambda r: r[1]) if rows else None
        if cheapest:
            LOGGER.error("Even ntrials=%d needs %d nodes to hit %.1f h -- more than the %d-node budget. "
                         "Options: raise --max_nodes, accept a longer wall, or cut --project_samples "
                         "(e.g. ~%d samples would fit %d nodes).",
                         cheapest[0], cheapest[1], args.target_hours, args.max_nodes,
                         int(n_jobs * args.max_nodes / cheapest[1]), args.max_nodes)
        else:
            LOGGER.error("Nothing fits %.1f h at any grid ntrials. Cut samples or raise the time.", args.target_hours)
        return

    # Highest ntrials that fits within the node budget = most accuracy for the allowed nodes.
    nt, nodes, wall, _ = max(fitting, key=lambda r: r[0])
    hh, mm = divmod(int(min(args.target_hours * 60, math.ceil(wall / 5) * 5 + 20)), 60)
    walltime = f"{hh:02d}:{mm:02d}:00"
    benchmarked = args.benchmark_min_per_mut is not None
    banner = f"================ RECOMMENDED (<{args.target_hours:.0f} h) ================" if benchmarked else \
             f"========== RECOMMENDED (<{args.target_hours:.0f} h) -- UNVERIFIED ESTIMATE =========="
    print(f"\n{banner}")
    if not benchmarked:
        print("  !! literature-anchored (+-~2x), NOT measured on this build/hardware.")
        print("  !! Run `plan_run.py --benchmark 8` once Rosetta is built, then re-run this with")
        print("  !! --benchmark_min_per_mut <measured> before trusting this node count at scale.")
    print(f"  ntrials={nt}  NODES={nodes}  ({nodes*args.cores_per_node} cores)  est wall {wall/60:.1f} h")
    print(f"  NJOBS={n_jobs}")
    print(f"  sbatch --array=0-$((NODES-1)) --time={walltime} --cpus-per-task={args.cores_per_node} \\")
    print(f"    --export=ALL,NJOBS=$NJOBS,NODES={nodes},CORES_PER_NODE={args.cores_per_node},"
          f"BACKRUB_TRIALS={nt},ROSETTA_SCRIPTS_BIN=$ROSETTA_SCRIPTS_BIN \\")
    print(f"    rosetta_flex/submit_array.sbatch")
    print("=" * len(banner))


if __name__ == "__main__":
    main()
