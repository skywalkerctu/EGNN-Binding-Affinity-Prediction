#!/usr/bin/env python
"""
Size the Tier-2 Flex ddG run so it finishes in <24h on UC ARC -- and print the exact
`sbatch` command to launch it.

The per-mutation cost of Flex ddG is unknown until Rosetta is built and timed on the real
node, and it drives everything (how many 64-core nodes to request, what --time to set). This
tool removes the guessing:

  1. BENCHMARK (needs Rosetta built): run a handful of real mutations sampled across
     jobs.csv single-threaded, and report the median / p95 wall time per mutation.
  2. PLAN: given that per-mutation time, the sample budget, cores/node, and a wall-clock
     ceiling, compute the *minimum* node count whose per-core serial slice fits the ceiling,
     then emit the ready-to-run `sbatch --array=... submit_array.sbatch` line (with a safety
     margin baked into --time).

Typical use on the cluster (after BUILD_ROSETTA.md):
    # measure, then plan, in one shot
    python rosetta_flex/plan_run.py --benchmark 12 --max_hours 24

Pre-planning before Rosetta exists (use an assumed rate):
    python rosetta_flex/plan_run.py --minutes_per_mut 12 --max_hours 24

This tool's printed `sbatch` command uses a FIXED node count. For the actual launch, prefer
`bash rosetta_flex/submit_all_nodes.sh -p <partition>` instead, which sizes the array to
however many nodes are idle in your partition right now (rather than a number chosen here) --
use this script to sanity-check expected wall time for that many nodes, not to pick the count.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("plan_run")


def count_jobs(jobs_csv: Path) -> int:
    with open(jobs_csv, newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def spread_indices(n_rows: int, k: int) -> List[int]:
    """k indices evenly spread across [0, n_rows) so the benchmark spans small+large interfaces."""
    if k >= n_rows:
        return list(range(n_rows))
    if k == 1:
        return [n_rows // 2]
    return sorted({round(i * (n_rows - 1) / (k - 1)) for i in range(k)})


def benchmark(args, n_rows: int) -> Optional[float]:
    """Time `--benchmark N` real mutations; return median minutes/mutation (or None on total failure)."""
    picks = spread_indices(n_rows, args.benchmark)
    LOGGER.info("Benchmarking %d mutations (indices %s) at backrub=%d nstruct=%d ...",
                len(picks), picks, args.backrub_trials, args.nstruct)
    times: List[float] = []
    for idx in picks:
        with tempfile.TemporaryDirectory(prefix="flexddg_bench_") as td:
            out = Path(td) / f"bench_{idx}.csv"
            cmd = [
                sys.executable, "rosetta_flex/run_flex_ddg.py",
                "--jobs_csv", str(args.jobs_csv), "--job-index", str(idx),
                "--rosetta_bin", args.rosetta_bin, "--protocol_xml", args.protocol_xml,
                "--backrub_trials", str(args.backrub_trials), "--nstruct", str(args.nstruct),
                "--workdir", str(Path(td) / f"work_{idx}"), "--output", str(out),
            ]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = (time.perf_counter() - t0) / 60.0
            if proc.returncode != 0:
                LOGGER.warning("benchmark job %d failed (rc=%d): %s", idx, proc.returncode,
                               proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "")
                continue
            LOGGER.info("  job %d: %.2f min", idx, dt)
            times.append(dt)
    if not times:
        LOGGER.error("All benchmark jobs failed -- is ROSETTA_SCRIPTS_BIN correct and Rosetta built?")
        return None
    times.sort()
    median = times[len(times) // 2]
    p95 = times[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)]
    LOGGER.info("Benchmark: n=%d  median=%.2f min  p95=%.2f min", len(times), median, p95)
    return median


def plan(n_jobs: int, minutes_per_mut: float, cores_per_node: int, max_hours: float,
         max_nodes: int, margin: float, benchmarked: bool) -> None:
    """Find the minimum node count whose per-core serial slice fits max_hours, and print sbatch."""
    ceiling_min = max_hours * 60.0
    total_core_hours = n_jobs * minutes_per_mut / 60.0
    LOGGER.info("Planning: %d jobs x %.2f min = %.0f core-min (%.0f core-hours) of work.",
                n_jobs, minutes_per_mut, n_jobs * minutes_per_mut, total_core_hours)
    LOGGER.info("One %d-core node in %.0fh delivers %.0f core-hours.",
                cores_per_node, max_hours, cores_per_node * max_hours)

    print("\n  nodes | cores | serial/core | est wall (h) | fits %.0fh?" % max_hours)
    print("  ------+-------+-------------+--------------+---------")
    chosen = None
    for nodes in range(1, max_nodes + 1):
        serial = math.ceil(n_jobs / (nodes * cores_per_node))  # jobs each core runs back-to-back
        wall_min = serial * minutes_per_mut * margin
        fits = wall_min <= ceiling_min
        star = " *" if (chosen is None and fits) else "  "
        print(f"  {nodes:5d} | {nodes*cores_per_node:5d} | {serial:11d} | {wall_min/60:12.1f} | {'yes' if fits else 'no ':>3}{star}")
        if chosen is None and fits:
            chosen = (nodes, serial, wall_min)

    if chosen is None:
        LOGGER.error(
            "Cannot fit %d jobs in %.0fh with <=%d nodes at %.1f min/mut. Options: raise --max_nodes, "
            "lower --target_samples in make_mutfiles.py, or cut --backrub_trials (e.g. 2500).",
            n_jobs, max_hours, max_nodes, minutes_per_mut,
        )
        return

    nodes, serial, wall_min = chosen
    # Request a little more walltime than the estimate (margin already applied), capped at the ceiling.
    req_min = min(ceiling_min, math.ceil(wall_min / 5) * 5 + 30)  # round up to 5 min, +30 min headroom
    hh, mm = divmod(int(req_min), 60)
    walltime = f"{hh:02d}:{mm:02d}:00"
    LOGGER.info("Recommended: %d node(s) (%d cores), est wall %.1fh, request --time=%s.",
                nodes, nodes * cores_per_node, wall_min / 60, walltime)
    banner = "================ LAUNCH COMMAND ================" if benchmarked else \
             "========== LAUNCH COMMAND (UNVERIFIED ESTIMATE) =========="
    print(f"\n{banner}")
    if not benchmarked:
        print("!! minutes_per_mut is a LITERATURE GUESS (+-~2x), not measured on this build/hardware.")
        print("!! Re-run with --benchmark 8+ (needs Rosetta built) before trusting this node count.")
    print(f"NJOBS={n_jobs}")
    print(f"NODES={nodes}")
    print("sbatch \\")
    print(f"  --array=0-$((NODES-1)) \\")
    print(f"  --time={walltime} \\")
    print(f"  --cpus-per-task={cores_per_node} \\")
    print(f"  --export=ALL,NJOBS=$NJOBS,NODES=$NODES,CORES_PER_NODE={cores_per_node},"
          f"ROSETTA_SCRIPTS_BIN=$ROSETTA_SCRIPTS_BIN \\")
    print(f"  rosetta_flex/submit_array.sbatch")
    print("=" * len(banner))
    print("(Export ROSETTA_SCRIPTS_BIN to your compiled binary first; add GAM_COEFFS=... to reweight.)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs_csv", type=Path, default=Path("rosetta_flex/jobs/jobs.csv"))
    ap.add_argument("--target_samples", type=int, default=None,
                    help="Plan for this many jobs (default: actual row count of jobs.csv).")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="Time this many real mutations to measure min/mut (needs Rosetta built). 0 = skip.")
    ap.add_argument("--minutes_per_mut", type=float, default=12.0,
                    help="Assumed per-mutation wall minutes when --benchmark is not used.")
    ap.add_argument("--cores_per_node", type=int, default=64,
                    help="Cores per node for this ESTIMATE (default 64; actual runs auto-detect "
                         "the real count per-node via run_node_chunk.sh, so this is just for planning math).")
    ap.add_argument("--max_hours", type=float, default=8.0, help="Wall-clock ceiling to plan against.")
    ap.add_argument("--max_nodes", type=int, default=64,
                    help="Node ceiling for this ESTIMATE's table (submit_all_nodes.sh uses however many "
                         "nodes are actually idle in your partition, not this value -- raise it here if "
                         "you just want to see the tradeoff table extend further).")
    ap.add_argument("--margin", type=float, default=1.3, help="Safety multiplier on the estimated wall time.")
    # Rosetta params (only used by --benchmark).
    ap.add_argument("--rosetta_bin", type=str,
                    default="rosetta_scripts.default.linuxgccrelease")
    ap.add_argument("--protocol_xml", type=str, default="rosetta_flex/ddG_backrub.xml")
    ap.add_argument("--backrub_trials", type=int, default=3500,
                    help="Benchmark at the value you'll run (default 3500 = Graphinity full accuracy).")
    ap.add_argument("--nstruct", type=int, default=1)
    args = ap.parse_args()

    if not args.jobs_csv.exists():
        LOGGER.error("jobs.csv not found at %s -- run make_mutfiles.py first.", args.jobs_csv)
        sys.exit(1)
    n_rows = count_jobs(args.jobs_csv)
    n_jobs = args.target_samples or n_rows
    LOGGER.info("jobs.csv has %d rows; planning for %d jobs.", n_rows, n_jobs)

    minutes_per_mut = args.minutes_per_mut
    benchmarked = False
    if args.benchmark > 0:
        measured = benchmark(args, n_rows)
        if measured is None:
            LOGGER.error("Benchmark failed; falling back to UNVERIFIED --minutes_per_mut=%.1f for the plan.", minutes_per_mut)
        else:
            minutes_per_mut = measured
            benchmarked = True
    else:
        LOGGER.info("No benchmark; using assumed %.1f min/mut. Run with --benchmark 12 on the cluster "
                    "for a real number.", minutes_per_mut)

    plan(n_jobs, minutes_per_mut, args.cores_per_node, args.max_hours, args.max_nodes, args.margin, benchmarked)


if __name__ == "__main__":
    main()
