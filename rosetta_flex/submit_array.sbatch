#!/bin/bash
#SBATCH --job-name=flex_ddg
#SBATCH --array=0-0              # OVERRIDDEN at submit time -- see submit_all_nodes.sh, which sizes
#                                 # this to every idle node in the partition. Do not sbatch this file
#                                 # directly unless you're intentionally passing your own --array.
#SBATCH --nodes=1
#SBATCH --exclusive              # own the whole node; we pack all its cores ourselves
#SBATCH --ntasks=1
#SBATCH --time=08:00:00          # generous default for backrub=3500 (Graphinity); trim via --time if
#                                 # plan_run.py/estimate_runtime.py size a shorter run for your node count.
#SBATCH --output=rosetta_flex/logs/%A_%a.out
#SBATCH --error=rosetta_flex/logs/%A_%a.err
#
# Tier-2 Rosetta Flex ddG NODE-PACKED array driver.
#
# Model: instead of one 1-core task per mutation (20k tasks blows past array-index caps
# and wastes multi-core nodes), each array task owns ONE node and runs its stride of
# jobs.csv all-cores-at-once via run_node_chunk.sh (which auto-detects the real core
# count of whatever node it lands on). So a tiny array (0..NODES-1) drives the whole set.
#
# Usage: prefer `bash rosetta_flex/submit_all_nodes.sh` (sizes NODES to every idle node
# in your partition automatically). To size/submit by hand instead:
#   NJOBS=$(($(wc -l < rosetta_flex/jobs/jobs.csv) - 1))
#   NODES=13
#   sbatch --array=0-$((NODES-1)) --export=ALL,NJOBS=$NJOBS,NODES=$NODES \
#          rosetta_flex/submit_array.sbatch
#
# Everything is resumable: run_node_chunk.sh / run_flex_ddg.py skip any mutation whose
# results/parts/job_<i>.csv already exists, so a requeue picks up where it left off.
#
# Site knobs (set via --export or the environment):
#   NODES               total nodes == array size            (must match --array=0-<NODES-1>)
#   NJOBS               rows in jobs.csv (auto if unset)
#   CORES_PER_NODE      workers per node                      (default: autodetect real core count)
#   ROSETTA_SCRIPTS_BIN path to rosetta_scripts binary       (or `module load rosetta`)
#   BACKRUB_TRIALS      backrub MC steps (default 3500 = Graphinity full; 1500 = throughput-tuned)   NSTRUCT (default 1)
#   GAM_COEFFS          optional reweighting JSON (default: auto-use gam_coeffs/flex_ddg_gam.json)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/site_env.sh" ] && . "$SCRIPT_DIR/site_env.sh"

# Load or locate Rosetta. UC ARC has no stock module yet (build per BUILD_ROSETTA.md), so
# ROSETTA_SCRIPTS_BIN should be exported to your compiled binary; the module load is a no-op
# fallback for sites that do ship one.
module load rosetta 2>/dev/null || true
export ROSETTA_SCRIPTS_BIN="${ROSETTA_SCRIPTS_BIN:-rosetta_scripts.default.linuxgccrelease}"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
export PROJECT_DIR
cd "$PROJECT_DIR"
mkdir -p rosetta_flex/logs rosetta_flex/results/parts rosetta_flex/results/work

export TASK_ID="${SLURM_ARRAY_TASK_ID}"
export NODES="${NODES:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}"
export NJOBS="${NJOBS:-0}"           # 0 => run_node_chunk.sh auto-counts from jobs.csv
# CORES_PER_NODE is intentionally left unset here: run_node_chunk.sh auto-detects the
# real core count of whichever node this task lands on (SLURM_CPUS_PER_TASK if the
# submitter set --cpus-per-task, else nproc), so every node is fully packed regardless
# of which partition/hardware it is.

bash rosetta_flex/run_node_chunk.sh
