#!/bin/bash
#SBATCH --job-name=flex_ddg
#SBATCH --array=0-12             # tuned <4h config = 13 nodes. OVERRIDE: --array=0-<NODES-1>. ONE task == ONE node.
#SBATCH --nodes=1
#SBATCH --exclusive              # own the whole 64-core EPYC node; we pack it ourselves
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # UC ARC dual-EPYC node = 64 cores; matches CORES_PER_NODE
#SBATCH --time=04:00:00          # <4h target at backrub=1500, 13 nodes (~3.8h est). plan_run.py/estimate_runtime.py size this.
#SBATCH --output=rosetta_flex/logs/%A_%a.out
#SBATCH --error=rosetta_flex/logs/%A_%a.err
#
# Tier-2 Rosetta Flex ddG NODE-PACKED array driver.
#
# Model: instead of one 1-core task per mutation (20k tasks blows past array-index caps
# and wastes 64-core nodes), each array task owns ONE node and runs its stride of jobs.csv
# 64-at-a-time via run_node_chunk.sh. So a tiny array (0..NODES-1) drives the whole 20k set.
#
# Usage (after make_mutfiles.py wrote jobs.csv; NODES/backrub from estimate_runtime.py):
#   NJOBS=$(($(wc -l < rosetta_flex/jobs/jobs.csv) - 1))
#   NODES=13                                        # tuned <4h operating point
#   sbatch --array=0-$((NODES-1)) --export=ALL,NJOBS=$NJOBS,NODES=$NODES,BACKRUB_TRIALS=1500 \
#          rosetta_flex/submit_array.sbatch
#
# Everything is resumable: run_node_chunk.sh / run_flex_ddg.py skip any mutation whose
# results/parts/job_<i>.csv already exists, so a requeue picks up where it left off.
#
# Site knobs (set via --export or the environment):
#   NODES               total nodes == array size            (must match --array=0-<NODES-1>; tuned=13)
#   NJOBS               rows in jobs.csv (auto if unset)
#   CORES_PER_NODE      workers per node                     (default 64)
#   ROSETTA_SCRIPTS_BIN path to rosetta_scripts binary       (or `module load rosetta`)
#   BACKRUB_TRIALS      backrub MC steps (default 1500 tuned; 3500 = Graphinity full)   NSTRUCT (default 1)
#   GAM_COEFFS          optional reweighting JSON (default: auto-use gam_coeffs/flex_ddg_gam.json)

set -euo pipefail

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
export CORES_PER_NODE="${CORES_PER_NODE:-${SLURM_CPUS_PER_TASK:-64}}"
export NJOBS="${NJOBS:-0}"           # 0 => run_node_chunk.sh auto-counts from jobs.csv

bash rosetta_flex/run_node_chunk.sh
