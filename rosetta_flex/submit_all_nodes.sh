#!/usr/bin/env bash
# Submit-time wrapper for submit_array.sbatch: sizes the array to use every
# currently-idle node in a partition, instead of a fixed/guessed node count.
#
# Why this has to be a separate submit-time script (not logic inside submit_array.sbatch
# itself): by the time an sbatch script's body runs, SLURM has already allocated the
# nodes for that job -- there's no way to ask "how many nodes are free" from inside the
# job. Querying `sinfo` and passing the answer via --array has to happen BEFORE sbatch.
#
# Usage:
#   bash rosetta_flex/submit_all_nodes.sh -p <partition> [-a <account>] [-m <max_nodes>] \
#       [-- <extra args passed through to sbatch, e.g. --time=08:00:00>]
#
# Env vars (same meaning as run_node_chunk.sh / submit_array.sbatch):
#   BACKRUB_TRIALS, NSTRUCT, GAM_COEFFS, ROSETTA_SCRIPTS_BIN, KEEP_WORKDIR
#
# Examples:
#   bash rosetta_flex/submit_all_nodes.sh -p gpu-preempt   # (any CPU partition name works)
#   bash rosetta_flex/submit_all_nodes.sh -p community -m 20 -- --time=06:00:00
set -euo pipefail

PARTITION=""
ACCOUNT=""
MAX_NODES=""
SBATCH_EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--partition) PARTITION="$2"; shift 2 ;;
    -a|--account)   ACCOUNT="$2"; shift 2 ;;
    -m|--max_nodes) MAX_NODES="$2"; shift 2 ;;
    --) shift; SBATCH_EXTRA=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PARTITION" ]; then
  echo "Usage: $0 -p <partition> [-a <account>] [-m <max_nodes>] [-- <extra sbatch args>]" >&2
  echo "(Need -p/--partition so sinfo knows which node pool to size the array against.)" >&2
  exit 2
fi

if ! command -v sinfo >/dev/null 2>&1; then
  echo "sinfo not found -- this must run on a SLURM login/submit node, not locally." >&2
  exit 1
fi

# Idle + mixed nodes are both usable capacity right now (mixed = partially allocated but
# SLURM will still schedule a whole-node --exclusive job onto it once it frees up enough,
# and --exclusive jobs queue rather than fail if none are free yet). We count idle nodes
# specifically since --exclusive needs a FULLY free node to start immediately; sizing off
# idle-only avoids requesting more nodes than can plausibly start soon.
idle_nodes="$(sinfo -h -p "$PARTITION" -t idle -o "%D" 2>/dev/null | paste -sd+ - | bc 2>/dev/null || true)"
if [ -z "$idle_nodes" ] || [ "$idle_nodes" -le 0 ] 2>/dev/null; then
  echo "sinfo reported 0 idle nodes in partition '$PARTITION' right now." >&2
  echo "(sbatch --exclusive jobs will still queue and start as nodes free up, but this" >&2
  echo "script only sizes the array off nodes that are idle THIS INSTANT. Pass -m to" >&2
  echo "force a specific node count instead, e.g. -m 8, if you'd rather queue for capacity.)" >&2
  exit 1
fi
NODES="$idle_nodes"

if [ -n "$MAX_NODES" ] && [ "$NODES" -gt "$MAX_NODES" ]; then
  echo "Capping requested nodes: $NODES idle -> using --max_nodes=$MAX_NODES."
  NODES="$MAX_NODES"
fi

JOBS_CSV="${JOBS_CSV:-rosetta_flex/jobs/jobs.csv}"
if [ ! -f "$JOBS_CSV" ]; then
  echo "jobs.csv not found at $JOBS_CSV -- run make_mutfiles.py first." >&2
  exit 1
fi
NJOBS=$(( $(wc -l < "$JOBS_CSV") - 1 ))

echo "Partition '$PARTITION': $NODES idle node(s) right now -- sizing --array=0-$((NODES-1))."
echo "jobs.csv: $NJOBS jobs across $NODES node(s)."

account_args=()
[ -n "$ACCOUNT" ] && account_args=(--account="$ACCOUNT")

sbatch \
  --partition="$PARTITION" \
  "${account_args[@]+"${account_args[@]}"}" \
  --array=0-$((NODES - 1)) \
  --export="ALL,NJOBS=$NJOBS,NODES=$NODES" \
  "${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"}" \
  rosetta_flex/submit_array.sbatch
