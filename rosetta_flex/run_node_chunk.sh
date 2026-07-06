#!/usr/bin/env bash
# Node-local Flex ddG driver: run THIS node's stride of jobs.csv through
# run_flex_ddg.py, CORES_PER_NODE mutations at a time. One invocation per node
# (one SLURM array task == one node); see submit_array.sbatch.
#
# Why a stride, not a contiguous chunk: node t of NODES takes rows t, t+NODES,
# t+2*NODES, ... so per-mutation cost (which varies a little with interface size)
# is evenly balanced across nodes instead of front-loading the big early structures.
#
# Runnable standalone for testing (no SLURM), e.g. a 8-wide dry slice of node 0:
#   TASK_ID=0 NODES=1 CORES_PER_NODE=8 NJOBS=200 bash rosetta_flex/run_node_chunk.sh
#
# All knobs are env vars (defaults chosen for UC ARC 64-core EPYC nodes):
#   TASK_ID              this node's index (default $SLURM_ARRAY_TASK_ID or 0)
#   NODES                total nodes / array size                     (default 1)
#   CORES_PER_NODE       parallel workers on this node                (default 64)
#   NJOBS                rows in jobs.csv (auto: wc -l minus header)
#   JOBS_CSV             (default rosetta_flex/jobs/jobs.csv)
#   PARTS_DIR            per-job output CSVs                           (default rosetta_flex/results/parts)
#   WORK_DIR             per-job Rosetta scratch                      (default rosetta_flex/results/work)
#   PROTOCOL_XML         (default rosetta_flex/ddG_backrub.xml)
#   PYTHON               interpreter                                  (default python)
#   ROSETTA_SCRIPTS_BIN  rosetta_scripts binary                       (default rosetta_scripts.default.linuxgccrelease)
#   BACKRUB_TRIALS       backrub MC steps (default 1500 tuned <4h; 3500 = Graphinity)   NSTRUCT (default 1)
#   GAM_COEFFS           optional {score_type: weight} JSON (empty => nogam / total_score)
#   KEEP_WORKDIR         non-empty => keep each job's Rosetta scratch dir (structures, ddG.db3,
#                        struct.db3) after success, for debugging. Default deletes it (unbounded
#                        disk otherwise across ~20k jobs).
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

export TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
export NODES="${NODES:-1}"
export CORES_PER_NODE="${CORES_PER_NODE:-64}"
export JOBS_CSV="${JOBS_CSV:-rosetta_flex/jobs/jobs.csv}"
export PARTS_DIR="${PARTS_DIR:-rosetta_flex/results/parts}"
export WORK_DIR="${WORK_DIR:-rosetta_flex/results/work}"
export PROTOCOL_XML="${PROTOCOL_XML:-rosetta_flex/ddG_backrub.xml}"
# Default the Python launcher to the uv-managed venv when uv is present (so a local
# `bash run_node_chunk.sh` uses the same deps as `uv sync`), else bare python. Override
# with PYTHON=... (e.g. PYTHON=.venv/bin/python) to skip uv's per-call resolution.
if [ -z "${PYTHON:-}" ]; then
    if command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
        PYTHON="uv run python"
    else
        PYTHON="python"
    fi
fi
export PYTHON
export ROSETTA_SCRIPTS_BIN="${ROSETTA_SCRIPTS_BIN:-rosetta_scripts.default.linuxgccrelease}"
export BACKRUB_TRIALS="${BACKRUB_TRIALS:-1500}"   # tuned <4h default; 3500 = Graphinity full accuracy
export NSTRUCT="${NSTRUCT:-1}"
export GAM_COEFFS="${GAM_COEFFS:-}"

# ------------------------------------------------------------------ single-job mode
# Re-invoked by the dispatcher below as `"$0" --one <index>` under xargs -P. Kept as a
# self-exec (rather than an exported bash function) so it is portable across shells and
# needs no GNU parallel.
run_one() {
  local i="$1"
  local out="$PARTS_DIR/job_${i}.csv"
  if [ -s "$out" ]; then
    return 0   # already done (resume) -- never recompute a finished mutation
  fi
  local gam_arg=()
  [ -n "$GAM_COEFFS" ] && gam_arg=(--gam-coeffs "$GAM_COEFFS")
  local keep_arg=()
  [ -n "${KEEP_WORKDIR:-}" ] && keep_arg=(--keep-workdir)
  # ${arr[@]+"${arr[@]}"} expands to nothing when the array is empty *without*
  # tripping `set -u` on bash < 4.4 (macOS ships bash 3.2). A bare "${gam_arg[@]}"
  # on an empty array is an "unbound variable" error there and kills every worker.
  "$PYTHON" rosetta_flex/run_flex_ddg.py \
    --jobs_csv "$JOBS_CSV" \
    --job-index "$i" \
    --rosetta_bin "$ROSETTA_SCRIPTS_BIN" \
    --protocol_xml "$PROTOCOL_XML" \
    --backrub_trials "$BACKRUB_TRIALS" \
    --nstruct "$NSTRUCT" \
    --workdir "$WORK_DIR/job_${i}" \
    --output "$out" \
    ${gam_arg[@]+"${gam_arg[@]}"} \
    ${keep_arg[@]+"${keep_arg[@]}"} \
    || { echo "$(date '+%F %T') job ${i} FAILED (rosetta or parse error)" >&2; return 1; }
}

if [ "${1:-}" = "--one" ]; then
  run_one "$2"
  exit $?
fi

# ------------------------------------------------------------------ dispatch mode
if [ -z "${NJOBS:-}" ] || [ "${NJOBS}" -le 0 ]; then
  NJOBS=$(( $(wc -l < "$JOBS_CSV") - 1 ))
fi
mkdir -p "$PARTS_DIR" "$WORK_DIR"

# Build this node's remaining (not-yet-done) index list.
todo=()
for (( i=TASK_ID; i<NJOBS; i+=NODES )); do
  [ -s "$PARTS_DIR/job_${i}.csv" ] || todo+=("$i")
done

echo "=================================================================="
echo " Flex ddG node ${TASK_ID}/${NODES}  |  cores=${CORES_PER_NODE}"
echo " jobs.csv rows   : ${NJOBS}"
echo " this node's todo: ${#todo[@]} (stride ${TASK_ID}::${NODES}, resume-filtered)"
echo " settings        : backrub=${BACKRUB_TRIALS} nstruct=${NSTRUCT} gam=${GAM_COEFFS:-<nogam>}"
echo " rosetta bin     : ${ROSETTA_SCRIPTS_BIN}"
echo "=================================================================="

if [ "${#todo[@]}" -eq 0 ]; then
  echo "Nothing to do on this node (all assigned jobs already have outputs)."
  exit 0
fi

# Feed the todo indices to CORES_PER_NODE concurrent workers. xargs -P is used (no GNU
# parallel dependency); each worker re-execs this script in single-job mode. Invoke via
# an explicit interpreter ("$BASH" or bash) rather than bare "$0": xargs execs its command
# directly, so a bare "$0" would need this file to carry the executable bit — which a fresh
# git checkout may not preserve, silently failing every worker with EACCES.
_self="${BASH_SOURCE[0]:-$0}"
printf '%s\n' "${todo[@]}" \
  | xargs -P "$CORES_PER_NODE" -I{} "${BASH:-bash}" "$_self" --one {}

# Report completion for this node's slice.
done_ct=0
for (( i=TASK_ID; i<NJOBS; i+=NODES )); do
  [ -s "$PARTS_DIR/job_${i}.csv" ] && done_ct=$((done_ct + 1))
done
total_slice=$(( (NJOBS - TASK_ID + NODES - 1) / NODES ))
echo "Node ${TASK_ID} finished: ${done_ct}/${total_slice} of its jobs have outputs."
