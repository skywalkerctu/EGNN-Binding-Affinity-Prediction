# Tier 2 — Rosetta Flex ddG pipeline (TCR-pMHC)

Generates the Tier-2 "plasticity" dataset: backbone-flexible ΔΔG values from the
**Flex ddG** protocol (Barlow et al., *J. Phys. Chem. B* 2018), run on **TCR-pMHC**
structures. Tier 2 teaches the EGNN how backbones flex to absorb a mutation —
the rigid MadraX physics of Tier 1 cannot.

> **This pipeline targets an HPC cluster.** Rosetta is a large, separately-licensed
> C++ package; **build it on UC ARC first — see [`BUILD_ROSETTA.md`](BUILD_ROSETTA.md).**
> The node driver (`run_node_chunk.sh`) also runs **standalone on one machine**, filling
> all local cores, once a local Rosetta build is available — see
> [Local runs](#local-runs-all-cores-no-slurm) below. All Python helpers run anywhere
> under `uv run`.

> **Before submitting the full array, run ONE real job and check its `ddG.db3`.**
> `ddG_backrub.xml`'s ddG output only exists because `InterfaceDdGMover` carries
> `db_reporter="dbreport"` and `BackrubProtocol` carries `trajectory_apply_mover=...` —
> both are load-bearing wiring, not decoration, and Rosetta will not error if either is
> silently dropped by a version mismatch (see [BUILD_ROSETTA.md](BUILD_ROSETTA.md) Step 0
> on the pinned Rosetta version). `run_flex_ddg.py`'s parser (`parse_ddg_db3`) raises a
> loud `ValueError` naming exactly which batch(es) are missing if this ever goes wrong —
> **that check only protects you if you run it before the 20k-job array, not after.**
> See BUILD_ROSETTA.md Step 4 for the one-job smoke test.

## Target: ~20,000 samples, full-accuracy Graphinity settings, all available nodes

This tier is budgeted to **~20,000** high-value mutations (not full saturation), run at
**`backrub=3500`** (Graphinity/Hummer et al. 2025's validated full-accuracy setting — see
[Protocol parameters](#protocol-parameters-graphinity-2025-defaults-with-a-throughput-tuned-option)
below) across **every node your SLURM partition currently has idle**, not a fixed node count:

- **Budgeted, contact-ranked sampling** (`make_mutfiles.py --target_samples 20000`): keeps only
  genuine inter-chain contacts (via the shared `Min_Interchain_Distance`), ranks by contact
  tightness, caps per structure, and round-robins across complexes for diversity.
- **Node-packed execution**: one SLURM array task owns a whole node and runs its stride of
  `jobs.csv` all-cores-at-once (`run_node_chunk.sh`, which auto-detects the real core count of
  whatever node it lands on), so a *tiny* array (`0..NODES-1`) drives all 20k jobs — no
  array-index-cap problems, no idle cores.
- **All-nodes launch**: `submit_all_nodes.sh -p <partition>` queries `sinfo` for how many nodes
  are idle in your partition right now and sizes `--array` to use all of them, rather than a
  number chosen ahead of time — more nodes idle means a shorter wall-clock run, automatically.
- **Wall time is a function of node count, not a fixed target.** More idle nodes ⇒ done faster;
  fewer idle nodes ⇒ done slower (but never fails — SLURM just spreads the same 20k jobs across
  however many nodes you got). Use `estimate_runtime.py` / `plan_run.py --benchmark N` to see the
  expected wall time for a given node count before submitting, not to pick the node count itself.

### Estimating the runtime *before* touching the cluster

`estimate_runtime.py` predicts the CPU cost with **no Rosetta run** — it reads pose size and the
8 Å mutation neighbourhood straight from the PDBs and applies a two-component cost model
(backrub ≈ fixed per `ntrials`; repack/minimize ≈ pose size), anchored to a literature central
of ~12 min/mut. Because TCR-pMHC complexes are size-homogeneous (~500–600 residues), a couple
of parsed structures generalise to the whole 20k set. Example (projected to 20k, at the default
`backrub=3500`, showing the tradeoff across up to 64 nodes):

```bash
python rosetta_flex/estimate_runtime.py --jobs_csv rosetta_flex/jobs/jobs.csv \
  --project_samples 20000 --target_hours 8 --max_nodes 64
# validated on 1d9k (519 residues). --max_nodes here only bounds the printed TABLE --
# the real launch (submit_all_nodes.sh) uses however many nodes are actually idle.
```

The estimate is literature-anchored (±~2×); one real timing from `plan_run.py --benchmark 8`
fed back via `estimate_runtime.py --benchmark_min_per_mut <measured>` makes the estimate exact
for your hardware. If you're node-constrained and want a smaller node count to still finish in a
target window, pass a lower `--backrub_trials`/`BACKRUB_TRIALS` (e.g. `1500`) — see
[Protocol parameters](#protocol-parameters-graphinity-2025-defaults-with-a-throughput-tuned-option).

> **Note on Graphinity.** Graphinity ships a published Flex ddG set, but it is
> **antibody–antigen (SAbDab)**, a different binding problem. We do **not** use
> those numbers as labels. We reproduce the *protocol* here and run it on our own
> TCR-pMHC structures so the labels are TCR-relevant.

## Prerequisites
- A built Rosetta with `rosetta_scripts` (academic/commercial license from
  RosettaCommons). Point `ROSETTA_SCRIPTS_BIN` at the binary, or `module load rosetta`.
- The standardized TCR-pMHC PDBs in `stcrdab_structures/` (produced by
  `scripts/fetch_stcrdab.py` — the same inputs as Tier 1).
- Python env from the repo root (`uv sync`).

## Files
| File | Role |
|------|------|
| `ddG_backrub.xml` | Canonical flex_ddG RosettaScripts protocol (backrub ensemble + `InterfaceDdGMover`, talaris2014). |
| `make_mutfiles.py` | Budgeted, contact-ranked sampling (via shared `interface_utils`) → per-mutation resfiles + `jobs.csv`. |
| `estimate_runtime.py` | **A-priori** CPU-hour/wall estimate from structure geometry (no Rosetta); optimizes NODES/`ntrials` for a target wall clock. |
| `plan_run.py` | Benchmarks real mutations, sizes NODES/`--time` for the target, prints the exact `sbatch` command. |
| `run_flex_ddg.py` | Runs one `jobs.csv` row through Rosetta and parses `ddG.db3` → one shared-schema CSV row (idempotent/resumable, cleans up its scratch dir on success — pass `--keep-workdir` to inspect it). |
| `run_node_chunk.sh` | Node-local driver: runs this node's stride of `jobs.csv`, all-cores-at-once via `xargs -P` (auto-detects the real core count of whatever node it's on). |
| `submit_array.sbatch` | Node-packed SLURM array: one task == one whole node. Not usually invoked directly — see `submit_all_nodes.sh`. |
| `submit_all_nodes.sh` | Submit-time wrapper: queries `sinfo` for idle nodes in a partition and sizes/launches the array to use all of them. |
| `fetch_gam_coeffs.py` | Installs official Flex ddG reweighting coefficients (never fabricated). |
| `merge_results.py` | Concatenates per-job CSVs into one dataset + coverage/missing-job report. |
| `BUILD_ROSETTA.md` | One-time Rosetta build on UC ARC. |

## Workflow (on UC ARC)
```bash
# 0. One-time: build Rosetta and export ROSETTA_SCRIPTS_BIN  (see BUILD_ROSETTA.md)
#    This checkout also carries rosetta_flex/site_env.sh, which auto-loads the verified
#    local build under scratch/ for the submit scripts if you do not export your own path.

# 1. Enumerate the 20k-sample budget (safe to run anywhere; no Rosetta needed)
uv run python rosetta_flex/make_mutfiles.py --pdb_dir stcrdab_structures/ \
  --out_dir rosetta_flex/jobs --target_samples 20000

# 2. (optional) Install GAM reweighting coefficients for higher accuracy
uv run python rosetta_flex/fetch_gam_coeffs.py --from-tsv my_flexddg_weights.tsv

# 3. REQUIRED smoke test -- run ONE real job before submitting the full array (see the
#    warning near the top of this file and BUILD_ROSETTA.md Step 4)
uv run python rosetta_flex/run_flex_ddg.py --jobs_csv rosetta_flex/jobs/jobs.csv \
  --job-index 0 --rosetta_bin "$ROSETTA_SCRIPTS_BIN" \
  --workdir /tmp/flexddg_smoketest --output /tmp/flexddg_smoketest/out.csv --keep-workdir

# 4. (optional) See expected wall time for a given node count -- doesn't pick the count
python rosetta_flex/plan_run.py --benchmark 8 --max_hours 8 --max_nodes 64

# 5. Launch across every currently-idle node in your partition
bash rosetta_flex/submit_all_nodes.sh -p <partition>
# add -a <account> if your site requires one; -m <n> to cap the node count instead of
# using all idle nodes; -- <extra sbatch args> (e.g. -- --time=12:00:00) to override defaults.

# 6. Merge per-job outputs + see coverage (which jobs, if any, are missing)
uv run python rosetta_flex/merge_results.py
```
The run is **resumable**: every mutation writes `results/parts/job_<i>.csv` and finished jobs
are skipped, so a requeue (or re-running `submit_all_nodes.sh`, even against a different node
count) fills only the gaps.

## Protocol parameters (Graphinity 2025 defaults, with a throughput-tuned option)

Defaults follow the Flex ddG settings validated by Hummer et al.
([*Nat. Comput. Sci.* 2025](https://www.nature.com/articles/s43588-025-00823-8)):

- **Backrub trials: `3500`** (`--backrub_trials` / `BACKRUB_TRIALS`) — Graphinity's validated
  full-accuracy setting. Barlow 2018 used `35000`. If you're node-constrained and want to trade
  some accuracy for a shorter run on fewer nodes, pass `BACKRUB_TRIALS=1500` (a modest reduction —
  backrub is ~60% of per-mutation cost — keep it ≥1000 to avoid under-relaxing large mutations;
  see `estimate_runtime.py`'s `--ntrials_grid` for the tradeoff table at other values).
- **Ensemble size: `nstruct = 1`** (`--nstruct` / `NSTRUCT`) — vs Barlow 2018's `35`.
- Score function: `talaris2014`.
- `chains_to_move`: request `DE` by default, but `make_mutfiles.py` now stores the subset of those
  TCR chains actually present in each structure before Rosetta is launched.

At `backrub=3500`/`nstruct=1` these are ~**350× cheaper per mutation** than the full Barlow protocol
for **near-identical ΔΔG**; `backrub=1500` trims a further ~40% of backrub cost if you need it.
Why it holds up:
- ΔΔG is a **difference** (mutant − WT) scored in the *same* locally-relaxed
  backbone context, so most single-model/limited-sampling noise **cancels** — one
  relaxed model (`nstruct=1`) already captures the signal.
- 3,500 backrub steps already relax the *local* backbone enough to relieve the
  strain a point mutation introduces; extending to 35,000 mostly re-samples the
  same energy basin (diminishing returns).
- The accuracy ceiling here is Flex ddG's intrinsic correlation to experiment
  (~0.46), **not** ensemble size — so spending 350× compute to shave ensemble
  noise barely moves it. For an EGNN training set, **volume + diversity beats
  per-label precision** (the paper's central finding), so that compute is far
  better spent on *more* mutations.

Pass `--backrub_trials 35000 --nstruct 35` (or `BACKRUB_TRIALS`/`NSTRUCT` env
vars) to recover the full Barlow protocol.

- **GAM reweighting (recommended).** In Hummer et al., GAM-reweighted Flex ddG
  reached Pearson 0.46 vs experiment, beating both non-GAM Flex ddG (0.42) and
  FoldX (0.20). Pass `--gam-coeffs <json>` to `run_flex_ddg.py` with the official
  `{score_type: weight}` coefficients from the
  [flex_ddG tutorial](https://github.com/Kortemme-Lab/flex_ddG_tutorial) (drop the
  file in e.g. `rosetta_flex/gam_coeffs.json`). Without it, ΔΔG uses Rosetta's raw
  `total_score` (the uncalibrated "nogam" variant). We do not ship invented coefficients.

- **Insertion codes.** `make_mutfiles.py` writes the PDB insertion code into each
  resfile (`<resnum><icode> <chain> PIKAA ...`), so IMGT CDR3 insertion-coded
  positions are mutated correctly in Rosetta (which distinguishes them, unlike the
  MadraX integer-only parser).

## Output schema (shared with Tier 1)
```
PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid, ddG, source
```
`source = "rosetta_flex"`. ΔΔG sign convention is **mutant − wild type**, matching
the MadraX generator.

## Local runs (all cores, no SLURM)
`run_node_chunk.sh` runs standalone off-cluster: set `NODES=1 TASK_ID=0` and it auto-detects
your machine's real core count (via `nproc`/`sysctl`) to fan `jobs.csv` across every local core
via `xargs -P` — no need to pass `CORES_PER_NODE` yourself (override it only to use fewer than
all cores). `PYTHON` defaults to `uv run python` when uv is present, so the workers use the
`uv sync` environment. A local Rosetta build is still required (override `ROSETTA_SCRIPTS_BIN`,
e.g. `rosetta_scripts.default.macosclangrelease`).
```bash
uv run python rosetta_flex/make_mutfiles.py --pdb_dir stcrdab_structures/ \
  --out_dir rosetta_flex/jobs --target_samples 20000
NODES=1 TASK_ID=0 ROSETTA_SCRIPTS_BIN=/path/to/rosetta_scripts.<platform>release \
  bash rosetta_flex/run_node_chunk.sh
uv run python rosetta_flex/merge_results.py
```

## Local validation without Rosetta
```bash
uv run python rosetta_flex/run_flex_ddg.py --self-test          # synthetic db3 → parser check
uv run python rosetta_flex/run_flex_ddg.py --parse-only path/to/ddG.db3   # parse a real db3
uv run python rosetta_flex/make_mutfiles.py --pdb_dir stcrdab_structures/ --limit 1   # resfiles + jobs.csv
```
`--self-test` and `--parse-only` only validate the SQL against `parse_ddg_db3`'s expected
schema (`structure_scores` joined to `batches` on `batch_id`, classified by `batches.name`
prefix — see `ddG_backrub.xml`'s header comment); they cannot catch a Rosetta version that
silently drops the `db_reporter`/`trajectory_apply_mover` mover attributes and writes no
scores at all. Only a real job on the target Rosetta build (BUILD_ROSETTA.md Step 4) verifies
that.
