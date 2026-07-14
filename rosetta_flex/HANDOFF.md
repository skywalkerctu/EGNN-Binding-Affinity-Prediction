# Rosetta Flex ddG Handoff

Date: 2026-07-14

## Final Output

- Final merged dataset: `rosetta_flex/results/Synthetic_FlexddG_TCR_pMHC.csv`
- Final coverage: `17152 / 17152` jobs complete (`100.0%`)
- Structures covered: `173`
- Per-job outputs present: `17152`

## Final CSV Sanity Check

- Schema: `PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid, ddG, source`
- Row count: `17152`
- Missing fields: none
- Non-numeric `ddG` values: none
- Duplicate mutation keys on `(PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid)`: none
- `source` values: all `rosetta_flex`

### ddG Distribution

- `min`: `-3729.4597`
- `p01`: `-351.1609`
- `p05`: `-144.6589`
- `median`: `0.1006`
- `p95`: `150.2378`
- `p99`: `346.3160`
- `max`: `3353.0732`
- `mean`: `1.4221`
- `stdev`: `137.2735`

### Large-Magnitude Tail

- `183` rows have `ddG == 0.0`
- `668` rows have `|ddG| > 250`
- `165` rows have `|ddG| > 500`
- `54` rows have `|ddG| > 1000`
- `8` rows have `|ddG| > 2000`

Interpretation: the CSV is structurally sound, but the ddG distribution has a heavy tail. That is a modeling concern, not a file-integrity problem. Downstream training may want clipping, winsorization, or a transformed target.

## What Broke During The Run

The main runtime bug was not Rosetta availability or SLURM itself. It was a mixed-chain assumption in the current `stcrdab_structures/` tree:

- Some structures are not uniformly remapped to a full `A/B/C/D/E` layout.
- Several valid complexes only contain one of the nominal TCR move chains (`D` or `E`).
- Passing a fixed `D,E` move set into `InterfaceDdGMover` caused Rosetta failures like:
  - `InterfaceDdGMover cannot add chain_name D to pose; out of range`

This was confirmed on real failing examples such as `6bga`, `4apq`, and `6omg`.

## Fixes Applied

### Runner / Rosetta invocation

File: `rosetta_flex/run_flex_ddg.py`

- Blank `--rosetta_bin` values are treated as unset instead of reaching `subprocess.run()` as an empty executable.
- Rosetta binary resolution now accepts:
  - explicit executable path
  - PATH lookup
  - glob-expanded path
  - Rosetta `bin/` directory
- `--protocol_xml` is resolved to an absolute existing path before launching Rosetta.
- Runtime work/output directories are normalized to absolute paths so `ReportToDB` does not reinterpret paths relative to the scratch cwd.
- Compact `chains_to_move` values from `jobs.csv` are normalized for Rosetta XML usage.
- Requested move chains are intersected with the chains actually present in the input PDB before constructing `chainstomove=...`.

### Job generation

File: `rosetta_flex/make_mutfiles.py`

- `jobs.csv` no longer blindly writes one global `DE` move set for every structure.
- It now stores the subset of requested TCR move chains actually present in each PDB.

### Cluster submission / resume behavior

Files:

- `rosetta_flex/site_env.sh`
- `rosetta_flex/submit_all_nodes.sh`
- `rosetta_flex/submit_array.sbatch`
- `rosetta_flex/run_node_chunk.sh`

Changes:

- Repo-local Rosetta site configuration was added.
- Submit/worker scripts were made cwd-independent and path-stable.
- Worker launch was fixed so the node driver actually runs `uv run python` correctly.
- Node driver now propagates worker failures clearly.
- Resume behavior was preserved: existing `results/parts/job_<i>.csv` files are skipped on reruns.

### Documentation

Files:

- `rosetta_flex/README.md`
- `rosetta_flex/BUILD_ROSETTA.md`

Changes:

- Rosetta setup instructions were corrected.
- `chains_to_move` documentation now matches the actual per-structure subset behavior.

## Run History Summary

1. Initial smoke test failed on empty or unresolved Rosetta binary handling.
2. Real Rosetta smoke test then exposed:
   - relative XML path issues
   - invalid compact chain formatting for Rosetta
3. Full array preparation exposed:
   - node driver launch bug
   - relative scratch/output path issue for sqlite outputs
4. First large array runs partially succeeded but left gaps because some structures only had one usable TCR move chain.
5. After patching chain-subset handling, a resumable backfill completed the remaining jobs.

## Final Operational State

- No remaining missing jobs.
- No additional array resubmission is needed.
- Final merged CSV is ready for downstream post-processing or training.

## Suggested Next Step

If this dataset is going directly into model training, decide how to handle the extreme ddG tail before fitting:

- clip or winsorize large `|ddG|` values
- keep the raw target and use a robust loss
- compare training with and without extreme-value filtering