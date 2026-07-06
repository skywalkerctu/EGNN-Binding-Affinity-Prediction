# Building Rosetta (`rosetta_scripts`) on UC ARC

Flex ddG needs the `rosetta_scripts` binary. Rosetta is separately licensed and must be
compiled — there is no stock module on UC ARC. This is a one-time setup.

## 0. Version — this is not "any recent Rosetta"

`ddG_backrub.xml` is ported from the Kortemme-Lab reference
([flex_ddG_tutorial](https://github.com/Kortemme-Lab/flex_ddG_tutorial)), whose own README
states: **"It is recommended that you use weekly release 'Rosetta 2017.52' ... More recent
versions of Rosetta may not be able to run this tutorial."** Concretely, this protocol relies
on:
- `InterfaceDdGMover`'s `db_reporter` attribute and `BackrubProtocol`'s
  `trajectory_apply_mover` attribute existing with these exact names/semantics — mover
  attributes are exactly the kind of thing that gets renamed/removed across Rosetta releases.
- The **talaris2014** score function plus `-restore_talaris_behavior`, which exists
  specifically to make newer Rosetta (which defaults to ref2015) reproduce talaris2014
  behavior. A build new enough to have dropped this compatibility flag entirely would
  silently score with the wrong function instead of erroring.

**Before trusting any real ddG numbers from a build newer than 2017.52**, run
`rosetta_scripts.default.<platform>release -parser:protocol rosetta_flex/ddG_backrub.xml -parser:view`
(or just Step 4 below) and confirm it doesn't warn/error about unrecognized mover attributes,
and diff a couple of resulting `ddG.db3` files' `batches` table against
`BATCH_STATE_PREFIXES` in `run_flex_ddg.py` — a version drift here fails as a clear
`ValueError` from `parse_ddg_db3` (see its diagnostic message), not a silent wrong number,
**but only if the mover attributes still exist at all**; if Rosetta parses the XML but drops
an attribute it doesn't recognize with just a warning, that could still run and produce
wrong output. Pin to a build reasonably close to 2017.52 unless you've verified otherwise.

## 1. Get the academic license + source (free for academics)
1. Request a license and download the source at
   <https://www.rosettacommons.org/software/license-and-download> (choose the **source**
   release, e.g. `rosetta.source.release-XXX.tar.bz2`). Academic/non-profit use is free.
   Prefer a release close to **weekly 2017.52** (see Step 0) if older archived releases are
   available from RosettaCommons; if only a current release is offered, budget time in Step 4
   to actually verify output before running the full array.
2. Copy the tarball to your ARC project/scratch space (source builds are large — use
   `/scratch` or your project dir, **not** `$HOME` if it has a small quota):
   ```bash
   cd /scratch/$USER            # or your ARC project space
   tar xjf rosetta.source.release-XXX.tar.bz2
   ```

## 2. Compile `rosetta_scripts` (EPYC / gcc)
Build on a compute node (compilation is CPU+RAM heavy). Grab an interactive node so you
don't compile on the login node:
```bash
# adjust --account/--partition to your ARC allocation
srun --nodes=1 --cpus-per-task=32 --time=02:00:00 --pty bash
module load gcc            # any recent gcc/g++ toolchain on ARC
module load python         # scons needs python

cd /scratch/$USER/rosetta*/main/source
# Build just the app we need, release mode, parallel:
./scons.py -j 32 mode=release bin/rosetta_scripts.default.linuxgccrelease
```
This produces:
```
/scratch/$USER/rosetta*/main/source/bin/rosetta_scripts.default.linuxgccrelease
```
Notes:
- The **database** (`rosetta*/main/database`) ships with the source and is found relative to
  the binary automatically. If a run complains, add `-database /scratch/$USER/rosetta*/main/database`.
- The protocol uses the **talaris2014** score function, which is in the standard database — no
  extra data needed.
- A single non-MPI binary is all we need; we parallelize across cores/nodes ourselves (SLURM
  array × `xargs -P`), so **do not** build the MPI variant.

## 3. Point the pipeline at your binary
```bash
export ROSETTA_SCRIPTS_BIN=/scratch/$USER/rosetta*/main/source/bin/rosetta_scripts.default.linuxgccrelease
# sanity check:
"$ROSETTA_SCRIPTS_BIN" -help >/dev/null && echo "rosetta_scripts OK"
```
`submit_array.sbatch`, `run_node_chunk.sh`, and `plan_run.py --benchmark` all read
`ROSETTA_SCRIPTS_BIN`. Put the `export` in your submit script or `~/.bashrc` on ARC.

## 4. Verify the protocol runs on one mutation — and produces the RIGHT db3 shape
After `make_mutfiles.py` has produced `jobs.csv` (see the main README), first run ONE job
directly (not through `plan_run.py`) so you can inspect its `ddG.db3` before trusting a batch:
```bash
uv run python rosetta_flex/run_flex_ddg.py --jobs_csv rosetta_flex/jobs/jobs.csv \
  --job-index 0 --rosetta_bin "$ROSETTA_SCRIPTS_BIN" \
  --workdir /tmp/flexddg_smoketest --output /tmp/flexddg_smoketest/out.csv
```
This should print a final `ddG=<number>` line with no `ValueError`. If it raises
`"missing required state(s) ..."`, **stop** — that means `db_reporter`/`trajectory_apply_mover`
did not execute as expected on your Rosetta build (see Step 0); do not submit the full array
until this one job parses cleanly, since every job would otherwise fail identically at scale.
For extra confidence, inspect the batches directly:
```bash
sqlite3 /tmp/flexddg_smoketest/ddG.db3 "SELECT name FROM batches;"
# expect exactly 4 rows: bound_wt_interface_ddG, unbound_wt_interface_ddG,
#                        bound_mut_interface_ddG, unbound_mut_interface_ddG
```

Once that single job is verified, benchmark a handful more for timing:
```bash
python rosetta_flex/plan_run.py --benchmark 4 --max_hours 24
```
If the 4 timed mutations succeed, it prints the exact `sbatch` line sized for a <24h run.
If they fail, re-check `ROSETTA_SCRIPTS_BIN` and that the build finished cleanly.
