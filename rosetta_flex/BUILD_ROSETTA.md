# Building Rosetta (`rosetta_scripts`) on UC ARC

Flex ddG needs the `rosetta_scripts` binary. Rosetta is separately licensed and must be
compiled — there is no stock module on UC ARC. This is a one-time setup.

## 1. Get the academic license + source (free for academics)
1. Request a license and download the source at
   <https://www.rosettacommons.org/software/license-and-download> (choose the **source**
   release, e.g. `rosetta.source.release-XXX.tar.bz2`). Academic/non-profit use is free.
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

## 4. Verify the protocol runs on one mutation
After `make_mutfiles.py` has produced `jobs.csv` (see the main README):
```bash
python rosetta_flex/plan_run.py --benchmark 4 --max_hours 24
```
If the 4 timed mutations succeed, it prints the exact `sbatch` line sized for a <24h run.
If they fail, re-check `ROSETTA_SCRIPTS_BIN` and that the build finished cleanly.
