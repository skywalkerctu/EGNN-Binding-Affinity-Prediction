"""
Tier 1 synthetic TCR-pMHC ddG dataset generator.

Pipeline
--------
1. Ingest IMGT-renumbered STCRDab structures (chains conventionally:
   A = MHC chain 1, B = MHC chain 2 / beta2-microglobulin, C = peptide,
   D = TCR alpha chain, E = TCR beta chain).
2. Identify the 6 CDR loops on the TCR alpha/beta chains using their IMGT
   residue numbers (CDR1: 27-38, CDR2: 56-65, CDR3: 105-117).
3. Identify the "binding interface" as every residue (on the TCR, peptide,
   or MHC) within `--interface_radius` Angstrom of any CDR atom.
4. Run saturation mutagenesis (19 point mutations per interface residue)
   through the real MadraX PyTorch force field, batching many mutants of
   the same structure into a single forward pass for GPU throughput.
5. Stream results to CSV incrementally, logging/skipping structures or
   mutants that blow up (steric clashes -> NaN/Inf/huge energies).

This replaces a previous version of this script that called a
`force_field.compute_energy(...)` method that does not exist in MadraX and
silently fell back to a random-number mock, identified CDR loops by
positional list slicing (wrong whenever a structure has missing residues or
IMGT insertion codes), and required `pandas`, which was never declared as a
project dependency. All of that is fixed here using MadraX's real API
(`madrax.utils.parsePDB`, `madrax.dataStructures.create_info_tensors`,
`madrax.mutate.mutatingEngine.mutate`, `madrax.ForceField.ForceField`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser, Select
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Interface detection is shared with the Rosetta Flex ddG pipeline
# (rosetta_flex/make_mutfiles.py) so both tiers mutate the same positions.
from interface_utils import (
    AMINO_ACIDS,
    DEFAULT_COMPLEX_CHAINS,
    DEFAULT_TCR_CHAINS,
    IMGT_CDR_RANGES,
    MutationTarget,
    find_interface_targets,
)

try:
    from madrax.ForceField import ForceField
    from madrax import dataStructures
    from madrax import utils as madrax_utils
    from madrax.mutate.mutatingEngine import mutate as madrax_mutate
except ImportError as exc:  # MadraX is a hard requirement, not optional.
    raise SystemExit(
        "MadraX is required to run this script but could not be imported. "
        "Install it with `uv sync` (see pyproject.toml) or "
        "`pip install git+https://bitbucket.org/grogdrinker/madrax/`. "
        f"Original error: {exc}"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("tcr_pmhc_dataset")

# Constants and interface-detection helpers live in interface_utils so the
# Rosetta Flex ddG pipeline can reuse the exact same definitions. The amino
# acid list, chain conventions, IMGT CDR ranges and MutationTarget are imported
# at the top of this module.

# Shared ddG-label schema (single source of truth across both data tiers).
# The first seven columns are identical for the MadraX and Rosetta CSVs;
# `Distance_to_CDR_center` and `Min_Interchain_Distance` are MadraX-only analysis
# extras (the latter flags genuine inter-chain contacts vs. second-shell hits).
# `ddG` is a *binding* ddG: complex change minus the mutated residue's own-chain
# change in isolation (chain separation), so it is comparable to ATLAS/TRAIT.
CSV_FIELDNAMES = [
    "PDB_ID",
    "Chain",
    "Residue_Position",
    "WT_Amino_Acid",
    "Mutant_Amino_Acid",
    "ddG",
    "source",
    "Distance_to_CDR_center",
    "Min_Interchain_Distance",
]
SOURCE_TAG = "madrax"

# MadraX's plain-text PDB parser keys atoms purely by integer residue number
# (madrax/utils.py: `resnum = int(line[22:26])`), so IMGT insertion codes
# (e.g. "111A", "111B") collapse onto the same integer position. We mirror
# that limitation deliberately: every step below identifies residues by
# (chain, integer resseq) only, so mutation directives sent to MadraX always
# refer to a residue MadraX can actually distinguish.


class InterfaceChainSelect(Select):
    """BioPython Select that keeps only standard residues on whitelisted chains."""

    def __init__(self, allowed_chains: Sequence[str]):
        self.allowed_chains = set(allowed_chains)

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains

    def accept_residue(self, residue):
        return residue.id[0] == " "  # drop waters / heteroatoms / ligands


@dataclass
class PdbJob:
    pdb_id: str
    clean_pdb_path: Optional[str]  # full complex (all complex_chains)
    clean_tcr_path: Optional[str] = None  # TCR-only subset (tcr_chains)
    clean_pmhc_path: Optional[str] = None  # pMHC-only subset (complex_chains - tcr_chains)
    targets: List[MutationTarget] = field(default_factory=list)
    error: Optional[str] = None


def clean_structure(source_path: Path, dest_path: Path, allowed_chains: Sequence[str]) -> None:
    """Write a copy of `source_path` containing only standard residues on `allowed_chains`."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source_path.stem, str(source_path))
    io = PDBIO()
    io.set_structure(structure)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(dest_path), select=InterfaceChainSelect(allowed_chains))


class TcrPmhcDataset(Dataset):
    """CPU-only stage: parse a PDB, find CDRs + interface residues, write a clean copy.

    Kept free of CUDA/madrax-forward-pass work so it can run safely inside
    multiple `DataLoader` worker processes while the GPU forward passes stay
    serialized in the main process.
    """

    def __init__(
        self,
        pdb_files: Sequence[str],
        clean_dir: Path,
        tcr_chains: Sequence[str],
        complex_chains: Sequence[str],
        interface_radius: float,
    ):
        self.pdb_files = list(pdb_files)
        self.clean_dir = clean_dir
        self.tcr_chains = tuple(tcr_chains)
        self.complex_chains = tuple(complex_chains)
        # pMHC = the complex minus the TCR chains; used to score the isolated
        # binding partner for the binding-ddG (chain-separation) calculation.
        self.pmhc_chains = tuple(c for c in complex_chains if c not in set(tcr_chains))
        self.interface_radius = interface_radius

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> PdbJob:
        pdb_path = Path(self.pdb_files[idx])
        pdb_id = pdb_path.stem.lower()
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, str(pdb_path))

            targets = find_interface_targets(
                structure,
                tcr_chains=self.tcr_chains,
                complex_chains=self.complex_chains,
                radius=self.interface_radius,
            )
            if not targets:
                return PdbJob(
                    pdb_id, None,
                    error="No mutable interface residues found (check chain IDs / IMGT numbering / radius)",
                )

            # Write the full complex plus the two isolated binding partners. The
            # partner subsets are scored separately so the per-mutation label is a
            # *binding* ddG (complex change minus the change to the mutated
            # residue's own chain in isolation), not whole-complex stability.
            complex_path = self.clean_dir / f"{pdb_id}.pdb"
            tcr_path = self.clean_dir / f"{pdb_id}_tcr.pdb"
            pmhc_path = self.clean_dir / f"{pdb_id}_pmhc.pdb"
            clean_structure(pdb_path, complex_path, self.complex_chains)
            clean_structure(pdb_path, tcr_path, self.tcr_chains)
            clean_structure(pdb_path, pmhc_path, self.pmhc_chains)

            return PdbJob(
                pdb_id,
                str(complex_path),
                clean_tcr_path=str(tcr_path),
                clean_pmhc_path=str(pmhc_path),
                targets=targets,
            )
        except Exception as exc:  # noqa: BLE001 - report and keep the pipeline alive
            return PdbJob(pdb_id, None, error=f"{type(exc).__name__}: {exc}")


def load_madrax_inputs(clean_pdb_path: str):
    """Parse a single cleaned PDB into MadraX's (coords, atom_names, pdb_names) batch-of-1 form."""
    clean_path = Path(clean_pdb_path)
    with tempfile.TemporaryDirectory(prefix="madrax_input_") as tmp_dir:
        staged = Path(tmp_dir) / clean_path.name
        shutil.copy2(clean_path, staged)
        coords, atom_names, pdb_names = madrax_utils.parsePDB(tmp_dir)
    return coords, atom_names, pdb_names


def run_mutation_batch(
    coords: torch.Tensor,
    atom_names: List[List[str]],
    directives: List[str],
    force_field: ForceField,
    device: torch.device,
) -> torch.Tensor:
    """Mutate one structure into `len(directives)` independent point mutants and
    score WT + all mutants in a single MadraX forward pass.

    Returns a 1D tensor of length `len(directives)` with MadraX_ddG (mutant
    total energy - WT total energy) for each directive, in the same order.
    """
    mutation_list = [[[directive] for directive in directives]]
    mutated_coords, mutated_atom_names = madrax_mutate(coords, atom_names, mutation_list)
    info_tensors = dataStructures.create_info_tensors(mutated_atom_names, device=str(device))
    with torch.no_grad():
        energy = force_field(mutated_coords.to(device), info_tensors)
    # energy shape: (Batch=1, nChains, nResi, nAlt, nEnergyComponents)
    # alt index 0 is always the wild type; 1..N are the requested mutants, in order.
    total_per_alt = energy[0].sum(dim=(0, 1, 3))
    ddg = total_per_alt[1:] - total_per_alt[0]
    return ddg.detach().cpu()


def chunked(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _score_partner_group(
    pdb_id: str,
    targets: Sequence[MutationTarget],
    complex_coords: torch.Tensor,
    complex_atom_names: List[List[str]],
    partner_coords: torch.Tensor,
    partner_atom_names: List[List[str]],
    partner_label: str,
    force_field: ForceField,
    device: torch.device,
    mutation_batch_size: int,
    clash_threshold: float,
    writer: csv.DictWriter,
    failure_log: csv.DictWriter,
    out_f,
    fail_f,
) -> int:
    """Score every mutant of ``targets`` as a *binding* ddG.

    For each batch we score the same mutations twice -- once in the full complex
    and once in the isolated binding partner that carries the mutated residue --
    and report ``complex_ddg - partner_ddg``. The non-mutated partner's energy is
    identical in WT and mutant, so it cancels and need not be scored.
    """
    flat_jobs: List[Tuple[MutationTarget, str, str]] = []
    for target in targets:
        for mutant_aa in AMINO_ACIDS:
            if mutant_aa == target.wt_aa:
                continue
            directive = f"{target.resnum}_{target.chain}_{mutant_aa}"
            flat_jobs.append((target, mutant_aa, directive))
    if not flat_jobs:
        return 0

    written = 0
    batch_size = mutation_batch_size
    pending = list(flat_jobs)
    progress = tqdm(
        total=len(flat_jobs),
        desc=f"Mutations {pdb_id}/{partner_label}",
        unit="mut",
        leave=False,
    )
    try:
        while pending:
            batch = pending[:batch_size]
            directives = [item[2] for item in batch]
            try:
                complex_ddg = run_mutation_batch(complex_coords, complex_atom_names, directives, force_field, device)
                partner_ddg = run_mutation_batch(partner_coords, partner_atom_names, directives, force_field, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    LOGGER.error("OOM on %s even at batch size 1; skipping remaining mutants.", pdb_id)
                    failure_log.writerow({"PDB_ID": pdb_id, "Reason": "CUDA OOM at batch size 1"})
                    pending = pending[1:]
                    progress.update(1)
                    continue
                batch_size = max(1, batch_size // 2)
                LOGGER.warning("CUDA OOM on %s; reducing mutation batch size to %d and retrying.", pdb_id, batch_size)
                continue
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("MadraX forward pass failed for %s batch starting at %s: %s", pdb_id, directives[0], exc)
                failure_log.writerow({"PDB_ID": pdb_id, "Reason": f"forward pass failure: {exc}"})
                pending = pending[len(batch) :]
                progress.update(len(batch))
                continue

            binding = complex_ddg - partner_ddg
            for (target, mutant_aa, _directive), ddg_tensor in zip(batch, binding):
                ddg = float(ddg_tensor.item())
                if not np.isfinite(ddg) or abs(ddg) > clash_threshold:
                    LOGGER.warning(
                        "Severe steric clash for %s %s%d %s->%s (binding ddG=%s); filtering out.",
                        pdb_id, target.chain, target.resnum, target.wt_aa, mutant_aa, ddg,
                    )
                    failure_log.writerow(
                        {"PDB_ID": pdb_id, "Reason": f"clash at {target.chain}{target.resnum}{target.wt_aa}->{mutant_aa} (ddG={ddg})"}
                    )
                    continue
                writer.writerow(
                    {
                        "PDB_ID": pdb_id,
                        "Chain": target.chain,
                        "Residue_Position": target.resnum,
                        "WT_Amino_Acid": target.wt_aa,
                        "Mutant_Amino_Acid": mutant_aa,
                        "ddG": round(ddg, 4),
                        "source": SOURCE_TAG,
                        "Distance_to_CDR_center": round(target.distance_to_cdr, 3),
                        "Min_Interchain_Distance": round(target.min_interchain_dist, 3),
                    }
                )
                written += 1

            pending = pending[len(batch) :]
            progress.update(len(batch))
            # Flush every batch so an interrupted long structure keeps the rows it
            # has already computed (a full structure can take many minutes).
            out_f.flush()
            fail_f.flush()
    finally:
        progress.close()

    return written


def process_job(
    job: PdbJob,
    force_field: ForceField,
    device: torch.device,
    mutation_batch_size: int,
    clash_threshold: float,
    tcr_chains: Sequence[str],
    writer: csv.DictWriter,
    failure_log: csv.DictWriter,
    out_f,
    fail_f,
) -> int:
    if job.error or job.clean_pdb_path is None:
        LOGGER.warning("Skipping %s: %s", job.pdb_id, job.error)
        failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": job.error or "unknown"})
        return 0

    try:
        complex_coords, complex_atom_names, _ = load_madrax_inputs(job.clean_pdb_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to parse %s for MadraX: %s", job.pdb_id, exc)
        failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": f"MadraX parse failure: {exc}"})
        return 0

    tcr_set = set(tcr_chains)
    tcr_targets = [t for t in job.targets if t.chain in tcr_set]
    pmhc_targets = [t for t in job.targets if t.chain not in tcr_set]

    written = 0
    for targets, partner_path, label in (
        (tcr_targets, job.clean_tcr_path, "TCR"),
        (pmhc_targets, job.clean_pmhc_path, "pMHC"),
    ):
        if not targets:
            continue
        if not partner_path:
            failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": f"missing {label} partner subset"})
            continue
        try:
            partner_coords, partner_atom_names, _ = load_madrax_inputs(partner_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to parse %s %s partner for MadraX: %s", job.pdb_id, label, exc)
            failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": f"{label} partner parse failure: {exc}"})
            continue
        written += _score_partner_group(
            job.pdb_id, targets, complex_coords, complex_atom_names,
            partner_coords, partner_atom_names, label,
            force_field, device, mutation_batch_size, clash_threshold,
            writer, failure_log, out_f, fail_f,
        )

    return written


def load_done_pdb_ids(done_path: Path) -> set:
    if not done_path.exists():
        return set()
    return {line.strip() for line in done_path.read_text().splitlines() if line.strip()}


def main() -> None:
    # Auto-detect a SLURM array environment so `sbatch --array=0-N` shards with zero
    # extra flags: SLURM_ARRAY_TASK_ID -> --shard_id, SLURM_ARRAY_TASK_COUNT -> --num_shards.
    # An explicit --shard_id/--num_shards on the command line still overrides these.
    _slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    _slurm_task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0))
    _default_num_shards = _slurm_task_count if _slurm_task_id >= 0 and _slurm_task_count > 0 else 1
    _default_shard_id = _slurm_task_id if _slurm_task_id >= 0 else 0

    parser = argparse.ArgumentParser(description="Generate synthetic TCR-pMHC ddG dataset using MadraX.")
    parser.add_argument("--pdb_dir", type=str, default="stcrdab_structures/", help="Directory containing STCRDab PDB files.")
    parser.add_argument("--output", type=str, default="tcr_pmhc_interface_ddg.csv", help="Output CSV file path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of PDB files to process (useful for dry runs).")
    parser.add_argument("--target_samples", type=int, default=None, help="Stop cleanly once this many ddG rows have been written (e.g. 1000000). Checked at PDB-job boundaries.")
    parser.add_argument("--interface_radius", type=float, default=10.0, help="Angstrom radius defining the binding interface around CDR atoms.")
    parser.add_argument("--tcr_chains", type=str, default=",".join(DEFAULT_TCR_CHAINS), help="Comma-separated chain IDs for the TCR alpha/beta chains.")
    parser.add_argument("--complex_chains", type=str, default=",".join(DEFAULT_COMPLEX_CHAINS), help="Comma-separated chain IDs that make up the full TCR-pMHC complex.")
    parser.add_argument("--mutation_batch_size", type=int, default=200, help="Number of point mutants scored per MadraX forward pass.")
    parser.add_argument("--clash_threshold", type=float, default=1000.0, help="|ddG| (kcal/mol) above which a mutant is treated as a steric clash and filtered out.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="CPU worker processes for PDB parsing / interface detection, PER SHARD. "
                             "Default: (cpu_count - 1) split evenly across --num_shards so concurrent "
                             "shards on one box do not oversubscribe the cores (total ~= cpu_count).")
    parser.add_argument("--clean_dir", type=str, default=None, help="Directory to cache cleaned, chain-filtered PDBs. Defaults to <pdb_dir>/_clean.")
    parser.add_argument("--resume", action="store_true", help="Skip PDB IDs already recorded as done in <output>.done.")
    parser.add_argument("--num_shards", type=int, default=_default_num_shards,
                        help="Partition the PDB list into this many shards for parallel runs (one process "
                             "per shard; they may share a single GPU since the bottleneck is CPU-side). "
                             "Auto-set from SLURM_ARRAY_TASK_COUNT.")
    parser.add_argument("--shard_id", type=int, default=_default_shard_id,
                        help="Which shard (0..num_shards-1) this process handles. Output/failure/.done "
                             "paths are auto-suffixed with .shardN. Auto-set from SLURM_ARRAY_TASK_ID.")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"Invalid sharding: shard_id={args.shard_id} must be in [0, num_shards={args.num_shards}).")

    # Resolve the per-shard worker count. The old default was (cpu_count - 1) PER PROCESS,
    # so N concurrent shards spawned N*(cpu_count-1) workers and thrashed the box. Divide
    # the core budget across the shards instead; a single-shard run is unchanged.
    if args.num_workers is None:
        total_budget = max(1, (os.cpu_count() or 2) - 1)
        args.num_workers = max(1, total_budget // args.num_shards)
    LOGGER.info(
        "Sharding: shard %d/%d, %d DataLoader worker(s) per shard (~%d cores total across shards).",
        args.shard_id, args.num_shards, args.num_workers, args.num_workers * args.num_shards,
    )

    pdb_dir = Path(args.pdb_dir)
    output_csv = Path(args.output)
    base_output = output_csv  # unsuffixed name, used for the shared shard-config marker
    # Keep concurrent shards from writing the same files.
    if args.num_shards > 1:
        output_csv = output_csv.with_name(f"{output_csv.stem}.shard{args.shard_id}{output_csv.suffix}")

    # Guard the strided split against a changed --num_shards on resume. Because a
    # structure's shard is `index % num_shards`, resuming with a different shard count
    # reassigns structures to different shards while each shard's .done only covers its
    # OLD slice -> silent double-computation of some structures and permanent gaps in
    # others. Record num_shards once next to the output and refuse a mismatched resume.
    shardmeta = base_output.with_name(f"{base_output.stem}.shardmeta")
    if shardmeta.exists():
        try:
            recorded = int(shardmeta.read_text().strip())
        except ValueError:
            recorded = None
        if recorded is not None and recorded != args.num_shards:
            raise SystemExit(
                f"Shard-count mismatch: this dataset was started with --num_shards {recorded}, "
                f"but you passed --num_shards {args.num_shards}. Resuming with a different shard "
                f"count corrupts coverage (the strided split reassigns structures). Either re-run "
                f"with --num_shards {recorded}, or start fresh (remove {shardmeta} and the "
                f"{base_output.stem}.shard*{base_output.suffix} outputs)."
            )
    else:
        shardmeta.parent.mkdir(parents=True, exist_ok=True)
        shardmeta.write_text(f"{args.num_shards}\n")

    tcr_chains = tuple(c.strip() for c in args.tcr_chains.split(",") if c.strip())
    complex_chains = tuple(c.strip() for c in args.complex_chains.split(",") if c.strip())
    clean_dir = Path(args.clean_dir) if args.clean_dir else pdb_dir / "_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_dir.exists():
        pdb_dir.mkdir(parents=True)
        LOGGER.info("Directory %s created. Please populate it with STCRDab PDB files.", pdb_dir)

    pdb_files = sorted(glob.glob(str(pdb_dir / "*.pdb")))
    if args.num_shards > 1:
        pdb_files = pdb_files[args.shard_id :: args.num_shards]
        LOGGER.info("Shard %d/%d: handling %d of the PDB files.", args.shard_id, args.num_shards, len(pdb_files))

    done_path = output_csv.with_suffix(output_csv.suffix + ".done")
    done_ids = load_done_pdb_ids(done_path) if args.resume else set()
    if done_ids:
        pdb_files = [f for f in pdb_files if Path(f).stem.lower() not in done_ids]
        LOGGER.info("Resuming: skipping %d already-completed PDB files.", len(done_ids))

    if args.limit:
        pdb_files = pdb_files[: args.limit]
        LOGGER.info("Limiting to %d PDB files for this run.", args.limit)

    LOGGER.info("Found %d PDB files to process.", len(pdb_files))
    if not pdb_files:
        LOGGER.error("No PDB files left to process. Exiting.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        LOGGER.warning("CUDA is not available; running on CPU. This will be far too slow for the full 1M-datapoint run.")
    LOGGER.info("Using device: %s", device)

    force_field = ForceField(device=str(device)).to(device)
    force_field.eval()

    dataset = TcrPmhcDataset(pdb_files, clean_dir, tcr_chains, complex_chains, args.interface_radius)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
    )

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    failure_log_path = output_csv.with_name(output_csv.stem + "_failures.csv")
    failure_write_header = not failure_log_path.exists() or failure_log_path.stat().st_size == 0

    total_written = 0
    with open(output_csv, "a", newline="") as out_f, open(failure_log_path, "a", newline="") as fail_f, open(done_path, "a") as done_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        failure_writer = csv.DictWriter(fail_f, fieldnames=["PDB_ID", "Reason"])
        if failure_write_header:
            failure_writer.writeheader()

        for i, job in enumerate(loader):
            LOGGER.info("[%d/%d] Processing %s (%d interface residues)", i + 1, len(dataset), job.pdb_id, len(job.targets))
            written = process_job(job, force_field, device, args.mutation_batch_size, args.clash_threshold, tcr_chains, writer, failure_writer, out_f, fail_f)
            total_written += written
            out_f.flush()
            fail_f.flush()
            done_f.write(job.pdb_id + "\n")
            done_f.flush()
            LOGGER.info("%s -> %d ddG points (running total: %d)", job.pdb_id, written, total_written)
            if args.target_samples is not None and total_written >= args.target_samples:
                LOGGER.info(
                    "Reached target of %d samples (%d written) after %d/%d structures; stopping. "
                    "Re-run with --resume to extend the dataset further.",
                    args.target_samples, total_written, i + 1, len(dataset),
                )
                break

    LOGGER.info("Successfully generated %d synthetic ddG points.", total_written)
    LOGGER.info("Results written to %s (failures logged to %s).", output_csv, failure_log_path)


if __name__ == "__main__":
    main()
