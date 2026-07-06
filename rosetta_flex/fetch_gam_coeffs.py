#!/usr/bin/env python
"""
Install the official Flex ddG reweighting coefficients into gam_coeffs/flex_ddg_gam.json.

Why this is an *installer*, not a hard-coded table
--------------------------------------------------
The published Flex ddG "GAM" (Barlow et al. 2018) is a generalized additive model -- a
nonlinear spline *per Rosetta score term*, fit in R (mgcv) against experimental ddG. There
is no honest way to reduce it to invented constants, and this project's rule is to never
fabricate biophysics. So:

* We ship NO numbers by default. With no coefficients file present, `run_flex_ddg.py`
  correctly falls back to Rosetta's raw `total_score` (the "nogam" variant).
* This script INSTALLS coefficients you obtained from the official source into the JSON
  schema the parser understands ({score_type_name: weight}), validating the term names.

`run_flex_ddg.py --gam-coeffs` (and its parser `_total_scores_by_struct`) applies a *linear*
per-term reweighting: total = sum_t weight[t] * score_value[t]. That exactly implements the
published *linear* reweighting variant when you supply those weights. The full nonlinear GAM
is best applied as a post-processing step on the exported per-term table (see README) -- this
installer targets the linear-reweight hook that already exists.

Where to get real coefficients
-------------------------------
Kortemme Lab flex_ddG tutorial: https://github.com/Kortemme-Lab/flex_ddG_tutorial
(its analysis/ scripts fit and report the reweighting; extract the per-term weights into a
TSV/JSON and pass them here). Nothing is downloaded automatically because the upstream form
is an R model, not a ready JSON.

Usage
-----
    # install from a two-column TSV you extracted (score_type <tab> weight):
    python rosetta_flex/fetch_gam_coeffs.py --from-tsv my_weights.tsv
    # or from a ready JSON dict:
    python rosetta_flex/fetch_gam_coeffs.py --from-json my_weights.json
    # just scaffold the folder + provenance README (no numbers written):
    python rosetta_flex/fetch_gam_coeffs.py --scaffold
    # list the talaris2014 score-type names the JSON should key on:
    python rosetta_flex/fetch_gam_coeffs.py --print-schema
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("fetch_gam_coeffs")

GAM_DIR = Path(__file__).resolve().parent / "gam_coeffs"
GAM_JSON = GAM_DIR / "flex_ddg_gam.json"

# Standard talaris2014 score-type NAMES (identifiers only -- no weights). These are the keys
# a valid coefficients JSON should use; supplied so you can format your extracted weights.
TALARIS2014_SCORE_TYPES = [
    "fa_atr", "fa_rep", "fa_sol", "fa_intra_rep", "fa_elec", "pro_close",
    "hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc", "dslf_fa13",
    "rama", "omega", "fa_dun", "p_aa_pp", "yhh_planarity", "ref",
]

README_TEXT = """# Flex ddG reweighting coefficients

This folder holds the OPTIONAL reweighting coefficients used by
`run_flex_ddg.py --gam-coeffs`. If `flex_ddg_gam.json` is present, the ddG parser computes
each structure's score as a linear reweighting `sum_t weight[t] * score_value[t]`; if absent,
it uses Rosetta's raw `total_score` (the "nogam" variant). Both are legitimate; GAM/reweighted
Flex ddG reached Pearson ~0.46 vs experiment in Hummer et al. 2025 (vs ~0.42 nogam, ~0.20 FoldX).

## We ship no numbers here on purpose
The published Flex ddG GAM is a nonlinear spline model (R/mgcv). We do not invent constants.
Obtain real coefficients from the Kortemme Lab flex_ddG tutorial and install them:

    https://github.com/Kortemme-Lab/flex_ddG_tutorial

Extract the per-term weights into a TSV (`score_type<TAB>weight`) or JSON dict, then:

    python rosetta_flex/fetch_gam_coeffs.py --from-tsv my_weights.tsv

The JSON keys must be talaris2014 score-type names; see
`python rosetta_flex/fetch_gam_coeffs.py --print-schema`.

## Full (nonlinear) GAM
The exact nonlinear GAM can be reproduced by exporting the per-term score breakdown from each
`ddG.db3` (the `structure_scores` table, populated by the protocol's ReportToDB) and applying
the tutorial's fitted mgcv model as a post-processing step. The linear installer here targets
the reweighting hook already wired into the pipeline.
"""


def validate(coeffs: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    unknown = []
    for k, v in coeffs.items():
        name = str(k).strip()
        try:
            out[name] = float(v)
        except (TypeError, ValueError):
            LOGGER.warning("Skipping non-numeric weight for %r: %r", name, v)
            continue
        if name not in TALARIS2014_SCORE_TYPES:
            unknown.append(name)
    if unknown:
        LOGGER.warning("These score-type names are not standard talaris2014 terms (installing anyway, "
                       "but double-check spelling): %s", ", ".join(sorted(unknown)))
    if not out:
        raise SystemExit("No valid {score_type: weight} pairs found; nothing installed.")
    return out


def load_tsv(path: Path) -> Dict[str, float]:
    coeffs: Dict[str, float] = {}
    with open(path, newline="") as fh:
        # Accept tab- or comma-separated, with or without a header line.
        for row in csv.reader(fh, delimiter="\t" if "\t" in path.read_text() else ","):
            if len(row) < 2:
                continue
            name, val = row[0].strip(), row[1].strip()
            if not name or name.lower() in ("score_type", "term", "name"):
                continue
            try:
                coeffs[name] = float(val)
            except ValueError:
                continue
    return coeffs


def write_readme() -> None:
    GAM_DIR.mkdir(parents=True, exist_ok=True)
    (GAM_DIR / "README.md").write_text(README_TEXT)
    LOGGER.info("Wrote provenance note to %s", GAM_DIR / "README.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-tsv", type=Path, help="Two-column (score_type, weight) TSV/CSV of real coefficients.")
    src.add_argument("--from-json", type=Path, help="JSON dict {score_type: weight} of real coefficients.")
    src.add_argument("--scaffold", action="store_true", help="Only create the folder + README (no numbers).")
    src.add_argument("--print-schema", action="store_true", help="Print the talaris2014 score-type names and exit.")
    args = ap.parse_args()

    if args.print_schema:
        print("talaris2014 score-type names (JSON keys):")
        for t in TALARIS2014_SCORE_TYPES:
            print(f"  {t}")
        return

    write_readme()  # always keep provenance current

    if args.scaffold:
        LOGGER.info("Scaffolded %s (no coefficients installed; parser will use raw total_score).", GAM_DIR)
        return

    if not args.from_tsv and not args.from_json:
        LOGGER.info("No --from-tsv/--from-json given. Folder scaffolded; obtain real coefficients from the "
                    "flex_ddG tutorial (see README), then re-run to install. Parser uses total_score meanwhile.")
        return

    if args.from_json:
        coeffs = json.loads(Path(args.from_json).read_text())
    else:
        coeffs = load_tsv(args.from_tsv)

    coeffs = validate(coeffs)
    GAM_JSON.write_text(json.dumps(coeffs, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Installed %d reweighting coefficients -> %s", len(coeffs), GAM_JSON)
    LOGGER.info("run_flex_ddg.py will now auto-use these (no --gam-coeffs flag needed).")


if __name__ == "__main__":
    main()
