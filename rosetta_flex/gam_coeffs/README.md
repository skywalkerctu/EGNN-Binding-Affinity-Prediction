# Flex ddG reweighting coefficients

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
