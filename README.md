# AMDA: manuscript-aligned Python reconstruction (PELT + TFPW)

This package is a **reconstruction from the submitted AMDA manuscript**, not the original numerical experiment source. It implements the paper's four-stage workflow as closely as the written specification permits, including:

- Seasonal residual-based quality control, persistence detection, physical limits and forward cross-validation of the seasonal smoother.
- Change-point detection using **PELT with pruning**, a penalised segment sum-of-squared-errors objective and a minimum segment length of 24 months.
- AR(1)-adjusted penalty and a post-detection minimum shift threshold `delta_min`.
- Theil–Sen slope with a distribution-free confidence interval.
- **Trend-free pre-whitening (TFPW) followed by tie-corrected Mann–Kendall significance testing**, in the Yue et al. (2002) style.
- Sensitivity evaluation for `delta_min = 0.3, 0.5, 0.7` and `c = 1.5, 3, 6`.

## Run

```bash
pip install -r requirements.txt
pytest -q
python amda.py --mode demo --output-dir demo_results
python amda.py --mode tune --n-seeds 12 --output-dir tuning_12
python amda_station.py --input station_reference.csv --output-dir station_results
```

The input CSV must have columns `date`, `station`, and `reference`, with date in YYYY-MM-DD format and both temperatures in °C. For real-data analysis, `reference` must be a correctly matched observational/reanalysis reference series; the code does not download or verify ERA5/GHCN records.

The output `summary.json` contains `trend.mann_kendall_TFPW.p` and `trend.mann_kendall_TFPW.rho_detrended` as well as the 95% Theil–Sen interval. The unadjusted MK result is retained separately as a descriptive cross-check.

## Critical provenance and reproducibility warning

The submitted manuscript includes numeric values for Tables 3–6 that **this reconstructed package has not reproduced**. It should not be described as the original code or as proof of those numerical values. Before stating that the article is fully reproducible, retrieve the original experiment implementation/inputs or independently rerun the full paper-specific benchmark and update/verify the tables. The reconstruction uses some simplifying choices, including month-specific climatology instead of directly aggregating the manuscript's exact deseasonalised daily series, and does not implement the complete baseline/ablation suite. It uses the single-pass TFPW variant, with lag-one correlation estimated from detrended adjacent annual residuals and only positive estimated lag-one correlation whitened.

Do not silently append historical tuning results to newly generated parameter-sensitivity results: these come from different implementations. The `tune` script's optional fixed `h=11` does **not** perform the manuscript's adaptive per-series window selection (run normal `analyze` with h omitted for that).

## Research traceability

- `amda.py`: source implementation
- `amda_station.py`: CSV wrapper
- `test_amda.py`: tests of PELT against exhaustive dynamic programming and TFPW outputs
- `MANUSCRIPT_CODE_MAP.md`: description of article-method correspondence

No repository URL or DOI has been created automatically. A public link can only be inserted after a genuine release.
