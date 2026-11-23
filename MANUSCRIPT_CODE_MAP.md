# Manuscript/code correspondence

| Described manuscript component | Function in amda.py | Status |
|---|---|---|
| Calendar conversion (remove 29 Feb) | clean_calendar | Implemented |
| Robust seasonal decomposition | seasonal_decompose | STL-type approximation, not exact LOESS STL |
| Forward origins for h={5,11,21} | choose_window | Implemented |
| Physical/persistence/seasonal residual QC | quality_control | Implemented |
| Monthly station-minus-reference differences | monthly_anomalies | Approximation: monthly climatological anomalies; see README |
| PELT SSE + beta, min segment 24 | exact_changepoints | Actual PELT pruning; regression-tested versus unpruned exact DP |
| AR(1) penalty, delta_min | homogenize | Implemented |
| Theil–Sen slope and 95% CI | annual_trend | Implemented |
| Trend-free pre-whitening + MK | trend_free_prewhitened_mk, mann_kendall | Implemented (single-pass Yue-style TFPW) |

| 100-simulation sensitivity tuning | sensitivity   |


