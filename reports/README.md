# Stage reports

Plain-text report written by each stage of the two runs reported in the
manuscript, plus the spatial cross-validation results
(`validacao_resultados.csv`) and the QA/QC summary (`qaqc_relatorio.csv`).
Every number cited in the paper can be traced to one of these files.

| | Area 1 | Area 2 |
|---|---|---|
| Region | Eastern Sao Paulo Plateau, Parana Basin border | Middle Paraiba do Sul valley, Taubate rift |
| `AOI_BBOX` (WGS84) | `-47.35,-23.25,-46.15,-22.05` | `-46.10,-23.35,-44.90,-22.15` |
| Quality-approved wells | 7,646 | 1,364 |
| Modelling matrix | 1,742 | 493 |
| Configuration | `pipeline/config/config.yaml` | `pipeline/config/config.area2.yaml` |

## Predictability ceilings

`s09_relatorio.txt` in each directory was produced on 2026-10-02 by
`stages/s09_teto_previsibilidade.py` as committed here, with the
exhaustive-pair variogram estimator and the bootstrap confidence intervals
described in Section 3.6 of the manuscript.

| | Area 1 | Area 2 |
|---|---|---|
| Neighbour AUC @250 m | 0.672 (95% CI 0.622-0.721) | 0.752 (95% CI 0.671-0.828) |
| Neighbour AUC @500 m | 0.624 (0.584-0.663) | 0.764 (0.698-0.826) |
| Neighbour AUC @1000 m | 0.594 (0.558-0.629) | 0.755 (0.705-0.806) |
| Neighbour AUC @2000 m | 0.585 (0.552-0.617) | 0.759 (0.708-0.809) |
| Real pairs below 250 m | 595 | 182 |
| Nugget fraction | 66% (95% CI 57-77%) | 44% (95% CI 33-57%) |
| Maximum theoretical R-squared | 0.34 | 0.56 |
| Sill reached at | 500-1,000 m | not reached at 16 km |

The nugget interval holds the sill fixed and is therefore a lower bound on
the total uncertainty of the fraction.

## Superseded estimator

The first version of the ceiling stage sampled 400,000 well pairs with
replacement instead of enumerating every pair below 2 km, and reported no
confidence intervals. With a fixed number of draws that estimator
undersamples the shortest lag in a dense inventory and oversamples it in a
sparse one: in Area 1 it recovered 158 of the 595 real pairs below 250 m and
returned a nugget fraction of 78%; in Area 2 it drew 588 times from the 182
real pairs and returned 45%, within one point of the exhaustive value. The
reports in this directory were produced by the current estimator; the
superseded ones are not distributed.
