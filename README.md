# GroundSight — code accompanying the manuscript

[![DOI](https://zenodo.org/badge/1346166542.svg)](https://doi.org/10.5281/zenodo.22097014)

Code for: **"How good can a groundwater favorability map be? Predictability
ceilings, ceiling-relative validation and cross-area transfer of
machine-learning models in two crystalline–sedimentary basin transition
regions of São Paulo State, Brazil"** (M. Alberto, submitted).

This repository contains the analysis pipeline exactly as described in
Section 3 of the manuscript. Product, deployment and client-facing
components of the wider platform are not part of this archive.

## What is here

| Stage | Manuscript section | Purpose |
|---|---|---|
| `s01` | 3.1 | SIAGAS extraction (cadastre + individual well records) |
| `s02` | 3.1 | QA/QC, eight rule families |
| `s03` | 3.2 | Target derivation: specific capacity, log10(Q/s) |
| `s04` | 3.3 | 3-D weathering model; lithological dictionary |
| `s04b` | 3.5(d) | Regression-kriging upgrade of surfaces (tested, rejected) |
| `s05`, `s05b`, `s05c` | 3.3 | Terrain, structural and airborne-geophysics covariates |
| `s06` | 3.4 | Random forest / XGBoost / ensemble and fuzzy baseline |
| `s07`, `s07_externa`, `s07_cross` | 3.5 | Spatial CV, external validation, cross-area transfer |
| `s08b`, `s08c` | 3.7 | Map products and five-class validated favorability |
| `s09`, `s09b` | 3.6 | Predictability-ceiling estimators |

`s09` enumerates **every** well pair closer than 2 km from a k-d tree
rather than sampling pairs, accepts a variogram lag only if it holds at
least 100 pairs, prints the lag actually used, and reports a bootstrap
95% confidence interval for both the neighbour AUC and the nugget
fraction. The nugget interval holds the sill fixed and is therefore a
lower bound on the total uncertainty of the fraction.

Every stage writes a plain-text report to `data/outputs/` containing the
numbers cited in the manuscript.

## Areas and configuration

The manuscript reports two areas. The only differences between the two runs
are the AOI bounding box (set in `.env`) and the measured ceiling (set in the
area configuration after running `s09`):

| | Area 1 — Parana Basin border | Area 2 — Taubate rift |
|---|---|---|
| `AOI_BBOX` (WGS84 lon/lat) | `-47.35,-23.25,-46.15,-22.05` | `-46.10,-23.35,-44.90,-22.15` |
| Working CRS | EPSG:31983 | EPSG:31983 |
| Configuration | `pipeline/config/config.yaml` | `pipeline/config/config.area2.yaml` |
| Neighbour-AUC ceiling @250 m | 0.672 (95% CI 0.622-0.721) | 0.752 (95% CI 0.671-0.828) |
| Nugget fraction | 66% (95% CI 57-77%) | 44% (95% CI 33-57%) |

`diff pipeline/config/config.yaml pipeline/config/config.area2.yaml` returns
exactly two differing lines. That is the "configuration-only changes" claim
of Section 4.5 in verifiable form.

## Protocol declared in advance

`PROTOCOL_AREA2.md` reproduces, verbatim, the assertions and the area
definition written for Area 2 before any Area 2 computation was run
(document dated 11 August 2026; the Area 2 stage reports carry file dates of
11-12 August 2026). It is the document behind Sections 3.6 and 4.5 of the
manuscript. It was not deposited in a public pre-registration registry, and
the file says so.

## Stage reports

`reports/area1/` and `reports/area2/` hold the plain-text report written by
each stage of the two runs, together with the spatial cross-validation
results and the QA/QC summary. See `reports/README.md`, which also lists the
reports still pending regeneration under the revised ceiling estimator.

## Data

All inputs are public and are **not** redistributed here:

- **SIAGAS** groundwater well inventory — Geological Survey of Brazil
  (https://siagasweb.sgb.gov.br); downloaded by `s01`.
- **Copernicus GLO-30** digital elevation model.
- **Structural datasets** — Geological Survey of Brazil geoportal;
  downloaded by `estrutural_download.py`.
- **Airborne geophysics** — Geological Survey of Brazil surveys 1039 and
  1105 (XYZ flight-line data), obtained from the SGB download area.

## Running

```bash
cp .env.example .env       # set AOI_BBOX and database credentials
docker compose build pipeline
docker compose run --rm pipeline python stages/s00_migrations.py
docker compose run --rm pipeline python stages/s01_extract_siagas.py
# ... stages in order, see table above
```

Reproducing a new region requires editing only `AOI_BBOX` in `.env` and
the ceiling value in `pipeline/config/config.yaml` after running `s09`.

The fraction of the ceiling captured is reported **chance-corrected**,
as `(AUC - 0.5) / (ceiling - 0.5)`, because an AUC of 0.5 is the
expectation of a random ranking and not a zero of skill. The blocking
bar itself (`fator_teto` x ceiling) stays in absolute terms.

## Citation

If you use this code, please cite the manuscript and this archive (see
`CITATION.cff`). The concept DOI
[10.5281/zenodo.22097014](https://doi.org/10.5281/zenodo.22097014) resolves
to the latest version; version 1.1.0, the one reported in the manuscript, is
[10.5281/zenodo.23102501](https://doi.org/10.5281/zenodo.23102501).

## License

GNU Affero General Public License v3.0 — see `LICENSE`.
