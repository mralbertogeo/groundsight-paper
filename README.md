# GroundSight — code accompanying the manuscript

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

Every stage writes a plain-text report to `data/outputs/` containing the
numbers cited in the manuscript.

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
cp .env.exemplo .env       # set AOI_BBOX and database credentials
docker compose build pipeline
docker compose run --rm pipeline python stages/s00_migrations.py
docker compose run --rm pipeline python stages/s01_extract_siagas.py
# ... stages in order, see table above
```

Reproducing a new region requires editing only `AOI_BBOX` in `.env` and
the ceiling value in `pipeline/config/config.yaml` after running `s09`.

## Citation

If you use this code, please cite the manuscript and this archive
(see `CITATION.cff`).

## License

GNU Affero General Public License v3.0 — see `LICENSE`.
