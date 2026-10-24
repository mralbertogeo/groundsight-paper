# Figures

Every figure in the manuscript is produced by a script in this directory.
No figure carries a number typed into its own source: each one reads the
archived stage report, CSV or vector layer that the pipeline produced, and
several abort rather than draw something they cannot account for.

Run each script from the repository root.

| Figure | Script | Inputs | In this repository? |
|---|---|---|---|
| 1 | `fig1_location.py` | IBGE state boundaries; the two input geopackages | **no** — external, see below |
| 2 | `fig2_roses.py` | Geological Survey of Brazil structural shapefiles | **no** — external, see below |
| 3 | `fig3_workflow.py` | none (diagram) | — |
| 4 | `fig4_trajectory.py` | `reports/area1/s09_relatorio.txt`, `validacao_resultados.csv`, `trajetoria_validacao.csv` | yes |
| 5 | `fig5_tetos.py` | `reports/area*/s09_relatorio.txt` | yes |
| 6 | `fig6_importances.py` | `reports/area*/importancia_permutacao.csv` | yes |
| 7 | `fig7_products.py` | `data/derived/covariaveis/*.tif` of each instance | raster inputs not redistributed |

```
python figures/fig3_workflow.py                      --saida figures
python figures/fig4_trajectory.py  --reports reports --saida figures
python figures/fig5_tetos.py       --reports reports --saida figures
python figures/fig6_importances.py --reports reports --saida figures
```

Figures 3 to 6 therefore regenerate from this repository
alone. Figures 1 and 2 need third-party vector data that is public but too
large, and not ours, to redistribute here:

* **Figure 1** — `BR_UF_2025` state boundaries (IBGE, *Malhas territoriais*),
  plus `fig1_insumos_area1.gpkg` and `fig1_insumos_area2.gpkg`, written by
  `pipeline/stages/fig_export.py` in each instance. They carry the AOI
  polygon and the well points with the `qaqc_aprovado` flag; the script
  counts the approved wells itself rather than taking the caption's word
  (7,646 and 1,364).
* **Figure 2** — `geol_linha_dobra`, `geol_linha_falha`, `geol_linha_fratura`
  (Geological Survey of Brazil, GIS do Brasil ao Milionésimo) and, for
  Area 2, `estruturas_gis_brasil_2004`.
  The azimuths the figure actually plots are archived as
  `reports/area*/azimutes_estruturais.csv`, so the rose is checkable without
  the source shapefiles.

```
python figures/fig1_location.py --uf <...>/BR_UF_2025.shp \
    --gpkg1 <...>/fig1_insumos_area1.gpkg \
    --gpkg2 <...>/fig1_insumos_area2.gpkg --saida figures
python figures/fig2_roses.py --raw1 <...>/area1/data/raw/estrutural \
    --raw2 <...>/area2/data/raw/estrutural --saida figures \
    --reports reports
```

## Two estimator notes, because the figures differ from earlier versions

**Figure 2 — azimuths.** The earlier `fig_export.py` took the azimuth of
each clipped feature from its two endpoints, which treats a sinuous fault as
a straight trace and credits its whole length to one bearing. `fig2_roses.py`
computes the azimuth of every segment and weights it by that segment's
length. In Area 2 the two agree (modal 55–60°); in Area 1 the modal class
moves from 35–40° to 25–30°. The script prints both so the change is audible.
The reproducible clipped-part counts are 198 (Area 1) and 422 (Area 2).

**Figure 4 — seed spread.** The error bars follow `s07_validacao.py`
(line 146): the spread is the *largest* sample standard deviation among the
RF, XGB and ensemble runs, not the one belonging to the reported model.
Recomputing the RF column alone from `validacao_resultados.csv` gives 0.028
for Area 2, not the 0.034 the manuscript reports — both are correct under
their own definition, and the conservative one is what the pipeline applies.

## Provenance of the trajectory

`reports/area1/trajetoria_validacao.csv` is the only figure input that is
partly transcribed rather than produced. Stages v1 to v2.3 predate the frozen
covariate set and their run artefacts were not archived individually; the
file's header says so. The final v3 point is **not** transcribed —
`fig4_trajectory.py` recomputes it from `validacao_resultados.csv` and stops
with an error if it disagrees with the CSV.

## House style

`figstyle.py` holds the typography, the axis treatment and the palette, so a
style change happens in one place. Column widths are the Elsevier ones
(90 mm, 140 mm, 190 mm); every script writes a vector PDF and a 300 dpi PNG.

The four covariate-family colours of Figure 6 were not chosen by eye: of the
210 four-colour combinations drawn from the ramps, 13 clear every gate on the
all-pairs list against a white surface, and the set used here is the one with
the widest separation — CVD ΔE 13.0 (target ≥ 8), normal-vision ΔE 16.1
(floor ≥ 15), every colour above 3:1 contrast, so none of them depends on its
label to be legible.
