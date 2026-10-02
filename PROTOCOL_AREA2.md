# Area 2 — reproducibility protocol, declared in advance

This file reproduces the scientifically load-bearing sections of the internal
protocol document written for Area 2 before any Area 2 computation was run.
The document is dated **11 August 2026**; the Area 2 stage reports in
`reports/area2/` carry file dates of 11–12 August 2026.

Sections 1 and 2 of the original are reproduced **verbatim** below, in the
Portuguese in which they were written, with an English translation beside
them. Sections 3 to 7 of the original are **omitted**: they are installation
and operating instructions (directory layout on the author's machine, file
copying, stage execution order) with no bearing on the assertions under test.
Nothing in sections 1 and 2 has been edited, reordered or added to.

This protocol was not deposited in a public pre-registration registry. It is
published here so that the assertions reported in Sections 3.6 and 4.5 of the
manuscript can be read as they were written, rather than only as described
after the fact.

---

## 1. Assertions under test (defined BEFORE the numbers)

### Original (Portuguese, verbatim)

> ## 1. Afirmações sob teste (definidas ANTES dos números)
>
> | # | Afirmação | Métrica | Critério de reprodução |
> |---|---|---|---|
> | A1 | O pipeline executa fim a fim sem alteração de código | cadeia s01→s09 concluída com config idêntico | binário |
> | A2 | O teto de previsibilidade é propriedade do alvo, não da área | AUC-vizinho (s09) e fração pepita | teto medido e reportado (sem critério de igualdade — PODE diferir) |
> | A3 | O modelo captura fração equivalente do teto local | AUC espacial ÷ teto-vizinho da própria área | fração ≥ 0,90 (na área 1: 0,98) |
> | A4 | A hierarquia de covariáveis se mantém | estruturais+geofísicas no top 8 do s06 | qualitativo, reportado |
> | A5 | ML supera baseline especialista | AUC > fuzzy | binário |
>
> Barra bloqueante da área 2 = 0,95 × teto-vizinho MEDIDO NA ÁREA 2
> (mesma regra relativa; nunca o 0,672 importado da área 1).

### English translation

| # | Assertion | Metric | Reproduction criterion |
|---|---|---|---|
| A1 | The pipeline runs end to end with no code change | chain s01→s09 completed with an identical config | binary |
| A2 | The predictability ceiling is a property of the target, not of the area | neighbour AUC (s09) and nugget fraction | ceiling measured and reported (no equality criterion — it MAY differ) |
| A3 | The model captures an equivalent fraction of the local ceiling | spatial AUC ÷ the area's own neighbour ceiling | fraction ≥ 0.90 (in Area 1: 0.98) |
| A4 | The covariate hierarchy is preserved | structural + geophysical covariates in the s06 top 8 | qualitative, reported |
| A5 | ML beats the expert baseline | AUC > fuzzy | binary |

Area 2 blocking bar = 0.95 × the neighbour ceiling MEASURED IN AREA 2
(the same relative rule; never the 0.672 imported from Area 1).

**Note added for this archive, not part of the original.** The A3 criterion
was written against the uncorrected ratio AUC ÷ ceiling, which is how Area 1
had been scored at the time (0.98). Section 3.5(c) of the manuscript reports
the fraction chance-corrected, as (AUC − 0.5) ÷ (ceiling − 0.5), because an
AUC of 0.5 is the expectation of a random ranking and not a zero of skill.
Under the corrected metric Area 1 scores 93% and Area 2 scores 94%, so A3 is
met either way; the change of metric is declared rather than silently applied.

---

## 2. Area chosen

### Original (Portuguese, verbatim)

> ## 2. Área escolhida
> Vale do Paraíba paulista (SJC–Taubaté–Guaratinguetá–Campos do Jordão):
>
>     AOI_BBOX=-46.10,-23.35,-44.90,-22.15
>
> Justificativa: mesmo cráton, terreno distinto (flancos cristalinos da
> Mantiqueira/Serra do Mar + Bacia de Taubaté no eixo — o filtro de
> domínio fissural+outro opera como sempre); densidade SIAGAS alta;
> INTEIRAMENTE coberta pelo levantamento moderno 1105 (linhas de 500 m)
> — teste adicional: geofísica de alta resolução em 100% da área.
>
> Sub-área para validação externa interna (config
> validacao.aoi_original_bbox da área 2 — metade central):
>
>     [-45.85, -23.10, -45.15, -22.40]

### English translation

Paraíba valley, São Paulo State (São José dos Campos – Taubaté –
Guaratinguetá – Campos do Jordão):

    AOI_BBOX=-46.10,-23.35,-44.90,-22.15

Rationale: same craton, distinct terrain (crystalline flanks of the
Mantiqueira / Serra do Mar plus the Taubaté Basin along the axis — the
fissural-plus-other domain filter operates as always); high SIAGAS density;
ENTIRELY covered by the modern survey 1105 (500 m line spacing) — an
additional test: high-resolution geophysics over 100% of the area.

Sub-area for the internal external-validation probe (the area's
`validacao.aoi_original_bbox`, the central half):

    [-45.85, -23.10, -45.15, -22.40]

---

## Where these values appear in the archive

| Declared here | Where it is used |
|---|---|
| `AOI_BBOX=-46.10,-23.35,-44.90,-22.15` | `reports/README.md`; set in `.env` at run time |
| `aoi_original_bbox: [-45.85, -23.10, -45.15, -22.40]` | `pipeline/config/config.area2.yaml`, line 67 |
| Blocking bar = 0.95 × the area's own ceiling | `pipeline/config/config.area2.yaml` (`fator_teto`), applied in `pipeline/stages/s07_validacao.py` |
| Ceiling measured in Area 2 | `reports/area2/s09_relatorio.txt` → 0.752 (95% CI 0.671–0.828) |
| A1–A5 outcomes | Section 4.5 of the manuscript; Table 4 |
