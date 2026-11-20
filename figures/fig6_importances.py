#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura 6 do manuscrito — importancia por permutacao das oito
covariaveis principais, colorida por familia.

(a) Area 1 — borda da Bacia do Parana
(b) Area 2 — rifte de Taubate

Os numeros NAO sao digitados: o script le o CSV de importancia
arquivado de cada area, de modo que a figura e sempre consistente com o
que o s06 produziu. A traducao nome-interno -> rotulo em ingles e a
atribuicao de familia ficam em tabelas explicitas aqui; qualquer
covariavel que o s06 produzir e que nao esteja nelas interrompe a
execucao em vez de ser desenhada sem familia.

VARIANTE (09/11/2026): o default passou a ser pre_perfuracao, que le
importancia_permutacao_pre_perfuracao.csv. Motivo: o mapa e entregue
para decidir ONDE perfurar, e as covariaveis de projeto do poco
(profundidade final, comprimento de intervalo) nao existem nesse
momento — alem de medirem causalidade reversa, ja que a profundidade
final e decidida durante a obra. No conjunto completo elas ocupavam as
DUAS primeiras posicoes nas duas areas, com importancia ~3x a da
primeira covariavel de sitio.

Uso:
    python fig6_importances.py [--reports reports] [--saida .]
                               [--variante pre_perfuracao|completo]

Saida: fig6_importances.pdf/.png (variante pre_perfuracao) ou
       fig6_importances_completo.pdf/.png (variante completo)
"""
import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from figstyle import (FAMILIA, FAMILIA_ROTULO, LARGURA_2COL, TINTA,
                      TINTA2, estilo, grava, limpa)

N_TOPO = 8

# Os titulos vao em tinta neutra de proposito: a cor aqui significa
# FAMILIA (ver legenda), e pintar "Area 2" de #C05000 a confundiria com
# a familia geofisica. A identidade da area fica no proprio texto.
AREAS = [("area1", "(a) Area 1 — Paraná Basin border"),
         ("area2", "(b) Area 2 — Taubaté rift")]

# nome interno -> (rotulo em ingles, familia)
COVARIAVEIS = {
    "profundidade_m":        ("Well depth",                    "well design"),
    "lf_m":                  ("Screen length",                 "well design"),
    "geof_k":                ("Radiometric K",                 "geophysics"),
    "geof_eth":              ("Radiometric eTh",               "geophysics"),
    "geof_eu":               ("Radiometric eU",                "geophysics"),
    "geof_ct":               ("Radiometric total count",       "geophysics"),
    "geof_mag":              ("Magnetic field",                "geophysics"),
    "geof_mag_ght":          ("Magnetic THG",                  "geophysics"),
    "geof_mag_tilt":         ("Magnetic tilt derivative",      "geophysics"),
    "geof_mag_1dv":          ("Magnetic 1st vert. derivative", "geophysics"),
    "dist_lin_ne_sw":        ("Dist. NE–SW lineaments",        "structural"),
    "dist_lin_ew":           ("Dist. E–W lineaments",          "structural"),
    "dist_lin_nw_se":        ("Dist. NW–SE lineaments",        "structural"),
    "dist_lin_ns":           ("Dist. N–S lineaments",          "structural"),
    "dist_lin_intersec":     ("Dist. lineament intersections", "structural"),
    "dist_lin_ne_sw_sgb":    ("Dist. NE–SW structures (SGB)",  "structural"),
    "dist_lin_ew_sgb":       ("Dist. E–W structures (SGB)",    "structural"),
    "dist_lin_nw_se_sgb":    ("Dist. NW–SE structures (SGB)",  "structural"),
    "dist_lin_ns_sgb":       ("Dist. N–S structures (SGB)",    "structural"),
    "dist_lin_intersec_sgb": ("Dist. structure intersections", "structural"),
    "mde":                   ("Elevation",                     "terrain"),
    "declividade":           ("Slope",                         "terrain"),
    "tri":                   ("Terrain ruggedness index",      "terrain"),
    "twi":                   ("Topographic wetness index",      "terrain"),
    "hand":                  ("HAND",                          "terrain"),
    "dist_drenagem":         ("Dist. drainage",                "terrain"),
    "esp_solo":              ("Soil thickness",                "weathering"),
    "esp_saprolito":         ("Saprolite thickness",           "weathering"),
    "esp_rocha_alterada":    ("Weathered-rock thickness",      "weathering"),
    "prof_base_solo":        ("Depth to soil base",            "weathering"),
    "prof_topo_zona_rocha":  ("Depth to weathered rock",       "weathering"),
    "prof_topo_rocha_sa":    ("Depth to sound rock",           "weathering"),
    # 09/11/2026: completadas as 29 covariaveis que o s06 produz e que
    # faltavam aqui — duas delas entram no top 8 da variante
    # pre_perfuracao (dist_lin_total_sgb na Area 1, tri_f250 na Area 2)
    "dist_lin_total":        ("Dist. all lineaments",           "structural"),
    "dist_lin_total_sgb":    ("Dist. all structures (SGB)",     "structural"),
    "dens_lin_ne_sw":        ("Dens. NE\u2013SW lineaments",       "structural"),
    "dens_lin_ew":           ("Dens. E\u2013W lineaments",         "structural"),
    "dens_lin_nw_se":        ("Dens. NW\u2013SE lineaments",       "structural"),
    "dens_lin_ns":           ("Dens. N\u2013S lineaments",         "structural"),
    "dens_lin_total":        ("Dens. all lineaments",           "structural"),
    "dens_lin_intersec":     ("Dens. lineament intersections",  "structural"),
    "dens_lin_ne_sw_sgb":    ("Dens. NE\u2013SW structures (SGB)", "structural"),
    "dens_lin_ew_sgb":       ("Dens. E\u2013W structures (SGB)",   "structural"),
    "dens_lin_nw_se_sgb":    ("Dens. NW\u2013SE structures (SGB)", "structural"),
    "dens_lin_ns_sgb":       ("Dens. N\u2013S structures (SGB)",   "structural"),
    "dens_lin_total_sgb":    ("Dens. all structures (SGB)",     "structural"),
    "dens_lin_intersec_sgb": ("Dens. structure intersections",  "structural"),
    "curvatura":             ("Curvature",                      "terrain"),
    "northness":             ("Northness",                      "terrain"),
    "dens_drenagem":         ("Drainage density",               "terrain"),
    "declividade_f250":      ("Slope (250 m focal)",            "terrain"),
    "declividade_f500":      ("Slope (500 m focal)",            "terrain"),
    "declividade_f1000":     ("Slope (1 km focal)",             "terrain"),
    "hand_f250":             ("HAND (250 m focal)",             "terrain"),
    "hand_f500":             ("HAND (500 m focal)",             "terrain"),
    "hand_f1000":            ("HAND (1 km focal)",              "terrain"),
    "twi_f250":              ("TWI (250 m focal)",              "terrain"),
    "twi_f500":              ("TWI (500 m focal)",              "terrain"),
    "twi_f1000":             ("TWI (1 km focal)",               "terrain"),
    "tri_f250":              ("TRI (250 m focal)",              "terrain"),
    "tri_f500":              ("TRI (500 m focal)",              "terrain"),
    "tri_f1000":             ("TRI (1 km focal)",               "terrain"),
}


def ler_importancias(caminho):
    """Le o CSV do s06 e devolve as N_TOPO primeiras, ja ordenadas."""
    with open(caminho, encoding="utf-8-sig", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    if not linhas:
        sys.exit("CSV vazio: %s" % caminho)
    campo = [c for c in linhas[0] if c != "covariavel"]
    if len(campo) != 1:
        sys.exit("Esperava uma coluna de valor em %s, achei %s"
                 % (caminho, campo))
    dados = [(r["covariavel"].strip(), float(r[campo[0]])) for r in linhas]
    dados.sort(key=lambda t: -t[1])
    topo = dados[:N_TOPO]

    desconhecidas = [n for n, _ in topo if n not in COVARIAVEIS]
    if desconhecidas:
        sys.exit("Covariavel sem rotulo/familia em COVARIAVEIS: %s\n"
                 "Acrescente antes de gerar a figura — nao desenho "
                 "covariavel sem familia." % ", ".join(desconhecidas))
    return topo


def painel(ax, topo, titulo, xmax):
    nomes = [COVARIAVEIS[n][0] for n, _ in topo]
    fams = [COVARIAVEIS[n][1] for n, _ in topo]
    vals = np.array([v for _, v in topo])

    # maior em cima
    y = np.arange(len(topo))[::-1]
    ax.barh(y, vals, height=0.62,
            color=[FAMILIA[f] for f in fams], linewidth=0, zorder=3)

    for yi, v in zip(y, vals):
        ax.text(v + vals.max() * 0.025, yi, "%.3f" % v, va="center",
                ha="left", fontsize=7.5, color=TINTA2, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(topo) - 0.3)
    ax.set_xlabel("Permutation importance (ΔAUC)")
    ax.set_title(titulo, loc="left", pad=8, color=TINTA)
    limpa(ax, grade="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports", default="reports")
    p.add_argument("--saida", default=".")
    p.add_argument("--variante", default="pre_perfuracao",
                   help="pre_perfuracao (default) | completo")
    a = p.parse_args()
    if a.variante not in ("pre_perfuracao", "completo"):
        sys.exit("Variante desconhecida: %s" % a.variante)
    sufixo = "" if a.variante == "completo" else "_" + a.variante
    nome_saida = ("fig6_importances" if a.variante == "pre_perfuracao"
                  else "fig6_importances_" + a.variante)
    print("Variante: %s" % a.variante)

    dados, usadas = {}, []
    for chave, _ in AREAS:
        caminho = os.path.join(a.reports, chave,
                               "importancia_permutacao%s.csv" % sufixo)
        if not os.path.exists(caminho):
            sys.exit("Nao encontrei %s" % caminho)
        dados[chave] = ler_importancias(caminho)
        for n, _ in dados[chave]:
            f = COVARIAVEIS[n][1]
            if f not in usadas:
                usadas.append(f)
        print("%s: %s" % (chave, ", ".join(
            "%s %.3f" % (COVARIAVEIS[n][0], v) for n, v in dados[chave])))

    estilo()
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_2COL, 3.45))
    # Escala x COMUM aos dois paineis: a importancia por permutacao esta
    # em unidades de AUC nas duas areas, entao comprimento de barra e
    # comparavel entre elas — escalas distintas convidariam a leitura
    # errada de que a hierarquia da area 2 e tao forte quanto a da 1.
    xmax = 1.26 * max(v for d in dados.values() for _, v in d)
    for ax, (chave, titulo) in zip(eixos, AREAS):
        painel(ax, dados[chave], titulo, xmax)

    # legenda na ordem em que as familias aparecem nos dois paineis
    ordem = ["structural", "geophysics", "terrain", "weathering",
             "well design"]
    manipulos = [plt.Rectangle((0, 0), 1, 1, color=FAMILIA[f],
                               label=FAMILIA_ROTULO[f])
                 for f in ordem if f in usadas]
    fig.legend(handles=manipulos, loc="lower center",
               ncol=len(manipulos), frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.1,
               handleheight=1.1, handletextpad=0.5, columnspacing=1.6)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    grava(fig, a.saida, nome_saida)


if __name__ == "__main__":
    main()
