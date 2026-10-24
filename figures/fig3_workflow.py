#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura 3 do manuscrito — fluxo do pipeline e estrutura de decisao.

Nao ha dado nesta figura: ela desenha a cadeia de estagios e os dois
mecanismos de decisao declarados antes dos resultados.

  espinha central   a cadeia s01 -> s08c, na ordem de dependencia real
  modulo esquerdo   o teto de previsibilidade (s09/s09b), que FORNECE a
                    barra de aceitacao usada na validacao
  modulo direito    o laco de adocao de covariavel, com os dois
                    vereditos que ele de fato produziu

Os vereditos sao escritos por extenso ("ADOPTED" / "NOT ADOPTED") e
distinguidos tambem por preenchimento, nunca so por cor: a figura
precisa sobreviver a impressao em preto e branco e a daltonismo.

O codigo do estagio fica FORA da caixa, a esquerda: dentro dela ele
disputava espaco com o texto e o sobrepunha.

Uso:
    python fig3_workflow.py [--saida .]

Saida: fig3_workflow.pdf e .png
"""
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import AREA1, AREA2, LARGURA_2COL, TINTA, TINTA2, estilo, grava

CAIXA = "#f2f3f4"
BORDA = "#b8bec4"
FUNDO_TETO = "#e8eff5"
FUNDO_ADOCAO = "#fbeee7"

CORPO = 7.0        # as caixas foram dimensionadas para este corpo
CORPO_PEQ = 6.6

ESPINHA = [
    ("SIAGAS well inventory\nQA/QC protocol", "s01–s02"),
    ("Lithological harmonisation\n3-D weathering surfaces", "s03–s04"),
    ("Covariates: terrain, structural,\nairborne geophysics", "s05–s05c"),
    ("Training matrix\nregion-relative q75 target", "s06"),
    ("Models: RF · XGB · ensemble\nfuzzy expert baseline", "s06"),
    ("Spatial block cross-validation\nexternal test", "s07 · s07e"),
    ("Favourability map, uncertainty,\nclasses with observed rates",
     "s08b–s08c"),
]
I_COVAR, I_MATRIZ, I_VALID = 2, 3, 5


def caixa(ax, x, y, w, h, texto, fundo=CAIXA, borda=BORDA, cor=TINTA,
          corpo=CORPO, negrito=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.010",
        facecolor=fundo, edgecolor=borda, linewidth=0.9, zorder=3))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=corpo, color=cor, zorder=4, linespacing=1.4,
            fontweight="bold" if negrito else "normal")


def seta(ax, p0, p1, cor=TINTA2, traco="-", lw=1.0, curva=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=8.5, color=cor,
        linewidth=lw, linestyle=traco, zorder=2,
        connectionstyle="arc3,rad=%.3f" % curva, shrinkA=1.5, shrinkB=1.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saida", default=".")
    a = p.parse_args()

    estilo()
    fig, ax = plt.subplots(figsize=(LARGURA_2COL, 4.9))
    # folga nos limites: um patch encostado em x=0 e cortado pelo clip do
    # eixo, e a borda arredondada some
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, 1.01)
    ax.set_axis_off()

    # ---------------------------------------------------- espinha
    xc, wc = 0.330, 0.300
    topo, alt, vao = 0.955, 0.093, 0.034
    ys = []
    for i, (rotulo, estagio) in enumerate(ESPINHA):
        y = topo - (i + 1) * alt - i * vao
        ys.append(y)
        caixa(ax, xc, y, wc, alt, rotulo)
        ax.text(xc + 0.004, y + alt + 0.005, estagio, ha="left",
                va="bottom", fontsize=6.0, color=TINTA2, style="italic",
                zorder=4)
        if i:
            seta(ax, (xc + wc / 2, ys[i - 1]), (xc + wc / 2, y + alt))

    # ---------------------------------------------------- teto (esquerda)
    xt, wt = 0.000, 0.238
    y_teto = ys[I_MATRIZ] - 0.010
    caixa(ax, xt, y_teto, wt, 0.108,
          "Predictability ceiling\nneighbour AUC · nugget\nzone-support test",
          fundo=FUNDO_TETO, borda=AREA1)
    ax.text(xt + wt / 2, y_teto + 0.116,
            "measured on the target,\nbefore any model is scored",
            ha="center", va="bottom", fontsize=6.6, color=AREA1,
            style="italic", linespacing=1.3)

    y_barra = ys[I_VALID] + 0.012
    caixa(ax, xt, y_barra, wt, 0.070,
          "Acceptance bar\n0.95 × measured ceiling", fundo=FUNDO_TETO,
          borda=AREA1, cor=AREA1, negrito=True)

    seta(ax, (xc, ys[I_MATRIZ] + alt / 2), (xt + wt, y_teto + 0.080),
         cor=AREA1, lw=1.2, curva=0.16)
    seta(ax, (xt + wt / 2, y_teto), (xt + wt / 2, y_barra + 0.070),
         cor=AREA1, lw=1.2)
    seta(ax, (xt + wt, y_barra + 0.035), (xc, ys[I_VALID] + alt / 2),
         cor=AREA1, lw=1.2)

    # ---------------------------------------------------- adocao (direita)
    xa, wa = 0.700, 0.288
    y_regra = ys[I_COVAR] - 0.014
    caixa(ax, xa, y_regra, wa, 0.106,
          "Covariate adoption rule\ndeclared before the result\n"
          "re-enters the matrix only if adopted",
          fundo=FUNDO_ADOCAO, borda=AREA2, corpo=CORPO_PEQ)
    seta(ax, (xc + wc, ys[I_COVAR] + alt * 0.66), (xa, y_regra + 0.072),
         cor=AREA2, lw=1.2)

    h = 0.100
    y1 = y_regra - 0.070 - h
    caixa(ax, xa, y1, wa, h,
          "Airborne geophysics\nspatial 0.650→0.660\nexternal 0.563→0.593",
          fundo="#ffffff", borda=AREA2, corpo=CORPO_PEQ)
    ax.text(xa + wa / 2, y1 - 0.014, "ADOPTED", ha="center", va="top",
            fontsize=7.4, color="#ffffff", fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.32", facecolor=AREA2,
                      edgecolor="none"))

    y2 = y1 - 0.130 - h
    caixa(ax, xa, y2, wa, h,
          "Regression-kriging of surfaces\nCV RMSE −0.3% to −2.1%\n"
          "(rule asked −5%)", fundo="#ffffff", borda=BORDA,
          corpo=CORPO_PEQ)
    ax.text(xa + wa / 2, y2 - 0.014, "NOT ADOPTED", ha="center", va="top",
            fontsize=7.4, color=TINTA2, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#ffffff",
                      edgecolor=BORDA, linewidth=0.9))

    seta(ax, (xa + wa / 2, y_regra), (xa + wa / 2, y1 + h), cor=AREA2,
         lw=1.1)

    seta(ax, (xa + wa / 2, y1 - 0.048), (xa + wa / 2, y2 + h), cor=TINTA2,
         lw=1.1, traco=(0, (3, 2)))

    # o veredito adotado volta para as covariaveis: e um laco
    seta(ax, (xa, y1 + h / 2), (xc + wc, ys[I_COVAR] + alt * 0.26),
         cor=AREA2, lw=1.2, curva=0.32)

    grava(fig, a.saida, "fig3_workflow")


if __name__ == "__main__":
    main()
