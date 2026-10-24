#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sistema visual comum das figuras do manuscrito.

Centraliza o que o fig3_tetos.py fixou, para que as figuras 1, 2, 4, 6
e 7 saiam com a mesma tipografia, os mesmos eixos e a mesma paleta, e
para que uma mudanca de estilo seja feita em um lugar so.

Larguras da Elsevier: 90 mm (1 coluna) e 190 mm (2 colunas).
Saida sempre em PDF (vetorial, para a revista) e PNG a 300 dpi.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- cores
# Areas (herdado do fig3_tetos.py): azul da casa #1F4E78 clareado para
# #205E8F, que passa no piso de luminancia e de croma; laranja mantido.
AREA1 = "#205E8F"
AREA2 = "#C05000"

# Familias de covariavel (figura 6). O conjunto foi escolhido por
# enumeracao: das 210 combinacoes de quatro cores das rampas, 13 passam
# todos os gates no pairlist all-pairs sobre superficie branca, e esta e
# a de melhor separacao — CVD dE 13.0 (alvo >= 8), visao normal 16.1
# (piso >= 15), todas acima de 3:1 contra o branco (nenhuma depende de
# rotulo para ser legivel). Verificavel com validate_palette.js.
FAMILIA = {
    "structural": "#2a78d6",
    "geophysics": "#C05000",
    "terrain": "#4a3aa7",
    "weathering": "#8c2d4a",
    "well design": "#8a8a8a",   # controle, deliberadamente acromatico
}
FAMILIA_ROTULO = {
    "structural": "Structural distance",
    "geophysics": "Airborne geophysics",
    "terrain": "Terrain morphometry",
    "weathering": "3-D weathering model",
    "well design": "Well design (control)",
}

TINTA = "#1a1a1a"
TINTA2 = "#5c5c5c"
EIXO = "#b0b0b0"
GRADE = "#e6e6e6"

LARGURA_1COL = 3.54   # 90 mm
LARGURA_15COL = 5.51  # 140 mm
LARGURA_2COL = 7.48   # 190 mm


def estilo():
    """Aplica o rcParams comum. Chamar antes de criar a figura."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": EIXO,
        "axes.linewidth": 0.8,
        "xtick.color": TINTA2,
        "ytick.color": TINTA2,
        "text.color": TINTA,
        "axes.labelcolor": TINTA,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def limpa(ax, grade="y"):
    """Remove as bordas superior e direita e poe a grade recuada.

    grade: "y", "x" ou None — o eixo que recebe a grade. Em barras
    horizontais a grade util e a do x.
    """
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.set_axisbelow(True)
    if grade:
        ax.grid(axis=grade, color=GRADE, linewidth=0.6, zorder=0)


def grava(fig, saida, nome):
    """Grava PDF e PNG com o mesmo nome base."""
    os.makedirs(saida, exist_ok=True)
    for ext in ("pdf", "png"):
        destino = os.path.join(saida, "%s.%s" % (nome, ext))
        fig.savefig(destino, bbox_inches="tight")
        print("gravado:", destino)
