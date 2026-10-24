#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura 7 do manuscrito — produtos finais validados, 2x2.

  coluna (a)  probabilidade da classe de alta favorabilidade
  coluna (b)  discordancia entre modelos |RF - XGB|
  linha 1     Area 1 — borda da Bacia do Parana
  linha 2     Area 2 — rifte de Taubate

As versoes anteriores (fig5_area1 e fig5_area2, geradas em separado pelo
fig_export.py) rotulavam os DOIS paineis de cada area como "(a)" e "(b)",
de modo que a figura montada tinha dois "(a)" e dois "(b)" — contra a
legenda, onde (a) e (b) sao COLUNAS e as areas sao LINHAS. Aqui o rotulo
de coluna aparece uma vez no topo e a area identifica a linha.

ESCALA DE COR, decisao declarada: cada painel tem sua propria escala,
com a barra ao lado e o limite superior no percentil 99. As duas areas
nao sao comparaveis em nivel absoluto — o alvo e o q75 da PROPRIA area,
entao "probabilidade alta" significa coisas diferentes em cada uma. O
que a legenda compara e o PADRAO espacial, nao a magnitude.

A rampa da probabilidade e escolhida por --cmap-prob. O default e
RdYlGn, decisao do autor em 24/10/2026 para manter a leitura
vermelho = baixo / verde = alto consagrada nos mapas de favorabilidade.
Alternativas avaliadas e disponiveis: azul_casa (monocromatica, segura
para daltonismo) e viridis.

Uso:
    python fig7_products.py --raiz1 <dir> --raiz2 <dir> [--saida .]
                            [--cmap-prob viridis] [--max-px 1200]

Saida: fig7_products.pdf e .png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from figstyle import LARGURA_2COL, TINTA, TINTA2, estilo, grava

# Rampa sequencial azul da casa (claro -> escuro), de uma unica matiz,
# monotonica em luminancia e segura para daltonismo.
AZUL_CASA = LinearSegmentedColormap.from_list("azul_casa", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
    "#184f95", "#0d366b"])
plt.colormaps.register(AZUL_CASA, force=True)

AREAS = [("area1", "Area 1 — Paraná Basin border"),
         ("area2", "Area 2 — Taubaté rift")]
COLUNAS = [("favorabilidade_prob", "(a) P(high favourability)",
            "probability", None),
           ("incerteza_modelos", "(b) Inter-model disagreement",
            "|RF − XGB|", "magma")]

SUB = "data/derived/covariaveis"


def le(caminho, max_px):
    """Le a banda 1 e reduz por passo inteiro ate caber em max_px."""
    with rasterio.open(caminho) as s:
        v = s.read(1).astype(float)
        if s.nodata is not None:
            v[v == s.nodata] = np.nan
        passo = max(1, int(np.ceil(max(v.shape) / max_px)))
        return v[::passo, ::passo], s.transform.a * passo


def barra_escala(ax, res_m, forma, km=20):
    """Barra de km no canto inferior esquerdo, em pixels da imagem."""
    n = km * 1000.0 / res_m
    x0 = forma[1] * 0.06
    y0 = forma[0] * 0.945
    alt = forma[0] * 0.009
    ax.plot([x0, x0 + n], [y0, y0], color=TINTA, lw=1.6, zorder=6,
            solid_capstyle="butt")
    for xx in (x0, x0 + n):
        ax.plot([xx, xx], [y0 - alt, y0 + alt], color=TINTA, lw=1.6,
                zorder=6)
    ax.text(x0 + n / 2, y0 - 2.0 * alt, "%d km" % km, ha="center",
            va="bottom", fontsize=6.8, color=TINTA, zorder=6)


def seta_norte(ax, forma):
    x, y = forma[1] * 0.945, forma[0] * 0.115
    d = forma[0] * 0.055
    ax.annotate("N", xy=(x, y - d), xytext=(x, y), ha="center", va="top",
                fontsize=7.6, fontweight="bold", color=TINTA, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=1.1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz1", required=True)
    p.add_argument("--raiz2", required=True)
    p.add_argument("--saida", default=".")
    p.add_argument("--cmap-prob", default="RdYlGn")
    p.add_argument("--max-px", type=int, default=1200)
    a = p.parse_args()

    raizes = {"area1": a.raiz1, "area2": a.raiz2}
    estilo()
    fig, eixos = plt.subplots(2, 2, figsize=(LARGURA_2COL, 7.25))

    for i, (chave, rotulo_area) in enumerate(AREAS):
        for j, (arq, titulo, unidade, cmap) in enumerate(COLUNAS):
            ax = eixos[i][j]
            caminho = os.path.join(raizes[chave], SUB, arq + ".tif")
            if not os.path.exists(caminho):
                raise SystemExit("Nao encontrei %s" % caminho)
            v, res = le(caminho, a.max_px)
            vmax = np.nanpercentile(v, 99)
            im = ax.imshow(np.ma.masked_invalid(v),
                           cmap=cmap or a.cmap_prob, vmin=0, vmax=vmax,
                           interpolation="nearest")
            # barra presa ao eixo do mapa com tamanho fixo: com
            # fig.colorbar(ax=...) as quatro sairam com larguras e
            # recuos diferentes, e o rotulo da linha 2 derivou para
            # dentro do painel vizinho
            cax = make_axes_locatable(ax).append_axes("right", size="4%",
                                                      pad=0.06)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(unidade, fontsize=7.4)
            cb.ax.tick_params(labelsize=6.8, length=2.5, width=0.7)
            cb.outline.set_linewidth(0.6)
            cb.outline.set_edgecolor("#b0b0b0")

            if j == 0:                      # a escala vale para a linha
                barra_escala(ax, res, v.shape)
            seta_norte(ax, v.shape)

            ax.set_xticks([])
            ax.set_yticks([])
            for lado in ax.spines.values():
                lado.set_color("#b0b0b0")
                lado.set_linewidth(0.7)
            if i == 0:                      # rotulo de coluna, uma vez
                ax.set_title(titulo, loc="left", pad=7, color=TINTA)
            if j == 0:                      # rotulo de linha = a area
                ax.set_ylabel(rotulo_area, fontsize=8.5, labelpad=8,
                              color=TINTA)
            print("%s/%s: %dx%d (res %.0f m) vmax=%.3f"
                  % (chave, arq, v.shape[1], v.shape[0], res, vmax))

    fig.text(0.5, 0.012,
             "Colour scales are per panel: the target is the q75 of each "
             "area's own distribution, so absolute levels are not "
             "comparable between areas.",
             ha="center", va="bottom", fontsize=6.8, color=TINTA2)
    fig.tight_layout(rect=(0, 0.028, 1, 1))
    grava(fig, a.saida, "fig7_products")


if __name__ == "__main__":
    main()
