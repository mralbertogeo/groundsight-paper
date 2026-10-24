#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura 1 do manuscrito — localizacao das duas areas de estudo.

(a) Posicao do Estado de Sao Paulo no Brasil, com as duas AOIs.
(b) Detalhe do Estado, com as AOIs e os pocos aprovados no QA/QC.

Os numeros NAO sao digitados: as contagens de poco saem do proprio
geopackage de insumos (camada 'pocos', coluna 'qaqc_aprovado') e as
AOIs saem da camada 'aoi'. O contorno estadual e a malha de UF do IBGE.

Projecoes, escolhidas e nao herdadas: painel (a) em SIRGAS 2000 /
Policonica do Brasil (EPSG:5880), que e a do IBGE para o pais inteiro;
painel (b) em SIRGAS 2000 / UTM 23S (EPSG:31983), a mesma do pipeline.
Desenhar qualquer um dos dois em coordenadas geograficas esticaria a
figura na horizontal nesta latitude.

Uso:
    python fig1_location.py --uf <BR_UF_2025.shp> --gpkg1 <...> \
        --gpkg2 <...> [--saida .]

Saida: fig1_location.pdf e .png
"""
import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

from figstyle import AREA1, AREA2, LARGURA_2COL, TINTA, TINTA2, estilo, grava

CRS_PAIS = 5880       # policonica do Brasil (IBGE)
CRS_ESTADO = 31983    # UTM 23S

# A malha do IBGE vem em resolucao plena e o PDF saiu com 19,8 MB — um
# arquivo que nenhuma revista aceita e que nenhum vertice justifica. As
# tolerancias abaixo estao MUITO abaixo de um pixel na escala de cada
# painel: no painel do pais, 1 mm impresso vale cerca de 50 km; no do
# estado, cerca de 9 km.
TOL_PAIS = 2_000      # m
TOL_ESTADO = 300      # m


def simplifica(gdf, tol):
    g = gdf.copy()
    g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
    return g
UF_ALVO = "SP"

CINZA_UF = "#ebebeb"
CINZA_BORDA = "#ffffff"
CINZA_SP = "#cfd6dc"

AREAS = [("area1", "Area 1\nParaná Basin border", AREA1),
         ("area2", "Area 2\nTaubaté rift", AREA2)]


def coluna_sigla(gdf):
    for c in ("SIGLA_UF", "SIGLA", "sigla_uf", "sigla", "CD_UF"):
        if c in gdf.columns:
            return c
    raise SystemExit("Nao achei a coluna de sigla da UF em %s"
                     % list(gdf.columns))


def barra_escala(ax, comprimento_m, rotulo, frac_x=0.06, frac_y=0.08):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + frac_x * (x1 - x0)
    y = y0 + frac_y * (y1 - y0)
    alt = 0.012 * (y1 - y0)
    ax.plot([x, x + comprimento_m], [y, y], color=TINTA, linewidth=1.6,
            solid_capstyle="butt", zorder=6)
    for xx in (x, x + comprimento_m):
        ax.plot([xx, xx], [y - alt, y + alt], color=TINTA, linewidth=1.6,
                zorder=6)
    ax.text(x + comprimento_m / 2, y + 1.8 * alt, rotulo, ha="center",
            va="bottom", fontsize=7, color=TINTA, zorder=6)


def seta_norte(ax, frac_x=0.94, frac_y=0.80):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + frac_x * (x1 - x0)
    y = y0 + frac_y * (y1 - y0)
    d = 0.07 * (y1 - y0)
    ax.annotate("N", xy=(x, y + d), xytext=(x, y), ha="center",
                va="top", fontsize=8.5, fontweight="bold", color=TINTA,
                arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=1.2),
                zorder=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uf", required=True)
    p.add_argument("--gpkg1", required=True)
    p.add_argument("--gpkg2", required=True)
    p.add_argument("--saida", default=".")
    a = p.parse_args()

    ufs = gpd.read_file(a.uf)
    col = coluna_sigla(ufs)
    sp = ufs[ufs[col] == UF_ALVO]
    if len(sp) != 1:
        raise SystemExit("Esperava 1 feicao para %s, achei %d"
                         % (UF_ALVO, len(sp)))

    aoi, pocos = {}, {}
    for (chave, _, _), caminho in zip(AREAS, (a.gpkg1, a.gpkg2)):
        aoi[chave] = gpd.read_file(caminho, layer="aoi")
        todos = gpd.read_file(caminho, layer="pocos")
        col_ok = "qaqc_aprovado"
        if col_ok not in todos.columns:
            raise SystemExit("Camada 'pocos' de %s sem %s" % (caminho, col_ok))
        ok = todos[todos[col_ok].astype(str).str.lower().isin(
            ("true", "1", "t", "sim"))]
        pocos[chave] = ok
        print("%s: %d pocos no geopackage, %d aprovados no QA/QC"
              % (chave, len(todos), len(ok)))

    estilo()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(LARGURA_2COL, 3.9),
        gridspec_kw={"width_ratios": [1.0, 1.35]})

    # ---------------- (a) Brasil ----------------
    pais = simplifica(ufs.to_crs(CRS_PAIS), TOL_PAIS)
    pais.plot(ax=ax1, facecolor=CINZA_UF, edgecolor=CINZA_BORDA,
              linewidth=0.35, zorder=2)
    simplifica(sp.to_crs(CRS_PAIS), TOL_PAIS).plot(
        ax=ax1, facecolor=CINZA_SP, edgecolor="#8a97a3", linewidth=0.6,
        zorder=3)
    for chave, rotulo, cor in AREAS:
        g = aoi[chave].to_crs(CRS_PAIS)
        x0, y0, x1, y1 = g.total_bounds
        ax1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor=cor, linewidth=1.1, zorder=5))
    # as AOIs tem ~1 grau num pais de 40: um marcador guia o olho ate elas
    cx = np.mean([aoi[k].to_crs(CRS_PAIS).total_bounds[[0, 2]].mean()
                  for k, _, _ in AREAS])
    cy = np.mean([aoi[k].to_crs(CRS_PAIS).total_bounds[[1, 3]].mean()
                  for k, _, _ in AREAS])
    # rotulo acima e a esquerda: abaixo ele cai sobre a barra de escala
    ax1.annotate("São Paulo State\n(detail in b)", xy=(cx, cy),
                 xytext=(-78, 46), textcoords="offset points",
                 fontsize=7.5, color=TINTA2, ha="center",
                 arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.7),
                 zorder=6)
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    ax1.set_title("(a) Brazil", loc="left", pad=6)
    barra_escala(ax1, 1_000_000, "1,000 km", frac_x=0.05, frac_y=0.06)

    # ---------------- (b) Sao Paulo ----------------
    sp_m = simplifica(sp.to_crs(CRS_ESTADO), TOL_ESTADO)
    # estados vizinhos em fundo claro: a AOI da area 2 cruza a divisa
    # com o Rio de Janeiro, e sem o contexto o transbordo parece erro
    simplifica(ufs.to_crs(CRS_ESTADO), TOL_ESTADO).plot(
        ax=ax2, facecolor="#f3f3f3", edgecolor="#d2d7db", linewidth=0.5,
        zorder=1)
    sp_m.plot(ax=ax2, facecolor="#e9edf0", edgecolor="#8a97a3",
              linewidth=0.9, zorder=2)
    for chave, rotulo, cor in AREAS:
        g = aoi[chave].to_crs(CRS_ESTADO)
        pocos[chave].to_crs(CRS_ESTADO).plot(
            ax=ax2, color=cor, markersize=0.7, linewidth=0, alpha=0.55,
            zorder=4)
        x0, y0, x1, y1 = g.total_bounds
        ax2.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor=cor, linewidth=1.2, zorder=5))
        # area 1 rotulada a esquerda da propria caixa, area 2 abaixo:
        # empilhados sob as duas caixas eles se sobrepunham entre si e
        # sobre o contorno estadual
        if chave == "area1":
            ax2.annotate("%s\nn = %s"
                         % (rotulo, format(len(pocos[chave]), ",")),
                         xy=(x0, 0.5 * (y0 + y1)), xytext=(-10, 0),
                         textcoords="offset points", ha="right",
                         va="center", fontsize=7.5, color=cor, zorder=6,
                         path_effects=[pe.withStroke(linewidth=2.4,
                                                     foreground="white")])
        else:
            ax2.annotate("%s\nn = %s"
                         % (rotulo, format(len(pocos[chave]), ",")),
                         xy=(0.5 * (x0 + x1), y0), xytext=(0, -12),
                         textcoords="offset points", ha="center",
                         va="top", fontsize=7.5, color=cor, zorder=6,
                         path_effects=[pe.withStroke(linewidth=2.4,
                                                     foreground="white")])
    minx, miny, maxx, maxy = sp_m.total_bounds
    folga = 0.05 * (maxx - minx)
    ax2.set_xlim(minx - folga, maxx + folga)
    ax2.set_ylim(miny - 1.6 * folga, maxy + folga)
    ax2.set_aspect("equal")
    ax2.set_axis_off()
    ax2.set_title("(b) São Paulo State", loc="left", pad=6)
    barra_escala(ax2, 100_000, "100 km", frac_x=0.04, frac_y=0.10)
    seta_norte(ax2)

    fig.tight_layout()
    grava(fig, a.saida, "fig1_location")


if __name__ == "__main__":
    main()
