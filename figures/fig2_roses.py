#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura 2 do manuscrito — rosetas de azimutes estruturais do SGB,
recortadas por AOI, ponderadas por comprimento.

(a) Area 1 — borda da Bacia do Parana
(b) Area 2 — rifte de Taubate

ESTIMADOR. O azimute e calculado POR SEGMENTO (par de vertices
consecutivos), ponderado pelo comprimento do segmento. O fig_export.py
original usava os EXTREMOS de cada feicao recortada, o que trata uma
falha sinuosa como um traco reto e credita todo o comprimento dela a um
unico rumo. Na Area 1 os dois estimadores discordam: extremos dao modo
em 35-40 graus, segmentos em 25-30. Na Area 2 coincidem (55-60). O
script imprime os dois, para que a troca seja audivel e nao silenciosa.

O eixo radial e PERCENTUAL do comprimento mapeado da propria area, nao
quilometragem: as duas areas tem comprimentos totais muito diferentes
(cerca de 2.000 km contra 5.300 km) e o que a figura compara e a FORMA
da distribuicao. Os totais vao impressos em cada painel.

Uso:
    python fig2_roses.py --raw1 <dir> --raw2 <dir> [--saida .]

Saida: fig2_roses.pdf e .png, mais reports/area*/azimutes_estruturais.csv
"""
import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from figstyle import AREA1, AREA2, LARGURA_2COL, TINTA2, estilo, grava

CRS_GEO = "EPSG:4674"      # SIRGAS 2000, o dos dados do SGB
CRS_METRICO = 31983        # SIRGAS 2000 / UTM 23S — comprimentos em m
N_BINS = 36                # 10 graus por classe, sobre 180

AREAS = {
    "area1": {"rotulo": "(a) Area 1 — Paraná Basin border",
              "cor": AREA1,
              "bbox": (-47.35, -23.25, -46.15, -22.05),
              "fontes": ["geol_linha_dobra.shp", "geol_linha_falha.shp",
                         "geol_linha_fratura.shp"]},
    "area2": {"rotulo": "(b) Area 2 — Taubaté rift",
              "cor": AREA2,
              "bbox": (-46.10, -23.35, -44.90, -22.15),
              "fontes": ["geol_linha_dobra.shp", "geol_linha_falha.shp",
                         "geol_linha_fratura.shp",
                         "estruturas_gis_brasil_2004.shp"]},
}


def carrega(diretorios, fontes, bb):
    """Le as fontes que existirem, recorta pela AOI e explode em partes."""
    partes, usadas = [], []
    for nome in fontes:
        caminho = None
        for d in diretorios:
            tentativa = os.path.join(d, nome)
            if os.path.exists(tentativa):
                caminho = tentativa
                break
        if caminho is None:
            continue
        g = gpd.read_file(caminho, bbox=bb)
        if len(g) == 0:
            continue
        if g.crs is None:
            g = g.set_crs(CRS_GEO)
        partes.append(g.to_crs(CRS_GEO)[["geometry"]])
        usadas.append(nome)
    if not partes:
        raise SystemExit("Nenhuma fonte estrutural encontrada em %s"
                         % ", ".join(diretorios))
    lin = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=CRS_GEO)
    lin = lin[lin.geometry.type.isin(["LineString", "MultiLineString"])]
    lin = lin.explode(index_parts=False).reset_index(drop=True)
    lin = gpd.clip(lin, gpd.GeoDataFrame(geometry=[box(*bb)], crs=CRS_GEO))
    lin = lin[~lin.geometry.is_empty & lin.geometry.notna()]
    lin = lin.explode(index_parts=False).reset_index(drop=True)
    return lin[lin.geometry.type == "LineString"].to_crs(CRS_METRICO), usadas


def azimutes(metrico):
    """Devolve (az_seg, w_seg, az_ext, w_ext) em graus 0-180 e metros."""
    az_s, w_s, az_e, w_e = [], [], [], []
    for geo in metrico.geometry:
        c = np.asarray(geo.coords)[:, :2]
        if len(c) < 2:
            continue
        dx, dy = c[-1, 0] - c[0, 0], c[-1, 1] - c[0, 1]
        az_e.append(np.degrees(np.arctan2(dx, dy)) % 180)
        w_e.append(geo.length)
        d = np.diff(c, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        ok = L > 0
        az_s.extend((np.degrees(np.arctan2(d[ok, 0], d[ok, 1])) % 180))
        w_s.extend(L[ok])
    return (np.array(az_s), np.array(w_s),
            np.array(az_e), np.array(w_e))


def concentracao(az, w):
    """Comprimento resultante medio de dados AXIAIS (angulo dobrado).

    0 = sem direcao preferencial, 1 = todo o comprimento em um rumo so.
    """
    t = np.radians(2 * az)
    r = np.hypot((w * np.cos(t)).sum(), (w * np.sin(t)).sum()) / w.sum()
    return r


def painel(ax, az, w, cor, rotulo, rmax):
    h, b = np.histogram(az, bins=N_BINS, range=(0, 180), weights=w)
    pct = 100.0 * h / w.sum()
    largura = np.radians(180.0 / N_BINS)
    centros = np.radians(0.5 * (b[:-1] + b[1:]))
    # eixo bidirecional: a roseta de dado axial e simetrica por construcao
    ax.bar(np.concatenate([centros, centros + np.pi]),
           np.concatenate([pct, pct]), width=largura, color=cor,
           edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 45),
                      ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_ylim(0, rmax)
    anel = np.linspace(0, rmax, 4)[1:]
    ax.set_yticks(anel)
    ax.set_yticklabels(["%.0f%%" % v for v in anel], fontsize=6.5,
                       color=TINTA2)
    ax.set_rlabel_position(112)
    ax.grid(color="#dcdcdc", linewidth=0.5)
    ax.set_title(rotulo, loc="left", pad=14)
    ax.spines["polar"].set_color("#d0d0d0")
    ax.spines["polar"].set_linewidth(0.8)
    return pct.max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw1", required=True, help="dir dos shapefiles da area 1")
    p.add_argument("--raw2", required=True, help="dir extra da area 2")
    p.add_argument("--saida", default=".")
    p.add_argument("--reports", default=None,
                   help="se dado, grava azimutes_estruturais.csv por area")
    a = p.parse_args()

    dados = {}
    for chave, cfg in AREAS.items():
        metrico, usadas = carrega([a.raw1, a.raw2], cfg["fontes"],
                                  cfg["bbox"])
        az_s, w_s, az_e, w_e = azimutes(metrico)
        h, b = np.histogram(az_s, bins=N_BINS, range=(0, 180), weights=w_s)
        he, _ = np.histogram(az_e, bins=N_BINS, range=(0, 180), weights=w_e)
        i, ie = h.argmax(), he.argmax()
        dados[chave] = {"az": az_s, "w": w_s, "n_partes": len(az_e),
                        "n_seg": len(az_s), "km": w_s.sum() / 1000.0,
                        "modo": (b[i], b[i + 1]), "r": concentracao(az_s, w_s),
                        "fontes": usadas}
        print("%s: %d partes / %d segmentos | %.0f km | modo por segmento "
              "%.0f-%.0f° (por extremos %.0f-%.0f°) | concentracao R=%.3f"
              % (chave, len(az_e), len(az_s), w_s.sum() / 1000.0,
                 b[i], b[i + 1], b[ie], b[ie + 1],
                 concentracao(az_s, w_s)))
        if a.reports:
            d = os.path.join(a.reports, chave)
            os.makedirs(d, exist_ok=True)
            pd.DataFrame({"azimute_graus": np.round(az_s, 3),
                          "comprimento_m": np.round(w_s, 2)}).to_csv(
                os.path.join(d, "azimutes_estruturais.csv"), index=False)

    estilo()
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_2COL, 3.9),
                              subplot_kw={"projection": "polar"})
    # limite radial comum: o eixo e percentual do comprimento da propria
    # area, entao as duas rosetas ficam diretamente comparaveis em forma
    rmax = 0
    for chave in AREAS:
        d = dados[chave]
        h, _ = np.histogram(d["az"], bins=N_BINS, range=(0, 180),
                            weights=d["w"])
        rmax = max(rmax, 100.0 * h.max() / d["w"].sum())
    rmax = np.ceil(rmax / 2.0) * 2.0 + 1.0

    for ax, chave in zip(eixos, AREAS):
        d, cfg = dados[chave], AREAS[chave]
        painel(ax, d["az"], d["w"], cfg["cor"], cfg["rotulo"], rmax)
        ax.text(0.5, -0.17,
                "%d clipped parts · %s km mapped\nmodal %.0f–%.0f° · "
                "concentration R = %.2f"
                % (d["n_partes"], format(int(round(d["km"])), ","),
                   d["modo"][0], d["modo"][1], d["r"]),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=TINTA2)

    fig.subplots_adjust(wspace=0.35)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    grava(fig, a.saida, "fig2_roses")


if __name__ == "__main__":
    main()
