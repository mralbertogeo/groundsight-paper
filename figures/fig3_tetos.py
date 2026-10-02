#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura 3 do manuscrito — estimadores do teto de previsibilidade.

(a) AUC-vizinho em funcao do raio de busca, com IC95%.
(b) Semivariograma empirico de log10(Q/s) em percentual do patamar.

Os numeros NAO sao digitados: o script le os relatorios arquivados do s09
das duas areas, de modo que a figura e sempre consistente com o que o
pipeline produziu.

Uso:
    python figures/fig3_tetos.py [--reports reports] [--saida figures]

Saida: fig3_tetos.pdf (vetorial, para a revista) e fig3_tetos.png (300 dpi).
"""
import argparse
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Paleta e marcadores herdados do conjunto de figuras do artigo
# (azul #1F4E78 clareado para #205E8F, que passa no piso de luminancia e
# de croma do validador; laranja mantido). Separacao CVD: dE 18.6 protan.
AREAS = [("area1", "Area 1 — Paraná Basin border", "#205E8F", "s", "--"),
         ("area2", "Area 2 — Taubaté rift", "#C05000", "o", "-")]
CORTE = 2000.0          # lags abaixo disso usam todos os pares reais
TINTA2 = "#5c5c5c"
EIXO = "#b0b0b0"


def ler_relatorio(caminho):
    """Extrai do s09_relatorio.txt tudo o que a figura precisa."""
    with open(caminho, encoding="utf-8") as fh:
        txt = fh.read()

    d = {}
    m = re.search(r"Poços:\s*(\d+)\s*\|\s*variância total de log Q/s:\s*"
                  r"([\d.]+)", txt)
    if not m:
        sys.exit("Nao achei a linha de poços/variância em %s" % caminho)
    d["n"] = int(m.group(1))
    d["patamar"] = float(m.group(2))

    # 1) AUC-vizinho:  "    250m      616        0.672    0.338   0.622-0.721"
    auc = re.findall(r"^\s*(\d+)m\s+(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s+"
                     r"([\d.]+)-([\d.]+)\s*$", txt, re.M)
    if not auc:
        sys.exit("Nao achei a tabela de AUC-vizinho em %s" % caminho)
    d["raio"] = np.array([float(r[0]) for r in auc])
    d["auc"] = np.array([float(r[2]) for r in auc])
    d["auc_lo"] = np.array([float(r[4]) for r in auc])
    d["auc_hi"] = np.array([float(r[5]) for r in auc])

    # 2) semivariograma: "     0–   250       595     0.334    66%     todos"
    var = re.findall(r"^\s*(\d+)[–-]\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)%",
                     txt, re.M)
    if not var:
        sys.exit("Nao achei a tabela do semivariograma em %s" % caminho)
    d["lag_min"] = np.array([float(v[0]) for v in var])
    d["lag_max"] = np.array([float(v[1]) for v in var])
    d["semivar"] = np.array([float(v[3]) for v in var])
    d["lag"] = 0.5 * (d["lag_min"] + d["lag_max"])
    d["exaustivo"] = d["lag_max"] <= CORTE

    m = re.search(r"Fração pepita \(lag [\d–\- ]+m / variância\):\s*(\d+)%",
                  txt)
    d["pepita"] = float(m.group(1)) if m else np.nan
    m = re.search(r"IC95% da fração pepita:\s*(\d+)%-(\d+)%", txt)
    d["pepita_ic"] = ((float(m.group(1)), float(m.group(2)))
                      if m else (np.nan, np.nan))
    return d


def estilo():
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
        "text.color": "#1a1a1a",
        "axes.labelcolor": "#1a1a1a",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def limpa(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports", default="reports")
    p.add_argument("--saida", default=".")
    a = p.parse_args()

    dados = {}
    for chave, _, _, _, _ in AREAS:
        caminho = os.path.join(a.reports, chave, "s09_relatorio.txt")
        if not os.path.exists(caminho):
            sys.exit("Nao encontrei %s" % caminho)
        dados[chave] = ler_relatorio(caminho)
        d = dados[chave]
        print("%s: n=%d patamar=%.3f pepita=%.0f%% (%.0f-%.0f%%)"
              % (chave, d["n"], d["patamar"], d["pepita"],
                 d["pepita_ic"][0], d["pepita_ic"][1]))

    estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 3.25))

    # ---------------- (a) AUC-vizinho ----------------
    for chave, rotulo, cor, marca, traco in AREAS:
        d = dados[chave]
        ax1.fill_between(d["raio"], d["auc_lo"], d["auc_hi"], color=cor,
                         alpha=0.15, linewidth=0, zorder=2)
        ax1.plot(d["raio"], d["auc"], traco, color=cor, linewidth=1.8,
                 marker=marca, markersize=5.5, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=3)
        ax1.annotate(rotulo.split(" —")[0],
                     xy=(d["raio"][-1], d["auc"][-1]),
                     xytext=(6, 7 if chave == "area2" else -14),
                     textcoords="offset points", color=cor,
                     fontsize=8.5, fontweight="bold", ha="right")

    ax1.axhline(0.5, color=EIXO, linewidth=0.9, linestyle=(0, (1, 2)),
                zorder=1)
    ax1.text(2150, 0.503, "chance", fontsize=7.5, color=TINTA2,
             va="bottom", ha="right")
    raios = dados["area1"]["raio"]
    ax1.set_xscale("log")
    ax1.set_xticks(raios)
    ax1.set_xticklabels(["{:,.0f}".format(r) for r in raios])
    ax1.set_xlim(raios.min() * 0.84, raios.max() * 1.20)
    ax1.set_ylim(0.48, 0.87)
    ax1.set_xlabel("Search radius (m)")
    ax1.set_ylabel("Neighbour AUC")
    ax1.set_title("(a) Nearest-neighbour discrimination", loc="left", pad=8)
    limpa(ax1)

    # ---------------- (b) semivariograma ----------------
    for chave, _, cor, marca, traco in AREAS:
        d = dados[chave]
        pct = 100.0 * d["semivar"] / d["patamar"]
        ex = d["exaustivo"]
        ax2.plot(d["lag"], pct, traco, color=cor, linewidth=1.8, zorder=3)
        ax2.plot(d["lag"][ex], pct[ex], linestyle="none", marker=marca,
                 markersize=5.5, color=cor, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=4)
        ax2.plot(d["lag"][~ex], pct[~ex], linestyle="none", marker=marca,
                 markersize=5.5, markerfacecolor="white",
                 markeredgecolor=cor, markeredgewidth=1.3, zorder=4)
        lo, hi = d["pepita_ic"]
        x0 = d["lag"][0]
        ax2.plot([x0, x0], [lo, hi], color=cor, linewidth=1.3, zorder=3)
        for y in (lo, hi):
            ax2.plot([x0 * 0.90, x0 * 1.11], [y, y], color=cor,
                     linewidth=1.3, zorder=3)
        ax2.annotate("nugget %.0f%%" % d["pepita"], xy=(x0, pct[0]),
                     xytext=(14, -2), textcoords="offset points",
                     color=cor, fontsize=8)

    ax2.axhline(100, color=EIXO, linewidth=0.9, linestyle=(0, (1, 2)),
                zorder=1)
    lags = dados["area1"]["lag"]
    ax2.text(lags.max() * 1.12, 101.5, "sill", fontsize=7.5, color=TINTA2,
             va="bottom", ha="right")
    ax2.set_xscale("log")
    ax2.set_xticks(lags)
    ax2.set_xticklabels(["{:,.0f}".format(x) for x in lags])
    ax2.set_xlim(lags.min() * 0.76, lags.max() * 1.29)
    ax2.set_ylim(25, 112)
    ax2.set_xlabel("Lag distance (m, bin midpoint)")
    ax2.set_ylabel("Semivariance (% of sill)")
    ax2.set_title("(b) Empirical semivariogram of log$_{10}$(Q/s)",
                  loc="left", pad=8)
    limpa(ax2)

    manipulos = []
    for _, rotulo, cor, marca, traco in AREAS:
        manipulos.append(plt.Line2D([], [], color=cor, linestyle=traco,
                                    linewidth=1.8, marker=marca,
                                    markersize=5.5, markeredgecolor="white",
                                    label=rotulo))
    manipulos.append(plt.Line2D([], [], color=TINTA2, linestyle="none",
                                marker="o", markersize=5.5,
                                markerfacecolor=TINTA2,
                                markeredgecolor="white",
                                label="(b) all real pairs (lag < 2 km)"))
    manipulos.append(plt.Line2D([], [], color=TINTA2, linestyle="none",
                                marker="o", markersize=5.5,
                                markerfacecolor="white",
                                markeredgecolor=TINTA2, markeredgewidth=1.3,
                                label="(b) sampled pairs (lag > 2 km)"))
    fig.legend(handles=manipulos, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.015), handletextpad=0.6,
               columnspacing=1.8)

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    os.makedirs(a.saida, exist_ok=True)
    for ext in ("pdf", "png"):
        destino = os.path.join(a.saida, "fig3_tetos.%s" % ext)
        fig.savefig(destino, bbox_inches="tight")
        print("gravado:", destino)


if __name__ == "__main__":
    main()
