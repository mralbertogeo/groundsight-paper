#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura 4 do manuscrito — trajetoria de validacao da Area 1 sob
validacao cruzada espacial em blocos.

Mostra por que o criterio a priori de 0,70 nunca foi atingivel: o teto
MEDIDO fica abaixo dele, e a barra de aceitacao derivada (0,95 x teto)
e o que o modelo de fato tinha de superar.

Procedencia dos numeros, em tres niveis:

  teto e IC95%   lidos de reports/area1/s09_relatorio.txt (arquivado)
  barra          calculada aqui como fator x teto, nunca digitada
  ponto v3       recalculado de reports/area1/validacao_resultados.csv
                 pela regra do s07_validacao.py, e CONFERIDO contra o
                 CSV da trajetoria — divergencia interrompe a execucao
  v1 a v2.3      transcritos (historicos); o cabecalho do CSV declara

Uso:
    python fig4_trajectory.py [--reports reports] [--saida .]

Saida: fig4_trajectory.pdf e .png
"""
import argparse
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from figstyle import (AREA1, EIXO, LARGURA_15COL, TINTA2, estilo, grava,
                      limpa)

FATOR_BARRA = 0.95        # config.yaml: fator_teto
CRITERIO_APRIORI = 0.70   # criterio declarado antes de medir o teto
TOL = 0.0015              # tolerancia da conferencia do ponto v3


def ler_teto(caminho):
    """Teto-vizinho e IC95% do menor raio, do relatorio do s09."""
    with open(caminho, encoding="utf-8") as fh:
        txt = fh.read()
    partes = txt.split("1) AUC-VIZINHO", 1)
    if len(partes) != 2:
        sys.exit("Nao achei a secao 1) AUC-VIZINHO em %s" % caminho)
    m = re.search(r"^\s*(\d+)m\s+\d+\s+([\d.]+)\s+-?[\d.]+\s+([\d.]+)-([\d.]+)",
                  partes[1], re.M)
    if not m:
        sys.exit("Nao achei a primeira linha da tabela em %s" % caminho)
    return (int(m.group(1)), float(m.group(2)),
            float(m.group(3)), float(m.group(4)))


def desvio_seeds(caminho):
    """Desvio entre sementes pela regra do s07: o MAIOR desvio-padrao
    amostral entre RF, XGB e ensemble."""
    with open(caminho, encoding="utf-8-sig", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    cols = [c for c in ("auc_rf", "auc_xgb", "auc_ens") if c in linhas[0]]
    por_modelo = {c: np.array([float(r[c]) for r in linhas]) for c in cols}
    media_rf = por_modelo["auc_rf"].mean()
    return media_rf, max(v.std(ddof=1) for v in por_modelo.values())


def ler_trajetoria(caminho):
    with open(caminho, encoding="utf-8-sig", newline="") as fh:
        linhas = [l for l in fh if not l.lstrip().startswith("#")]
    out = []
    for r in csv.DictReader(linhas):
        out.append({
            "estagio": r["estagio"],
            "n": int(r["n_matriz"]),
            "auc": float(r["auc"]) if r["auc"] else None,
            "lo": float(r["auc_lo"]) if r["auc_lo"] else None,
            "hi": float(r["auc_hi"]) if r["auc_hi"] else None,
            "desvio": float(r["auc_desvio"]) if r["auc_desvio"] else None,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports", default="reports")
    p.add_argument("--saida", default=".")
    a = p.parse_args()

    base = os.path.join(a.reports, "area1")
    raio, teto, teto_lo, teto_hi = ler_teto(
        os.path.join(base, "s09_relatorio.txt"))
    media_v3, desvio_v3 = desvio_seeds(
        os.path.join(base, "validacao_resultados.csv"))
    traj = ler_trajetoria(os.path.join(base, "trajetoria_validacao.csv"))
    barra = FATOR_BARRA * teto

    # O ponto final nao e de confianca: e conferido.
    fim = traj[-1]
    if abs(media_v3 - fim["auc"]) > TOL:
        sys.exit("v3 do CSV (%.3f) diverge do recalculo de "
                 "validacao_resultados.csv (%.4f) — reconcilie antes de "
                 "publicar a figura." % (fim["auc"], media_v3))
    fim["desvio"] = desvio_v3
    print("teto %.3f (%.3f-%.3f) a %d m | barra %.3f | v3 %.4f +- %.4f"
          % (teto, teto_lo, teto_hi, raio, barra, media_v3, desvio_v3))

    estilo()
    fig, ax = plt.subplots(figsize=(LARGURA_15COL, 3.4))
    x = np.arange(len(traj))

    # faixa do teto: o teto tambem e estimativa
    ax.fill_between([-0.6, len(traj) - 0.4], teto_lo, teto_hi,
                    color=AREA1, alpha=0.10, linewidth=0, zorder=1)
    ax.axhline(teto, color=AREA1, linewidth=1.3, zorder=2)
    ax.axhline(barra, color=AREA1, linewidth=1.1, linestyle=(0, (4, 2)),
               zorder=2)
    ax.axhline(CRITERIO_APRIORI, color="#8a8a8a", linewidth=1.1,
               linestyle=(0, (1, 2)), zorder=2)

    # trajetoria
    pontos = [(i, d) for i, d in enumerate(traj) if d["auc"] is not None]
    xi = np.array([i for i, _ in pontos])
    yi = np.array([d["auc"] for _, d in pontos])
    er = np.array([d["desvio"] or 0.0 for _, d in pontos])
    ax.errorbar(xi, yi, yerr=er, fmt="-o", color=AREA1, linewidth=1.8,
                markersize=5.5, markeredgecolor="white",
                markeredgewidth=0.8, elinewidth=1.1, capsize=3,
                ecolor=AREA1, zorder=4)

    # v1/v1.1 entrou como faixa, nao como ponto
    faixa = [(i, d) for i, d in enumerate(traj) if d["auc"] is None]
    for i, d in faixa:
        ax.plot([i, i], [d["lo"], d["hi"]], color=AREA1, linewidth=1.8,
                zorder=4)
        for y in (d["lo"], d["hi"]):
            ax.plot([i - 0.07, i + 0.07], [y, y], color=AREA1,
                    linewidth=1.8, zorder=4)
        ax.plot([i, xi[0]], [0.5 * (d["lo"] + d["hi"]), yi[0]],
                color=AREA1, linewidth=1.8, zorder=3)

    # Rotulos das linhas de referencia a ESQUERDA: a trajetoria sobe para
    # a direita e qualquer rotulo ali colide com os pontos.
    ex = -0.5
    ax.text(ex, CRITERIO_APRIORI + 0.012, "a priori criterion 0.70",
            fontsize=7.5, color=TINTA2, ha="left", va="bottom")
    ax.text(ex, teto + 0.012, r"measured ceiling AUC$_{\mathrm{nn}}$ = %.3f" % teto,
            fontsize=7.5, color=AREA1, ha="left", va="bottom")
    # a barra vai a DIREITA: a subida v1->v2 atravessa o canto inferior
    # esquerdo e cortaria o texto
    # folga generosa: com 0,007 as letras encostavam na linha tracejada
    # assim que o PDF era rasterizado em resolucao de tela, e o ponto da
    # v2.2 (0,635) passava rente ao texto
    ax.text(len(traj) - 0.5, barra - 0.019,
            r"acceptance bar %.2f × AUC$_{\mathrm{nn}}$ = %.3f" % (FATOR_BARRA, barra),
            fontsize=7.5, color=AREA1, ha="right", va="top")
    # a faixa e rotulada do lado oposto, onde nao ha dado nem linha
    ax.text(len(traj) - 0.5, teto_hi - 0.010, r"95% CI of AUC$_{\mathrm{nn}}$",
            fontsize=7, color=TINTA2, ha="right", va="top")

    # O tamanho da matriz entra no proprio rotulo do eixo, em vez de uma
    # anotacao solta: informa os seis estagios, nao so o da expansao.
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (d["estagio"], format(d["n"], ","))
                        for d in traj])
    ax.set_xlim(-0.6, len(traj) - 0.4)
    ax.set_ylim(0.46, 0.775)
    ax.set_xlabel("Development stage (training-matrix n below)")
    ax.set_ylabel("Spatial AUC (block cross-validation)")
    limpa(ax, grade="y")
    grava(fig, a.saida, "fig4_trajectory")


if __name__ == "__main__":
    main()
