"""s09 — Teto de previsibilidade do alvo (Fase B).

Pergunta: 0,70 de AUC é atingível NESTA base, por QUALQUER modelo?

Dois estimadores empíricos do teto:
  1. AUC-VIZINHO: prediz a classe de cada poço usando apenas o log Q/s
     do vizinho mais próximo (dentro de um raio). Se nem o vizinho
     imediato discrimina, nenhuma covariável espacial discriminará —
     é o teto prático de modelos espaciais naquela escala;
  2. VARIOGRAFIA do log Q/s: fração pepita/patamar. Pepita alta =
     ruído não-espacial (erro de medição do teste, construção, sorte
     de fratura) que NENHUM mapa captura.

Interpretação impressa ao final. Este estágio não altera nada — é
somente diagnóstico, com relatório em tela + txt.

Execução:  docker compose run --rm pipeline python stages/s09_teto_previsibilidade.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import DERIVED

OUT = Path("/data/outputs")
RELATORIO = OUT / "s09_relatorio.txt"


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def main():
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")
    g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m.lon, m.lat),
                         crs="EPSG:4674").to_crs(epsg=31983)
    xy = np.c_[g.geometry.x, g.geometry.y]
    z = m["log_qs"].values
    y = m["alvo_binario"].values
    n = len(m)

    print("===== s09 — TETO DE PREVISIBILIDADE DO ALVO =====")
    print(f"Poços: {n} | variância total de log Q/s: {z.var():.3f}\n")

    # ---------- 1. AUC-vizinho ----------
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=2)          # k=1 é o próprio poço
    viz_d, viz_i = dist[:, 1], idx[:, 1]

    print("1) AUC-VIZINHO (classe do poço predita pelo log Q/s do "
          "vizinho mais próximo):")
    print(f"{'raio':>8s} {'n pares':>8s} {'AUC-vizinho':>12s} "
          f"{'corr(z, z_viz)':>15s}")
    for raio in (250, 500, 1000, 2000):
        sel = viz_d <= raio
        if sel.sum() < 30 or len(set(y[sel])) < 2:
            print(f"{raio:7.0f}m {int(sel.sum()):8d}          — (n insuf.)")
            continue
        auc_v = roc_auc_score(y[sel], z[viz_i[sel]])
        corr = np.corrcoef(z[sel], z[viz_i[sel]])[0, 1]
        print(f"{raio:7.0f}m {int(sel.sum()):8d} {auc_v:12.3f} "
              f"{corr:15.3f}")

    # ---------- 2. Variografia empírica ----------
    print("\n2) SEMIVARIOGRAMA EMPÍRICO de log Q/s:")
    rng = np.random.default_rng(42)
    # amostra de pares para custo controlado
    max_pares = 400_000
    ii = rng.integers(0, n, size=max_pares)
    jj = rng.integers(0, n, size=max_pares)
    ok = ii != jj
    ii, jj = ii[ok], jj[ok]
    h = np.hypot(*(xy[ii] - xy[jj]).T)
    gama = 0.5 * (z[ii] - z[jj]) ** 2
    bins = [0, 250, 500, 1000, 2000, 4000, 8000, 16000]
    print(f"{'lag (m)':>14s} {'n pares':>9s} {'semivar':>9s} "
          f"{'% do patamar':>13s}")
    sill = z.var()
    gama0 = None
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (h > a) & (h <= b)
        if sel.sum() < 100:
            continue
        gm = gama[sel].mean()
        if gama0 is None:
            gama0 = gm
        print(f"{a:6.0f}–{b:6.0f} {int(sel.sum()):9d} {gm:9.3f} "
              f"{100 * gm / sill:12.0f}%")

    pepita_frac = min(1.0, gama0 / sill) if gama0 else float("nan")
    r2_max = 1 - pepita_frac
    print(f"\nFração pepita (lag mais curto / variância): "
          f"{pepita_frac:.0%}")
    print(f"R² máximo teórico de qualquer mapa: ~{r2_max:.2f}")

    print("""
===== INTERPRETAÇÃO =====
AUC-vizinho no raio de 250–500 m ≈ teto prático de modelos espaciais:
  >= 0,80  -> há sinal local forte; 0,70 é alcançável com covariáveis
              melhores (geofísica) — investir na Fase C;
  0,70–0,80-> 0,70 é a fronteira; alcançável, com margem estreita;
  < 0,70   -> NENHUM modelo espacial atinge 0,70 nesta base: o ruído
              do alvo (erro de teste, construção, aleatoriedade de
              fratura) domina. A barra deve ser recalibrada pelo teto
              medido (decisão declarável) ou o alvo redefinido
              (ex.: mediana por bloco em vez de poço a poço).
A fração pepita conta a mesma história em regressão: pepita de 70%
implica R² máximo ~0,30 — compare com o R² ~0,17 já obtido.""")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
