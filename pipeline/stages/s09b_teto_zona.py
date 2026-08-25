"""s09b — Teto de previsibilidade do ALVO DE ZONA (Rota 2).

Re-mede o teto do s09 com o alvo agregado: mediana de log Q/s por
bloco espacial (tamanhos testados: 1, 2 e 4 km). A agregação cancela
o ruído poço-a-poço (pepita de 78% medida no s09); a pergunta é
quanto o teto sobe — e, portanto, se a barra de 0,70 volta a ser
alcançável contra um alvo de zona.

Estimadores por tamanho de bloco:
  1. AUC-BLOCO-VIZINHO: classe do bloco (mediana >= q75 das medianas)
     predita pela mediana do bloco vizinho mais próximo;
  2. Fração pepita do semivariograma das medianas de bloco;
  3. n de blocos válidos (n_min poços) — o "novo n" da modelagem.

Não altera nada — diagnóstico puro, relatório em tela + txt.

Execução:  docker compose run --rm pipeline python stages/s09b_teto_zona.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import DERIVED

OUT = Path("/data/outputs")
RELATORIO = OUT / "s09b_relatorio.txt"


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def analisar_bloco(m, xy, tam_m, n_min):
    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score

    bx = (xy[:, 0] // tam_m).astype(int)
    by = (xy[:, 1] // tam_m).astype(int)
    df = pd.DataFrame({"bx": bx, "by": by, "z": m["log_qs"].values})
    bl = (df.groupby(["bx", "by"])
          .agg(n=("z", "size"), med=("z", "median"))
          .reset_index())
    bl = bl[bl["n"] >= n_min].reset_index(drop=True)
    if len(bl) < 30:
        return {"tam_km": tam_m / 1000, "n_blocos": len(bl),
                "auc_viz": np.nan, "pepita": np.nan, "corr": np.nan}

    q75 = bl["med"].quantile(0.75)
    y = (bl["med"] >= q75).astype(int).values
    cx = (bl["bx"].values + 0.5) * tam_m
    cy = (bl["by"].values + 0.5) * tam_m
    tree = cKDTree(np.c_[cx, cy])
    _, idx = tree.query(np.c_[cx, cy], k=2)
    viz = idx[:, 1]
    auc = roc_auc_score(y, bl["med"].values[viz]) \
        if len(set(y)) == 2 else np.nan
    corr = np.corrcoef(bl["med"].values, bl["med"].values[viz])[0, 1]

    # pepita: semivar no lag mais curto (blocos adjacentes) / variância
    dist_m = np.hypot(cx[:, None] - cx[None, :],
                      cy[:, None] - cy[None, :])
    adj = (dist_m > 0) & (dist_m <= tam_m * 1.5)
    ii, jj = np.where(adj)
    gama_adj = 0.5 * np.mean((bl["med"].values[ii]
                              - bl["med"].values[jj]) ** 2)
    pepita = min(1.0, gama_adj / bl["med"].var())

    return {"tam_km": tam_m / 1000, "n_blocos": len(bl),
            "auc_viz": auc, "pepita": pepita, "corr": corr}


def main():
    import geopandas as gpd

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")
    g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m.lon, m.lat),
                         crs="EPSG:4674").to_crs(epsg=31983)
    xy = np.c_[g.geometry.x, g.geometry.y]

    print("===== s09b — TETO DO ALVO DE ZONA =====")
    print(f"Poços de entrada: {len(m)} | referência do s09 "
          f"(poço a poço): AUC-vizinho 0,672 | pepita 78%\n")
    print(f"{'bloco':>7s} {'n_min':>6s} {'n blocos':>9s} "
          f"{'AUC-bloco-viz':>14s} {'pepita':>8s} {'corr viz':>9s}")
    linhas = []
    for tam_km, n_min in [(1, 3), (2, 4), (2, 6), (4, 6), (4, 10)]:
        r = analisar_bloco(m, xy, tam_km * 1000, n_min)
        linhas.append(r)
        print(f"{r['tam_km']:5.0f}km {n_min:6d} {r['n_blocos']:9d} "
              f"{r['auc_viz']:14.3f} {r['pepita']:8.0%} "
              f"{r['corr']:9.2f}")

    print("""
===== INTERPRETAÇÃO =====
- AUC-bloco-vizinho ≈ teto prático da favorabilidade DE ZONA naquele
  tamanho de bloco. Compare com 0,70:
    >= 0,80 -> barra de 0,70 plenamente alcançável contra o alvo de
               zona; retreinar (s06/s07 com alvo agregado) e a
               geofísica entra para fechar o gap;
    0,70–0,80 -> alcançável com margem estreita;
    < 0,70 -> mesmo agregado o alvo não sustenta 0,70 — recalibrar
              a barra pelo teto medido (declarado).
- Pepita menor que os 78% do s09 confirma que a agregação está
  cancelando o ruído poço-a-poço.
- n_blocos é o novo n da modelagem: abaixo de ~150, a CV espacial
  perde poder — preferir o menor bloco que já eleve o teto.""")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
