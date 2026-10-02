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

Convenções da variografia (manuscrito, seção 3.6-ii):
  - patamar = variância total de log Q/s (z.var(), ddof=0);
  - abaixo de 2 km TODOS os pares reais são enumerados por k-d tree,
    sem sorteio; acima do corte usa-se amostra de pares, pois esses
    lags não entram na pepita;
  - um lag só é aceito com ao menos 100 pares, e o lag efetivamente
    usado é impresso — se o mais curto for descartado, o relatório
    avisa que a pepita está subestimada;
  - IC95% por reamostragem (bootstrap). A amostra é ordenada antes do
    sorteio, de modo que o IC não depende da ordem de leitura dos
    registros. O patamar é tratado como fixo, logo o IC é um LIMITE
    INFERIOR da incerteza total da fração.

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

CORTE = 2000        # abaixo disso: todos os pares reais
MIN_PARES = 100     # lag com menos pares que isso é descartado
REPS = 2000         # reamostragens do bootstrap
SEMENTE = 20260101


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def ic_media(amostra, reps=REPS, semente=SEMENTE):
    """IC95% da média por reamostragem com reposição, em ordem canônica."""
    amostra = np.sort(np.asarray(amostra, dtype=float))
    if len(amostra) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(semente)
    idx = r.integers(0, len(amostra), size=(reps, len(amostra)))
    return tuple(np.percentile(amostra[idx].mean(axis=1), [2.5, 97.5]))


def ic_auc(yv, zv, reps=REPS, semente=SEMENTE):
    """IC95% da AUC por reamostragem dos pares, em ordem canônica."""
    from sklearn.metrics import roc_auc_score
    yv = np.asarray(yv)
    zv = np.asarray(zv)
    ordem = np.lexsort((yv, zv))
    yv, zv = yv[ordem], zv[ordem]
    r = np.random.default_rng(semente)
    vals = []
    for _ in range(reps):
        k = r.integers(0, len(yv), size=len(yv))
        if len(set(yv[k])) < 2:
            continue
        vals.append(roc_auc_score(yv[k], zv[k]))
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


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
          f"{'corr(z, z_viz)':>15s} {'IC95% da AUC':>18s}")
    for raio in (250, 500, 1000, 2000):
        sel = viz_d <= raio
        if sel.sum() < 30 or len(set(y[sel])) < 2:
            print(f"{raio:7.0f}m {int(sel.sum()):8d}          — (n insuf.)")
            continue
        auc_v = roc_auc_score(y[sel], z[viz_i[sel]])
        corr = np.corrcoef(z[sel], z[viz_i[sel]])[0, 1]
        lo, hi = ic_auc(y[sel], z[viz_i[sel]])
        ic = f"{lo:.3f}-{hi:.3f}"
        print(f"{raio:7.0f}m {int(sel.sum()):8d} {auc_v:12.3f} "
              f"{corr:15.3f} {ic:>18s}")

    # ---------- 2. Variografia empírica ----------
    print("\n2) SEMIVARIOGRAMA EMPÍRICO de log Q/s:")
    bins = [0, 250, 500, 1000, 2000, 4000, 8000, 16000]

    # abaixo do corte: enumeração exaustiva dos pares reais
    curtos = np.array(sorted(tree.query_pairs(r=CORTE)),
                      dtype=int).reshape(-1, 2)
    # acima do corte: amostra de pares (não entram na pepita)
    rng = np.random.default_rng(42)
    ii = rng.integers(0, n, size=400_000)
    jj = rng.integers(0, n, size=400_000)
    ok = ii != jj
    longos = np.column_stack((ii[ok], jj[ok]))

    def semivar(pares, a, b):
        if len(pares) == 0:
            return np.empty(0)
        d = np.hypot(*(xy[pares[:, 0]] - xy[pares[:, 1]]).T)
        s = (d > a) & (d <= b)
        return 0.5 * (z[pares[s, 0]] - z[pares[s, 1]]) ** 2

    sill = z.var()
    gama0, amostra0, lag0 = None, None, None
    descartados = []
    print(f"{'lag (m)':>14s} {'n pares':>9s} {'semivar':>9s} "
          f"{'% do patamar':>13s} {'origem':>9s}")
    for a, b in zip(bins[:-1], bins[1:]):
        todos = b <= CORTE
        origem = "todos" if todos else "amostra"
        gvals = semivar(curtos if todos else longos, a, b)
        if len(gvals) < MIN_PARES:
            print(f"{a:6.0f}–{b:6.0f} {len(gvals):9d} {'—':>9s} "
                  f"{'—':>13s} {origem:>9s}  (descartado)")
            if gama0 is None:
                descartados.append((a, b, len(gvals)))
            continue
        gm = gvals.mean()
        if gama0 is None:
            gama0, amostra0, lag0 = gm, gvals, (a, b)
        print(f"{a:6.0f}–{b:6.0f} {len(gvals):9d} {gm:9.3f} "
              f"{100 * gm / sill:12.0f}% {origem:>9s}")

    if gama0 is None:
        print(f"\nNenhum lag atingiu {MIN_PARES} pares: fração pepita não "
              "estimável nesta base.")
        pepita_frac = float("nan")
        r2_max = float("nan")
    else:
        pepita_frac = min(1.0, gama0 / sill)
        r2_max = 1 - pepita_frac
        print(f"\nFração pepita (lag {lag0[0]:.0f}–{lag0[1]:.0f} m / "
              f"variância): {pepita_frac:.0%}")
        lo, hi = np.clip(np.array(ic_media(amostra0)) / sill, 0, 1)
        print(f"IC95% da fração pepita: {lo:.0%}-{hi:.0%} "
              f"({len(amostra0)} pares no lag usado; patamar fixo, "
              "logo é limite inferior da incerteza)")
        if descartados:
            print(f"ATENÇÃO: lag(s) mais curto(s) descartado(s) por terem "
                  f"menos de {MIN_PARES} pares:")
            for a, b, k in descartados:
                print(f"  {a:.0f}–{b:.0f} m: {k} pares")
            print("  A pepita acima foi medida num lag maior, onde a "
                  "semivariância já cresceu,")
            print("  e portanto SUBESTIMA a pepita real desta base.")
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
A fração pepita conta a mesma história em regressão: o R² máximo é
~(1 - pepita). Compare o R² obtido no s07 com esse valor, e compare a
fração capturada com o IC da pepita, não só com o ponto estimado.
O alcance também informa: se o semivariograma satura antes do
espaçamento típico entre poços, interpolar entre poços opera além do
comprimento de correlação.""")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
