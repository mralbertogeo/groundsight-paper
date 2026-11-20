"""s09 — Teto de previsibilidade do alvo (Fase B; R2 em 25/10/2026).

Pergunta: 0,70 de AUC é atingível NESTA base, por QUALQUER modelo?

Dois estimadores empíricos do teto:
  1. AUC-VIZINHO: prediz a classe de cada poço usando apenas o log Q/s
     do vizinho mais próximo (dentro de um raio). É uma REFERÊNCIA de
     vizinhança, não um limite superior demonstrado — ver (1b);
  2. VARIOGRAFIA do log Q/s: fração pepita/patamar. Pepita alta =
     ruído não-espacial (erro de medição do teste, construção, sorte
     de fratura) que NENHUM mapa captura.

R2 (25/10/2026) — quatro acréscimos, em resposta à revisão externa:
  1b. k-NN (k = 1, 3, 5, 10) com peso 1/d dentro do mesmo raio. Se
      k > 1 supera k = 1, o vizinho único NÃO é teto: é uma observação
      ruidosa. O número a reportar como referência é o melhor k.
  1c. AUC-vizinho por FAIXA de distância NÃO CUMULATIVA, com n e
      prevalência por faixa. A curva por raio cumulativo mistura
      distância com composição da amostra; a por faixa, não.
  2b. Semivariograma após remoção de tendência linear em (x, y), para
      bases que não saturam no maior lag (caso da Área 2).
  2c. Semivariograma direcional em quatro classes de azimute.
  (*) IC95% da pepita por reamostragem DE POÇOS, não de pares: pares
      que compartilham um poço não são independentes, e o patamar é
      reamostrado junto em vez de tratado como fixo. O IC antigo
      (por pares, patamar fixo) continua impresso para comparação.
  (*) Lags longos passam a ser enumerados exaustivamente; não há mais
      sorteio de pares em nenhum lag.

Saídas legíveis por máquina, ao lado do relatório:
  s09_auc_vizinho.csv, s09_auc_faixas.csv, s09_variograma.csv,
  s09_pepita.csv

Este estágio não altera nada — é somente diagnóstico.

Execução:  docker compose run --rm pipeline python stages/s09_teto_previsibilidade.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import DERIVED

OUT = Path("/data/outputs")
RELATORIO = OUT / "s09_relatorio.txt"

CORTE = 2000        # histórico: abaixo disso sempre foram todos os pares
MIN_PARES = 100     # lag com menos pares que isso é descartado
REPS = 2000         # reamostragens do bootstrap
SEMENTE = 20260101
KS = (1, 3, 5, 10)  # R2: vizinhos no k-NN
FAIXAS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, None)]
DIRECOES = [("N-S", 0, 22.5), ("NE-SW", 22.5, 67.5),
            ("E-W", 67.5, 112.5), ("NW-SE", 112.5, 157.5),
            ("N-S", 157.5, 180.0)]


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def ic_media(amostra, reps=REPS, semente=SEMENTE):
    """IC95% da média por reamostragem com reposição, em ordem canônica.
    (Mantido para comparação: reamostra PARES, patamar fixo.)"""
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


def ic_auc_por_poco(y, score, sel, viz_i, n, reps=REPS, semente=SEMENTE):
    """R2: IC95% da AUC reamostrando POÇOS, não linhas de avaliação.

    Cada poço avaliado depende do seu vizinho; um mesmo poço é vizinho
    de vários. Aqui sorteia-se o conjunto de poços: uma linha só entra
    se o poço avaliado E o seu vizinho foram sorteados, com peso igual
    à multiplicidade do poço avaliado.
    """
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(semente)
    alvo = np.where(sel)[0]
    vals = []
    for _ in range(reps):
        c = r.multinomial(n, np.full(n, 1.0 / n))
        keep = (c[alvo] > 0) & (c[viz_i[alvo]] > 0)
        if keep.sum() < 10:
            continue
        idx = alvo[keep]
        yv, zv, w = y[idx], score[idx], c[idx].astype(float)
        if len(set(yv)) < 2:
            continue
        vals.append(roc_auc_score(yv, zv, sample_weight=w))
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def ic_pepita_por_poco(pares, gvals, z, n, reps=REPS, semente=SEMENTE):
    """R2: IC95% da FRAÇÃO pepita reamostrando POÇOS.

    O par (i, j) recebe peso c_i * c_j, de modo que um poço não sorteado
    retira TODOS os seus pares — é assim que a dependência entre pares
    que compartilham poço entra no intervalo. O patamar é recalculado em
    cada reamostra (ponderado por c), em vez de tratado como fixo, logo
    o IC cobre a incerteza do numerador e do denominador.
    Pares são reponderados, nunca criados: duplicatas do mesmo poço não
    geram pares de distância zero.
    """
    r = np.random.default_rng(semente)
    i, j = pares[:, 0], pares[:, 1]
    vals = []
    for _ in range(reps):
        c = r.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        w = c[i] * c[j]
        sw = w.sum()
        if sw <= 0:
            continue
        gama = float((w * gvals).sum() / sw)
        zbar = float((c * z).sum() / c.sum())
        sill = float((c * (z - zbar) ** 2).sum() / c.sum())
        if sill <= 0:
            continue
        vals.append(gama / sill)
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

    linhas_auc = []
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
        plo, phi = ic_auc_por_poco(y, z[viz_i], sel, viz_i, n)
        ic = f"{lo:.3f}-{hi:.3f}"
        print(f"{raio:7.0f}m {int(sel.sum()):8d} {auc_v:12.3f} "
              f"{corr:15.3f} {ic:>18s}")
        linhas_auc.append({"raio_m": raio, "k": 1, "n": int(sel.sum()),
                           "prevalencia": float(y[sel].mean()),
                           "auc": auc_v, "ic_lo_pares": lo,
                           "ic_hi_pares": hi, "ic_lo_pocos": plo,
                           "ic_hi_pocos": phi})
    print("  (IC acima = reamostragem de pares, formato histórico; o IC "
          "por poços está no CSV)")

    # ---------- 1b. k-NN: o vizinho único é teto? ----------
    print("\n1b) k-NN COM PESO 1/d, no mesmo subconjunto de cada raio "
          "(R2):")
    print(f"{'raio':>8s} {'n':>7s} " +
          " ".join(f"{'k=' + str(k):>9s}" for k in KS))
    kmax = max(KS)
    dk, ik = tree.query(xy, k=kmax + 1)
    dk, ik = dk[:, 1:], ik[:, 1:]
    for raio in (250, 500, 1000, 2000):
        sel = viz_d <= raio
        if sel.sum() < 30 or len(set(y[sel])) < 2:
            continue
        celula = []
        for k in KS:
            w = np.where(dk[:, :k] <= raio, 1.0 / np.maximum(dk[:, :k], 1.0),
                         0.0)
            sw = w.sum(axis=1)
            score = np.where(sw > 0,
                             (z[ik[:, :k]] * w).sum(axis=1) /
                             np.where(sw > 0, sw, 1.0), np.nan)
            ok = sel & (sw > 0)
            a = roc_auc_score(y[ok], score[ok])
            celula.append(a)
            linhas_auc.append({"raio_m": raio, "k": k, "n": int(ok.sum()),
                               "prevalencia": float(y[ok].mean()),
                               "auc": a, "ic_lo_pares": np.nan,
                               "ic_hi_pares": np.nan,
                               "ic_lo_pocos": np.nan,
                               "ic_hi_pocos": np.nan})
        print(f"{raio:7.0f}m {int(sel.sum()):7d} " +
              " ".join(f"{a:9.3f}" for a in celula))
    print("  Se k>1 supera k=1, o vizinho único é uma observação ruidosa,")
    print("  não um limite superior: reportar como REFERÊNCIA (benchmark).")

    # ---------- 1c. faixas não cumulativas ----------
    print("\n1c) AUC-VIZINHO POR FAIXA DE DISTÂNCIA (não cumulativa, R2):")
    print(f"{'faixa (m)':>14s} {'n':>7s} {'prev':>7s} {'AUC':>8s} "
          f"{'IC95% (poços)':>18s}")
    linhas_faixa = []
    for lo_f, hi_f in FAIXAS:
        hi_e = np.inf if hi_f is None else hi_f
        sel = (viz_d > lo_f) & (viz_d <= hi_e)
        rot = f"{lo_f:.0f}–{'inf' if hi_f is None else f'{hi_f:.0f}'}"
        if sel.sum() < 30 or len(set(y[sel])) < 2:
            print(f"{rot:>14s} {int(sel.sum()):7d}   (n insuficiente)")
            continue
        a = roc_auc_score(y[sel], z[viz_i[sel]])
        plo, phi = ic_auc_por_poco(y, z[viz_i], sel, viz_i, n)
        print(f"{rot:>14s} {int(sel.sum()):7d} {y[sel].mean():7.2f} "
              f"{a:8.3f} {f'{plo:.3f}-{phi:.3f}':>18s}")
        linhas_faixa.append({"faixa_lo_m": lo_f,
                             "faixa_hi_m": (np.nan if hi_f is None
                                            else hi_f),
                             "n": int(sel.sum()),
                             "prevalencia": float(y[sel].mean()),
                             "auc": a, "ic_lo": plo, "ic_hi": phi})
    print("  Prevalência estável entre faixas => a queda (ou a constância)")
    print("  da AUC reflete distância, não composição da amostra.")

    # ---------- 2. Variografia empírica ----------
    print("\n2) SEMIVARIOGRAMA EMPÍRICO de log Q/s:")
    bins = [0, 250, 500, 1000, 2000, 4000, 8000, 16000]

    # R2: enumeração EXAUSTIVA de todos os pares, em qualquer lag.
    iu, ju = np.triu_indices(n, k=1)
    dpar = np.hypot(xy[iu, 0] - xy[ju, 0], xy[iu, 1] - xy[ju, 1])
    # azimute do par em graus [0, 180)
    azi = np.degrees(np.arctan2(xy[iu, 0] - xy[ju, 0],
                                xy[iu, 1] - xy[ju, 1])) % 180.0

    def tabela_variograma(zv, rotulo, mascara=None, csv_rows=None,
                          variante="bruto"):
        """Imprime a tabela de um semivariograma e devolve (gama0,
        pares0, g0vals, lag0, descartados)."""
        sill_l = zv.var()
        gl = 0.5 * (zv[iu] - zv[ju]) ** 2
        gama0 = g0 = lag0 = None
        pares0 = None
        descartados = []
        print(f"{'lag (m)':>14s} {'n pares':>9s} {'semivar':>9s} "
              f"{'% do patamar':>13s} {'origem':>9s}")
        for a, b in zip(bins[:-1], bins[1:]):
            s = (dpar > a) & (dpar <= b)
            if mascara is not None:
                s &= mascara
            k = int(s.sum())
            origem = "todos"
            if k < MIN_PARES:
                print(f"{a:6.0f}–{b:6.0f} {k:9d} {'—':>9s} "
                      f"{'—':>13s} {origem:>9s}  (descartado)")
                if gama0 is None:
                    descartados.append((a, b, k))
                continue
            gm = float(gl[s].mean())
            if gama0 is None:
                gama0, g0, lag0 = gm, gl[s], (a, b)
                pares0 = np.column_stack((iu[s], ju[s]))
            print(f"{a:6.0f}–{b:6.0f} {k:9d} {gm:9.3f} "
                  f"{100 * gm / sill_l:12.0f}% {origem:>9s}")
            if csv_rows is not None:
                csv_rows.append({"variante": variante, "lag_lo_m": a,
                                 "lag_hi_m": b, "n_pares": k,
                                 "semivar": gm, "patamar": sill_l,
                                 "pct_patamar": 100 * gm / sill_l})
        return gama0, pares0, g0, lag0, descartados, sill_l

    linhas_vario = []
    sill = z.var()
    gama0, pares0, g0, lag0, descartados, _ = tabela_variograma(
        z, "bruto", csv_rows=linhas_vario, variante="bruto")

    linhas_pepita = []
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
        lo, hi = np.clip(np.array(ic_media(g0)) / sill, 0, 1)
        print(f"IC95% da fração pepita: {lo:.0%}-{hi:.0%} "
              f"({len(g0)} pares no lag usado; patamar fixo, "
              "logo é limite inferior da incerteza)")
        # R2: IC por reamostragem de poços, patamar reamostrado junto
        blo, bhi = ic_pepita_por_poco(pares0, g0, z, n)
        print(f"IC95% por reamostragem de POÇOS (R2, patamar "
              f"reamostrado): {blo:.0%}-{bhi:.0%}")
        print(f"  -> R² máximo teórico entre {1 - bhi:.2f} e {1 - blo:.2f}")
        if descartados:
            print(f"ATENÇÃO: lag(s) mais curto(s) descartado(s) por terem "
                  f"menos de {MIN_PARES} pares:")
            for a, b, k in descartados:
                print(f"  {a:.0f}–{b:.0f} m: {k} pares")
            print("  A pepita acima foi medida num lag maior, onde a "
                  "semivariância já cresceu,")
            print("  e portanto SUBESTIMA a pepita real desta base.")
        print(f"R² máximo teórico de qualquer mapa: ~{r2_max:.2f}")
        linhas_pepita.append({"variante": "bruto",
                              "lag_lo_m": lag0[0], "lag_hi_m": lag0[1],
                              "n_pares": len(g0), "pepita": pepita_frac,
                              "ic_lo_pares": lo, "ic_hi_pares": hi,
                              "ic_lo_pocos": blo, "ic_hi_pocos": bhi,
                              "r2_max": r2_max})

    # ---------- 2b. sem tendência linear ----------
    print("\n2b) SEMIVARIOGRAMA APÓS REMOÇÃO DE TENDÊNCIA LINEAR (R2):")
    A = np.column_stack((np.ones(n), xy[:, 0] - xy[:, 0].mean(),
                         xy[:, 1] - xy[:, 1].mean()))
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    zr = z - A @ coef
    r2_tend = 1 - zr.var() / z.var()
    print(f"Tendência: z = {coef[0]:.3f} + {coef[1]:.3e}·x' + "
          f"{coef[2]:.3e}·y'   (R² da tendência: {r2_tend:.3f}; "
          f"variância residual {zr.var():.3f})")
    g0b, pares0b, g0bv, lag0b, _, sill_b = tabela_variograma(
        zr, "sem_tendencia", csv_rows=linhas_vario,
        variante="sem_tendencia")
    if g0b is not None:
        pep_b = min(1.0, g0b / sill_b)
        blo_b, bhi_b = ic_pepita_por_poco(pares0b, g0bv, zr, n)
        print(f"\nFração pepita sem tendência: {pep_b:.0%} "
              f"(IC95% por poços: {blo_b:.0%}-{bhi_b:.0%})")
        print(f"R² máximo teórico sobre o resíduo: ~{1 - pep_b:.2f}")
        linhas_pepita.append({"variante": "sem_tendencia",
                              "lag_lo_m": lag0b[0], "lag_hi_m": lag0b[1],
                              "n_pares": len(g0bv), "pepita": pep_b,
                              "ic_lo_pares": np.nan, "ic_hi_pares": np.nan,
                              "ic_lo_pocos": blo_b, "ic_hi_pocos": bhi_b,
                              "r2_max": 1 - pep_b})
        print("  Se a pepita cai muito aqui, parte do 'ruído' era"
              " tendência regional;")
        print("  se não muda, o patamar global era uma referência"
              " adequada.")

    # ---------- 2c. direcional ----------
    print("\n2c) SEMIVARIOGRAMA DIRECIONAL (R2) — % do patamar global:")
    nomes = ["N-S", "NE-SW", "E-W", "NW-SE"]
    mascaras = {
        "N-S": (azi < 22.5) | (azi >= 157.5),
        "NE-SW": (azi >= 22.5) & (azi < 67.5),
        "E-W": (azi >= 67.5) & (azi < 112.5),
        "NW-SE": (azi >= 112.5) & (azi < 157.5),
    }
    print(f"{'lag (m)':>14s} " + " ".join(f"{d:>12s}" for d in nomes))
    for a, b in zip(bins[:-1], bins[1:]):
        base = (dpar > a) & (dpar <= b)
        celula = []
        for d in nomes:
            s = base & mascaras[d]
            k = int(s.sum())
            if k < MIN_PARES:
                celula.append("        —")
                continue
            gm = float((0.5 * (z[iu[s]] - z[ju[s]]) ** 2).mean())
            celula.append(f"{100 * gm / sill:7.0f}% ({k})")
            linhas_vario.append({"variante": f"dir_{d}", "lag_lo_m": a,
                                 "lag_hi_m": b, "n_pares": k,
                                 "semivar": gm, "patamar": sill,
                                 "pct_patamar": 100 * gm / sill})
        print(f"{a:6.0f}–{b:6.0f} " + " ".join(f"{c:>12s}" for c in celula))
    print("  Anisotropia aparece como direções que saturam em lags "
          "diferentes.")

    # ---------- CSVs ----------
    pd.DataFrame(linhas_auc).to_csv(OUT / "s09_auc_vizinho.csv",
                                    index=False)
    pd.DataFrame(linhas_faixa).to_csv(OUT / "s09_auc_faixas.csv",
                                      index=False)
    pd.DataFrame(linhas_vario).to_csv(OUT / "s09_variograma.csv",
                                      index=False)
    pd.DataFrame(linhas_pepita).to_csv(OUT / "s09_pepita.csv", index=False)

    print("""
===== INTERPRETAÇÃO =====
A AUC-vizinho no raio de 250–500 m é uma REFERÊNCIA de vizinhança, não
um limite superior demonstrado: compare-a com o k-NN de (1b). Se k>1
supera k=1, qualquer fração calculada contra k=1 está inflada, e o
denominador honesto é o melhor k.
A fração pepita conta a mesma história em regressão: o R² máximo é
~(1 - pepita). Use o IC por reamostragem de poços, não o de pares.
O alcance informa mais que a pepita: se o semivariograma satura antes
do espaçamento típico entre poços, interpolar entre poços opera além
do comprimento de correlação. Em base que não satura no maior lag,
leia (2b) antes de tratar a variância global como patamar.""")
    print(f"Relatório salvo em: {RELATORIO}")
    print(f"CSVs em: {OUT}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()