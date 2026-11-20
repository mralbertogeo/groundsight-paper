"""s10 — Fração capturada da referência de vizinhança, com IC conjunto.

R3 (25/10/2026). Responde aos pontos 8 e 9 da revisão externa.

O artigo reporta "o modelo captura X% do teto" como razão de duas AUCs
estimadas, e deriva a faixa variando só o denominador — o que produz
truncamento arbitrário em 100% e ignora que numerador e denominador são
medidos nos MESMOS poços, logo são correlacionados.

Aqui a razão é reamostrada inteira, por BLOCOS ESPACIAIS de bloco_km
(os mesmos da validação cruzada): em cada reamostra recalcula-se o
numerador e o denominador sobre o mesmo conjunto de poços sorteado, e
só então a razão. Nada é truncado: frações acima de 100% são contadas
e reportadas, porque são a evidência de que a referência de vizinhança
não é um limite superior.

Duas populações de comparação, porque não são equivalentes:
  (A) numerador em TODOS os poços, denominador no subconjunto com
      vizinho dentro do raio — é a comparação que o artigo faz hoje;
  (B) numerador e denominador no MESMO subconjunto — comparação justa.

Duas razões:
  AUC:  (AUC_modelo - 0,5) / (AUC_knn - 0,5)
  R²:   R²_modelo / (1 - pepita), com a pepita recalculada em cada
        reamostra (patamar incluído), não tratada como constante.

Insumos: matriz_treino.parquet e validacao_oof*.parquet (gravados pelo
s07 a partir de 25/10). Este estágio não altera nada.

Execução:  docker compose run --rm pipeline python stages/s10_fracao_capturada.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import config, DERIVED

OUT = Path("/data/outputs")
RELATORIO = OUT / "s10_relatorio.txt"

RAIO = 250          # raio da referência de vizinhança
K_BENCH = (1, 5)    # k=1 é o do artigo; k=5 é a referência recomendada
LAG_PEPITA = 250    # mesmo lag do s09
REPS = 2000
SEMENTE = 20260101

VARIANTES = [("completo", "validacao_oof.parquet"),
             ("pre_perfuracao", "validacao_oof_pre_perfuracao.parquet")]


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


class AUCRapida:
    """AUC ponderada com os escores FIXOS e os pesos variando.

    A ordenação por escore é feita uma vez; cada reamostra custa poucas
    operações vetoriais. Empates entram com meio ponto, como na
    definição de Mann-Whitney (idêntico ao roc_auc_score do sklearn).
    """

    def __init__(self, escore, y):
        self.ordem = np.argsort(escore, kind="mergesort")
        s = np.asarray(escore)[self.ordem]
        self.y = np.asarray(y)[self.ordem]
        self.inicios = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])

    def __call__(self, w):
        wo = np.asarray(w, dtype=float)[self.ordem]
        wneg = wo * (self.y == 0)
        wpos = wo * (self.y == 1)
        gneg = np.add.reduceat(wneg, self.inicios)
        gpos = np.add.reduceat(wpos, self.inicios)
        Wn, Wp = gneg.sum(), gpos.sum()
        if Wn <= 0 or Wp <= 0:
            return np.nan
        abaixo = np.cumsum(gneg) - gneg
        return float((gpos * (abaixo + 0.5 * gneg)).sum() / (Wp * Wn))


def r2_ponderado(y, pred, w):
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return np.nan
    ybar = (w * y).sum() / sw
    sse = (w * (y - pred) ** 2).sum()
    sst = (w * (y - ybar) ** 2).sum()
    return float(1 - sse / sst) if sst > 0 else np.nan


def pct(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return (np.nan, np.nan)
    return tuple(np.percentile(v, [2.5, 97.5]))


def main():
    import geopandas as gpd
    from scipy.spatial import cKDTree

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    cfg = config()["validacao"]
    bloco_km = float(cfg.get("bloco_km", 10))

    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")
    g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m.lon, m.lat),
                         crs="EPSG:4674").to_crs(epsg=31983)
    xy = np.c_[g.geometry.x, g.geometry.y]
    z = m["log_qs"].values
    y = m["alvo_binario"].values
    n = len(m)

    print("===== s10 — FRAÇÃO CAPTURADA, IC CONJUNTO POR BLOCOS =====")
    print(f"Poços: {n} | blocos de {bloco_km:.0f} km | {REPS} reamostras\n")

    # ---------- blocos espaciais (mesma regra do s06/s07) ----------
    bx = (xy[:, 0] // (bloco_km * 1000)).astype(int)
    by = (xy[:, 1] // (bloco_km * 1000)).astype(int)
    rotulo = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))
    blocos, bloco_idx = np.unique(rotulo, return_inverse=True)
    nb = len(blocos)
    tam = np.bincount(bloco_idx)
    print(f"Blocos com poço: {nb} | mediana de poços por bloco: "
          f"{np.median(tam):.0f} | maior: {tam.max()}")

    # ---------- referência de vizinhança (k-NN) ----------
    tree = cKDTree(xy)
    kmax = max(K_BENCH)
    dk, ik = tree.query(xy, k=kmax + 1)
    dk, ik = dk[:, 1:], ik[:, 1:]
    dentro = dk[:, 0] <= RAIO            # tem ao menos 1 vizinho no raio
    escore_bench = {}
    for k in K_BENCH:
        w = np.where(dk[:, :k] <= RAIO, 1.0 / np.maximum(dk[:, :k], 1.0), 0.0)
        sw = w.sum(axis=1)
        escore_bench[k] = np.where(
            sw > 0, (z[ik[:, :k]] * w).sum(axis=1) / np.where(sw > 0, sw, 1.0),
            np.nan)
    print(f"Poços com vizinho a <= {RAIO} m: {int(dentro.sum())}\n")

    # ---------- pares do lag da pepita ----------
    pares = np.array(sorted(tree.query_pairs(r=LAG_PEPITA)),
                     dtype=int).reshape(-1, 2)
    gpar = 0.5 * (z[pares[:, 0]] - z[pares[:, 1]]) ** 2
    print(f"Pares a <= {LAG_PEPITA} m (pepita): {len(pares)}")

    # ---------- escores do modelo, por variante e semente ----------
    modelos = {}
    for nome, arquivo in VARIANTES:
        caminho = DERIVED / arquivo
        if not caminho.exists():
            print(f"AVISO: {arquivo} nao encontrado — variante '{nome}' "
                  "ignorada. Rode o s07 correspondente.")
            continue
        oof = pd.read_parquet(caminho)
        seeds = sorted(oof["seed"].unique())
        pr, prl = [], []
        for s in seeds:
            bloco = oof[oof["seed"] == s]
            if not np.array_equal(bloco["codigo_siagas"].values,
                                  m["codigo_siagas"].values):
                bloco = bloco.set_index("codigo_siagas").reindex(
                    m["codigo_siagas"].values).reset_index()
            pr.append(bloco["p_rf"].values.astype(float))
            prl.append(bloco["pred_log_qs"].values.astype(float))
        modelos[nome] = {"seeds": seeds, "p": pr, "reg": prl}
        print(f"Variante '{nome}': {len(seeds)} sementes carregadas")
    if not modelos:
        sys.exit("Nenhum arquivo de OOF encontrado.")

    # ---------- objetos de AUC rápida (ordenação feita uma vez) ----------
    auc_bench = {k: AUCRapida(np.where(np.isfinite(escore_bench[k]),
                                       escore_bench[k], -1e9), y)
                 for k in K_BENCH}
    auc_mod = {nome: [AUCRapida(p, y) for p in d["p"]]
               for nome, d in modelos.items()}

    rng = np.random.default_rng(SEMENTE)
    pesos_pop = {"A_todos": np.ones(n), "B_subconjunto": dentro.astype(float)}

    # ---------- ponto ----------
    linhas = []
    print("\n--- ESTIMATIVAS PONTUAIS ---")
    ponto_bench = {}
    for k in K_BENCH:
        ponto_bench[k] = auc_bench[k](dentro.astype(float))
        print(f"Referencia k={k} a {RAIO} m: AUC = {ponto_bench[k]:.3f}")
    ponto_mod, ponto_r2 = {}, {}
    for nome, d in modelos.items():
        for pop, wpop in pesos_pop.items():
            a = float(np.mean([f(wpop) for f in auc_mod[nome]]))
            ponto_mod[(nome, pop)] = a
        r2 = float(np.mean([r2_ponderado(z, reg, np.ones(n))
                            for reg in d["reg"]]))
        ponto_r2[nome] = r2
        print(f"Modelo '{nome}': AUC(todos) = "
              f"{ponto_mod[(nome, 'A_todos')]:.3f} | "
              f"AUC(subconjunto) = "
              f"{ponto_mod[(nome, 'B_subconjunto')]:.3f} | R2 = {r2:.3f}")
    pep_ponto = float(gpar.mean() / z.var())
    print(f"Pepita (lag 0-{LAG_PEPITA} m): {pep_ponto:.0%} | "
          f"R2 maximo: {1 - pep_ponto:.2f}")

    # ---------- bootstrap por blocos ----------
    print(f"\n--- BOOTSTRAP POR BLOCOS ({REPS} reamostras) ---")
    chaves_auc = [(nome, pop, k) for nome in modelos
                  for pop in pesos_pop for k in K_BENCH]
    dist_auc = {c: [] for c in chaves_auc}
    dist_bench = {k: [] for k in K_BENCH}
    dist_r2 = {nome: [] for nome in modelos}
    dist_pep = []

    pi, pj = pares[:, 0], pares[:, 1]
    for _ in range(REPS):
        c = rng.multinomial(nb, np.full(nb, 1.0 / nb)).astype(float)
        w = c[bloco_idx]                      # peso de cada poço
        if w.sum() <= 0:
            continue
        # denominador: referencia de vizinhanca no subconjunto
        wsub = w * dentro
        bench = {k: auc_bench[k](wsub) for k in K_BENCH}
        for k in K_BENCH:
            dist_bench[k].append(bench[k])
        # pepita e R2 maximo, recalculados na mesma reamostra
        wp = w[pi] * w[pj]
        swp = wp.sum()
        zbar = (w * z).sum() / w.sum()
        sill = (w * (z - zbar) ** 2).sum() / w.sum()
        pep = (wp * gpar).sum() / swp / sill if (swp > 0 and sill > 0) else np.nan
        dist_pep.append(pep)
        for nome in modelos:
            amod = {pop: float(np.mean([f(w * wpop)
                                        for f in auc_mod[nome]]))
                    for pop, wpop in pesos_pop.items()}
            for pop in pesos_pop:
                for k in K_BENCH:
                    den = bench[k] - 0.5
                    dist_auc[(nome, pop, k)].append(
                        (amod[pop] - 0.5) / den if den > 0 else np.nan)
            r2b = float(np.mean([r2_ponderado(z, reg, w)
                                 for reg in modelos[nome]["reg"]]))
            r2max = 1 - pep
            dist_r2[nome].append(r2b / r2max if r2max > 0 else np.nan)

    # ---------- relatório ----------
    print("\nReferencia de vizinhanca (denominador):")
    for k in K_BENCH:
        lo, hi = pct(dist_bench[k])
        print(f"  k={k}: {ponto_bench[k]:.3f}  IC95% {lo:.3f}-{hi:.3f}")
    lo, hi = pct(dist_pep)
    print(f"Pepita: {pep_ponto:.0%}  IC95% {lo:.0%}-{hi:.0%}  "
          f"-> R2 maximo {1 - hi:.2f} a {1 - lo:.2f}")

    print("\nFRACAO DA REFERENCIA CAPTURADA (AUC), sem truncamento:")
    print(f"{'variante':>16s} {'populacao':>15s} {'k':>3s} {'ponto':>8s} "
          f"{'IC95%':>17s} {'>100%':>7s}")
    for (nome, pop, k) in chaves_auc:
        v = np.asarray(dist_auc[(nome, pop, k)], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            continue
        p = (ponto_mod[(nome, pop)] - 0.5) / (ponto_bench[k] - 0.5)
        lo, hi = pct(v)
        acima = 100.0 * (v > 1.0).mean()
        print(f"{nome:>16s} {pop:>15s} {k:>3d} {p:8.0%} "
              f"{f'{lo:.0%} - {hi:.0%}':>17s} {acima:6.1f}%")
        linhas.append({"razao": "auc", "variante": nome, "populacao": pop,
                       "k": k, "ponto": p, "ic_lo": lo, "ic_hi": hi,
                       "pct_acima_100": acima})

    print("\nFRACAO DO R2 MAXIMO CAPTURADA, sem truncamento:")
    print(f"{'variante':>16s} {'ponto':>8s} {'IC95%':>17s} {'>100%':>7s}")
    for nome in modelos:
        v = np.asarray(dist_r2[nome], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            continue
        p = ponto_r2[nome] / (1 - pep_ponto)
        lo, hi = pct(v)
        acima = 100.0 * (v > 1.0).mean()
        print(f"{nome:>16s} {p:8.0%} {f'{lo:.0%} - {hi:.0%}':>17s} "
              f"{acima:6.1f}%")
        linhas.append({"razao": "r2", "variante": nome,
                       "populacao": "A_todos", "k": np.nan, "ponto": p,
                       "ic_lo": lo, "ic_hi": hi, "pct_acima_100": acima})

    pd.DataFrame(linhas).to_csv(OUT / "s10_fracao.csv", index=False)

    print("""
===== COMO LER =====
populacao A_todos      = numerador em todos os pocos, denominador no
                         subconjunto com vizinho no raio (e a comparacao
                         que o artigo faz hoje; as populacoes diferem).
populacao B_subconjunto= numerador e denominador nos MESMOS pocos.
k=1  reproduz o denominador do artigo; k=5 e a referencia recomendada
     depois do R2 (o vizinho unico e superavel, logo nao e teto).
A coluna '>100%' e a proporcao de reamostras em que o modelo supera a
referencia de vizinhanca. Se nao for desprezivel, a referencia nao pode
ser apresentada como limite superior — e esse e o ponto.
O IC e conjunto: numerador e denominador variam juntos na mesma
reamostra de blocos, o que capta a correlacao entre eles. Nao cobre a
incerteza de QUEM e o vizinho (a vizinhanca e tratada como fixa).""")
    print(f"\nRelatorio: {RELATORIO}")
    print(f"CSV: {OUT / 's10_fracao.csv'}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()