"""s07_cross — Transferência entre áreas (pré-registrado 11/08/2026).

PERGUNTA: um modelo treinado em uma região prevê a favorabilidade de
outra que ele nunca viu? É o teste externo genuíno da plataforma
(substitui a sonda fraca núcleo×anel da área 2).

PROTOCOLO (declarado antes de qualquer número):
  1. Treino: matriz completa da área de origem (domínios fissural +
     outro, alvo = q75 do log Q/s DA PRÓPRIA área — a favorabilidade
     é região-relativa por definição do produto);
  2. Teste: TODOS os poços da área de destino, classe pelo q75 LOCAL
     do destino (mesma definição região-relativa);
  3. Covariáveis: interseção das colunas numéricas comuns, imputação
     pela MEDIANA DO TREINO (realismo de produção); as geofísicas são
     z-score por área — a padronização intra-área É o protocolo de
     harmonização testado;
  4. Modelos: RF classificador (n=500, seed 42) e RF regressor,
     retreinados do zero nas colunas comuns;
  5. DUAS direções: A1→A2 e A2→A1;
    6. ABLAÇÃO (R4): três conjuntos de covariáveis sobre a MESMA
     interseção — (a) todas as comuns; (b) sem geofísica (geof_*),
     que mede quanto da transferência depende da harmonização
     z-score intra-área; (c) sem construtivas (profundidade_m, lf_m,
     tipo_*), que é a variante de pré-perfuração do s06/s07;
  7. IC 95% do AUC transferido por bootstrap de BLOCOS de 10 km da
     área de TESTE (mesmo esquema do s07/s10), predições fixas;
  8. Bandas de interpretação (pré-registradas — experimento de
     MEDIÇÃO, sem barra de bloqueio):
       AUC >= 0,65  transferência útil (supera com folga o fuzzy
                    local 0,43/0,52 e empata com a sonda interna);
       0,55–0,65    transferência parcial — treino local recomendado;
       <  0,55      não transfere — mapa multi-região EXIGE treino
                    local (resultado igualmente reportável).

Pré-requisito: copiar da ÁREA 1 para esta instância:
  C:\\HydroAI\\groundsight\\data\\derived\\covariaveis_pocos.parquet
  C:\\HydroAI\\groundsight\\data\\derived\\hidraulica_derivada.parquet
        -> C:\\HydroAI\\groundsight-vp\\data\\derived\\cross_area1\\

Execução:  docker compose run --rm pipeline python stages/s07_cross.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import DERIVED

OUT = Path("/data/outputs")
RELATORIO = OUT / "s07_cross_relatorio.txt"
DIR_A1 = DERIVED / "cross_area1"
DOMINIOS = ["fissural", "outro"]


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def carregar(dir_base, rotulo):
    cov = pd.read_parquet(dir_base / "covariaveis_pocos.parquet")
    hid = pd.read_parquet(dir_base / "hidraulica_derivada.parquet")
    dcol = next((c for c in hid.columns if "domin" in c.lower()), None)
    if dcol is None:
        raise RuntimeError(
            f"[{rotulo}] coluna de domínio ausente em hidraulica_derivada; "
            f"colunas: {list(hid.columns)}")
    alvo = next((c for c in ("log_qs", "alvo_log_qs")
                 if c in hid.columns), None)
    if alvo is None:
        raise RuntimeError(f"[{rotulo}] coluna log_qs ausente; "
                           f"colunas: {list(hid.columns)}")
    m = cov.merge(hid[["codigo_siagas", alvo, dcol]], on="codigo_siagas")
    m = m[m[dcol].isin(DOMINIOS)].dropna(subset=[alvo])
    m = m.rename(columns={alvo: "log_qs"})
    print(f"[{rotulo}] matriz: {len(m)} poços "
          f"({dict(m[dcol].value_counts())})")
    return m.drop(columns=[dcol])


CONSTRUTIVOS = ("profundidade_m", "lf_m", "tipo_filtros",
                "tipo_trecho_aberto", "tipo_coluna_saturada")
BLOCO_KM = 10.0
N_BOOT = 1000


def blocos_10km(df):
    """Rótulo de bloco de BLOCO_KM na projeção metrica (UTM 23S)."""
    import geopandas as gpd
    faltando = [c for c in ("lon", "lat") if c not in df.columns]
    if faltando:
        raise RuntimeError(
            f"lon/lat ausentes ({faltando}) — o bootstrap de blocos "
            "precisa das coordenadas em covariaveis_pocos.parquet")
    g = gpd.GeoDataFrame(
        df[["lon", "lat"]],
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4674").to_crs(epsg=31983)
    bx = (g.geometry.x // (BLOCO_KM * 1000)).astype(int).values
    by = (g.geometry.y // (BLOCO_KM * 1000)).astype(int).values
    return np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))


def ic_auc_blocos(y, p, blocos, seed=42):
    """IC 95% do AUC por reamostragem de blocos espaciais do teste."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    unicos = np.array(sorted(set(blocos)))
    idx_por_bloco = {b: np.where(blocos == b)[0] for b in unicos}
    amostras = []
    for _ in range(N_BOOT):
        escolha = rng.choice(unicos, size=len(unicos), replace=True)
        ii = np.concatenate([idx_por_bloco[b] for b in escolha])
        yb = y[ii]
        if len(set(yb)) < 2:
            continue
        amostras.append(roc_auc_score(yb, p[ii]))
    if len(amostras) < 100:
        return np.nan, np.nan, len(amostras)
    a = np.array(amostras)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
            len(a))


def direcao(treino, teste, nome, conjunto="todas"):
    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.metrics import roc_auc_score, r2_score

    meta = {"codigo_siagas", "log_qs"}
    cols = sorted(
        c for c in set(treino.columns) & set(teste.columns)
        if c not in meta
        and pd.api.types.is_numeric_dtype(treino[c])
        and pd.api.types.is_numeric_dtype(teste[c])
        and treino[c].notna().mean() >= 0.6
        and teste[c].notna().mean() >= 0.6)
    n_comuns = len(cols)
    if conjunto == "sem_geofisica":
        cols = [c for c in cols if not c.startswith("geof_")]
    elif conjunto == "sem_construtivas":
        cols = [c for c in cols if c not in CONSTRUTIVOS]
    if not cols:
        raise RuntimeError(f"conjunto '{conjunto}' ficou sem covariáveis")

    med = treino[cols].median()
    Xtr = treino[cols].fillna(med).values
    Xte = teste[cols].fillna(med).values
    # q75 de CADA área sobre a própria distribuição: o alvo é
    # região-relativo por definição do produto, então o limiar do
    # teste é o q75 DO TESTE (nunca o do treino).
    q75_tr = treino["log_qs"].quantile(.75)
    q75_te = teste["log_qs"].quantile(.75)
    ytr = (treino["log_qs"] >= q75_tr).values
    yte = (teste["log_qs"] >= q75_te).values

    clf = RandomForestClassifier(n_estimators=500, random_state=42,
                                 n_jobs=-1, min_samples_leaf=2)
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_te)
    lo, hi, n_ok = ic_auc_blocos(yte, p_te, blocos_10km(teste))

    reg = RandomForestRegressor(n_estimators=500, random_state=42,
                                n_jobs=-1, min_samples_leaf=2)
    reg.fit(Xtr, treino["log_qs"].values)
    r2 = r2_score(teste["log_qs"].values, reg.predict(Xte))

    print(f"\n----- {nome} | conjunto: {conjunto} -----")
    print(f"Covariáveis: {len(cols)} de {n_comuns} comuns")
    print(f"Limiar de classe: q75 do TREINO {q75_tr:.3f} (treino) | "
          f"q75 do TESTE {q75_te:.3f} (teste)")
    print(f"Treino: {len(treino)} | Teste: {len(teste)} "
          f"(positivos: {int(yte.sum())})")
    print(f"AUC transferido: {auc:.3f} "
          f"[IC95 blocos {BLOCO_KM:.0f} km: {lo:.3f}–{hi:.3f}; "
          f"{n_ok} reamostras validas] | R² transferido: {r2:.3f}")
    top = sorted(zip(clf.feature_importances_, cols), reverse=True)[:5]
    print("Top 5 (importância no treino): "
          + ", ".join(f"{c} {v:.3f}" for v, c in top))
    return {"direcao": nome.split(" (")[0], "conjunto": conjunto,
            "n_cov": len(cols), "auc": auc, "auc_ic_lo": lo,
            "auc_ic_hi": hi, "r2": r2}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    print("===== s07_cross — TRANSFERÊNCIA ENTRE ÁREAS =====")
    print("Protocolo pré-registrado no cabeçalho do estágio.\n")

    if not (DIR_A1 / "covariaveis_pocos.parquet").exists():
        print("ERRO: copie covariaveis_pocos.parquet e "
              "hidraulica_derivada.parquet da área 1 para "
              "data/derived/cross_area1/")
        sys.exit(1)

    a1 = carregar(DIR_A1, "área 1")
    a2 = carregar(DERIVED, "área 2 (VP)")

    pares = [(a1, a2, "ÁREA 1 -> ÁREA 2 (treina SP-leste, "
                      "prevê Vale do Paraíba)"),
             (a2, a1, "ÁREA 2 -> ÁREA 1 (treina VP, prevê SP-leste)")]
    conjuntos = ["todas", "sem_geofisica", "sem_construtivas"]

    res = []
    for tr, te, nome in pares:
        for conj in conjuntos:
            res.append(direcao(tr, te, nome, conj))

    tab = pd.DataFrame(res)
    tab.to_csv(OUT / "s07_cross_ablacao.csv", index=False)

    print("\n===== ABLAÇÃO (AUC transferido, IC95 por blocos) =====")
    print(f"{'direção':>10s} {'conjunto':>18s} {'n cov':>6s} "
          f"{'AUC':>6s} {'IC95':>14s} {'R²':>7s}")
    for r in res:
        print(f"{r['direcao']:>10s} {r['conjunto']:>18s} "
              f"{r['n_cov']:6d} {r['auc']:6.3f} "
              f"{r['auc_ic_lo']:6.3f}-{r['auc_ic_hi']:6.3f} "
              f"{r['r2']:7.3f}")

    print("\n===== LEITURA PELAS BANDAS PRÉ-REGISTRADAS "
          "(conjunto 'todas') =====")
    for r in [x for x in res if x["conjunto"] == "todas"]:
        auc = r["auc"]
        banda = ("TRANSFERÊNCIA ÚTIL" if auc >= 0.65 else
                 "TRANSFERÊNCIA PARCIAL — treino local recomendado"
                 if auc >= 0.55 else
                 "NÃO TRANSFERE — treino local obrigatório")
        print(f"{r['direcao']}: AUC {auc:.3f} -> {banda}")
    print("\nTabela de ablação: data/outputs/s07_cross_ablacao.csv")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()    