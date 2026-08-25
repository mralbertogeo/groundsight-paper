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
  6. Bandas de interpretação (pré-registradas — experimento de
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


def direcao(treino, teste, nome):
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
    med = treino[cols].median()
    Xtr = treino[cols].fillna(med).values
    Xte = teste[cols].fillna(med).values
    ytr = (treino["log_qs"] >= treino["log_qs"].quantile(.75)).values
    yte = (teste["log_qs"] >= teste["log_qs"].quantile(.75)).values

    clf = RandomForestClassifier(n_estimators=500, random_state=42,
                                 n_jobs=-1, min_samples_leaf=2)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    reg = RandomForestRegressor(n_estimators=500, random_state=42,
                                n_jobs=-1, min_samples_leaf=2)
    reg.fit(Xtr, treino["log_qs"].values)
    r2 = r2_score(teste["log_qs"].values, reg.predict(Xte))

    print(f"\n----- {nome} -----")
    print(f"Colunas comuns usadas: {len(cols)}")
    print(f"Treino: {len(treino)} | Teste: {len(teste)} "
          f"(positivos: {int(yte.sum())})")
    print(f"AUC transferido: {auc:.3f} | R² transferido: {r2:.3f}")
    top = sorted(zip(clf.feature_importances_, cols), reverse=True)[:5]
    print("Top 5 (importância no treino): "
          + ", ".join(f"{c} {v:.3f}" for v, c in top))
    return auc, r2


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

    auc12, r212 = direcao(a1, a2, "ÁREA 1 -> ÁREA 2 (treina SP-leste, "
                                  "prevê Vale do Paraíba)")
    auc21, r221 = direcao(a2, a1, "ÁREA 2 -> ÁREA 1 (treina VP, "
                                  "prevê SP-leste)")

    print("\n===== LEITURA PELAS BANDAS PRÉ-REGISTRADAS =====")
    for nome, auc in (("A1->A2", auc12), ("A2->A1", auc21)):
        banda = ("TRANSFERÊNCIA ÚTIL" if auc >= 0.65 else
                 "TRANSFERÊNCIA PARCIAL — treino local recomendado"
                 if auc >= 0.55 else
                 "NÃO TRANSFERE — treino local obrigatório")
        print(f"{nome}: AUC {auc:.3f} -> {banda}")
    print("\nReferências: CV espacial A1 0,660 | A2 0,736; "
          "fuzzy A1 0,519 | A2 0,434; sonda interna A2 0,653.")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
