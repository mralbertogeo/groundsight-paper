"""s07_externa — Validação EXTERNA da v2.3.

Treina com configuração congelada usando SOMENTE os poços da AOI
original (config: validacao.aoi_original_bbox) e mede o desempenho
exclusivamente nos poços NOVOS da expansão — dados que nenhuma
iteração anterior jamais viu. Este é o teste de generalização mais
honesto do projeto; a barra bloqueante (0,70) segue aplicada à CV
espacial do conjunto completo (s07), e este resultado é REPORTADO.

O limiar do alvo binário (q75) é calculado APENAS no conjunto de
treino original — o teste externo não informa nada ao treino.

Execução:  docker compose run --rm pipeline python stages/s07_externa.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import config, DERIVED

RELATORIO = Path("/data/outputs/s07_externa_relatorio.txt")


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def main():
    import json
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score, r2_score
    import xgboost as xgb

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    cfg = config()["validacao"]
    bbox = cfg["aoi_original_bbox"]

    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")
    preds = json.loads((DERIVED / "modelos" / "preditores.json").read_text())

    orig = ((m.lon >= bbox[0]) & (m.lon <= bbox[2])
            & (m.lat >= bbox[1]) & (m.lat <= bbox[3])).values
    novos = ~orig

    print("===== s07_externa — VALIDAÇÃO EXTERNA (v2.3) =====")
    print(f"Poços de treino (AOI original): {int(orig.sum())}")
    print(f"Poços de teste (expansão):      {int(novos.sum())}")
    if novos.sum() < 50:
        print("REPORTE INVÁLIDO: menos de 50 poços novos — expansão "
              "insuficiente.")
        sys.exit(1)

    q75 = m.loc[orig, "log_qs"].quantile(0.75)
    y = (m["log_qs"] >= q75).astype(int).values
    print(f"Limiar 'alta' (q75 do TREINO): log10(Q/s) = {q75:.2f}")
    print(f"Positivos no teste externo: {int(y[novos].sum())} "
          f"de {int(novos.sum())}")
    if len(set(y[novos])) < 2:
        print("REPORTE INVÁLIDO: teste externo sem as duas classes.")
        sys.exit(1)

    X = m[preds].values

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X[orig], y[orig])
    auc_rf = roc_auc_score(y[novos], rf.predict_proba(X[novos])[:, 1])

    xg = xgb.XGBClassifier(n_estimators=400, max_depth=4,
                           learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, random_state=42,
                           eval_metric="auc")
    xg.fit(X[orig], y[orig])
    auc_xgb = roc_auc_score(y[novos], xg.predict_proba(X[novos])[:, 1])

    rr = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)
    rr.fit(X[orig], m.loc[orig, "log_qs"])
    r2 = r2_score(m.loc[novos, "log_qs"], rr.predict(X[novos]))

    print(f"\nAUC externo  RF: {auc_rf:.3f} | XGB: {auc_xgb:.3f}")
    print(f"R² externo (log Q/s): {r2:.3f}")
    print("\nInterpretação: este número mede transferência para área "
          "nunca vista por nenhuma iteração — é a estimativa mais "
          "conservadora do desempenho real do produto.")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
