"""Etapa 7 — Validação (M6, v1).

Validação cruzada ESPACIAL (blocos de bloco_km agrupados em n_folds):
poços do mesmo bloco nunca se separam entre treino e teste — evita a
superestimação clássica da CV aleatória sob autocorrelação espacial.

Métricas:
  - AUC (alvo binário 'alta favorabilidade') — RF e XGBoost;
  - R² e RMSE da regressão de log10(Q/s);
  - AUC do baseline fuzzy-gamma nos MESMOS folds (comparação justa);
  - Estabilidade: repetição com todas as seeds do config.

Critérios (config.validacao):
  - BLOQUEANTE: melhor AUC médio < auc_minimo_bloqueante -> exit 1;
  - BLOQUEANTE: melhor ML não supera o baseline -> exit 1;
  - Alvo (não bloqueante): AUC >= auc_alvo; variação entre seeds <= 0.02.

Execução:  docker compose run --rm pipeline python stages/s07_validacao.py
"""
import sys

import numpy as np
import pandas as pd

from _comum import config, DERIVED

PREDITORES = ["mde", "declividade", "northness", "curvatura", "tri",
              "twi", "hand", "dist_drenagem", "dens_drenagem",
              "esp_solo", "esp_saprolito", "esp_rocha_alterada",
              "prof_base_solo", "prof_topo_zona_rocha",
              "prof_topo_rocha_sa"]


def cv_espacial(m, seed, n_folds, bloco_km):
    from s06_treino import blocos_espaciais
    return blocos_espaciais(m, bloco_km, n_folds, seed)


def main():
    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
    import xgboost as xgb
    from s06_treino import baseline_fuzzy

    cfg = config()["validacao"]
    n_folds = cfg["n_folds"]
    bloco_km = float(cfg.get("bloco_km", 10))
    seeds = cfg["seeds"]

    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")
    # v2: usa exatamente os preditores persistidos pelo s06
    import json
    pred_file = DERIVED / "modelos" / "preditores.json"
    preds = (json.loads(pred_file.read_text()) if pred_file.exists()
             else PREDITORES)
    X = m[preds].values
    y_reg = m["log_qs"].values
    y_cls = m["alvo_binario"].values
    fuzzy = baseline_fuzzy(m)

    # v2.2: tuning por CV espacial ANINHADA — a seleção de
    # hiperparâmetros acontece dentro dos folds de treino, nunca vendo
    # o fold de teste. Grades pequenas e pré-registradas.
    GRADE_RF = [{"min_samples_leaf": v} for v in (2, 5, 10)]
    GRADE_XGB = [{"max_depth": 3, "learning_rate": 0.05},
                 {"max_depth": 4, "learning_rate": 0.05},
                 {"max_depth": 6, "learning_rate": 0.03}]

    def fit_rf(params, Xtr, ytr, seed):
        mdl = RandomForestClassifier(n_estimators=500, random_state=seed,
                                     n_jobs=-1, **params)
        mdl.fit(Xtr, ytr); return mdl

    def fit_xgb(params, Xtr, ytr, seed):
        mdl = xgb.XGBClassifier(n_estimators=400, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed,
                                eval_metric="auc", **params)
        mdl.fit(Xtr, ytr); return mdl

    def escolher(grade, fit_fn, tr_idx, folds, seed):
        """Seleção interna: folds de treino viram 3 pseudo-folds
        espaciais (blocos preservados)."""
        folds_tr = folds[tr_idx]
        restantes = sorted(set(folds_tr))
        inner = np.array([restantes.index(f) % 3 for f in folds_tr])
        melhor, melhor_auc = grade[0], -1
        for params in grade:
            pred = np.zeros(len(tr_idx))
            for k in range(3):
                itr, ite = inner != k, inner == k
                if ite.sum() == 0 or len(set(y_cls[tr_idx][itr])) < 2:
                    continue
                mdl = fit_fn(params, X[tr_idx][itr], y_cls[tr_idx][itr],
                             seed)
                pred[ite] = mdl.predict_proba(X[tr_idx][ite])[:, 1]
            try:
                a = roc_auc_score(y_cls[tr_idx], pred)
            except ValueError:
                a = 0.5
            if a > melhor_auc:
                melhor, melhor_auc = params, a
        return melhor

    resultados = []
    for seed in seeds:
        folds = cv_espacial(m, seed, n_folds, bloco_km)
        pred_rf = np.zeros(len(m)); pred_xgb = np.zeros(len(m))
        pred_reg = np.zeros(len(m))
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            if te.sum() == 0 or len(set(y_cls[tr])) < 2:
                continue
            tr_idx = np.where(tr)[0]

            p_rf = escolher(GRADE_RF, fit_rf, tr_idx, folds, seed)
            rf = fit_rf(p_rf, X[tr], y_cls[tr], seed)
            pred_rf[te] = rf.predict_proba(X[te])[:, 1]

            p_xg = escolher(GRADE_XGB, fit_xgb, tr_idx, folds, seed)
            xg = fit_xgb(p_xg, X[tr], y_cls[tr], seed)
            pred_xgb[te] = xg.predict_proba(X[te])[:, 1]

            rr = RandomForestRegressor(n_estimators=500,
                                       min_samples_leaf=3,
                                       random_state=seed, n_jobs=-1)
            rr.fit(X[tr], y_reg[tr])
            pred_reg[te] = rr.predict(X[te])

        pred_ens = (pred_rf + pred_xgb) / 2.0
        resultados.append({
            "seed": seed,
            "auc_rf": roc_auc_score(y_cls, pred_rf),
            "auc_xgb": roc_auc_score(y_cls, pred_xgb),
            "auc_ens": roc_auc_score(y_cls, pred_ens),
            "auc_fuzzy": roc_auc_score(y_cls, fuzzy),
            "r2_rf": r2_score(y_reg, pred_reg),
            "rmse_rf": float(np.sqrt(mean_squared_error(y_reg, pred_reg))),
        })

    df = pd.DataFrame(resultados)
    df.to_csv(DERIVED / "validacao_resultados.csv", index=False)

    med = df.mean(numeric_only=True)
    melhor_ml = max(med["auc_rf"], med["auc_xgb"], med["auc_ens"])
    var_seeds = max(df["auc_rf"].std(), df["auc_xgb"].std(),
                    df["auc_ens"].std())

    print("\n===== RELATÓRIO s07 (validação espacial) =====")
    print(df.round(3).to_string(index=False))
    print(f"\nAUC médio  RF: {med['auc_rf']:.3f} | XGB: {med['auc_xgb']:.3f}"
          f" | ENSEMBLE: {med['auc_ens']:.3f}"
          f" | baseline fuzzy: {med['auc_fuzzy']:.3f}")
    print(f"R² médio (log Q/s): {med['r2_rf']:.3f} | "
          f"RMSE: {med['rmse_rf']:.3f}")
    print(f"Desvio entre seeds (AUC): {var_seeds:.3f} (alvo <= 0.02)")
    print(f"CV: blocos de {bloco_km:.0f} km, {n_folds} folds, "
          f"seeds {seeds}")

    # Barra RELATIVA AO TETO medido (pré-registro de 10/08/2026):
    # exigir mais discriminação do que o próprio local replicado a
    # 250 m oferece é exigir o impossível — o critério passa a ser a
    # fração capturada do discriminável (s09: teto-vizinho).
    teto = float(cfg.get("teto_vizinho_auc", 0.672))
    fator = float(cfg.get("fator_teto", 0.95))
    barra = fator * teto
    fracao = melhor_ml / teto
    print(f"Barra bloqueante = {fator:.2f} × teto-vizinho {teto:.3f} "
          f"= {barra:.3f} | fração do teto capturada: {fracao:.0%}")

    ok = True
    if melhor_ml < barra:
        print(f"\nREPROVADO (bloqueante): melhor AUC {melhor_ml:.3f} < "
              f"barra {barra:.3f}")
        ok = False
    if melhor_ml <= med["auc_fuzzy"]:
        print(f"\nREPROVADO (bloqueante): ML ({melhor_ml:.3f}) não supera "
              f"o baseline fuzzy ({med['auc_fuzzy']:.3f})")
        ok = False
    if ok:
        print(f"\nAPROVADO: melhor AUC {melhor_ml:.3f} "
              f"({fracao:.0%} do teto mensurável)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
