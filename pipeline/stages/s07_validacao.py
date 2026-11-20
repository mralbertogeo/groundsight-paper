"""Etapa 7 — Validação (M6, v1; R1 em 25/10/2026).

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

R1 (25/10/2026): variantes de validação sem alterar o modelo final.
  --excluir  lista de preditores retirados SÓ nesta validação (o s06 e o
             preditores.json não mudam);
  --variante rótulo dos arquivos de saída ('completo' mantém os nomes
             antigos; qualquer outro vira sufixo).
  Grava também as probabilidades fora-da-amostra (OOF) por seed, para
  calibração e bootstrap conjunto.

R5 (10/11/2026): controles de vazamento espacial, sem alterar o modelo.
  --bloco_km   sobrepõe o bloco_km do config (teste de 20 km na área 2,
               cujo variograma não saturou em 16 km);
  --buffer_km  remove da AVALIAÇÃO os poços de teste a menos de X km do
               poço de treino mais próximo (o treino não muda, só o
               conjunto avaliado) — mede quanto do AUC vem de vizinhança
               imediata sobrevivente ao bloqueio;
  grava validacao_distancias{sufixo}.csv com a distribuição da distância
  de cada poço de teste ao treino mais próximo, por seed.

Execução:
  docker compose run --rm pipeline python stages/s07_validacao.py
  docker compose run --rm pipeline python stages/s07_validacao.py \
      --variante pre_perfuracao \
      --excluir profundidade_m lf_m tipo_filtros tipo_trecho_aberto tipo_coluna_saturada
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from _comum import config, DERIVED

PREDITORES = ["mde", "declividade", "northness", "curvatura", "tri",
              "twi", "hand", "dist_drenagem", "dens_drenagem",
              "esp_solo", "esp_saprolito", "esp_rocha_alterada",
              "prof_base_solo", "prof_topo_zona_rocha",
              "prof_topo_rocha_sa"]


def cv_espacial(m, seed, n_folds, bloco_km):
    from s06_treino import blocos_espaciais
    return blocos_espaciais(m, bloco_km, n_folds, seed)

def coords_m(m):
    """Coordenadas projetadas (UTM 23S) para distância em metros."""
    import geopandas as gpd
    g = gpd.GeoDataFrame(m[["lon", "lat"]],
                         geometry=gpd.points_from_xy(m.lon, m.lat),
                         crs="EPSG:4674").to_crs(epsg=31983)
    return np.c_[g.geometry.x.values, g.geometry.y.values]

def main():
    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
    import xgboost as xgb
    from s06_treino import baseline_fuzzy

    # R1: variante de validação por linha de comando
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", default="completo")
    ap.add_argument("--excluir", nargs="*", default=[])
    ap.add_argument("--bloco_km", type=float, default=None)
    ap.add_argument("--buffer_km", type=float, default=0.0)    
    args = ap.parse_args()
    sufixo = "" if args.variante == "completo" else f"_{args.variante}"

    cfg = config()["validacao"]
    n_folds = cfg["n_folds"]
    bloco_km = (args.bloco_km if args.bloco_km is not None
                else float(cfg.get("bloco_km", 10)))
    buffer_m = float(args.buffer_km) * 1000.0
    seeds = cfg["seeds"]

    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")
    # v2: usa exatamente os preditores persistidos pelo s06
    import json
    pred_file = DERIVED / "modelos" / "preditores.json"
    preds = (json.loads(pred_file.read_text()) if pred_file.exists()
             else PREDITORES)
    # R1: exclusão declarada — erro se o nome não existir, para não
    # "excluir" silenciosamente um preditor grafado errado
    if args.excluir:
        faltam = [c for c in args.excluir if c not in preds]
        if faltam:
            sys.exit(f"--excluir com preditores desconhecidos: {faltam}")
        preds = [c for c in preds if c not in args.excluir]
    print(f"Variante '{args.variante}': {len(preds)} preditores"
          + (f" (excluídos: {args.excluir})" if args.excluir else ""))
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

    xy = coords_m(m)
    resultados, oof, dists = [], [], []
    print(f"CV: blocos de {bloco_km:.0f} km | buffer de avaliação "
          f"{args.buffer_km:.1f} km")                          # R1
    for seed in seeds:
        folds = cv_espacial(m, seed, n_folds, bloco_km)
        pred_rf = np.zeros(len(m)); pred_xgb = np.zeros(len(m))
        pred_reg = np.zeros(len(m))
        d_nn = np.full(len(m), np.nan)       # R5
        av = np.zeros(len(m), dtype=bool)    # R5: avaliados
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            if te.sum() == 0 or len(set(y_cls[tr])) < 2:
                continue
            tr_idx = np.where(tr)[0]
            # R5: distância de cada poço de teste ao treino mais próximo
            d_te, _ = cKDTree(xy[tr]).query(xy[te], k=1)
            d_nn[te] = d_te
            av[np.where(te)[0][d_te >= buffer_m]] = True

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
        if av.sum() < 50 or len(set(y_cls[av])) < 2:
            sys.exit(f"seed {seed}: buffer de {args.buffer_km:.1f} km "
                     f"deixou {int(av.sum())} poços avaliáveis — "
                     "reduza o buffer ou o bloco")
        resultados.append({
            "seed": seed,
            "n_avaliados": int(av.sum()),
            "n_removidos_buffer": int(len(m) - av.sum()),
            "auc_rf": roc_auc_score(y_cls[av], pred_rf[av]),
            "auc_xgb": roc_auc_score(y_cls[av], pred_xgb[av]),
            "auc_ens": roc_auc_score(y_cls[av], pred_ens[av]),
            "auc_fuzzy": roc_auc_score(y_cls[av], fuzzy[av]),
            "r2_rf": r2_score(y_reg[av], pred_reg[av]),
            "rmse_rf": float(np.sqrt(
                mean_squared_error(y_reg[av], pred_reg[av]))),
        })
        # R5: distribuição da distância treino-teste
        d_ok = d_nn[~np.isnan(d_nn)]
        dists.append({
            "seed": seed, "n": int(len(d_ok)),
            "d_min_m": float(d_ok.min()),
            "d_p10_m": float(np.percentile(d_ok, 10)),
            "d_mediana_m": float(np.median(d_ok)),
            "d_p90_m": float(np.percentile(d_ok, 90)),
            "d_max_m": float(d_ok.max()),
            "frac_abaixo_250m": float((d_ok < 250).mean()),
            "frac_abaixo_1km": float((d_ok < 1000).mean()),
        })
        # R1: OOF por seed — insumo da calibração (R6) e do bootstrap
        # conjunto por bloco (R3)
        oof.append(pd.DataFrame({
            "seed": seed, "codigo_siagas": m["codigo_siagas"].values,
            "fold": folds, "lon": m["lon"].values, "lat": m["lat"].values,
            "y": y_cls, "log_qs": y_reg,
            "p_rf": pred_rf, "p_xgb": pred_xgb, "p_ens": pred_ens,
            "pred_log_qs": pred_reg,
            "dist_treino_m": d_nn, "avaliado": av,
        }))
        print(f"  seed {seed}: AUC RF {resultados[-1]['auc_rf']:.3f} "
              f"XGB {resultados[-1]['auc_xgb']:.3f} "
              f"ENS {resultados[-1]['auc_ens']:.3f} "
              f"(n={int(av.sum())})", flush=True)
        
    df = pd.DataFrame(resultados)
    df.to_csv(DERIVED / f"validacao_resultados{sufixo}.csv", index=False)
    pd.concat(oof, ignore_index=True).to_parquet(
        DERIVED / f"validacao_oof{sufixo}.parquet", index=False)

    dft = pd.DataFrame(dists)
    dft.to_csv(DERIVED / f"validacao_distancias{sufixo}.csv", index=False)    

    med = df.mean(numeric_only=True)
    melhor_ml = max(med["auc_rf"], med["auc_xgb"], med["auc_ens"])
    var_seeds = max(df["auc_rf"].std(), df["auc_xgb"].std(),
                    df["auc_ens"].std())

    print(f"\n===== RELATÓRIO s07 (validação espacial) — variante "
          f"'{args.variante}' =====")
    print(df.round(3).to_string(index=False))
    print(f"\nAUC médio  RF: {med['auc_rf']:.3f} | XGB: {med['auc_xgb']:.3f}"
          f" | ENSEMBLE: {med['auc_ens']:.3f}"
          f" | baseline fuzzy: {med['auc_fuzzy']:.3f}")
    print(f"R² médio (log Q/s): {med['r2_rf']:.3f} | "
          f"RMSE: {med['rmse_rf']:.3f}")
    print(f"Desvio entre seeds (AUC): {var_seeds:.3f} (alvo <= 0.02)")
    print(f"CV: blocos de {bloco_km:.0f} km, {n_folds} folds, "
          f"seeds {seeds}")
    print(f"Preditores ({len(preds)}): {preds}")

    print(f"Buffer de avaliação: {args.buffer_km:.1f} km | poços "
          f"avaliados: {med['n_avaliados']:.0f} de {len(m)}")
    print("\nDistância de cada poço de teste ao poço de treino mais "
          "próximo (m):")
    print(dft.round(3).to_string(index=False))
    
    # Barra RELATIVA AO TETO medido (pré-registro de 10/08/2026):
    # exigir mais discriminação do que o próprio local replicado a
    # 250 m oferece é exigir o impossível — o critério passa a ser a
    # fração capturada do discriminável (s09: teto-vizinho).
    teto = float(cfg.get("teto_vizinho_auc", 0.672))
    fator = float(cfg.get("fator_teto", 0.95))
    barra = fator * teto
    fracao_bruta = melhor_ml / teto
    # R1: fração corrigida pelo acaso — a que o manuscrito reporta
    fracao = (melhor_ml - 0.5) / (teto - 0.5)
    print(f"Barra bloqueante = {fator:.2f} × teto-vizinho {teto:.3f} "
          f"= {barra:.3f} (equivale a "
          f"{(barra - 0.5) / (teto - 0.5):.0%} corrigido pelo acaso)")
    print(f"Fração do teto capturada: {fracao:.0%} corrigida pelo acaso "
          f"[(AUC-0.5)/(teto-0.5)] | {fracao_bruta:.0%} bruta")

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
              f"({fracao:.0%} do teto mensurável, corrigido)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()