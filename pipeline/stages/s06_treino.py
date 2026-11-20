"""Etapa 6 — Matriz de treino e modelagem (M6, v1).

  1. Matriz: covariáveis extraídas nos poços (s05) × alvo log10(Q/s)
     (s03). SOMENTE covariáveis disponíveis em raster entram como
     preditores — garante que o modelo treinado gere o mapa direto no
     s08. Coordenadas ficam FORA dos preditores (evitam atalho de
     autocorrelação espacial na CV). Grava a tabela matriz_treino
     (com fold espacial da seed principal) e o parquet.
  2. Classes por quantis (config) e alvo binário 'alta favorabilidade'
     = log_qs >= q75 (declarado).
  3. Baseline knowledge-driven: combinação fuzzy-gamma de 5 membros
     normalizados (esp_saprolito+, twi+, dens_drenagem+, hand-,
     declividade-), pesos iguais, gamma=0.9 — documentado e auditável.
  4. Modelos data-driven: RandomForest e XGBoost (classificador p/ AUC
     e regressor p/ log_qs), treinados no conjunto completo para o
     mapa; a avaliação honesta é papel do s07 (CV espacial).
  5. Interpretabilidade: importância por permutação -> CSV. SHAP saiu em
     08/11/2026: nao era citado no manuscrito e a versao instalada e
     incompativel com numpy 2.x.

VARIANTES (v3, 26/10/2026) — mesma convenção do s07/s08b/s08c:

  --variante pre_perfuracao  (default; ver config produtos.variante)
      Treina SEM as covariáveis de projeto do poço (profundidade final,
      comprimento de intervalo, tipo construtivo). Motivo: o mapa é
      entregue para decidir ONDE perfurar, e nenhuma dessas existe
      nesse momento. Treinar com elas e depois fixá-las num poço-padrão
      na predição (como fazia a v2.2) produz um mapa cuja ordenação
      nunca foi validada: o s07 media a AUC com as profundidades reais.
  --variante completo
      Conjunto completo, para comparação e para o modelo condicionado
      ao projeto do poço (quando a locação já está decidida).

Cada variante grava seus próprios arquivos com sufixo; 'completo'
mantém os nomes históricos, sem sufixo. A MATRIZ é a mesma nas duas
(as colunas excluídas continuam no parquet, apenas não entram como
preditores), portanto reescrevê-la é idempotente.

Saídas: matriz_treino (tabela + parquet), modelos .joblib em
/data/derived/modelos/, importância por permutação em CSV.

Execução:
  docker compose run --rm pipeline python stages/s06_treino.py
  docker compose run --rm pipeline python stages/s06_treino.py --variante completo
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from _comum import engine, config, DERIVED

MOD = DERIVED / "modelos"

# v3: covariáveis que só existem DEPOIS de perfurar. A variante
# pre_perfuracao as remove por definição.
CONSTRUTIVOS = ["profundidade_m", "lf_m", "tipo_filtros",
                "tipo_trecho_aberto", "tipo_coluna_saturada"]

PREDITORES = (["mde", "declividade", "northness", "curvatura", "tri",
               "twi", "hand", "dist_drenagem", "dens_drenagem",
               "esp_solo", "esp_saprolito", "esp_rocha_alterada",
               "prof_base_solo", "prof_topo_zona_rocha",
               "prof_topo_rocha_sa"]
              # v1.1: focais (robustez à incerteza posicional)
              + [f"{v}_f{r}" for v in ("declividade", "hand", "twi", "tri")
                 for r in (250, 500, 1000)]
              # v1.1: profundidade como covariável de CONTROLE — usada
              # apenas na variante 'completo' (ver docstring)
              + ["profundidade_m"])


def resolver_variante(arg=None):
    """Variante efetiva: argumento > config > pre_perfuracao."""
    cfg = config().get("produtos", {}) or {}
    cfg_cls = cfg.get("classes", {}) or {}
    v = (arg or cfg.get("variante") or cfg_cls.get("variante")
         or "pre_perfuracao")
    if v not in ("completo", "pre_perfuracao"):
        raise SystemExit(f"Variante desconhecida: {v}")
    return v, ("" if v == "completo" else f"_{v}")


def montar_matriz(excluir=()):
    """Devolve (matriz, lista de preditores da variante).

    v3: deixou de alterar a global PREDITORES — a lista efetiva é
    devolvida, para que duas variantes possam coexistir no mesmo
    processo sem contaminar uma à outra.
    """
    eng = engine()
    cov = pd.read_parquet(DERIVED / "covariaveis_pocos.parquet")
    with eng.connect() as conn:
        alvo = pd.read_sql(text("""
            SELECT codigo_siagas, log_qs
            FROM hidraulica_derivada WHERE log_qs IS NOT NULL"""), conn)
    m = alvo.merge(cov, on="codigo_siagas", how="inner")
    # v1.1: domínio (do s03) e profundidade; filtro fissural+outro
    hd = pd.read_parquet(DERIVED / "hidraulica_derivada.parquet")
    m = m.merge(hd[["codigo_siagas", "dominio", "lf_m", "tipo_intervalo"]],
                on="codigo_siagas", how="left")
    # v2.2: controles construtivos (presentes na matriz em qualquer
    # variante; entram como preditor apenas em 'completo')
    for t in ("filtros", "trecho_aberto", "coluna_saturada"):
        m[f"tipo_{t}"] = (m["tipo_intervalo"] == t).astype(int)
    with eng.connect() as conn:
        prof = pd.read_sql(text(
            "SELECT codigo_siagas, profundidade_m FROM pocos"), conn)
    m = m.merge(prof, on="codigo_siagas", how="left")
    n_antes = len(m)
    m = m[m["dominio"].isin(["fissural", "outro"])].reset_index(drop=True)
    print(f"Filtro de domínio (fissural+outro): {n_antes} -> {len(m)}")
    # v2: covariáveis estruturais (s05b) entram dinamicamente
    lin_cols = sorted(c for c in m.columns
                      if c.startswith(("dens_lin", "dist_lin", "geof_")))
    controles = ["lf_m", "tipo_filtros", "tipo_trecho_aberto",
                 "tipo_coluna_saturada"]
    preds = ([c for c in PREDITORES if c in m.columns]
             + lin_cols + controles)
    if lin_cols:
        print(f"Covariáveis estruturais incluídas: {len(lin_cols)}")
    faltantes = [c for c in preds if c not in m.columns]
    if faltantes:
        raise RuntimeError(f"Covariáveis ausentes: {faltantes} — rode o s05.")
    # v3: exclusão declarada da variante — erro se o nome não existir,
    # para que um preditor grafado errado nao seja "excluido" em silencio
    if excluir:
        desconhecidos = [c for c in excluir if c not in preds]
        if desconhecidos:
            raise RuntimeError(
                f"--excluir com preditores desconhecidos: {desconhecidos}")
        preds = [c for c in preds if c not in excluir]
    # imputação simples por mediana (declarada) — aplicada a TODAS as
    # colunas candidatas, para que a matriz sirva as duas variantes
    for c in dict.fromkeys(list(preds) + CONSTRUTIVOS):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
            m[c] = m[c].fillna(m[c].median())
    return m, preds


def blocos_espaciais(m: pd.DataFrame, bloco_km: float, n_folds: int,
                     seed: int) -> np.ndarray:
    """Blocos quadrados de bloco_km, distribuídos entre folds."""
    import geopandas as gpd
    g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m.lon, m.lat),
                         crs="EPSG:4674").to_crs(epsg=31983)
    bx = (g.geometry.x // (bloco_km * 1000)).astype(int)
    by = (g.geometry.y // (bloco_km * 1000)).astype(int)
    bloco_id = (bx.astype(str) + "_" + by.astype(str)).values
    rng = np.random.default_rng(seed)
    unicos = np.array(sorted(set(bloco_id)))
    rng.shuffle(unicos)
    fold_do_bloco = {b: i % n_folds for i, b in enumerate(unicos)}
    return np.array([fold_do_bloco[b] for b in bloco_id])


def baseline_fuzzy(m: pd.DataFrame, gamma: float = 0.9) -> np.ndarray:
    """Combinação fuzzy-gamma de membros normalizados [0,1]."""
    def norm01(s, inverter=False):
        s = pd.to_numeric(s, errors="coerce")
        lo, hi = s.quantile(0.02), s.quantile(0.98)
        v = ((s - lo) / (hi - lo)).clip(0, 1).fillna(0.5).values
        return 1 - v if inverter else v
    membros = np.column_stack([
        norm01(m["esp_saprolito"]),
        norm01(m["twi"]),
        norm01(m["dens_drenagem"]),
        norm01(m["hand"], inverter=True),
        norm01(m["declividade"], inverter=True),
    ])
    soma = 1 - np.prod(1 - membros, axis=1)      # fuzzy OR
    produto = np.prod(membros, axis=1)           # fuzzy AND
    return (soma ** gamma) * (produto ** (1 - gamma))


def main():
    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.inspection import permutation_importance
    import xgboost as xgb

    # v3: argparse DENTRO de main() — o s07 importa blocos_espaciais e
    # baseline_fuzzy deste modulo, e um parse_args em nivel de modulo
    # engoliria os argumentos do estagio que importa
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", default=None,
                    help="pre_perfuracao | completo")
    ap.add_argument("--excluir", nargs="*", default=None,
                    help="preditores extras a remover nesta variante")
    args = ap.parse_args()
    variante, sufixo = resolver_variante(args.variante)

    excluir = list(args.excluir or [])
    if variante == "pre_perfuracao":
        excluir = list(dict.fromkeys(CONSTRUTIVOS + excluir))

    cfg = config()
    quantis = cfg["modelagem"]["classes_quantis"]
    seed = cfg["validacao"]["seeds"][0]
    n_folds = cfg["validacao"]["n_folds"]
    bloco_km = float(cfg["validacao"].get("bloco_km", 10))

    print(f"Variante: {variante}"
          + (f" | excluindo {excluir}" if excluir else ""))

    m, preds = montar_matriz(excluir=excluir)
    q = m["log_qs"].quantile(quantis).values
    m["classe"] = pd.cut(m["log_qs"],
                         [-np.inf, *q, np.inf],
                         labels=["baixa", "media", "alta", "muito_alta"])
    m["alvo_binario"] = (m["log_qs"] >= q[-1]).astype(int)
    m["fold_espacial"] = blocos_espaciais(m, bloco_km, n_folds, seed)
    m["score_fuzzy"] = baseline_fuzzy(m)

    X, y_reg, y_cls = m[preds], m["log_qs"], m["alvo_binario"]

    # ---- Modelos finais (conjunto completo — avaliação é no s07) ----
    MOD.mkdir(parents=True, exist_ok=True)
    modelos = {
        "rf_cls": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=seed,
            n_jobs=-1),
        "rf_reg": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, random_state=seed,
            n_jobs=-1),
        "xgb_cls": xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            eval_metric="auc"),
        "xgb_reg": xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed),
    }
    escritos = []
    for nome, mod in modelos.items():
        alvo = y_cls if nome.endswith("cls") else y_reg
        mod.fit(X, alvo)
        destino = MOD / f"{nome}{sufixo}.joblib"
        joblib.dump(mod, destino)
        escritos.append(destino.name)

    # ---- Interpretabilidade ----
    pi = permutation_importance(modelos["rf_cls"], X, y_cls,
                                n_repeats=10, random_state=seed, n_jobs=-1)
    imp = (pd.DataFrame({"covariavel": preds,
                         "importancia_permutacao": pi.importances_mean})
           .sort_values("importancia_permutacao", ascending=False))
    imp.to_csv(MOD / f"importancia_permutacao{sufixo}.csv", index=False)
    escritos.append(f"importancia_permutacao{sufixo}.csv")

    # ---- Persistência da matriz ----
    # A matriz é idêntica nas duas variantes (as colunas excluídas
    # continuam aqui), portanto reescrevê-la é idempotente.
    m.to_parquet(DERIVED / "matriz_treino.parquet", index=False)
    import json
    (MOD / f"preditores{sufixo}.json").write_text(json.dumps(preds, indent=1))
    escritos.append(f"preditores{sufixo}.json")
    with engine().begin() as conn:
        conn.execute(text("DELETE FROM matriz_treino"))
        for _, r in m.iterrows():
            conn.execute(text("""
                INSERT INTO matriz_treino
                  (codigo_siagas, alvo_log_qs, classe, fold_espacial)
                VALUES (:c,:a,:k,:f)"""),
                {"c": r["codigo_siagas"], "a": float(r["log_qs"]),
                 "k": str(r["classe"]), "f": int(r["fold_espacial"])})

    # ---- Relatório ----
    print(f"\n===== RELATÓRIO s06 (treino) — variante '{variante}' =====")
    print(f"Poços na matriz: {len(m)}")
    print(f"Preditores: {len(preds)}"
          + (f" (excluídos: {excluir})" if excluir else ""))
    print(f"Limiares de classe (log10 Q/s): "
          f"{', '.join(f'{v:.2f}' for v in q)}")
    print(f"Alvo binário 'alta' (>= q75): {int(y_cls.sum())} positivos")
    print(f"Blocos espaciais de {bloco_km:.0f} km "
          f"-> {m['fold_espacial'].nunique()} folds")
    print("\nTop 8 covariáveis (importância por permutação):")
    for _, r in imp.head(8).iterrows():
        print(f"   {r['covariavel']:22s} {r['importancia_permutacao']:.4f}")
    print("\nArquivos escritos em data/derived/modelos/:")
    for nome in escritos:
        print(f"   {nome}")
    print("A avaliação honesta (CV espacial, AUC bloqueante) é o s07, "
          f"na MESMA variante:")
    print(f"   python stages/s07_validacao.py"
          + ("" if variante == "completo" else
             f" --variante {variante} --excluir " + " ".join(CONSTRUTIVOS)))


if __name__ == "__main__":
    main()