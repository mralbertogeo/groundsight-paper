"""s07_diag.py — Diagnóstico da reprovação do M6.

Quatro verificações:
  A) Integridade das covariáveis nos poços: estatísticas (min/mediana/
     max/nº de valores distintos) — detecta coluna constante, NaN
     imputado em massa ou valores fora de faixa física;
  B) Sanidade da amostragem espacial: cota do MDE nos poços vs faixa
     esperada, e correlação MDE × declividade no ponto (deve ser
     coerente, não aleatória);
  C) Correlação univariada de cada covariável com log10(Q/s);
  D) CV ALEATÓRIA vs CV ESPACIAL (mesmo modelo RF): o discriminador
     entre defeito técnico (ambas ~0,5) e sinal puramente local
     (aleatória >> espacial).

Execução:  docker compose run --rm pipeline python stages/s07_diag.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import DERIVED, config
from s06_treino import PREDITORES, baseline_fuzzy, blocos_espaciais

RELATORIO = Path("/data/outputs/s07_diag_relatorio.txt")


class Tee:
    """Duplica stdout para um arquivo (relatório persistente)."""

    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush()
        self.arquivo.flush()


def main():
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")
    X = m[PREDITORES].values
    y = m["alvo_binario"].values

    print("===== A) INTEGRIDADE DAS COVARIÁVEIS (nos 564 poços) =====")
    print(f"{'covariavel':22s} {'min':>10s} {'mediana':>10s} {'max':>10s} "
          f"{'n_distintos':>11s}")
    for c in PREDITORES:
        s = m[c]
        print(f"{c:22s} {s.min():10.2f} {s.median():10.2f} "
              f"{s.max():10.2f} {s.nunique():11d}")

    print("\n===== B) SANIDADE DA AMOSTRAGEM =====")
    print(f"Cota MDE nos poços: {m['mde'].min():.0f}–{m['mde'].max():.0f} m "
          f"(esperado ~550–1500)")
    corr_md = np.corrcoef(m["mde"], m["tri"])[0, 1]
    print(f"Correlação mde × tri no ponto: {corr_md:.2f} "
          f"(esperado positivo moderado — relevo mais alto é mais rugoso)")
    fz = baseline_fuzzy(m)
    print(f"Score fuzzy: min {fz.min():.3f}, max {fz.max():.3f}, "
          f"desvio {fz.std():.3f} (se ~0, memberships degenerados)")

    print("\n===== C) CORRELAÇÃO UNIVARIADA COM log10(Q/s) =====")
    corrs = {c: np.corrcoef(m[c], m["log_qs"])[0, 1] for c in PREDITORES}
    for c, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1])):
        print(f"{c:22s} {v:+.3f}")

    print("\n===== D) CV ALEATÓRIA vs ESPACIAL =====")
    cfg = config()["validacao"]

    def auc_cv(folds_iter):
        pred = np.zeros(len(m))
        for tr, te in folds_iter:
            rf = RandomForestClassifier(n_estimators=500,
                                        min_samples_leaf=3,
                                        random_state=42, n_jobs=-1)
            rf.fit(X[tr], y[tr])
            pred[te] = rf.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, pred)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_rand = auc_cv(skf.split(X, y))

    folds = blocos_espaciais(m, float(cfg.get("bloco_km", 10)),
                             cfg["n_folds"], 42)
    esp_iter = [(folds != f, folds == f) for f in range(cfg["n_folds"])]
    auc_esp = auc_cv(esp_iter)

    print(f"AUC CV aleatória (estratificada): {auc_rand:.3f}")
    print(f"AUC CV espacial (blocos 10 km):   {auc_esp:.3f}")
    print("""
Leitura:
  ambas ~0,5      -> covariáveis sem informação (defeito técnico ou
                     ausência total de relação) — investigar A/B/C;
  aleatória >>    -> dados íntegros; o sinal existe mas é apenas
  espacial           autocorrelação local, não transferível — caminho é
                     enriquecer preditores (lineamentos etc.), não
                     afrouxar a validação.""")
    print(f"\nRelatório completo salvo em: data/outputs/s07_diag_relatorio.txt")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
