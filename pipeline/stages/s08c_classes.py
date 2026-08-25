"""s08c — Classes de favorabilidade (5 graus) com validação por classe.

PRÉ-REGISTRO v2 (revisão declarada em 13/08/2026, após o resultado
da v1 no VP e ANTES de qualquer número da v2):
  1. Quebras = QUINTIS DAS PROBABILIDADES OUT-OF-FOLD DOS POÇOS
     (percentis 20/40/60/80) -> muito_baixa..muito_alta. Justificativa
     da revisão: quintis de ÁREA (v1) diluíram a classe muito_alta —
     os poços concentram-se nas zonas favoráveis, e a classe com 20%
     do território engoliu a maioria deles, aproximando sua taxa da
     taxa-base (lift 1,15x no VP). Classes devem particionar o espaço
     das DECISÕES DE PERFURAÇÃO validadas; a fração de área por
     classe passa a ser REPORTADA, não imposta. v1 fica registrada
     como testada e revisada.
  2. Validação por classe com predições OUT-OF-FOLD da CV espacial
     (blocos de 10 km, 5 folds, seed 42): para cada classe, taxa
     observada de poços de alta produtividade e lift sobre a taxa-base
     (25%). A legenda do produto carrega esses números.
  3. Checagem de monotonicidade: as taxas observadas devem crescer de
     muito_baixa a muito_alta; inversões são REPORTADAS.

Saídas: classes_favorabilidade.tif (1..5, com quebras e taxas nos
metadados), quicklook com legenda validada, relatório txt.

Execução:  docker compose run --rm pipeline python stages/s08c_classes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _comum import config, DERIVED

OUT = Path("/data/outputs")
RELATORIO = OUT / "s08c_relatorio.txt"
CLASSES = ["muito_baixa", "baixa", "media", "alta", "muito_alta"]
CORES = {1: (165, 0, 38), 2: (244, 109, 67), 3: (254, 224, 139),
         4: (166, 217, 106), 5: (0, 104, 55)}
DOMINIOS = ["fissural", "outro"]


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def achar_raster_prob(cfg):
    """Localiza a superfície de probabilidade do s08b."""
    manual = cfg.get("classes", {}).get("arquivo_prob")
    if manual:
        p = Path(manual)
        if p.exists():
            return p
        raise RuntimeError(f"arquivo_prob configurado não existe: {p}")
    candidatos = []
    for padrao in ("prob*alta*.tif", "*prob*.tif", "*favorab*.tif"):
        for base in (DERIVED / "produtos", DERIVED / "covariaveis",
                     DERIVED):
            candidatos += sorted(base.glob(padrao))
    candidatos = [c for c in dict.fromkeys(candidatos)
                  if "classe" not in c.name and "incert" not in c.name]
    if not candidatos:
        raise RuntimeError(
            "Nenhum raster de probabilidade encontrado em derived/ — "
            "configure produtos.classes.arquivo_prob no config.yaml")
    if len(candidatos) > 1:
        print("Candidatos encontrados (usando o 1º; configure "
              "produtos.classes.arquivo_prob para trocar):")
        for c in candidatos:
            print(f"   {c}")
    return candidatos[0]


def oof_probabilidades():
    """Predições out-of-fold da CV espacial (mesmo protocolo do s07)."""
    from sklearn.ensemble import RandomForestClassifier

    cov = pd.read_parquet(DERIVED / "covariaveis_pocos.parquet")
    hid = pd.read_parquet(DERIVED / "hidraulica_derivada.parquet")
    dcol = next(c for c in hid.columns if "domin" in c.lower())
    alvo = next(c for c in ("log_qs", "alvo_log_qs") if c in hid.columns)
    m = cov.merge(hid[["codigo_siagas", alvo, dcol]], on="codigo_siagas")
    m = m[m[dcol].isin(DOMINIOS)].dropna(subset=[alvo])
    num = [c for c in m.columns
           if c not in ("codigo_siagas", alvo, dcol)
           and pd.api.types.is_numeric_dtype(m[c])
           and m[c].notna().mean() >= 0.6]
    X = m[num].fillna(m[num].median()).values
    y = (m[alvo] >= m[alvo].quantile(.75)).values

    # blocos espaciais de 10 km (precisa de x/y)
    import geopandas as gpd
    if not {"lon", "lat"}.issubset(m.columns):
        from sqlalchemy import text
        from _comum import engine
        with engine().connect() as conn:
            pos = pd.read_sql(text(
                "SELECT codigo_siagas, ST_X(geom) lon, ST_Y(geom) lat "
                "FROM pocos WHERE qaqc_aprovado"), conn)
        m = m.merge(pos, on="codigo_siagas")
    g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m["lon"], m["lat"]),
                         crs="EPSG:4674").to_crs(epsg=31983)
    bx = (g.geometry.x // 10_000).astype(int)
    by = (g.geometry.y // 10_000).astype(int)
    ids = bx.astype(str) + "_" + by.astype(str)
    rng = np.random.default_rng(42)
    unicos = np.array(sorted(ids.unique())); rng.shuffle(unicos)
    fold = ids.map({b: i % 5 for i, b in enumerate(unicos)}).values

    oof = np.full(len(m), np.nan)
    for f in range(5):
        tr, te = fold != f, fold == f
        clf = RandomForestClassifier(n_estimators=500, random_state=42,
                                     n_jobs=-1, min_samples_leaf=2)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof, y


def main():
    import rasterio
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    print("===== s08c — CLASSES DE FAVORABILIDADE (5 graus) =====")
    cfg = config().get("produtos", {})

    raster = achar_raster_prob(cfg)
    print(f"Superfície de probabilidade: {raster}")
    with rasterio.open(raster) as src:
        prob = src.read(1).astype("float64")
        perfil = src.profile

    validos = np.isfinite(prob)

    # ---------- OOF primeiro: as quebras vêm dos POÇOS (v2) ----------
    print("\nCalculando predições out-of-fold (CV espacial)...")
    oof, y = oof_probabilidades()
    ok = np.isfinite(oof)
    quebras = np.quantile(oof[ok], [0.2, 0.4, 0.6, 0.8])
    print("Quebras (quintis das probabilidades OOF dos poços): "
          + " | ".join(f"{q:.3f}" for q in quebras))

    classes = np.full(prob.shape, 0, dtype="uint8")
    classes[validos] = np.digitize(prob[validos], quebras) + 1

    print("\nValidação por classe (out-of-fold):")
    cls_pocos = np.digitize(oof[ok], quebras) + 1
    base = y[ok].mean()
    linhas = []
    print(f"{'classe':>12s} {'n poços':>8s} {'% área':>7s} "
          f"{'taxa alta':>10s} {'lift':>6s}")
    taxas = []
    for c in range(1, 6):
        sel = cls_pocos == c
        n = int(sel.sum())
        taxa = float(y[ok][sel].mean()) if n else float("nan")
        area = 100 * float((classes == c).sum()) / validos.sum()
        lift = taxa / base if n else float("nan")
        taxas.append(taxa)
        linhas.append((CLASSES[c - 1], n, area, taxa, lift))
        print(f"{CLASSES[c-1]:>12s} {n:8d} {area:6.1f}% "
              f"{100*taxa:9.1f}% {lift:5.1f}x")
    print(f"Taxa-base (fração de poços 'alta'): {100*base:.1f}%")

    mono = all(a <= b + 0.02 for a, b in zip(taxas[:-1], taxas[1:]))
    print("Monotonicidade: "
          + ("OK — taxas crescem de muito_baixa a muito_alta"
             if mono else
             "VIOLADA — inversão de taxa entre classes (REPORTAR: "
             "as quebras por quintil de poços não ordenam o risco nesta área)"))

    # ---------- gravação ----------
    perfil.update(dtype="uint8", nodata=0)
    destino = raster.parent / "classes_favorabilidade.tif"
    with rasterio.open(destino, "w", **perfil) as dst:
        dst.write(classes, 1)
        dst.write_colormap(1, CORES)
        dst.update_tags(
            QUEBRAS="|".join(f"{q:.4f}" for q in quebras),
            METODO="quintis das probabilidades OOF dos pocos (pre-registro v2)",
            VALIDACAO=" ; ".join(
                f"{n}: taxa {100*t:.0f}% lift {l:.1f}x"
                for n, _, _, t, l in linhas),
            MONOTONICIDADE="OK" if mono else "VIOLADA")
    print(f"\nRaster de classes: {destino}")

    # ---------- quicklook ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([tuple(v / 255 for v in CORES[c])
                               for c in range(1, 6)])
        fig, axs = plt.subplots(1, 2, figsize=(15, 6.5),
                                gridspec_kw={"width_ratios": [1.35, 1]})
        mostra = np.ma.masked_equal(classes, 0)
        axs[0].imshow(mostra, cmap=cmap, vmin=1, vmax=5)
        axs[0].set_title("Classes de favorabilidade (quintis validados)")
        axs[0].axis("off")
        nomes = [l[0] for l in linhas]
        axs[1].bar(nomes, [100 * l[3] for l in linhas],
                   color=[tuple(v / 255 for v in CORES[c])
                          for c in range(1, 6)])
        axs[1].axhline(100 * base, ls="--", c="k", lw=1,
                       label=f"taxa-base {100*base:.0f}%")
        axs[1].set_ylabel("Taxa observada de poços de alta produtividade (%)")
        axs[1].set_title("Validação por classe (out-of-fold)")
        axs[1].legend()
        plt.setp(axs[1].get_xticklabels(), rotation=20, ha="right")
        fig.savefig(OUT / "quicklook_classes.png", dpi=120,
                    bbox_inches="tight")
        print("Quicklook: data/outputs/quicklook_classes.png")
    except Exception as e:
        print(f"Quicklook não gerado: {e}")

    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
