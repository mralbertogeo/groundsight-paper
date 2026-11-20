"""s08c — Classes de favorabilidade (5 graus) com validação por classe.

PRÉ-REGISTRO v2 (13/08/2026): quebras = quintis das probabilidades
out-of-fold dos poços (percentis 20/40/60/80). A fração de ÁREA por
classe é REPORTADA, não imposta. Mantido.

REVISÃO v3 (25/10/2026) — declarada após a auditoria de 25/10 e antes
de qualquer número novo. Três defeitos corrigidos:

  1. O estágio TREINAVA UM MODELO PRÓPRIO para obter as probabilidades
     out-of-fold, diferente do que o s07 valida e do que o s08b usa no
     mapa. Eram três modelos distintos no mesmo pipeline, e a tabela de
     taxas — o número que sustenta a proposta de valor — vinha do único
     que ninguém validava. Agora o estágio CONSOME o OOF gravado pelo
     s07 (10 sementes, grade aninhada, preditores de preditores.json).
  2. Aquele modelo usava `lon` e `lat` como preditores (o filtro por
     tipo numérico não as excluía). Num produto de mapa isso é um
     interpolador de coordenadas disfarçado de modelo de covariáveis.
     Eliminado junto com o refit.
  3. A variante padrão passa a ser PRE_PERFURACAO: o mapa é entregue
     para decidir ONDE perfurar, logo tem de ser validado só com o que
     se sabe ANTES de perfurar. Profundidade final, comprimento de
     filtros e tipo de intervalo não existem nesse momento.

Convenções mantidas: probabilidade de referência = RF (p_rf),
que é exatamente o que o s08b grava em favorabilidade_prob.tif;
monotonicidade reportada, nunca imposta.

Acrescentado: IC95% das taxas por reamostragem de blocos espaciais
(mesma unidade da validação cruzada) e dispersão entre sementes.

AVISO DE CONSISTÊNCIA: enquanto o s06/s08b não forem regerados na mesma
variante, o RASTER vem do modelo completo com poço-padrão e as QUEBRAS
vêm do OOF da variante escolhida. As TAXAS por classe (medidas nos
poços) são válidas; a atribuição de classe no raster é provisória. O
estágio avisa em toda execução.

Execução:
  docker compose run --rm pipeline python stages/s08c_classes.py
  docker compose run --rm pipeline python stages/s08c_classes.py \
      --variante completo --modelo rf
"""
import argparse
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
REPS = 2000
SEMENTE_BOOT = 20260101


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


def carregar_oof(variante, coluna):
    """Probabilidades out-of-fold gravadas pelo s07 (v3)."""
    sufixo = "" if variante == "completo" else f"_{variante}"
    caminho = DERIVED / f"validacao_oof{sufixo}.parquet"
    if not caminho.exists():
        sys.exit(
            f"Nao encontrei {caminho.name}.\n"
            f"Rode antes:\n"
            f"  python stages/s07_validacao.py"
            + ("" if variante == "completo" else
               f" --variante {variante} --excluir profundidade_m lf_m "
               "tipo_filtros tipo_trecho_aberto tipo_coluna_saturada")
            + "\n(o s07 grava validacao_oof*.parquet a partir de 25/10/2026)")
    oof = pd.read_parquet(caminho)
    if coluna not in oof.columns:
        sys.exit(f"Coluna {coluna} ausente em {caminho.name}")
    seeds = sorted(oof["seed"].unique())
    prim = oof[oof["seed"] == seeds[0]]
    y = prim["y"].values.astype(int)
    lon = prim["lon"].values
    lat = prim["lat"].values
    P = np.column_stack([oof[oof["seed"] == s][coluna].values
                         for s in seeds])
    return y, P, lon, lat, seeds


def taxas_por_classe(y, p, quebras):
    """Taxa observada em cada classe, dadas quebras fixas."""
    cls = np.digitize(p, quebras) + 1
    return np.array([y[cls == c].mean() if (cls == c).any() else np.nan
                     for c in range(1, 6)]), cls


def main():
    import rasterio
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", default=None,
                    help="completo | pre_perfuracao (default: config ou "
                         "pre_perfuracao)")
    ap.add_argument("--modelo", default=None,
                    help="ens | rf | xgb (default: ens, igual ao raster)")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    print("===== s08c — CLASSES DE FAVORABILIDADE (5 graus) =====")
    cfg_prod = config().get("produtos", {})
    cfg_cls = cfg_prod.get("classes", {}) or {}
    cfg_val = config().get("validacao", {})

    variante = (args.variante or cfg_prod.get("variante")
                or cfg_cls.get("variante") or "pre_perfuracao")
    modelo = args.modelo or cfg_cls.get("modelo", "rf")
    coluna = {"ens": "p_ens", "rf": "p_rf", "xgb": "p_xgb"}[modelo]
    bloco_km = float(cfg_val.get("bloco_km", 10))
    print(f"Variante de validacao: {variante} | probabilidade: {modelo} "
          f"({coluna})")
    if variante == "pre_perfuracao":
        print("  -> classes validadas SO com informacao disponivel antes "
              "de perfurar")
    else:
        print("  -> ATENCAO: inclui covariaveis de projeto do poco "
              "(profundidade, filtros),")
        print("     que o cliente NAO tem ao decidir onde perfurar. "
              "As taxas abaixo nao")
        print("     descrevem o que um mapa entrega.")

    raster = achar_raster_prob(cfg_prod)
    print(f"\nSuperficie de probabilidade: {raster}")
    with rasterio.open(raster) as src:
        prob = src.read(1).astype("float64")
        perfil = src.profile
        tag_var = src.tags().get("VARIANTE")
        tag_mod = src.tags().get("MODELO")
    validos = np.isfinite(prob)
    # v3: o raster declara a variante do modelo que o gerou. Quebras
    # derivadas de uma variante aplicadas ao raster de outra atribuem
    # classes erradas — melhor parar do que entregar mapa inconsistente.
    if tag_var is None:
        print("AVISO: o raster nao declara VARIANTE (foi gerado antes de "
              "26/10/2026).")
        print("As TAXAS abaixo sao validas (medidas nos pocos), mas a "
              "atribuicao de classe")
        print("no raster e provisoria. Regere o s08b nesta variante.")
    elif tag_var != variante:
        sys.exit(
            f"INCONSISTENCIA: o raster foi gerado na variante "
            f"'{tag_var}' e as quebras viriam da variante "
            f"'{variante}'.\nRode:\n"
            f"  python stages/s08b_mapa_exploratorio.py --variante "
            f"{variante}\nou passe --variante {tag_var} aqui.")
    else:
        print(f"Raster e quebras na mesma variante ({variante}) — "
              "consistente.")

    if tag_mod is None:
        print("AVISO: o raster nao declara MODELO (gerado antes de "
              "13/11/2026). Regere o s08b.")
    elif tag_mod != modelo:
        sys.exit(
            f"INCONSISTENCIA: o raster foi gerado com o modelo "
            f"'{tag_mod}' e as taxas viriam do modelo '{modelo}'.\n"
            f"Rode:\n  python stages/s08b_mapa_exploratorio.py "
            f"--variante {variante}\nou passe --modelo {tag_mod} aqui.")
    else:
        print(f"Raster e taxas no mesmo modelo ({modelo}) — consistente.")

    # ---------- OOF do s07 ----------
    y, P, lon, lat, seeds = carregar_oof(variante, coluna)
    n = len(y)
    p = P.mean(axis=1)          # media entre sementes = o mapa entregue
    base = float(y.mean())
    print(f"OOF do s07: {n} pocos, {len(seeds)} sementes {seeds}")

    quebras = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
    print("Quebras (quintis da probabilidade OOF media dos pocos): "
          + " | ".join(f"{q:.3f}" for q in quebras))

    classes = np.full(prob.shape, 0, dtype="uint8")
    classes[validos] = np.digitize(prob[validos], quebras) + 1

    taxas, cls_pocos = taxas_por_classe(y, p, quebras)

    # ---------- dispersao entre sementes (quebras fixas) ----------
    por_seed = np.array([taxas_por_classe(y, P[:, j], quebras)[0]
                         for j in range(P.shape[1])])

    # ---------- IC por reamostragem de blocos espaciais ----------
    import geopandas as gpd
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat),
                         crs="EPSG:4674").to_crs(epsg=31983)
    bx = (g.geometry.x // (bloco_km * 1000)).astype(int).astype(str)
    by = (g.geometry.y // (bloco_km * 1000)).astype(int).astype(str)
    _, bidx = np.unique(np.char.add(np.char.add(bx.values, "_"), by.values),
                        return_inverse=True)
    nb = int(bidx.max()) + 1
    por_bloco = [np.where(bidx == b)[0] for b in range(nb)]
    rng = np.random.default_rng(SEMENTE_BOOT)
    dist = np.full((args.reps, 5), np.nan)
    dist_base = np.full(args.reps, np.nan)
    for r in range(args.reps):
        idx = np.concatenate([por_bloco[b] for b in
                              rng.integers(0, nb, size=nb)])
        if len(set(y[idx])) < 2:
            continue
        dist[r] = taxas_por_classe(y[idx], p[idx], quebras)[0]
        dist_base[r] = y[idx].mean()
    lo, hi = np.nanpercentile(dist, [2.5, 97.5], axis=0)
    ganho5 = dist[:, 4] / dist_base
    ganho5 = ganho5[np.isfinite(ganho5)]
    g5lo, g5hi = np.nanpercentile(ganho5, [2.5, 97.5])
    print(f"IC95% por reamostragem de {nb} blocos de {bloco_km:.0f} km, "
          f"{args.reps} repeticoes")

    # ---------- tabela ----------
    print("\nValidacao por classe (out-of-fold do s07):")
    print(f"{'classe':>12s} {'n pocos':>8s} {'% area':>7s} "
          f"{'taxa alta':>10s} {'IC95%':>16s} {'lift':>6s} "
          f"{'entre sementes':>16s}")
    linhas = []
    for c in range(1, 6):
        sel = cls_pocos == c
        npoc = int(sel.sum())
        taxa = float(taxas[c - 1])
        area = 100 * float((classes == c).sum()) / validos.sum()
        lift = taxa / base if npoc else float("nan")
        smin, smax = por_seed[:, c - 1].min(), por_seed[:, c - 1].max()
        linhas.append({"classe": CLASSES[c - 1], "n_pocos": npoc,
                       "pct_area": area, "taxa": taxa,
                       "ic_lo": lo[c - 1], "ic_hi": hi[c - 1],
                       "lift": lift, "seed_min": smin, "seed_max": smax})
        print(f"{CLASSES[c-1]:>12s} {npoc:8d} {area:6.1f}% "
              f"{100*taxa:9.1f}% "
              f"{f'{100*lo[c-1]:.1f}-{100*hi[c-1]:.1f}%':>16s} "
              f"{lift:5.2f}x {f'{100*smin:.1f}-{100*smax:.1f}%':>16s}")
    print(f"Taxa-base (fracao de pocos 'alta'): {100*base:.1f}%")

    # ---------- leitura comercial ----------
    t5 = float(taxas[4]); t1 = float(taxas[0])
    print("\nLeitura operacional:")
    print(f"  Ganho da classe muito_alta sobre a base: "
          f"{t5/base:.2f}x [{g5lo:.2f}-{g5hi:.2f}]")
    print(f"  Pocos a perfurar para 10 acertos: {10/t5:.0f} na classe "
          f"muito_alta | {10/base:.0f} ao acaso"
          + (f" | {10/t1:.0f} na classe muito_baixa" if t1 > 0 else ""))
    rec = t5 * (cls_pocos == 5).sum() / max(y.sum(), 1)
    print(f"  Dos pocos de alta favorabilidade, {100*rec:.0f}% caem na "
          f"classe muito_alta")
    print("  Ressalva: taxas medidas EM POCOS existentes, que estao onde "
          "ja se perfurou;")
    print("  a taxa numa celula arbitraria da mesma classe pode diferir.")

    mono = all(a <= b + 0.02 for a, b in zip(taxas[:-1], taxas[1:]))
    print("\nMonotonicidade: "
          + ("OK — taxas crescem de muito_baixa a muito_alta"
             if mono else
             "VIOLADA — inversao de taxa entre classes (REPORTAR: "
             "as quebras por quintil de pocos nao ordenam o risco nesta area)"))

    # ---------- gravacao ----------
    perfil.update(dtype="uint8", nodata=0)
    destino = raster.parent / "classes_favorabilidade.tif"
    with rasterio.open(destino, "w", **perfil) as dst:
        dst.write(classes, 1)
        dst.write_colormap(1, CORES)
        dst.update_tags(
            QUEBRAS="|".join(f"{q:.4f}" for q in quebras),
            METODO=("quintis da probabilidade OOF do s07 "
                    f"(variante {variante}, {modelo}; pre-registro v3)"),
            VARIANTE=variante,
            VALIDACAO=" ; ".join(
                f"{l['classe']}: taxa {100*l['taxa']:.0f}% "
                f"(IC {100*l['ic_lo']:.0f}-{100*l['ic_hi']:.0f}%) "
                f"lift {l['lift']:.2f}x" for l in linhas),
            TAXA_BASE=f"{100*base:.1f}%",
            MONOTONICIDADE="OK" if mono else "VIOLADA")
    print(f"\nRaster de classes: {destino}")

    pd.DataFrame(linhas).to_csv(OUT / "s08c_taxas_classe.csv", index=False)
    print(f"CSV: {OUT / 's08c_taxas_classe.csv'}")

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
        axs[0].imshow(np.ma.masked_equal(classes, 0), cmap=cmap,
                      vmin=1, vmax=5)
        axs[0].set_title("Classes de favorabilidade "
                         f"(quintis validados — {variante})")
        axs[0].axis("off")
        nomes = [l["classe"] for l in linhas]
        vals = np.array([100 * l["taxa"] for l in linhas])
        err = np.vstack([vals - 100 * lo, 100 * hi - vals])
        axs[1].bar(nomes, vals, yerr=err, capsize=4,
                   color=[tuple(v / 255 for v in CORES[c])
                          for c in range(1, 6)])
        axs[1].axhline(100 * base, ls="--", c="k", lw=1,
                       label=f"taxa-base {100*base:.0f}%")
        axs[1].set_ylabel("Taxa observada de poços de alta produtividade (%)")
        axs[1].set_title(f"Validação por classe (OOF, {len(seeds)} sementes, "
                         f"IC95% por bloco)")
        axs[1].legend()
        plt.setp(axs[1].get_xticklabels(), rotation=20, ha="right")
        fig.savefig(OUT / "quicklook_classes.png", dpi=120,
                    bbox_inches="tight")
        print("Quicklook: data/outputs/quicklook_classes.png")
    except Exception as e:
        print(f"Quicklook nao gerado: {e}")

    print(f"Relatorio salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()