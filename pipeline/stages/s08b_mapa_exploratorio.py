"""s08b — Mapa EXPLORATÓRIO de favorabilidade (Fase A).

Gera o mapa de probabilidade de alta favorabilidade (quartil superior
de Q/s) aplicando os modelos treinados (s06) ao grid completo, com a
validação DECLARADA no próprio produto.

VARIANTES (v3, 26/10/2026) — mesma convenção do s06/s07/s08c:

  pre_perfuracao (default)  usa os modelos treinados SÓ com informação
      disponível antes de perfurar. Não há controle construtivo a
      fixar: o mapa responde "quão bom seria um poço aqui" e a AUC que
      o acompanha foi medida nessas mesmas condições.
  completo  usa os modelos do conjunto completo e fixa os controles
      construtivos no poço-padrão (comportamento da v2.2). O aviso
      declara que a ordenação resultante não foi validada nessas
      condições — o s07 mede a AUC com as profundidades reais.

Mapa = RandomForest (v4, 13/11/2026: o ensemble foi abandonado porque
o RF domina em AUC e em Brier nas duas areas e nas duas variantes;
o XGBoost tem Brier skill score NEGATIVO em todas, isto e, as
probabilidades dele sao piores que prever a taxa base); camadas de
incerteza:
  (a) divergência entre modelos |p_RF − p_XGB|;
  (b) distância ao poço de treino mais próximo (suporte amostral).

Os rasters mantêm o NOME CANÔNICO em qualquer variante (GeoTIFF de AOI
é grande e duplicar não se justifica) e passam a carregar a tag
VARIANTE, que o s08c confere antes de derivar as classes.

Correções de v3 no aviso do produto:
  - `teto_vizinho_auc` era lido de config()["produtos"], onde a chave
    não existe — todo raster saiu com o default 0,672, errado na Área 2
    (0,752). A fração "do teto" saiu do aviso: a revisão de 25/10 (R3)
    mostrou que a referência de vizinhança é superável pelo próprio
    modelo e que a razão tem intervalo largo demais para ir num rótulo.
  - a AUC externa estava cravada em "0,59" no código. Saiu.

Saídas (data/derived/covariaveis/ + outputs/): favorabilidade_prob.tif,
incerteza_modelos.tif, dist_poco_treino.tif, quicklook e relatório.

Execução:
  docker compose run --rm pipeline python stages/s08b_mapa_exploratorio.py
  docker compose run --rm pipeline python stages/s08b_mapa_exploratorio.py --variante completo
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from _comum import config, DERIVED

OUT = Path("/data/outputs")
COV = DERIVED / "covariaveis"
GEO = DERIVED / "geologia_3d"
MOD = DERIVED / "modelos"
RELATORIO = OUT / "s08b_relatorio.txt"

AVISO = ("EXPLORATORIO - variante {variante} | AUC espacial {auc:.3f} "
         "(media de {ns} sementes, CV em blocos de {bloco:.0f} km) - "
         "triagem regional; nao substitui investigacao local")

FIXOS_POCO_PADRAO = {"profundidade_m": None, "lf_m": None,
                     "tipo_filtros": 0.0, "tipo_trecho_aberto": 1.0,
                     "tipo_coluna_saturada": 0.0}


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def resolver_variante(arg=None):
    cfg = config().get("produtos", {}) or {}
    cfg_cls = cfg.get("classes", {}) or {}
    v = (arg or cfg.get("variante") or cfg_cls.get("variante")
         or "pre_perfuracao")
    if v not in ("completo", "pre_perfuracao"):
        raise SystemExit(f"Variante desconhecida: {v}")
    return v, ("" if v == "completo" else f"_{v}")


def carregar_raster(nome, shape, transform, epsg):
    import rasterio
    from rasterio.warp import reproject, Resampling
    for base in (COV, GEO):
        p = base / f"{nome}.tif"
        if p.exists():
            with rasterio.open(p) as src:
                if (src.transform == transform and
                        src.shape == shape):
                    a = src.read(1).astype("float32")
                    a[a == src.nodata] = np.nan
                    return a
                destino = np.full(shape, np.nan, dtype="float32")
                reproject(rasterio.band(src, 1), destino,
                          dst_transform=transform,
                          dst_crs=f"EPSG:{epsg}",
                          resampling=Resampling.bilinear,
                          dst_nodata=np.nan)
                return destino
    return None


def main():
    import rasterio
    from scipy.spatial import cKDTree
    from s05_covariaveis import salvar

    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", default=None,
                    help="pre_perfuracao | completo")
    args = ap.parse_args()
    variante, sufixo = resolver_variante(args.variante)

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    cfg = config().get("produtos", {}) or {}
    cfg_val = config().get("validacao", {}) or {}
    bloco_km = float(cfg_val.get("bloco_km", 10))

    with rasterio.open(COV / "mde.tif") as src:
        shape = (src.height, src.width)
        transform, epsg = src.transform, src.crs.to_epsg()
        res = transform.a

    arq_preds = MOD / f"preditores{sufixo}.json"
    if not arq_preds.exists():
        sys.exit(f"Nao encontrei {arq_preds.name}.\nRode antes:\n"
                 f"  python stages/s06_treino.py --variante {variante}")
    preds = json.loads(arq_preds.read_text())
    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")

    # ---- validacao da MESMA variante para o aviso ----
    vr = DERIVED / f"validacao_resultados{sufixo}.csv"
    if vr.exists():
        dfv = pd.read_csv(vr)
        auc = float(dfv[["auc_rf", "auc_xgb", "auc_ens"]].mean().max())
        ns = len(dfv)
    else:
        print(f"AVISO: {vr.name} nao existe — o aviso do produto sai sem "
              "AUC. Rode o s07 nesta variante.")
        auc, ns = float("nan"), 0
    aviso = AVISO.format(variante=variante, auc=auc, ns=ns, bloco=bloco_km)
    print("===== s08b — MAPA DE FAVORABILIDADE =====")
    print(aviso)
    if variante == "pre_perfuracao":
        print("Variante pre_perfuracao: nenhum controle construtivo entre "
              "os preditores,")
        print("logo o mapa e a AUC acima descrevem as MESMAS condicoes.")

    # ---- poço-padrão: só para os controles que forem preditores ----
    fixos = {}
    if any(k in preds for k in FIXOS_POCO_PADRAO):
        prof_padrao = float(cfg.get("profundidade_padrao_m", 150))
        lf_padrao = float(cfg.get("lf_padrao_m",
                                  m["lf_m"].median(skipna=True)))
        base = dict(FIXOS_POCO_PADRAO)
        base["profundidade_m"] = prof_padrao
        base["lf_m"] = lf_padrao
        fixos = {k: v for k, v in base.items() if k in preds}
        print(f"Poço-padrão: profundidade {prof_padrao:.0f} m, "
              f"intervalo {lf_padrao:.0f} m, trecho aberto")
        print("ATENCAO: a AUC do aviso foi medida com as profundidades "
              "REAIS dos pocos,")
        print("nao com o poco-padrao — a ordenacao deste mapa nao foi "
              "validada nessas")
        print("condicoes. Use a variante pre_perfuracao para o produto.")

    # ---- pilha de covariáveis ----
    pilha, ausentes = {}, []
    for p in preds:
        if p in fixos:
            continue
        a = carregar_raster(p, shape, transform, epsg)
        if a is None:
            ausentes.append(p)
        else:
            med = float(np.nanmedian(a))
            pilha[p] = np.nan_to_num(a, nan=med)
    if ausentes:
        raise RuntimeError(f"Rasters ausentes para predição: {ausentes}")

    rf = joblib.load(MOD / f"rf_cls{sufixo}.joblib")
    xg = joblib.load(MOD / f"xgb_cls{sufixo}.joblib")
    # v3: a predicao monta as colunas na ordem de preds; se o modelo foi
    # ajustado com outra ordem, o erro seria silencioso
    nomes_mod = list(getattr(rf, "feature_names_in_", preds))
    if nomes_mod != list(preds):
        sys.exit("ORDEM DE COVARIAVEIS divergente entre o modelo e "
                 f"preditores{sufixo}.json — regere o s06 nesta variante.")

    # ---- predição em blocos de linhas ----
    n_lin, n_col = shape
    p_rf = np.zeros(shape, dtype="float32")
    p_xg = np.zeros(shape, dtype="float32")
    passo = max(1, int(2e6 / n_col))
    for i0 in range(0, n_lin, passo):
        i1 = min(i0 + passo, n_lin)
        bloco = np.column_stack(
            [np.full(((i1 - i0) * n_col,), fixos[p], dtype="float32")
             if p in fixos else pilha[p][i0:i1].ravel()
             for p in preds])
        p_rf[i0:i1] = rf.predict_proba(bloco)[:, 1].reshape(i1 - i0, n_col)
        p_xg[i0:i1] = xg.predict_proba(bloco)[:, 1].reshape(i1 - i0, n_col)
        print(f"   linhas {i0}–{i1} de {n_lin}", flush=True)

    fav = p_rf
    incerteza = np.abs(p_rf - p_xg)

    # suporte amostral: distância ao poço de treino
    import geopandas as gpd
    g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m.lon, m.lat),
                         crs="EPSG:4674").to_crs(epsg=epsg)
    tree = cKDTree(np.c_[g.geometry.x, g.geometry.y])
    xs = transform.c + (np.arange(n_col) + 0.5) * res
    ys = transform.f - (np.arange(n_lin) + 0.5) * res
    gxx, gyy = np.meshgrid(xs, ys)
    dist, _ = tree.query(np.c_[gxx.ravel(), gyy.ravel()], k=1)
    dist_poco = dist.reshape(shape).astype("float32")

    for nome, arr in [("favorabilidade_prob", fav),
                      ("incerteza_modelos", incerteza),
                      ("dist_poco_treino", dist_poco)]:
        salvar(nome, arr, transform, epsg)
        with rasterio.open(COV / f"{nome}.tif", "r+") as dst:
            # v3: a tag VARIANTE e conferida pelo s08c
            dst.update_tags(
                AVISO=aviso, VARIANTE=variante, MODELO="rf",
                N_PREDITORES=str(len(preds)),
                POCO_PADRAO=("sim" if fixos else "nao_aplicavel"),
                PRODUTO="GroundSight s08b")

    # ---- quicklook com o aviso ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        im0 = axs[0].imshow(fav, cmap="RdYlGn", vmin=0, vmax=1)
        axs[0].set_title("Probabilidade de ALTA favorabilidade "
                         "(poço-padrão, RandomForest)")
        fig.colorbar(im0, ax=axs[0], shrink=.75)
        im1 = axs[1].imshow(incerteza, cmap="magma", vmin=0, vmax=0.5)
        axs[1].set_title("Incerteza (divergência RF × XGB)")
        fig.colorbar(im1, ax=axs[1], shrink=.75)
        for ax in axs:
            ax.axis("off")
        fig.suptitle("MAPA DE FAVORABILIDADE — " + aviso, fontsize=11,
                     color="#1F4E78")
        fig.savefig(OUT / "quicklook_mapa_exploratorio.png", dpi=120,
                    bbox_inches="tight")
        print("Quicklook: data/outputs/quicklook_mapa_exploratorio.png")
    except Exception as e:
        print(f"Quicklook não gerado: {e}")

    print(f"\nDistribuição da probabilidade: "
          f"p10={np.nanpercentile(fav, 10):.2f}  "
          f"med={np.nanmedian(fav):.2f}  "
          f"p90={np.nanpercentile(fav, 90):.2f}")
    print(f"Fração do grid com prob >= 0,5: "
          f"{float((fav >= 0.5).mean()):.1%}")
    print(f"Variante gravada na tag dos rasters: {variante}")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()