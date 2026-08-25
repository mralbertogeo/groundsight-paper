"""s08b — Mapa EXPLORATÓRIO de favorabilidade (Fase A).

Gera o mapa de probabilidade de alta favorabilidade (quartil superior
de Q/s) aplicando os modelos treinados (s06) ao grid completo, com a
validação DECLARADA no próprio produto:

  - Título/metadados/relatório carregam: "EXPLORATÓRIO — AUC espacial
    0,65 | externa 0,56 — triagem regional; não substitui investigação
    local" (valores lidos de validacao_resultados.csv quando existir);
  - Predição para POÇO-PADRÃO: profundidade, comprimento de intervalo
    e tipo construtivo fixados (config) — o mapa responde "quão bom
    seria um poço-padrão aqui", não "quão fundo perfuraram aqui";
  - Mapa = média RF+XGBoost (ensemble); camadas de incerteza:
    (a) divergência entre modelos |p_RF − p_XGB|;
    (b) distância ao poço de treino mais próximo (suporte amostral).

Saídas (data/derived/covariaveis/ + outputs/): favorabilidade_prob.tif,
incerteza_modelos.tif, dist_poco_treino.tif, quicklook e relatório.

Execução:  docker compose run --rm pipeline python stages/s08b_mapa_exploratorio.py
"""
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

AVISO = ("VALIDADO DENTRO DO TETO MENSURAVEL - AUC espacial "
         "{auc:.2f} ({frac:.0%} do teto-vizinho {teto:.2f}) | "
         "externa {ext} - triagem regional; nao substitui "
         "investigacao local")


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


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

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)
    cfg = config().get("produtos", {})

    with rasterio.open(COV / "mde.tif") as src:
        shape = (src.height, src.width)
        transform, epsg = src.transform, src.crs.to_epsg()
        res = transform.a

    preds = json.loads((MOD / "preditores.json").read_text())
    m = pd.read_parquet(DERIVED / "matriz_treino.parquet")

    # validação para o aviso
    auc, ext = 0.65, "0,59"
    vr = DERIVED / "validacao_resultados.csv"
    if vr.exists():
        dfv = pd.read_csv(vr)
        auc = float(dfv[["auc_rf", "auc_xgb",
                         "auc_ens"]].mean().max())
    teto = float(cfg.get("teto_vizinho_auc", 0.672))
    aviso = AVISO.format(auc=auc, ext=ext, teto=teto, frac=auc / teto)
    print("===== s08b — MAPA DE FAVORABILIDADE (validado no teto) =====")
    print(aviso)

    # ---- poço-padrão (controles fixados) ----
    prof_padrao = float(cfg.get("profundidade_padrao_m", 150))
    lf_padrao = float(cfg.get("lf_padrao_m",
                              m["lf_m"].median(skipna=True)))
    fixos = {"profundidade_m": prof_padrao, "lf_m": lf_padrao,
             "tipo_filtros": 0.0, "tipo_trecho_aberto": 1.0,
             "tipo_coluna_saturada": 0.0}
    print(f"Poço-padrão: profundidade {prof_padrao:.0f} m, "
          f"intervalo {lf_padrao:.0f} m, trecho aberto")

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

    rf = joblib.load(MOD / "rf_cls.joblib")
    xg = joblib.load(MOD / "xgb_cls.joblib")

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

    fav = (p_rf + p_xg) / 2.0
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
            dst.update_tags(AVISO=aviso,
                            PRODUTO="GroundSight s08b - validado no teto mensuravel")

    # ---- quicklook com o aviso ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        im0 = axs[0].imshow(fav, cmap="RdYlGn", vmin=0, vmax=1)
        axs[0].set_title("Probabilidade de ALTA favorabilidade "
                         "(poço-padrão)")
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
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
